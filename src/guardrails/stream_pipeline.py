"""
Phase 5 (FR-5) + Phase 6 (FR-6): Streaming guard pipeline.

Pipeline:  Agent token -> buffer -> PII masking -> budget -> client
           (the judge evaluates the masked output once the stream ends; the
            final verdict is published before the stream is marked done)

Design notes:
  - Tokens are concatenated losslessly. Client output == concatenation of
    all "token" event texts.
  - The buffer is released when either `flush_tokens` tokens have arrived
    (FR-5.3: 10-50) or `flush_interval_ms` has passed since the last release.
  - A holdback of `holdback_words` trailing words is never released until a
    later flush, so PII spanning a flush boundary is detected whole. Cut
    points are also moved back so no detected entity is split.
  - Masking runs in a worker thread (asyncio.to_thread) so it never blocks
    token generation (FR-5.4).
  - Budget (FR-6): the query is charged as input. Each released fragment is
    charged as output. When the budget is reached the released text is
    truncated at the limit, the stream stops, a "budget" event carries the
    FR-6.6 message, and the stream is escalated without a judge verdict.
  - The judge evaluates the *masked* text the user actually saw (FR-5.5).
    If the judge fails or times out the stream is held for review (fail closed).
  - Every StreamSession keeps an ordered event log with sequence numbers, so
    a client that drops can reconnect and resume from its last seq (FR-5.8).
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from src.guardrails.token_budget import (
    BUDGET_EXCEEDED_MESSAGE,
    SessionBudgetService,
    TokenEstimator,
)

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")


@dataclass
class PipelineConfig:
    """Tuning knobs for the streaming pipeline."""

    flush_tokens: int = 20  # FR-5.3: buffer 10-50 tokens
    flush_interval_ms: float = 100.0  # release at least this often when tokens are pending
    holdback_words: int = 3  # trailing words kept back for cross-boundary PII
    judge_timeout_seconds: float = 5.0


@dataclass
class PipelineTimings:
    """Latency instrumentation (FR-5.6). All durations in milliseconds."""

    first_token_ms: Optional[float] = None
    total_ms: float = 0.0
    agent_ms: float = 0.0
    mask_ms: float = 0.0
    judge_ms: float = 0.0
    tokens_in: int = 0
    token_events: int = 0
    masking_overhead_per_token_ms: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "first_token_ms": None if self.first_token_ms is None else round(self.first_token_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "agent_ms": round(self.agent_ms, 3),
            "mask_ms": round(self.mask_ms, 3),
            "judge_ms": round(self.judge_ms, 3),
            "tokens_in": self.tokens_in,
            "token_events": self.token_events,
            "masking_overhead_per_token_ms": round(self.masking_overhead_per_token_ms, 4),
        }


@dataclass
class _RunState:
    """Mutable state shared by the flush helpers during one run."""

    emitted_parts: List[str] = field(default_factory=list)
    emitted_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    budget_service: Optional[SessionBudgetService] = None
    budget_session_id: Optional[str] = None
    remaining: Optional[int] = None  # None when no budget is attached
    truncated: bool = False

    @property
    def budget_active(self) -> bool:
        return self.budget_service is not None and self.budget_session_id is not None


class StreamSession:
    """
    Ordered, replayable event log for one streaming response.

    The producer (pipeline) appends events; any number of subscribers can
    read from any sequence number, including after the producer finished.
    Sequence numbers start at 1 and are contiguous.
    """

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.events: List[Dict[str, Any]] = []
        self.finished = False
        self.finished_at: Optional[float] = None
        self.created_at = time.monotonic()
        self.task: Optional[asyncio.Task] = None
        self._cond = asyncio.Condition()

    async def publish(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        event = {"seq": len(self.events) + 1, "type": event_type, **(data or {})}
        self.events.append(event)
        async with self._cond:
            self._cond.notify_all()
        return event

    async def finish(self) -> None:
        self.finished = True
        self.finished_at = time.monotonic()
        async with self._cond:
            self._cond.notify_all()

    async def subscribe(self, from_seq: int = 0) -> AsyncIterator[Dict[str, Any]]:
        """Yield every event with seq > from_seq, waiting for new ones until finished."""
        index = max(0, from_seq)
        while True:
            while index < len(self.events):
                event = self.events[index]
                index += 1
                yield event
            if self.finished:
                return
            async with self._cond:
                while index >= len(self.events) and not self.finished:
                    await self._cond.wait()


class StreamingGuardPipeline:
    """Runs one token stream through buffering, PII masking, budgeting and judging."""

    def __init__(self, detector, masker, judge, config: Optional[PipelineConfig] = None):
        self.detector = detector
        self.masker = masker
        self.judge = judge
        self.config = config or PipelineConfig()

    async def run(
        self,
        session: StreamSession,
        token_source: AsyncIterator[str],
        query: str = "",
        agent_id: str = "agent_unknown",
        budget_service: Optional[SessionBudgetService] = None,
        budget_session_id: Optional[str] = None,
    ) -> PipelineTimings:
        """Consume token_source, publish events into session, and finalize."""
        timings = PipelineTimings()
        state = _RunState(budget_service=budget_service, budget_session_id=budget_session_id)
        pending = ""  # raw text not yet released (holdback + unflushed tokens)
        tokens_since_flush = 0
        interval = self.config.flush_interval_ms / 1000.0
        last_flush = time.perf_counter()
        next_token: Optional[asyncio.Future] = None

        started = {"stream_id": session.stream_id}
        if state.budget_active:
            started["session_id"] = budget_session_id
        await session.publish("started", started)

        try:
            if state.budget_active:
                await self._charge_input(session, state, query)

            if not state.truncated:
                next_token = asyncio.ensure_future(token_source.__anext__())

            while not state.truncated:
                t0 = time.perf_counter()
                remaining = max(0.0, last_flush + interval - t0) if pending else None
                done, _ = await asyncio.wait({next_token}, timeout=remaining)

                if not done:
                    # Time trigger: release what is safe, keep waiting for the agent.
                    pending = await self._flush(session, pending, timings, state, final=False)
                    tokens_since_flush = 0
                    last_flush = time.perf_counter()
                    continue

                timings.agent_ms += (time.perf_counter() - t0) * 1000
                try:
                    token = next_token.result()
                except StopAsyncIteration:
                    next_token = None
                    break
                next_token = asyncio.ensure_future(token_source.__anext__())

                timings.tokens_in += 1
                pending += token
                tokens_since_flush += 1

                if tokens_since_flush >= self.config.flush_tokens:
                    pending = await self._flush(session, pending, timings, state, final=False)
                    tokens_since_flush = 0
                    last_flush = time.perf_counter()

            if next_token is not None and not next_token.done():
                next_token.cancel()

            if state.truncated:
                return await self._finish_budget_exhausted(session, timings, state)

            pending = await self._flush(session, pending, timings, state, final=True)
            if state.truncated:
                return await self._finish_budget_exhausted(session, timings, state)

            await session.publish("status", {"state": "judging"})
            verdict = await self._judge(timings, "".join(state.emitted_parts), query, agent_id)
            await session.publish("final", verdict)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            logger.exception("Streaming pipeline failed for %s", session.stream_id)
            if next_token is not None and not next_token.done():
                next_token.cancel()
            await session.publish("error", {"code": "pipeline_error", "message": str(exc)})
            timings.total_ms = (time.perf_counter() - timings.started_at) * 1000
            await session.finish()
            return timings

        timings.total_ms = (time.perf_counter() - timings.started_at) * 1000
        if timings.tokens_in:
            timings.masking_overhead_per_token_ms = timings.mask_ms / timings.tokens_in
        await session.publish("done", {"metrics": timings.to_dict(), "status": verdict["status"]})
        await session.finish()
        return timings

    # ------------------------------------------------------------------ budget

    async def _charge_input(self, session: StreamSession, state: _RunState, query: str) -> None:
        tokens = TokenEstimator.tokens_for_chars(len(query))
        decision = await state.budget_service.charge(state.budget_session_id, tokens, "input")
        state.input_tokens = tokens
        state.remaining = decision.remaining

        estimate = TokenEstimator.project_request(query)
        if estimate:
            await session.publish("budget", {"state": "estimate", **estimate, "remaining": decision.remaining})
            state.budget_service.record_event(
                state.budget_session_id, "estimate", **estimate
            )

        await self._publish_threshold_events(session, state, decision)
        if state.remaining <= 0:
            state.truncated = True

    async def _charge_output(self, session: StreamSession, state: _RunState, masked: str) -> str:
        """Return the part of `masked` that fits in the remaining budget; charge for it."""
        if not state.budget_active:
            return masked

        # Output may use whatever the budget has left after input and earlier output.
        max_output_chars = (state.output_tokens + state.remaining) * TokenEstimator.CHARS_PER_TOKEN
        allowed = max(0, max_output_chars - state.emitted_chars)
        if len(masked) > allowed:
            masked = masked[:allowed]
            state.truncated = True

        new_output_tokens = TokenEstimator.tokens_for_chars(state.emitted_chars + len(masked))
        delta = new_output_tokens - state.output_tokens
        if delta > 0:
            decision = await state.budget_service.charge(state.budget_session_id, delta, "output")
            state.output_tokens = new_output_tokens
            state.remaining = decision.remaining
            await self._publish_threshold_events(session, state, decision)
        if state.remaining is not None and state.remaining <= 0 and masked:
            # Budget spent exactly on this fragment: no further fragments may be released.
            state.truncated = True
        return masked

    async def _publish_threshold_events(self, session, state: _RunState, decision) -> None:
        if decision.newly_warned:
            await session.publish(
                "budget",
                {"state": "warning", "used": decision.used, "budget": decision.budget},
            )
        if decision.newly_exhausted:
            state.budget_service.record_event(
                state.budget_session_id, "budget_reached", used=decision.used, budget=decision.budget
            )

    async def _finish_budget_exhausted(
        self, session: StreamSession, timings: PipelineTimings, state: _RunState
    ) -> PipelineTimings:
        used = state.input_tokens + state.output_tokens
        budget = used + (state.remaining or 0)
        await session.publish(
            "budget",
            {"state": "exhausted", "message": BUDGET_EXCEEDED_MESSAGE, "used": used, "budget": budget},
        )
        state.budget_service.record_event(
            state.budget_session_id, "truncated", used=used, budget=budget,
            output_chars=state.emitted_chars,
        )
        state.budget_service.record_event(
            state.budget_session_id, "escalation", reason="budget_exhausted"
        )
        await session.publish(
            "final",
            {
                "status": "escalated",
                "reason": "budget_exhausted",
                "escalate": True,
                "truncated": True,
                "message": BUDGET_EXCEEDED_MESSAGE,
                "overall_score": None,
                "quality_tier": None,
            },
        )
        timings.total_ms = (time.perf_counter() - timings.started_at) * 1000
        if timings.tokens_in:
            timings.masking_overhead_per_token_ms = timings.mask_ms / timings.tokens_in
        await session.publish("done", {"metrics": timings.to_dict(), "status": "escalated"})
        await session.finish()
        return timings

    # ------------------------------------------------------------- masking

    async def _flush(
        self,
        session: StreamSession,
        pending: str,
        timings: PipelineTimings,
        state: _RunState,
        final: bool,
    ) -> str:
        """Mask and release everything in `pending` that is safe to release. Returns new pending."""
        if not pending:
            return ""

        t0 = time.perf_counter()
        entities = await asyncio.to_thread(self.detector.detect, pending)

        if final:
            # Stream is over: everything is released, nothing is held back.
            head, head_entities, rest = pending, entities, ""
        else:
            cut = self._holdback_cut(pending)
            # Never split an entity across the release boundary.
            cut = self._adjust_cut_for_entities(cut, entities)
            if cut == 0:
                timings.mask_ms += (time.perf_counter() - t0) * 1000
                return pending
            head = pending[:cut]
            head_entities = [e for e in entities if e.end <= cut]
            rest = pending[cut:]

        masked = await asyncio.to_thread(self.masker.mask_text, head, head_entities)
        timings.mask_ms += (time.perf_counter() - t0) * 1000

        masked = await self._charge_output(session, state, masked)

        if masked:
            if timings.first_token_ms is None:
                timings.first_token_ms = (time.perf_counter() - timings.started_at) * 1000
            state.emitted_parts.append(masked)
            state.emitted_chars += len(masked)
            timings.token_events += 1
            await session.publish("token", {"text": masked})
        return rest

    def _holdback_cut(self, text: str) -> int:
        """Index before which text may be released, keeping the last N words back."""
        keep = self.config.holdback_words
        if keep <= 0:
            return len(text)
        matches = list(_WORD_RE.finditer(text))
        if len(matches) <= keep:
            return 0
        return matches[-keep].start()

    @staticmethod
    def _adjust_cut_for_entities(cut: int, entities) -> int:
        changed = True
        while changed:
            changed = False
            for entity in entities:
                if entity.start < cut < entity.end:
                    cut = entity.start
                    changed = True
        return cut

    # -------------------------------------------------------------- judge

    async def _judge(
        self, timings: PipelineTimings, masked_output: str, query: str, agent_id: str
    ) -> Dict[str, Any]:
        """Judge the masked output. Fails closed: errors or timeouts hold the stream."""
        t0 = time.perf_counter()
        try:
            evaluation = await asyncio.wait_for(
                self.judge.evaluate(masked_output, query=query, agent_id=agent_id),
                timeout=self.config.judge_timeout_seconds,
            )
        except asyncio.TimeoutError:
            timings.judge_ms = (time.perf_counter() - t0) * 1000
            return self._held("judge_timeout")
        except Exception as exc:  # noqa: BLE001
            timings.judge_ms = (time.perf_counter() - t0) * 1000
            logger.warning("Judge failed, holding stream: %s", exc)
            return self._held(f"judge_error: {exc}")
        timings.judge_ms = (time.perf_counter() - t0) * 1000

        escalate = bool(self.judge.should_escalate(evaluation))
        return {
            "status": "escalated" if escalate else "approved",
            "reason": None,
            "escalate": escalate,
            "overall_score": round(evaluation.overall_score, 4),
            "quality_tier": evaluation.quality_tier,
            "evaluation_method": evaluation.evaluation_method,
            "dimensions": evaluation.dimension_scores(),
        }

    @staticmethod
    def _held(reason: str) -> Dict[str, Any]:
        return {
            "status": "held",
            "reason": reason,
            "escalate": True,
            "overall_score": None,
            "quality_tier": None,
        }
