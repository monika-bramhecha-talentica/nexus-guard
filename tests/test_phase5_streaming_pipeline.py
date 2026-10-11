"""
Phase 5 (FR-5): Unit tests for the streaming guard pipeline and its event log.

Uses the real PII detector (custom patterns; Presidio disabled), real masker,
and real judge (rule-based fallback, since Ollama is not running in CI).
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from src.guardrails.judge import JudgeLLM
from src.guardrails.pii_detector import PIIDetector, PIIEntity
from src.guardrails.pii_masker import PIIMasker
from src.guardrails.stream_pipeline import (
    PipelineConfig,
    StreamingGuardPipeline,
    StreamSession,
)

# Aadhaar-format number that the custom detector catches. Used as a PII canary.
CANARY = "4111 1111 1111"


def make_pipeline(judge=None, **config):
    return StreamingGuardPipeline(
        detector=PIIDetector(use_presidio=False),
        masker=PIIMasker(),
        judge=judge or JudgeLLM(),
        config=PipelineConfig(**config),
    )


async def source(tokens, delay=0.0):
    for token in tokens:
        if delay:
            await asyncio.sleep(delay)
        yield token


async def collect(session, from_seq=0):
    return [event async for event in session.subscribe(from_seq)]


def output_text(events):
    return "".join(e["text"] for e in events if e["type"] == "token")


def types(events):
    return [e["type"] for e in events]


class TestSessionEventLog:
    @pytest.mark.asyncio
    async def test_sequence_numbers_are_contiguous_from_one(self):
        session = StreamSession("s")
        for i in range(5):
            await session.publish("token", {"text": str(i)})
        await session.finish()
        events = await collect(session)
        assert [e["seq"] for e in events] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_subscribe_from_seq_replays_only_later_events(self):
        session = StreamSession("s")
        for i in range(6):
            await session.publish("token", {"text": str(i)})
        await session.finish()
        events = await collect(session, from_seq=4)
        assert [e["seq"] for e in events] == [5, 6]

    @pytest.mark.asyncio
    async def test_live_subscriber_receives_events_as_they_are_published(self):
        session = StreamSession("s")

        async def producer():
            for i in range(3):
                await asyncio.sleep(0.01)
                await session.publish("token", {"text": str(i)})
            await session.finish()

        producer_task = asyncio.create_task(producer())
        events = await collect(session)
        await producer_task
        assert [e["text"] for e in events] == ["0", "1", "2"]

    @pytest.mark.asyncio
    async def test_finished_session_replays_for_late_subscriber(self):
        session = StreamSession("s")
        await session.publish("token", {"text": "a"})
        await session.finish()
        assert [e["text"] for e in await collect(session)] == ["a"]
        assert [e["text"] for e in await collect(session)] == ["a"]


class TestMaskingCorrectness:
    @pytest.mark.asyncio
    async def test_clean_text_is_released_losslessly(self):
        text = "Your order has shipped and will arrive on Friday. Thank you."
        tokens = [w + " " for w in text.split(" ")]
        pipeline = make_pipeline(flush_tokens=3)
        session = StreamSession("s")
        await pipeline.run(session, source(tokens), query="order?")
        events = await collect(session)
        assert output_text(events) == "".join(tokens)

    @pytest.mark.asyncio
    async def test_pii_split_across_tokens_is_masked(self):
        tokens = ["Your Aadhaar ", "4111 ", "1111 ", "1111 ", "is on file."]
        pipeline = make_pipeline(flush_tokens=50)
        session = StreamSession("s")
        await pipeline.run(session, source(tokens))
        out = output_text(await collect(session))
        assert CANARY not in out
        assert "MASKED" in out

    @pytest.mark.asyncio
    async def test_pii_spanning_a_flush_boundary_is_masked(self):
        # flush_tokens=2 forces a flush in the middle of the number.
        tokens = ["Card ", "4111 ", "1111 ", "1111 ", "ends here ok"]
        pipeline = make_pipeline(flush_tokens=2, holdback_words=1)
        session = StreamSession("s")
        await pipeline.run(session, source(tokens))
        out = output_text(await collect(session))
        assert CANARY not in out
        assert "1111" not in out

    @pytest.mark.asyncio
    async def test_output_never_contains_digits_of_detected_entity_in_any_event(self):
        tokens = ["x ", "4111 ", "1111 ", "1111 ", "y ", "z ", "w"]
        pipeline = make_pipeline(flush_tokens=1, holdback_words=0)
        session = StreamSession("s")
        await pipeline.run(session, source(tokens))
        for event in await collect(session):
            if event["type"] == "token":
                assert CANARY not in event["text"]

    @pytest.mark.asyncio
    async def test_judge_receives_masked_text_not_raw_text(self):
        seen = {}

        class RecordingJudge:
            async def evaluate(self, response, query="", agent_id=""):
                seen["text"] = response
                return SimpleNamespace(
                    overall_score=0.9,
                    quality_tier="excellent",
                    evaluation_method="rule-based",
                    dimension_scores=lambda: {"accuracy": 0.9},
                )

            def should_escalate(self, evaluation):
                return False

        pipeline = make_pipeline(judge=RecordingJudge(), flush_tokens=50)
        session = StreamSession("s")
        await pipeline.run(session, source(["id ", "4111 ", "1111 ", "1111 ", "ok"]))
        assert CANARY not in seen["text"]
        assert seen["text"] == output_text(await collect(session))


class TestBufferingBehaviour:
    @pytest.mark.asyncio
    async def test_held_tail_is_released_when_stream_ends(self):
        pipeline = make_pipeline(flush_tokens=50, holdback_words=3)
        session = StreamSession("s")
        await pipeline.run(session, source(["Hi ", "there"]))
        assert output_text(await collect(session)) == "Hi there"

    @pytest.mark.asyncio
    async def test_size_trigger_releases_output_every_n_tokens(self):
        tokens = [f"w{i} " for i in range(20)]
        pipeline = make_pipeline(flush_tokens=5, holdback_words=0, flush_interval_ms=10_000)
        session = StreamSession("s")
        await pipeline.run(session, source(tokens))
        token_events = [e for e in await collect(session) if e["type"] == "token"]
        assert len(token_events) == 4
        assert "".join(e["text"] for e in token_events) == "".join(tokens)

    @pytest.mark.asyncio
    async def test_time_trigger_releases_output_while_agent_is_slow(self):
        """First token must reach the client before the slow agent produces its next token."""
        session = StreamSession("s")
        seen_before_second_token = {}

        async def slow_agent():
            yield "First "
            yield "words "
            await asyncio.sleep(0.3)
            # By now the time trigger should have released something.
            seen_before_second_token["count"] = sum(
                1 for e in session.events if e["type"] == "token"
            )
            yield "later "

        pipeline = make_pipeline(flush_tokens=1000, flush_interval_ms=30, holdback_words=0)
        await pipeline.run(session, slow_agent())
        assert seen_before_second_token["count"] >= 1

    @pytest.mark.asyncio
    async def test_masking_runs_off_the_event_loop(self):
        """A slow masker must not block other coroutines (FR-5.4)."""
        masker = PIIMasker()
        original = masker.mask_text

        def slow_mask(text, entities):
            time.sleep(0.2)  # blocking work
            return original(text, entities)

        masker.mask_text = slow_mask
        pipeline = StreamingGuardPipeline(
            PIIDetector(use_presidio=False), masker, JudgeLLM(),
            PipelineConfig(flush_tokens=2, holdback_words=0),
        )
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        tick_task = asyncio.create_task(ticker())
        session = StreamSession("s")
        await pipeline.run(session, source(["a ", "b ", "c ", "d "]))
        tick_task.cancel()
        assert ticks >= 10  # event loop kept running while masking blocked a thread

    def test_holdback_cut_keeps_trailing_words(self):
        pipeline = make_pipeline(holdback_words=2)
        text = "one two three four"
        assert text[pipeline._holdback_cut(text):] == "three four"

    def test_holdback_cut_returns_zero_when_text_is_short(self):
        pipeline = make_pipeline(holdback_words=3)
        assert pipeline._holdback_cut("only two") == 0

    def test_holdback_zero_releases_everything(self):
        pipeline = make_pipeline(holdback_words=0)
        assert pipeline._holdback_cut("abc def") == len("abc def")

    def test_entity_cut_moves_back_to_entity_start(self):
        entity = PIIEntity(
            entity_type="X", text="abcdef", start=4, end=10, confidence=0.9, source="custom"
        )
        assert StreamingGuardPipeline._adjust_cut_for_entities(7, [entity]) == 4

    def test_entity_cut_unchanged_when_no_entity_crosses(self):
        entity = PIIEntity(
            entity_type="X", text="ab", start=0, end=2, confidence=0.9, source="custom"
        )
        assert StreamingGuardPipeline._adjust_cut_for_entities(5, [entity]) == 5


class TestJudgeGating:
    @pytest.mark.asyncio
    async def test_status_judging_then_final_then_done(self):
        pipeline = make_pipeline()
        session = StreamSession("s")
        await pipeline.run(session, source(["Hello ", "there ", "friend"]))
        assert types(await collect(session))[-3:] == ["status", "final", "done"]

    @pytest.mark.asyncio
    async def test_final_event_carries_verdict_fields(self):
        pipeline = make_pipeline()
        session = StreamSession("s")
        await pipeline.run(session, source(["Hello ", "there"]))
        final = next(e for e in await collect(session) if e["type"] == "final")
        assert final["status"] in ("approved", "escalated")
        assert "overall_score" in final and "quality_tier" in final
        assert isinstance(final["escalate"], bool)

    @pytest.mark.asyncio
    async def test_escalation_status_when_judge_says_escalate(self):
        class EscalatingJudge:
            async def evaluate(self, response, query="", agent_id=""):
                return SimpleNamespace(
                    overall_score=0.2,
                    quality_tier="critical",
                    evaluation_method="rule-based",
                    dimension_scores=lambda: {},
                )

            def should_escalate(self, evaluation):
                return True

        pipeline = make_pipeline(judge=EscalatingJudge())
        session = StreamSession("s")
        await pipeline.run(session, source(["Bad ", "answer"]))
        final = next(e for e in await collect(session) if e["type"] == "final")
        assert final["status"] == "escalated"
        assert final["escalate"] is True

    @pytest.mark.asyncio
    async def test_judge_timeout_holds_stream_fail_closed(self):
        class SlowJudge:
            async def evaluate(self, response, query="", agent_id=""):
                await asyncio.sleep(1.0)

            def should_escalate(self, evaluation):
                return False

        pipeline = make_pipeline(judge=SlowJudge(), judge_timeout_seconds=0.05)
        session = StreamSession("s")
        await pipeline.run(session, source(["Hi ", "there"]))
        final = next(e for e in await collect(session) if e["type"] == "final")
        assert final["status"] == "held"
        assert final["reason"] == "judge_timeout"
        assert final["escalate"] is True

    @pytest.mark.asyncio
    async def test_judge_exception_holds_stream_fail_closed(self):
        class BrokenJudge:
            async def evaluate(self, response, query="", agent_id=""):
                raise RuntimeError("judge exploded")

            def should_escalate(self, evaluation):
                return False

        pipeline = make_pipeline(judge=BrokenJudge())
        session = StreamSession("s")
        await pipeline.run(session, source(["Hi ", "there"]))
        final = next(e for e in await collect(session) if e["type"] == "final")
        assert final["status"] == "held"
        assert "judge exploded" in final["reason"]


class TestFailureAndInstrumentation:
    @pytest.mark.asyncio
    async def test_agent_error_publishes_error_and_no_done(self):
        async def broken_agent():
            yield "partial "
            raise ValueError("agent crashed")

        pipeline = make_pipeline()
        session = StreamSession("s")
        await pipeline.run(session, broken_agent())
        events = await collect(session)
        assert events[-1]["type"] == "error"
        assert events[-1]["code"] == "pipeline_error"
        assert "done" not in types(events)
        assert session.finished

    @pytest.mark.asyncio
    async def test_done_event_contains_timing_metrics(self):
        pipeline = make_pipeline(flush_tokens=2)
        session = StreamSession("s")
        await pipeline.run(session, source(["a ", "b ", "c ", "d ", "e"]))
        done = (await collect(session))[-1]
        assert done["type"] == "done"
        metrics = done["metrics"]
        for key in (
            "first_token_ms", "total_ms", "agent_ms", "mask_ms", "judge_ms",
            "tokens_in", "token_events", "masking_overhead_per_token_ms",
        ):
            assert key in metrics
        assert metrics["tokens_in"] == 5
        assert metrics["first_token_ms"] is not None
        assert metrics["total_ms"] >= metrics["first_token_ms"]

    @pytest.mark.asyncio
    async def test_started_event_is_first(self):
        pipeline = make_pipeline()
        session = StreamSession("s")
        await pipeline.run(session, source(["ok"]))
        first = (await collect(session))[0]
        assert first["type"] == "started"
        assert first["stream_id"] == "s"

    @pytest.mark.asyncio
    async def test_run_returns_timings_object(self):
        pipeline = make_pipeline()
        session = StreamSession("s")
        timings = await pipeline.run(session, source(["ok ", "fine"]))
        assert timings.tokens_in == 2
        assert timings.total_ms > 0
