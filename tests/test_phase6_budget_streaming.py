"""
Phase 6 (FR-6): Budget enforcement inside the streaming pipeline.

Covers the PRD acceptance example: "Repeat 'VERIFY' 10,000 times, then check
TXN-90811 status." with a 5,000-token budget:
  - the request is projected at 17,500 tokens before generation
  - a warning is issued at 80% (4,000 tokens)
  - output is truncated at the budget, the FR-6.6 message is sent, and the
    stream is escalated with no judge verdict
  - the audit trail records consumption, warning, truncation and escalation
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.guardrails.stream_pipeline import PipelineConfig, StreamingGuardPipeline, StreamSession
from src.guardrails.token_budget import (
    BUDGET_EXCEEDED_MESSAGE,
    BudgetAuditLog,
    BudgetPolicy,
    InMemoryBudgetStore,
    SessionBudgetService,
    TokenEstimator,
)
from tests.test_phase5_streaming_pipeline import CANARY, make_pipeline


class CountingJudge:
    def __init__(self):
        self.calls = 0

    async def evaluate(self, response, query="", agent_id=""):
        self.calls += 1
        return SimpleNamespace(
            overall_score=0.9,
            quality_tier="excellent",
            evaluation_method="rule-based",
            dimension_scores=lambda: {},
        )

    def should_escalate(self, evaluation):
        return False


def budget_service(budget=5000, warn_fraction=0.8):
    policy = BudgetPolicy(default_budget=budget, warn_fraction=warn_fraction)
    return SessionBudgetService(InMemoryBudgetStore(), policy, BudgetAuditLog())


class CountingSource:
    """Async token source that records how many tokens were pulled from the agent."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pulled = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.pulled >= len(self.tokens):
            raise StopAsyncIteration
        token = self.tokens[self.pulled]
        self.pulled += 1
        await asyncio.sleep(0)
        return token


def verify_tokens(repetitions):
    return ["VERIFY "] * repetitions


async def run_stream(pipeline, tokens, query, service, sid):
    session = StreamSession(f"stream-{sid}")
    await service.open(sid)
    source = CountingSource(tokens)
    timings = await pipeline.run(
        session, source, query=query, agent_id="billing_agent",
        budget_service=service, budget_session_id=sid,
    )
    return session, source, timings


def events_of(session, kind):
    return [e for e in session.events if e["type"] == kind]


class TestPrdRepetitionScenario:
    @pytest.mark.asyncio
    async def test_request_is_projected_before_generation(self):
        judge = CountingJudge()
        service = budget_service(budget=5000)
        pipeline = make_pipeline(judge=judge, flush_tokens=50)
        query = "Repeat 'VERIFY' 10,000 times, then check TXN-90811 status."
        session, _, _ = await run_stream(pipeline, verify_tokens(20), query, service, "s1")

        estimate = events_of(session, "budget")[0]
        assert estimate["state"] == "estimate"
        assert estimate["projected_tokens"] == 17500
        assert estimate["repetitions"] == 10000

    @pytest.mark.asyncio
    async def test_warning_is_issued_once_at_eighty_percent(self):
        service = budget_service(budget=5000)
        pipeline = make_pipeline(judge=CountingJudge(), flush_tokens=50, holdback_words=0)
        query = "Repeat 'VERIFY' 10,000 times"
        session, _, _ = await run_stream(pipeline, verify_tokens(10000), query, service, "s2")

        warnings = [e for e in events_of(session, "budget") if e["state"] == "warning"]
        assert len(warnings) == 1
        assert warnings[0]["used"] >= 4000
        assert warnings[0]["budget"] == 5000

    @pytest.mark.asyncio
    async def test_output_is_truncated_at_budget_with_fr66_message(self):
        service = budget_service(budget=5000)
        pipeline = make_pipeline(judge=CountingJudge(), flush_tokens=50, holdback_words=0)
        query = "Repeat 'VERIFY' 10,000 times"
        session, _, _ = await run_stream(pipeline, verify_tokens(10000), query, service, "s3")

        exhausted = [e for e in events_of(session, "budget") if e["state"] == "exhausted"]
        assert len(exhausted) == 1
        assert exhausted[0]["message"] == BUDGET_EXCEEDED_MESSAGE
        assert exhausted[0]["message"] == "Session budget exceeded. Escalating to support team."

        delivered = "".join(e["text"] for e in events_of(session, "token"))
        assert TokenEstimator.estimate_text(delivered) <= 5000
        assert len(delivered) < 10000 * 7  # far less than the full repetition

    @pytest.mark.asyncio
    async def test_stream_stops_pulling_from_agent_after_budget(self):
        service = budget_service(budget=500)
        pipeline = make_pipeline(judge=CountingJudge(), flush_tokens=10, holdback_words=0)
        session, source, _ = await run_stream(
            pipeline, verify_tokens(10000), "Repeat 'VERIFY' 10,000 times", service, "s4"
        )
        assert source.pulled < 10000
        assert source.pulled <= 500 * 4  # cannot pull far beyond what the budget allows

    @pytest.mark.asyncio
    async def test_exhausted_stream_is_escalated_without_judge_verdict(self):
        judge = CountingJudge()
        service = budget_service(budget=5000)
        pipeline = make_pipeline(judge=judge, flush_tokens=50, holdback_words=0)
        session, _, _ = await run_stream(
            pipeline, verify_tokens(10000), "Repeat 'VERIFY' 10,000 times", service, "s5"
        )
        final = events_of(session, "final")[-1]
        assert final["status"] == "escalated"
        assert final["reason"] == "budget_exhausted"
        assert final["escalate"] is True
        assert final["truncated"] is True
        assert judge.calls == 0
        assert events_of(session, "done")[-1]["status"] == "escalated"

    @pytest.mark.asyncio
    async def test_audit_trail_records_consumption_warning_truncation_escalation(self):
        service = budget_service(budget=5000)
        pipeline = make_pipeline(judge=CountingJudge(), flush_tokens=50, holdback_words=0)
        await run_stream(pipeline, verify_tokens(10000), "Repeat 'VERIFY' 10,000 times", service, "s6")

        kinds = [e["event"] for e in service.audit.events_for("s6")]
        assert "session_opened" in kinds
        assert "estimate" in kinds
        assert "charge" in kinds
        assert "warning" in kinds
        assert "exhausted" in kinds
        assert "truncated" in kinds
        assert "escalation" in kinds
        assert kinds.index("warning") < kinds.index("truncated") < kinds.index("escalation")

    @pytest.mark.asyncio
    async def test_status_after_stream_never_exceeds_budget(self):
        service = budget_service(budget=5000)
        pipeline = make_pipeline(judge=CountingJudge(), flush_tokens=50, holdback_words=0)
        await run_stream(pipeline, verify_tokens(10000), "Repeat 'VERIFY' 10,000 times", service, "s7")
        status = await service.status("s7")
        assert status["used"] <= 5000
        assert status["state"] == "exhausted"


class TestWithinBudget:
    @pytest.mark.asyncio
    async def test_normal_stream_within_budget_completes_and_is_judged(self):
        judge = CountingJudge()
        service = budget_service(budget=5000)
        pipeline = make_pipeline(judge=judge, flush_tokens=3, holdback_words=0)
        tokens = ["Your order has shipped. ", "It arrives Friday. "]
        session, _, _ = await run_stream(pipeline, tokens, "where is my order", service, "w1")

        assert events_of(session, "budget") == []
        assert events_of(session, "final")[-1]["status"] == "approved"
        assert judge.calls == 1
        delivered = "".join(e["text"] for e in events_of(session, "token"))
        assert delivered == "".join(tokens)

    @pytest.mark.asyncio
    async def test_output_charged_matches_delivered_text(self):
        service = budget_service(budget=5000)
        pipeline = make_pipeline(judge=CountingJudge(), flush_tokens=2, holdback_words=0)
        tokens = ["abcd " for _ in range(10)]  # 50 chars
        session, _, _ = await run_stream(pipeline, tokens, "hi", service, "w2")
        delivered = "".join(e["text"] for e in events_of(session, "token"))
        input_tokens = TokenEstimator.tokens_for_chars(len("hi"))
        expected = input_tokens + TokenEstimator.tokens_for_chars(len(delivered))
        assert (await service.status("w2"))["used"] == expected

    @pytest.mark.asyncio
    async def test_pii_stays_masked_even_when_budget_truncates(self):
        service = budget_service(budget=40)
        pipeline = make_pipeline(judge=CountingJudge(), flush_tokens=2, holdback_words=1)
        tokens = ["Aadhaar ", "4111 ", "1111 ", "1111 ", "then "] + ["more text "] * 50
        session, _, _ = await run_stream(pipeline, tokens, "x", service, "w3")
        delivered = "".join(e["text"] for e in events_of(session, "token"))
        assert CANARY not in delivered

    @pytest.mark.asyncio
    async def test_input_alone_over_budget_exhausts_before_any_generation(self):
        service = budget_service(budget=5)
        pipeline = make_pipeline(judge=CountingJudge(), flush_tokens=2)
        long_query = "q" * 200  # 50 tokens of input
        session, source, _ = await run_stream(pipeline, ["never"], long_query, service, "w4")
        assert source.pulled == 0
        assert events_of(session, "token") == []
        assert events_of(session, "final")[-1]["status"] == "escalated"

    @pytest.mark.asyncio
    async def test_session_budget_is_shared_across_queries(self):
        service = budget_service(budget=30)
        pipeline = make_pipeline(judge=CountingJudge(), flush_tokens=2, holdback_words=0)
        # 3 x 40 chars = 30 output tokens; with the 1-token query that exceeds 30.
        first, _, _ = await run_stream(pipeline, ["a" * 40] * 3, "q", service, "shared")
        assert events_of(first, "final")[-1]["reason"] == "budget_exhausted"

        second_session = StreamSession("second")
        source = CountingSource(["fresh text"])
        await pipeline.run(
            second_session, source, query="again", agent_id="x",
            budget_service=service, budget_session_id="shared",
        )
        assert source.pulled == 0
        assert events_of(second_session, "final")[-1]["reason"] == "budget_exhausted"
