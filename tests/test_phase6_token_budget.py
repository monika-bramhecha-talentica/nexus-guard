"""
Phase 6 (FR-6): Unit tests for token estimation, policy, stores, audit and the budget service.

The Redis store tests run against a live redis-server on localhost:6379 and are
skipped when none is reachable.
"""

import asyncio
import json
import socket
import uuid

import pytest

from src.guardrails import token_budget as tb
from src.guardrails.token_budget import (
    BudgetAuditLog,
    BudgetPolicy,
    InMemoryBudgetStore,
    RedisBudgetStore,
    SessionBudgetService,
    TokenEstimator,
)


def _redis_available() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 6379), timeout=0.5):
            return True
    except OSError:
        return False


# ----------------------------------------------------------------- estimator

class TestTokenEstimator:
    def test_four_chars_is_one_token_rounded_up(self):
        assert TokenEstimator.tokens_for_chars(0) == 0
        assert TokenEstimator.tokens_for_chars(4) == 1
        assert TokenEstimator.tokens_for_chars(5) == 2

    def test_negative_chars_are_treated_as_zero(self):
        assert TokenEstimator.tokens_for_chars(-10) == 0

    def test_prd_example_projects_ten_thousand_verify_repetitions(self):
        # "VERIFY " is 7 characters, 10,000 times = 70,000 chars = 17,500 tokens.
        projection = TokenEstimator.project_request("Repeat 'VERIFY' 10,000 times, then check TXN-90811 status.")
        assert projection == {"unit": "VERIFY", "repetitions": 10000, "projected_tokens": 17500}

    def test_projection_handles_unquoted_unit(self):
        projection = TokenEstimator.project_request("repeat hello 3 times")
        assert projection["unit"] == "hello"
        assert projection["repetitions"] == 3

    def test_no_projection_for_ordinary_query(self):
        assert TokenEstimator.project_request("Where is my refund?") is None

    def test_no_projection_for_empty_query(self):
        assert TokenEstimator.project_request("") is None
        assert TokenEstimator.project_request(None) is None


# -------------------------------------------------------------------- policy

class TestBudgetPolicy:
    def test_default_budget_is_five_thousand(self):
        assert BudgetPolicy().budget_for() == 5000

    def test_tier_budget_applies(self):
        policy = BudgetPolicy(tier_budgets={"premium": 20000})
        assert policy.budget_for(tier="premium") == 20000

    def test_org_budget_overrides_tier(self):
        policy = BudgetPolicy(tier_budgets={"premium": 20000}, org_budgets={"acme": 1000})
        assert policy.budget_for(org_id="acme", tier="premium") == 1000

    def test_unknown_tier_falls_back_to_default(self):
        assert BudgetPolicy().budget_for(tier="platinum") == 5000

    def test_warning_threshold_is_eighty_percent(self):
        assert BudgetPolicy().warn_at(5000) == 4000

    def test_cost_is_simulated_per_thousand_tokens(self):
        policy = BudgetPolicy(usd_per_1k_tokens=0.002)
        assert policy.cost_usd(5000) == pytest.approx(0.01)
        assert policy.cost_usd(0) == 0.0


# --------------------------------------------------------------------- audit

class TestBudgetAuditLog:
    def test_records_and_filters_by_session(self):
        log = BudgetAuditLog()
        log.record("s1", "charge", tokens=5)
        log.record("s2", "charge", tokens=7)
        assert [e["tokens"] for e in log.events_for("s1")] == [5]
        assert len(log.all_events()) == 2

    def test_entries_carry_timestamp_and_event_name(self):
        entry = BudgetAuditLog().record("s1", "warning", used=4000)
        assert entry["event"] == "warning"
        assert entry["timestamp"].endswith("Z")

    def test_mirrors_to_json_lines_file(self, tmp_path):
        path = tmp_path / "budget.jsonl"
        log = BudgetAuditLog(path=str(path))
        log.record("s1", "exhausted", used=5000)
        lines = path.read_text().strip().splitlines()
        assert json.loads(lines[0])["event"] == "exhausted"


# ------------------------------------------------------------ in-memory store

class TestInMemoryBudgetStore:
    @pytest.mark.asyncio
    async def test_init_creates_once(self):
        store = InMemoryBudgetStore()
        assert await store.init("s", 100, 60) is True
        assert await store.init("s", 999, 60) is False
        assert (await store.read("s"))["budget"] == 100

    @pytest.mark.asyncio
    async def test_incr_accumulates(self):
        store = InMemoryBudgetStore()
        await store.init("s", 100, 60)
        assert await store.incr_used("s", 30, 60) == 30
        assert await store.incr_used("s", 12, 60) == 42

    @pytest.mark.asyncio
    async def test_incr_unknown_session_raises(self):
        with pytest.raises(KeyError):
            await InMemoryBudgetStore().incr_used("missing", 1, 60)

    @pytest.mark.asyncio
    async def test_flags_round_trip(self):
        store = InMemoryBudgetStore()
        await store.init("s", 100, 60)
        await store.set_flag("s", "warned", True, 60)
        assert (await store.read("s"))["warned"] is True

    @pytest.mark.asyncio
    async def test_session_expires_after_ttl(self, monkeypatch):
        clock = {"now": 1000.0}
        monkeypatch.setattr(tb.time, "monotonic", lambda: clock["now"])
        store = InMemoryBudgetStore()
        await store.init("s", 100, ttl=10)
        clock["now"] += 11
        assert await store.read("s") is None


# ---------------------------------------------------------------- redis store

@pytest.mark.skipif(not _redis_available(), reason="redis-server not running on localhost:6379")
class TestRedisBudgetStore:
    @pytest.fixture
    def store(self):
        import redis.asyncio as redis

        client = redis.from_url("redis://127.0.0.1:6379/0")
        return RedisBudgetStore(client, prefix=f"test:{uuid.uuid4().hex}:")

    @pytest.mark.asyncio
    async def test_init_is_idempotent_and_readable(self, store):
        assert await store.init("s", 5000, 60) is True
        assert await store.init("s", 1, 60) is False
        record = await store.read("s")
        assert record == {"used": 0, "budget": 5000, "warned": False, "exhausted": False}
        await store.close()

    @pytest.mark.asyncio
    async def test_incr_is_atomic_under_concurrency(self, store):
        await store.init("s", 10**9, 60)
        await asyncio.gather(*[store.incr_used("s", 1, 60) for _ in range(200)])
        assert (await store.read("s"))["used"] == 200
        await store.close()

    @pytest.mark.asyncio
    async def test_flags_persist(self, store):
        await store.init("s", 100, 60)
        await store.set_flag("s", "exhausted", True, 60)
        assert (await store.read("s"))["exhausted"] is True
        await store.close()

    @pytest.mark.asyncio
    async def test_key_has_ttl(self, store):
        await store.init("s", 100, 120)
        ttl = await store.client.ttl(store._key("s"))
        assert 0 < ttl <= 120
        await store.close()

    @pytest.mark.asyncio
    async def test_incr_unknown_session_raises(self, store):
        with pytest.raises(KeyError):
            await store.incr_used("nope", 1, 60)
        await store.close()


# ------------------------------------------------------------ budget service

class TestSessionBudgetService:
    def _service(self, budget=100, warn_fraction=0.8):
        policy = BudgetPolicy(default_budget=budget, warn_fraction=warn_fraction)
        return SessionBudgetService(InMemoryBudgetStore(), policy, BudgetAuditLog())

    @pytest.mark.asyncio
    async def test_open_reports_fresh_state(self):
        service = self._service(budget=100)
        status = await service.open("s")
        assert status["state"] == "ok"
        assert status["used"] == 0
        assert status["remaining"] == 100

    @pytest.mark.asyncio
    async def test_warning_fires_once_at_threshold(self):
        service = self._service(budget=100)
        await service.open("s")
        first = await service.charge("s", 79, "output")
        assert first.newly_warned is False
        crossing = await service.charge("s", 1, "output")
        assert crossing.newly_warned is True
        again = await service.charge("s", 5, "output")
        assert again.newly_warned is False
        warnings = [e for e in service.audit.events_for("s") if e["event"] == "warning"]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_exhaustion_fires_once_at_budget(self):
        service = self._service(budget=10)
        await service.open("s")
        decision = await service.charge("s", 10, "output")
        assert decision.newly_exhausted is True
        assert decision.remaining == 0
        await service.charge("s", 1, "output")
        exhausted = [e for e in service.audit.events_for("s") if e["event"] == "exhausted"]
        assert len(exhausted) == 1

    @pytest.mark.asyncio
    async def test_status_states_progress(self):
        service = self._service(budget=10)
        await service.open("s")
        assert (await service.status("s"))["state"] == "ok"
        await service.charge("s", 8, "output")
        assert (await service.status("s"))["state"] == "warning"
        await service.charge("s", 2, "output")
        assert (await service.status("s"))["state"] == "exhausted"

    @pytest.mark.asyncio
    async def test_status_includes_simulated_cost(self):
        service = self._service(budget=10000)
        await service.open("s")
        await service.charge("s", 2000, "output")
        status = await service.status("s")
        assert status["simulated_cost_usd"] == service.policy.cost_usd(2000)
        assert status["percent_used"] == 20.0

    @pytest.mark.asyncio
    async def test_unknown_session_status_is_error(self):
        service = self._service()
        assert (await service.status("missing"))["error"] == "session_not_found"

    @pytest.mark.asyncio
    async def test_charge_unknown_session_raises(self):
        with pytest.raises(KeyError):
            await self._service().charge("missing", 1, "output")

    @pytest.mark.asyncio
    async def test_open_is_idempotent_and_audits_once(self):
        service = self._service()
        await service.open("s")
        await service.open("s")
        opened = [e for e in service.audit.events_for("s") if e["event"] == "session_opened"]
        assert len(opened) == 1

    @pytest.mark.asyncio
    async def test_every_charge_is_audited_with_reason(self):
        service = self._service()
        await service.open("s")
        await service.charge("s", 3, "input")
        await service.charge("s", 4, "output")
        charges = [e for e in service.audit.events_for("s") if e["event"] == "charge"]
        assert [(c["reason"], c["tokens"]) for c in charges] == [("input", 3), ("output", 4)]
        assert charges[-1]["used"] == 7
