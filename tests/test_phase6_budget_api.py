"""
Phase 6 (FR-6): Budget behaviour through the WebSocket API and budget endpoint.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from src.api.streaming_server import SessionRegistry, create_app
from src.guardrails.token_budget import (
    BudgetAuditLog,
    BudgetPolicy,
    InMemoryBudgetStore,
    SessionBudgetService,
)
from tests.test_phase5_streaming_pipeline import make_pipeline


def make_source(tokens):
    async def gen(query):
        for token in tokens:
            await asyncio.sleep(0)
            yield token

    return lambda query: gen(query)


def build(tokens, default_budget=50, premium_budget=500):
    policy = BudgetPolicy(
        default_budget=default_budget,
        tier_budgets={"standard": default_budget, "premium": premium_budget},
    )
    service = SessionBudgetService(InMemoryBudgetStore(), policy, BudgetAuditLog())
    app = create_app(
        make_source(tokens),
        make_pipeline(flush_interval_ms=20, holdback_words=0),
        registry=SessionRegistry(),
        budget_service=service,
    )
    return TestClient(app), service


def run_query(client, payload):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "query", **payload})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("done", "error"):
                break
    return events


class TestBudgetThroughWebSocket:
    def test_started_event_reports_budget_session_id(self):
        client, _ = build(["hello there"])
        with client:
            events = run_query(client, {"query": "hi", "session_id": "cust-1"})
        assert events[0]["session_id"] == "cust-1"

    def test_generated_session_id_is_returned_when_not_supplied(self):
        client, _ = build(["hello there"])
        with client:
            events = run_query(client, {"query": "hi"})
        assert events[0]["session_id"]
        assert len(events[0]["session_id"]) == 32

    def test_budget_endpoint_reports_consumption_after_stream(self):
        client, _ = build(["Your order has shipped."])
        with client:
            run_query(client, {"query": "where", "session_id": "cust-2"})
            body = client.get("/budget/cust-2").json()
        assert body["state"] == "ok"
        assert body["used"] > 0
        assert body["remaining"] == body["budget"] - body["used"]

    def test_budget_endpoint_unknown_session(self):
        client, _ = build(["x"])
        with client:
            body = client.get("/budget/never-seen").json()
        assert body["error"] == "session_not_found"

    def test_same_session_budget_is_shared_across_queries(self):
        client, _ = build(["a" * 80], default_budget=20)
        with client:
            first = run_query(client, {"query": "q", "session_id": "shared-1"})
            second = run_query(client, {"query": "q", "session_id": "shared-1"})
            status = client.get("/budget/shared-1").json()
        assert first[-2]["type"] == "final"
        assert first[-2]["reason"] == "budget_exhausted"
        assert second[-2]["reason"] == "budget_exhausted"
        assert status["state"] == "exhausted"

    def test_exhausted_message_is_sent_to_client(self):
        client, _ = build(["b" * 400], default_budget=10)
        with client:
            events = run_query(client, {"query": "q", "session_id": "msg-1"})
        budget_events = [e for e in events if e["type"] == "budget" and e["state"] == "exhausted"]
        assert budget_events[0]["message"] == "Session budget exceeded. Escalating to support team."

    def test_premium_tier_gets_larger_budget(self):
        client, _ = build(["ok"], default_budget=50, premium_budget=500)
        with client:
            run_query(client, {"query": "q", "session_id": "vip", "tier": "premium"})
            body = client.get("/budget/vip").json()
        assert body["budget"] == 500

    def test_org_and_tier_are_recorded_in_audit(self):
        client, service = build(["ok"])
        with client:
            run_query(client, {"query": "q", "session_id": "audited", "org_id": "acme", "tier": "standard"})
        opened = [e for e in service.audit.events_for("audited") if e["event"] == "session_opened"]
        assert opened[0]["org_id"] == "acme"
        assert opened[0]["tier"] == "standard"

    def test_done_status_is_escalated_when_budget_exhausted(self):
        client, _ = build(["c" * 400], default_budget=10)
        with client:
            events = run_query(client, {"query": "q", "session_id": "done-1"})
        assert events[-1]["type"] == "done"
        assert events[-1]["status"] == "escalated"
