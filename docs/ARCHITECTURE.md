# Nexus Guard: Architecture and API Reference

This document describes the system as it is implemented today. Where a PRD requirement is not yet met, it is listed under Known Gaps.

## Request flow

Every customer query follows the same path:

1. The client opens `WebSocket /ws/stream` and sends a `query` frame.
2. Input PII is masked before any agent sees the query (FR-3.5). Each detected entity is written to the PII audit log with its type and user.
3. The session budget is opened and the query is projected against it (FR-6).
4. The orchestrator routes the query. The LLM router chooses the first agent. After each agent runs, the next hop comes from that agent's `next_agent_hint`, and the orchestrator validates it (FR-1, FR-2). Loop detection and the hop limit (`max_hops`, default 3) escalate the session when breached.
5. The agent's answer is tokenised and streamed through the guard pipeline. Output PII is masked with a holdback window, so entities are never split across emitted chunks (FR-3.7, FR-5).
6. The budget is charged for input and output. At 80% a warning is sent once. At 100% the output is truncated, the FR-6.6 message is sent, and the stream is escalated without a judge verdict.
7. The LLM judge scores the masked output after streaming (FR-4). If it times out or errors, the verdict is `held`, which fails closed.
8. The stream ends with a `final` event and then a `done` event.

## Modules

| Area | Module | Purpose |
|------|--------|---------|
| API | `src/api/streaming_server.py` | WebSocket streaming, resume, budget endpoint, session registry |
| API | `src/api/assistant.py` | `GuardedAssistant`: input masking, orchestrator call, tokenised answer |
| API | `src/api/app.py` | Production entry point, configured from environment variables |
| Orchestrator | `src/orchestrator/core.py` | Session state, routing, loop detection, hop limit, audit log |
| Orchestrator | `src/orchestrator/router.py` | LLM routing decision and deterministic step validation |
| Agents | `src/agents/*` | Billing, policy, tech support, escalation |
| Guardrails | `src/guardrails/pii_detector.py` | Regex patterns for Aadhaar, card numbers, API keys, and others |
| Guardrails | `src/guardrails/pii_masker.py` | Replaces detected entities with `[MASKED_<TYPE>]` placeholders |
| Guardrails | `src/guardrails/stream_pipeline.py` | Streaming masking, judge gating, budget enforcement |
| Guardrails | `src/guardrails/judge.py` | LLM-as-a-judge scoring |
| Guardrails | `src/guardrails/token_budget.py` | Estimator, policy, in-memory and Redis stores, budget audit |
| Guardrails | `src/guardrails/audit_logger.py` | PII audit events |

## WebSocket protocol (`/ws/stream`)

All frames are JSON text.

Client frames:

- Start a stream: `{"type": "query", "query": "...", "session_id": "optional", "org_id": "optional", "tier": "optional", "agent_id": "optional"}`
- Resume after a disconnect: `{"type": "resume", "stream_id": "...", "from_seq": N}`

Server events, each with an increasing `seq` number (the sequence is replayable):

- `started`: includes `stream_id` and `session_id`
- `token`: a masked output fragment, `text`
- `status`: for example `judging`
- `budget`: `state` is `estimate`, `warning`, or `exhausted`. The exhausted event carries the message "Session budget exceeded. Escalating to support team."
- `final`: `status` is `approved`, `escalated`, or `held`, plus `reason` and `truncated`
- `done`: ends the stream, with metrics

Errors are sent as `{"type": "error", "code": "...", "message": "..."}` and are not part of the sequence. Codes: `bad_json`, `empty_query`, `query_too_long` (limit 4000 characters), `bad_type`, `unknown_stream`, `bad_from_seq`.

If the client disconnects, the pipeline keeps running. The client can resume with the stream ID, and events are replayed from `from_seq`. Finished streams are dropped after a TTL (default 300 seconds).

## REST endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness and count of active streams |
| GET | `/budget/{session_id}` | Budget state: `used`, `budget`, `remaining`, `state` (`ok`, `warning`, `exhausted`), or `session_not_found` |

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `localhost` | Ollama server for routing and the judge |
| `OLLAMA_PORT` | `11434` | Ollama port |
| `OLLAMA_MODEL` | `mistral` | Model name |
| `SESSION_TOKEN_BUDGET` | `5000` | Default per-session budget (FR-6.2) |
| `PREMIUM_TOKEN_BUDGET` | `20000` | Budget for tier `premium` |
| `REDIS_URL` | unset | When set, budgets are shared across instances via Redis (FR-6.7) |
| `BUDGET_AUDIT_FILE` | unset | When set, budget audit events are appended to this JSONL file |
| `MAX_AGENT_HOPS` | `3` | Hop limit used by the default orchestrator configuration |

## Known gaps

- **Masking format.** FR-3.2 specifies `[REDACTED_<TYPE>]`. The masker emits `[MASKED_<TYPE>]`. The end-to-end test for the PRD format is an expected failure until this is decided.
- **Entity coverage without Presidio.** Presidio is not installed in this environment, so names, emails and phone numbers are not detected. Tests run with `use_presidio=False`.
- **API key patterns.** Only `api_key=...` / `api-key: ...` style assignments are detected. Bare tokens such as Stripe-style `sk_live_` keys are not yet detected.
- **Orchestrator budget.** `route_query` uses fixed per-step estimates (100 and 500 tokens). It is not yet connected to `SessionBudgetService`, which drives the stream budget.
- **Authentication.** `org_id` and `tier` are supplied by the client and are not verified.
- **Live judge and router.** Routing and judging were tested with scripted or stubbed LLM responses. Three Phase 2 integration tests need a running Ollama server.
