# Nexus Guard: Project Structure

## Overview

Nexus Guard is organized into modular components following clean architecture principles. Each module has a single responsibility and can be developed/tested independently.

---

## Directory Tree

```
nexus-guard/
├── src/                           # Main application code
│   ├── __init__.py               # Package initialization
│   │
│   ├── agents/                   # Agent Interface & Implementations
│   │   ├── __init__.py
│   │   ├── base.py              # BaseAgent abstract class, AgentResponse dataclass
│   │   ├── billing_agent.py      # BillingAgent: refund & transaction handling
│   │   ├── support_agent.py      # TechSupportAgent: troubleshooting & escalation
│   │   ├── policy_agent.py       # PolicyAgent: policy validation & eligibility
│   │   └── escalation_agent.py   # EscalationAgent: human escalation routing
│   │
│   ├── orchestrator/              # Core Orchestration Logic
│   │   ├── __init__.py
│   │   ├── core.py              # NexusGuardOrchestrator: main coordinator
│   │   │                         # - Agent registry management
│   │   │                         # - Session state tracking
│   │   │                         # - Token budget enforcement
│   │   │                         # - Audit logging
│   │   └── router.py            # DeterministicRouter: agent routing decisions
│   │
│   ├── guardrails/                # Safety Mechanisms
│   │   ├── __init__.py
│   │   ├── pii_masker.py        # PIIMasker: real-time PII detection & masking
│   │   │                         # - Credit card, Aadhaar, API key masking
│   │   │                         # - Token-level streaming masking
│   │   │                         # - Audit trail logging
│   │   ├── loop_detector.py     # LoopDetector: infinite loop prevention
│   │   │                         # - Direct loop detection (A→B→A)
│   │   │                         # - Circular loop detection (A→B→C→A)
│   │   │                         # - Max hop enforcement (default: 3)
│   │   └── judge.py             # JudgeLLM: response quality evaluation
│   │                             # - Relevance, Factuality, Safety scoring
│   │                             # - Ollama/Mistral 7B integration
│   │                             # - Auto-correction triggering
│   │
│   ├── api/                       # REST & WebSocket API Layer
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application & route definitions
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── websocket_handler.py # WebSocket streaming implementation
│   │   └── endpoints.py         # REST endpoints (health, audit, config)
│   │
│   ├── models/                    # Data Models & Type Definitions
│   │   └── __init__.py
│   │
│   └── utils/                     # Utility Functions
│       ├── __init__.py
│       ├── logging.py            # Structured JSON logging setup
│       ├── metrics.py            # Prometheus metrics collection
│       └── helpers.py            # Helper functions (hashing, validation)
│
├── tests/                         # Unit & Integration Tests
│   ├── __init__.py
│   ├── test_agents.py           # Agent interface & implementation tests
│   ├── test_orchestrator.py      # Orchestrator & routing tests
│   ├── test_loop_detection.py    # Loop detection edge cases
│   ├── test_pii_masking.py       # PII masking accuracy tests
│   ├── test_judge.py            # Judge evaluation tests
│   ├── test_api.py              # API endpoint tests
│   └── fixtures.py              # Shared test fixtures & mocks
│
├── docs/                          # Documentation
│   ├── API_SPECIFICATION.md      # OpenAPI/WebSocket endpoint specs
│   ├── AGENT_GOVERNANCE_DESIGN.md # Agent state machine, rubrics, algorithms
│   ├── ARCHITECTURE.md           # System architecture & data flow diagrams
│   └── EXAMPLES.md              # Usage examples & walkthroughs
│
├── .env.example                   # Environment variables template
├── .dockerignore                  # Docker build exclusions
├── .gitignore                     # Git exclusions
├── config.py                      # Centralized configuration management
├── docker-compose.yml             # Multi-service orchestration
├── Dockerfile                     # Application container image
├── requirements.txt               # Python dependencies
├── PRD.md                        # Product Requirements Document
├── README.md                     # Project README & setup guide
├── PROJECT_STRUCTURE.md          # This file
└── LICENSE                       # Project license
```

---

## Core Modules Explained

### 1. **agents/** - Agent Interface & Implementations

**Responsibility**: Define the contract that all specialized agents must follow, and implement concrete agent logic.

**Key Classes**:
- `BaseAgent`: Abstract base class
  - `execute(query, context, attempt, feedback)` → `AgentResponse`
  - `validate_context(context)` → checks required fields
  - `correct(query, context, feedback)` → auto-correction handler

- `AgentResponse`: Dataclass
  - `agent_id`, `output_text`, `confidence`, `next_agent_hint`
  - `metadata`, `status`, `error_message`, `timestamp`

**Concrete Agents**:
- `BillingAgent`: Refund requests, transaction queries, payment status
- `TechSupportAgent`: Service failure diagnosis, troubleshooting, escalations
- `PolicyAgent`: Policy validation, eligibility checks, escalation criteria
- `EscalationAgent`: Routes complex issues to human support

**Phase 2**: Full agent implementation with LLM integration

---

### 2. **orchestrator/** - Core Routing & Orchestration

**Responsibility**: Central coordinator that manages agent registry, session state, token budgets, and audit trails.

**Key Classes**:
- `NexusGuardOrchestrator`:
  - `register_agent(agent)` → add agent to registry
  - `create_session(user_id, org_id)` → create session state
  - `route_query(session_id, query)` → execute routing logic
  - `check_loop_detection(session_id, agent)` → loop safety check
  - `check_token_budget(session_id, tokens)` → budget enforcement
  - `get_audit_log(session_id)` → retrieve session history

- `SessionState`: Dataclass tracking:
  - Routing path `[Agent1 → Agent2 → ...]`
  - Token consumption per session
  - PII events, loop detection events, judge evaluations
  - Escalation status & reason

- `DeterministicRouter`:
  - `decide_next_agent(query, context)` → routing decision
  - `is_valid_next_step(current, next, path)` → loop prevention

**Phase 1**: Base implementation complete  
**Phase 2**: Full LLM-based routing logic

---

### 3. **guardrails/** - Safety Mechanisms

**Responsibility**: Implement three critical safety guardrails.

#### 3a. **loop_detector.py** - LoopDetector
- Detects infinite routing loops (max 3 hops)
- Loop types:
  - Direct: A → B → A
  - Circular: A → B → C → A
  - Max hop exceeded: >3 hops
- Methods:
  - `detect_loop(routing_path)` → (is_loop, loop_type)
  - `is_valid_next_agent(path, next_agent)` → boolean
  - `record_loop_event(path, type, resolution)` → audit log

**Phase 2**: Full implementation with state machine design

#### 3b. **pii_masker.py** - PIIMasker
- Real-time detection and masking of sensitive data
- Detects:
  - Credit cards (16-digit)
  - Aadhaar numbers (Indian ID)
  - API keys
  - Emails, phone numbers
- Methods:
  - `mask_text(text, user_id)` → sanitized text
  - `mask_streaming_tokens(tokens)` → token-level masking
  - `detect_pii(text)` → list of PII entities
  - `get_audit_log()` → all detection events

**Phase 3**: Full Presidio integration + streaming implementation

#### 3c. **judge.py** - JudgeLLM
- Response quality evaluation against structured rubric
- Scoring dimensions:
  - **Relevance** (0-10): Does it answer the query?
  - **Factuality** (0-10): Are facts accurate?
  - **Safety** (0-10): No PII? No harmful content?
- Methods:
  - `evaluate(response)` → `JudgeVerdictDict`
  - `should_correct(verdict)` → boolean (score < 7.0)
  - `should_escalate(verdict, was_corrected)` → boolean
  - `cache_verdict(hash, verdict)` → Redis caching
  - `get_audit_log()` → all verdicts

**Phase 4**: Full Ollama/Mistral 7B integration

---

### 4. **api/** - REST & WebSocket API Layer

**Responsibility**: Expose orchestrator functionality via HTTP/WebSocket APIs.

**Key Components**:
- `main.py`: FastAPI app initialization, middleware, error handlers
- `schemas.py`: Pydantic models for request/response validation
  - `ChatRequest`, `ChatResponse`, `StreamToken`, etc.
- `websocket_handler.py`: WebSocket streaming logic
  - Token-level streaming with masking/judge async execution
- `endpoints.py`: REST endpoints
  - `GET /health` → liveness/readiness checks
  - `GET /agents` → list available agents
  - `GET /session/{id}/audit` → audit log
  - `POST /session` → create session
  - `WS /ws/chat` → streaming chat endpoint

**Phase 5**: Full streaming implementation

---

### 5. **models/** - Data Models & Type Definitions

**Responsibility**: Centralized type definitions and domain models.

**To be filled in Phase 2+** with specific models for:
- Agent execution contexts
- Routing decisions
- Judge verdicts
- PII events
- etc.

---

### 6. **utils/** - Utility Functions

**Responsibility**: Shared helper functions and infrastructure.

**Key Modules**:
- `logging.py`: Structured JSON logging with audit trail
- `metrics.py`: Prometheus metrics for monitoring
- `helpers.py`: Validation, hashing, serialization helpers

---

### 7. **tests/** - Unit & Integration Tests

**Responsibility**: Comprehensive test coverage for all components.

**Test Modules**:
- `test_agents.py`: Agent interface contract, implementation tests
- `test_orchestrator.py`: Router, session management, budget enforcement
- `test_loop_detection.py`: Edge cases (direct, circular, max-hop loops)
- `test_pii_masking.py`: Detection accuracy, false positive rates
- `test_judge.py`: Evaluation logic, caching, corrections
- `test_api.py`: Endpoint functionality, WebSocket streaming
- `fixtures.py`: Shared test utilities (mock agents, sample queries)

**Phase 7**: Comprehensive test suite with >90% coverage

---

## Data Flow Architecture

```
User Query
    ↓
[API Layer] → WebSocket /ws/chat
    ↓
[Orchestrator] Create/retrieve session
    ↓
[PII Masker] Mask input query
    ↓
[Router] Decide next agent (deterministic)
    ↓
[Agent] Execute (BillingAgent, PolicyAgent, etc.)
    ↓
[Loop Detector] Check for cycles
    ↓
[Judge LLM] Evaluate response (async)
    ↓
[PII Masker] Mask output
    ↓
[Streaming] Token-level response to user
    ↓
[Audit Log] Record session events
    ↓
User Response (with PII masked, quality assured)
```

---

## Development Workflow

### Phase 1: Core Architecture (CURRENT)
- ✅ Initialize project structure
- ✅ Create base classes & interfaces
- ✅ Set up configuration system
- [ ] Create main API app

### Phase 2: Multi-Agent Routing
- [ ] Implement deterministic router
- [ ] Full agent implementations
- [ ] Loop detection algorithms
- [ ] Integration tests

### Phase 3: PII Masking
- [ ] Presidio analyzer integration
- [ ] Custom regex patterns
- [ ] Token-level streaming masking
- [ ] Accuracy tests & benchmarks

### Phase 4: Judge LLM
- [ ] Ollama setup
- [ ] Mistral 7B integration
- [ ] Rubric evaluation logic
- [ ] Auto-correction loops

### Phase 5: Streaming & Latency
- [ ] WebSocket streaming
- [ ] Async PII masking
- [ ] Async judge evaluation
- [ ] Performance testing

### Phase 6: Token Budgeting
- [ ] Budget tracking
- [ ] Cost simulation
- [ ] Graceful degradation
- [ ] Budget enforcement tests

### Phase 7: Integration Testing
- [ ] End-to-end test scenarios
- [ ] Edge case testing
- [ ] Load testing
- [ ] Performance benchmarks

### Phase 8: Documentation
- [ ] API specification
- [ ] Architecture diagrams
- [ ] Design decision log
- [ ] Runbook & operations guide

### Phase 9: Docker & Deployment
- [ ] Dockerfile
- [ ] docker-compose.yml
- [ ] Health checks
- [ ] Graceful shutdown

### Phase 10: Demo & Presentation
- [ ] Recording & editing
- [ ] OneDrive upload
- [ ] GitHub repository setup

---

## Key Design Principles

1. **Modularity**: Each component can be developed/tested independently
2. **Extensibility**: New agents added via `BaseAgent` inheritance
3. **Observability**: Full audit trails and structured logging
4. **Safety**: Three layers of guardrails (loop, PII, quality)
5. **Performance**: Async/streaming architecture for low latency
6. **Determinism**: Reproducible routing paths for auditing

---

## Important Files

| File | Purpose | Phase |
|------|---------|-------|
| `src/agents/base.py` | Agent interface contract | 1 ✅ |
| `src/orchestrator/core.py` | Main orchestrator | 1 ✅ |
| `src/guardrails/loop_detector.py` | Loop prevention | 2 |
| `src/guardrails/pii_masker.py` | PII masking | 3 |
| `src/guardrails/judge.py` | Response evaluation | 4 |
| `src/api/main.py` | FastAPI app | 5 |
| `config.py` | Centralized config | 1 ✅ |
| `docker-compose.yml` | Deployment | 9 |
| `tests/` | Test suite | 7 |

---

**Last Updated**: 2026-09-30  
**Next Step**: Proceed to Phase 1.3 (Base Orchestrator Class completion)
