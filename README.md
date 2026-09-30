# Nexus Guard
## Enterprise Agentic Workflow & Guardrail System

**Version**: 1.0.0  
**Status**: Active Development (Phase 1)  
**Author**: Monika B (v-monikab@affle.com)

---

## Quick Start

### What is Nexus Guard?

Nexus Guard is an orchestrator framework that safely enables organizations to deploy autonomous multi-agent customer operations networks. It solves three critical failures in early agentic deployments:

1. **Multi-hop Hallucination** → Prevents infinite agent routing loops (max 3 hops)
2. **Data Privacy Breach** → Real-time PII masking (credit cards, Aadhaar, API keys)
3. **Quality Assurance Gap** → LLM judge evaluation with auto-correction

---

## System Architecture

```
User Query
    ↓
[PII Masking] Sanitize input
    ↓
[Orchestrator] Route to agent
    ↓
[Agent] Execute (Billing, Support, Policy)
    ↓
[Loop Detector] Check for cycles
    ↓
[Judge LLM] Evaluate quality
    ↓
[PII Masking] Sanitize output
    ↓
[Streaming API] Return to user
    ↓
User Response ✅
```

---

## Hardware Requirements

### Minimum (Development)
- **CPU**: 4 vCPU (x86-64)
- **RAM**: 8 GB
- **Storage**: 10 GB (for models, logs, Docker)
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Recommended (Production)
- **CPU**: 8 vCPU
- **RAM**: 16 GB (Mistral 7B model + Redis + App)
- **Storage**: 20 GB
- **GPU**: Optional (for faster Mistral inference, but not required)

### Software Requirements
- **Docker** >= 20.10
- **Docker Compose** >= 1.29
- **Python** >= 3.11 (for local development)
- **Git**

---

## Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/nexus-guard.git
cd nexus-guard
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano .env
```

**Key settings to configure**:
- `OLLAMA_HOST`: URL to Ollama server (default: http://ollama:11434)
- `REDIS_HOST`: Redis hostname (default: redis in docker-compose)
- `DEFAULT_TOKEN_BUDGET`: Token limit per session (default: 5000)
- `MAX_AGENT_HOPS`: Max agent hops before escalation (default: 3)

### Step 3: Start Services (Docker Compose)

```bash
# Start all services (App + Ollama + Redis)
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f app
docker-compose logs -f ollama
docker-compose logs -f redis
```

**Services Started**:
- **App**: http://localhost:8000 (FastAPI)
- **Ollama**: http://localhost:11434 (LLM inference)
- **Redis**: localhost:6379 (Session cache)

### Step 4: Verify Setup

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "1.0.0"}

# List agents
curl http://localhost:8000/agents

# Expected response:
# {
#   "agents": [
#     {"name": "BillingAgent", "description": "..."},
#     ...
#   ]
# }
```

---

## Development Setup (Local)

### Install Dependencies

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for PII detection)
python -m spacy download en_core_web_sm
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_agents.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Run Local App (without Docker)

```bash
# Start Ollama server separately (in another terminal)
ollama serve

# In another terminal, start Redis
redis-server

# Start the app
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Usage Examples

### 1. Create a Session

```bash
curl -X POST http://localhost:8000/session \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "org_id": "org_acme"
  }'

# Response:
# {
#   "session_id": "user_123_org_acme_1696046400.123",
#   "user_id": "user_123",
#   "token_budget": 5000,
#   "created_at": "2026-09-30T12:00:00"
# }
```

### 2. Send Query via WebSocket (Coming in Phase 5)

```javascript
// JavaScript example
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onopen = () => {
  ws.send(JSON.stringify({
    session_id: 'user_123_org_acme_1696046400.123',
    query: 'I want a refund for transaction TXN-90811'
  }));
};

ws.onmessage = (event) => {
  const token = JSON.parse(event.data);
  console.log(token.content);  // Print token as it arrives
};
```

### 3. Check Budget Status

```bash
curl http://localhost:8000/session/user_123_org_acme_1696046400.123/budget

# Response:
# {
#   "used": 1250,
#   "budget": 5000,
#   "remaining": 3750,
#   "percent_used": 25.0
# }
```

### 4. View Audit Log

```bash
curl http://localhost:8000/session/user_123_org_acme_1696046400.123/audit

# Response includes:
# {
#   "routing_path": ["BillingAgent", "PolicyAgent"],
#   "pii_events": [
#     {"type": "CREDIT_CARD", "timestamp": "...", "action": "masked"}
#   ],
#   "loop_detection_events": [],
#   "judge_evaluations": [...]
# }
```

---

## Configuration Reference

### Core Configuration (`config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | dev/staging/production |
| `DEBUG` | true | Enable debug logging |
| `API_HOST` | 0.0.0.0 | Server bind address |
| `API_PORT` | 8000 | Server port |
| `DEFAULT_TOKEN_BUDGET` | 5000 | Tokens per session |
| `MAX_AGENT_HOPS` | 3 | Max routing hops |
| `OLLAMA_HOST` | http://ollama:11434 | Ollama server URL |
| `OLLAMA_MODEL` | mistral | LLM model name |
| `REDIS_HOST` | redis | Redis hostname |
| `PII_MASKING_ENABLED` | true | Enable PII masking |
| `PROMETHEUS_ENABLED` | true | Enable metrics |

See `.env.example` for complete list.

---

## Project Structure

```
nexus-guard/
├── src/
│   ├── agents/          # Agent implementations
│   ├── orchestrator/    # Core orchestration logic
│   ├── guardrails/      # Safety mechanisms (PII, loops, judge)
│   ├── api/             # REST & WebSocket API
│   ├── models/          # Data models
│   └── utils/           # Utilities
├── tests/               # Unit & integration tests
├── docs/                # Documentation
├── config.py            # Configuration
├── docker-compose.yml   # Service orchestration
├── Dockerfile           # App container image
├── requirements.txt     # Python dependencies
├── PRD.md              # Product requirements
└── README.md           # This file
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed explanation.

---

## API Endpoints

### REST Endpoints

| Method | Endpoint | Description | Phase |
|--------|----------|-------------|-------|
| GET | `/health` | Health check | 1 ✅ |
| GET | `/agents` | List agents | 1 ✅ |
| POST | `/session` | Create session | 1 ✅ |
| GET | `/session/{id}/budget` | Budget status | 6 |
| GET | `/session/{id}/audit` | Audit log | 1 ✅ |
| GET | `/config` | Get configuration | 1 ✅ |

### WebSocket Endpoint

| Endpoint | Purpose | Phase |
|----------|---------|-------|
| `/ws/chat` | Streaming chat interaction | 5 |

Full API specification in [ARCHITECTURE.md](docs/ARCHITECTURE.md) (Phase 8)

---

## Docker Compose Services

### `app`
- FastAPI application
- Exposes port 8000
- Depends on Ollama and Redis

### `ollama`
- Mistral 7B LLM inference
- Exposes port 11434
- Model auto-downloads on first run

### `redis`
- Session & cache storage
- Exposes port 6379
- Persistent volume: `redis_data`

---

## Monitoring & Logs

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app

# Last 100 lines
docker-compose logs -f --tail=100 app
```

### Prometheus Metrics (Phase 8)

```bash
# Metrics endpoint
curl http://localhost:9090/metrics

# Common metrics:
# - nexus_requests_total: Total requests
# - nexus_latency_seconds: Request latency
# - nexus_pii_detected_total: PII detections
# - nexus_loops_detected_total: Loop detections
```

### Audit Log

```bash
# View audit log file
docker-compose exec app cat logs/audit.jsonl | head -20

# Follow audit log
docker-compose exec app tail -f logs/audit.jsonl
```

---

## Troubleshooting

### Issue: Ollama model download timeout

**Solution**: Download model manually before starting app
```bash
docker-compose exec ollama ollama pull mistral
```

### Issue: Redis connection refused

**Solution**: Ensure Redis is running
```bash
docker-compose up -d redis
docker-compose logs redis
```

### Issue: API returns 503 (Ollama unavailable)

**Solution**: Wait for Ollama to initialize (first run can take 1-2 minutes)
```bash
docker-compose logs ollama
# Wait for: "Listening on 127.0.0.1:11434"
```

### Issue: High latency (>200ms)

**Causes**: 
- Ollama running on CPU (normal, 1-2s inference time)
- Redis latency (check network)
- PII masking slow (tune buffer size)

**Solutions**:
- Add GPU support (update docker-compose.yml)
- Reduce `PII_MASKING_BUFFER_SIZE` (default 25)
- Scale to multiple API instances

---

## Development Phases

| Phase | Task | Status | Est. Duration |
|-------|------|--------|---|
| 0 | Requirements & Design | ✅ Complete | 2 days |
| 1 | Architecture & Framework Setup | 🔄 In Progress | 3 days |
| 2 | Multi-Agent Routing | ⏳ Planned | 5 days |
| 3 | PII Masking Engine | ⏳ Planned | 4 days |
| 4 | LLM-as-a-Judge | ⏳ Planned | 4 days |
| 5 | Streaming & Latency | ⏳ Planned | 3 days |
| 6 | Token Budgeting | ⏳ Planned | 2 days |
| 7 | Integration Testing | ⏳ Planned | 5 days |
| 8 | Documentation | ⏳ Planned | 3 days |
| 9 | Docker & Deployment | ⏳ Planned | 2 days |
| 10 | Demo & Presentation | ⏳ Planned | 2 days |

---

## Key Files & Documentation

- **[PRD.md](PRD.md)**: Product requirements, personas, goals, NFRs
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Directory tree & module descriptions
- **[AGENT_GOVERNANCE_DESIGN.md](docs/AGENT_GOVERNANCE_DESIGN.md)**: Agent state machine, rubrics, algorithms (Phase 8)
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: System architecture & data flow (Phase 8)
- **[API_SPECIFICATION.md](docs/API_SPECIFICATION.md)**: REST & WebSocket specs (Phase 8)
- **[CHAT_HISTORY.md](CHAT_HISTORY.md)**: Design decisions log (Phase 8)
- **[TEST_REPORT.md](TEST_REPORT.md)**: Test results & coverage (Phase 7)

---

## Contributing

### Code Style
- **Linting**: `black`, `flake8`, `pylint`
- **Type Hints**: 100% coverage required
- **Testing**: Minimum 85% code coverage
- **Documentation**: Docstrings on all public methods

### Pull Request Process
1. Fork repository
2. Create feature branch: `git checkout -b feature/agent-x`
3. Implement feature with tests
4. Run: `pytest tests/ --cov=src`
5. Format code: `black src/ tests/`
6. Create PR with detailed description

---

## Support & Contact

- **Issues**: GitHub Issues
- **Questions**: Slack #nexus-guard
- **Reports**: v-monikab@affle.com

---

## License

Proprietary - Internal Use Only

---

## Changelog

### Version 1.0.0 (2026-09-30)
- **Initial Release**: Project structure, Phase 0-1 complete
- Agent interface defined
- Core orchestrator scaffolding
- Guardrail framework (loop, PII, judge)
- Configuration system

---

**Last Updated**: 2026-09-30  
**Next Milestone**: Phase 2 - Multi-Agent Routing (Est. 2026-10-05)
