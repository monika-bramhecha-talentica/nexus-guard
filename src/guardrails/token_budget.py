"""
Phase 6 (FR-6): Token budgeting, cost simulation and budget audit.

Components:
  - TokenEstimator: chars -> tokens (4 chars ~= 1 token) and projection of
    requests that ask for large repeated output (FR-6 acceptance example).
  - BudgetPolicy: default budget (5000), per-tier and per-org budgets,
    warning threshold (80%), simulated cost rate (FR-6.2, FR-6.3, FR-6.4).
  - BudgetStore: session counters with TTL. InMemoryBudgetStore for tests and
    single-process use; RedisBudgetStore for shared deployments (FR-6.7).
  - BudgetAuditLog: every budget event (open, warning, exhausted, truncation,
    escalation) with session id and consumption (FR-6.8).
  - SessionBudgetService: the one place that charges, warns, and enforces.

Token counts are estimates (chars / 4). Swap TokenEstimator for a real
tokenizer if exact billing counts are required.
"""

import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BUDGET_EXCEEDED_MESSAGE = "Session budget exceeded. Escalating to support team."


class TokenEstimator:
    """Approximate token counting. 4 characters ~= 1 token."""

    CHARS_PER_TOKEN = 4

    # e.g. "Repeat 'VERIFY' 10,000 times" -> unit "VERIFY", 10000 repetitions
    _REPEAT_RE = re.compile(r"repeat\s+['\"]?(.+?)['\"]?\s+(\d[\d,]*)\s+times", re.IGNORECASE)

    @classmethod
    def tokens_for_chars(cls, chars: int) -> int:
        return math.ceil(max(0, chars) / cls.CHARS_PER_TOKEN)

    @classmethod
    def estimate_text(cls, text: str) -> int:
        return cls.tokens_for_chars(len(text))

    @classmethod
    def project_request(cls, query: str) -> Optional[Dict[str, Any]]:
        """
        Estimate output size for requests that ask for repetition.

        Returns None when the query does not request repeated output.
        """
        match = cls._REPEAT_RE.search(query or "")
        if not match:
            return None
        unit = match.group(1).strip()
        repetitions = int(match.group(2).replace(",", ""))
        projected_chars = len(unit + " ") * repetitions
        return {
            "unit": unit,
            "repetitions": repetitions,
            "projected_tokens": cls.tokens_for_chars(projected_chars),
        }


@dataclass
class BudgetPolicy:
    """Budget limits and cost simulation settings."""

    default_budget: int = 5000  # FR-6.2
    tier_budgets: Dict[str, int] = field(default_factory=lambda: {"standard": 5000, "premium": 20000})
    org_budgets: Dict[str, int] = field(default_factory=dict)
    warn_fraction: float = 0.8  # FR-6.4
    usd_per_1k_tokens: float = 0.0005  # FR-6.3: simulated rate, not a real price
    session_ttl_seconds: int = 1800  # FR-6.7

    def budget_for(self, org_id: Optional[str] = None, tier: Optional[str] = None) -> int:
        if org_id and org_id in self.org_budgets:
            return self.org_budgets[org_id]
        if tier and tier in self.tier_budgets:
            return self.tier_budgets[tier]
        return self.default_budget

    def warn_at(self, budget: int) -> int:
        return int(budget * self.warn_fraction)

    def cost_usd(self, tokens: int) -> float:
        return round(tokens / 1000 * self.usd_per_1k_tokens, 6)


class BudgetAuditLog:
    """Append-only record of budget events. Optionally mirrored to a JSON-lines file."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, session_id: str, event: str, **detail: Any) -> Dict[str, Any]:
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "event": event,
            **detail,
        }
        with self._lock:
            self._events.append(entry)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    def events_for(self, session_id: str) -> List[Dict[str, Any]]:
        return [e for e in self._events if e["session_id"] == session_id]

    def all_events(self) -> List[Dict[str, Any]]:
        return list(self._events)


class BudgetStore:
    """Interface for per-session budget counters."""

    async def init(self, session_id: str, budget: int, ttl: int) -> bool:
        """Create the session record if absent. Returns True if created."""
        raise NotImplementedError

    async def incr_used(self, session_id: str, tokens: int, ttl: int) -> int:
        """Atomically add tokens. Returns the new total."""
        raise NotImplementedError

    async def read(self, session_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def set_flag(self, session_id: str, name: str, value: bool, ttl: int) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class InMemoryBudgetStore(BudgetStore):
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._expires: Dict[str, float] = {}

    def _alive(self, session_id: str) -> Optional[Dict[str, Any]]:
        expires = self._expires.get(session_id)
        if expires is not None and time.monotonic() > expires:
            self._data.pop(session_id, None)
            self._expires.pop(session_id, None)
            return None
        return self._data.get(session_id)

    def _touch(self, session_id: str, ttl: int) -> None:
        self._expires[session_id] = time.monotonic() + ttl

    async def init(self, session_id: str, budget: int, ttl: int) -> bool:
        if self._alive(session_id) is not None:
            self._touch(session_id, ttl)
            return False
        self._data[session_id] = {
            "used": 0,
            "budget": budget,
            "warned": False,
            "exhausted": False,
        }
        self._touch(session_id, ttl)
        return True

    async def incr_used(self, session_id: str, tokens: int, ttl: int) -> int:
        record = self._alive(session_id)
        if record is None:
            raise KeyError(f"budget session not found: {session_id}")
        record["used"] += tokens
        self._touch(session_id, ttl)
        return record["used"]

    async def read(self, session_id: str) -> Optional[Dict[str, Any]]:
        record = self._alive(session_id)
        return dict(record) if record is not None else None

    async def set_flag(self, session_id: str, name: str, value: bool, ttl: int) -> None:
        record = self._alive(session_id)
        if record is None:
            raise KeyError(f"budget session not found: {session_id}")
        record[name] = value
        self._touch(session_id, ttl)


class RedisBudgetStore(BudgetStore):
    """Budget counters in Redis hashes with TTL. Safe across processes."""

    def __init__(self, client, prefix: str = "nexus:budget:"):
        self.client = client
        self.prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    @staticmethod
    def _decode(raw: Dict[Any, Any]) -> Optional[Dict[str, Any]]:
        if not raw:
            return None

        def text(value):
            return value.decode() if isinstance(value, bytes) else value

        data = {text(k): text(v) for k, v in raw.items()}
        return {
            "used": int(data["used"]),
            "budget": int(data["budget"]),
            "warned": data["warned"] == "1",
            "exhausted": data["exhausted"] == "1",
        }

    async def init(self, session_id: str, budget: int, ttl: int) -> bool:
        key = self._key(session_id)
        created = await self.client.hsetnx(key, "used", 0)
        if created:
            await self.client.hset(
                key, mapping={"budget": budget, "warned": 0, "exhausted": 0}
            )
        await self.client.expire(key, ttl)
        return bool(created)

    async def incr_used(self, session_id: str, tokens: int, ttl: int) -> int:
        key = self._key(session_id)
        if not await self.client.exists(key):
            raise KeyError(f"budget session not found: {session_id}")
        pipe = self.client.pipeline()
        pipe.hincrby(key, "used", tokens)
        pipe.expire(key, ttl)
        new_total, _ = await pipe.execute()
        return int(new_total)

    async def read(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._decode(await self.client.hgetall(self._key(session_id)))

    async def set_flag(self, session_id: str, name: str, value: bool, ttl: int) -> None:
        key = self._key(session_id)
        await self.client.hset(key, name, 1 if value else 0)
        await self.client.expire(key, ttl)

    async def close(self) -> None:
        await self.client.aclose()


@dataclass
class BudgetDecision:
    """Result of a charge: what the caller must do next."""

    used: int
    budget: int
    newly_warned: bool = False
    newly_exhausted: bool = False

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)


class SessionBudgetService:
    """Charges sessions, emits warnings once, enforces the hard limit, and audits."""

    def __init__(
        self,
        store: BudgetStore,
        policy: Optional[BudgetPolicy] = None,
        audit: Optional[BudgetAuditLog] = None,
    ):
        self.store = store
        self.policy = policy or BudgetPolicy()
        self.audit = audit or BudgetAuditLog()

    async def open(
        self,
        session_id: str,
        org_id: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        budget = self.policy.budget_for(org_id=org_id, tier=tier)
        created = await self.store.init(session_id, budget, self.policy.session_ttl_seconds)
        if created:
            self.audit.record(session_id, "session_opened", budget=budget, org_id=org_id, tier=tier)
        return await self.status(session_id)

    async def charge(self, session_id: str, tokens: int, reason: str) -> BudgetDecision:
        """Add tokens for a reason ('input', 'output', ...) and report threshold crossings."""
        ttl = self.policy.session_ttl_seconds
        record = await self.store.read(session_id)
        if record is None:
            raise KeyError(f"budget session not found: {session_id}")
        budget = record["budget"]

        used = await self.store.incr_used(session_id, tokens, ttl)
        decision = BudgetDecision(used=used, budget=budget)
        self.audit.record(session_id, "charge", reason=reason, tokens=tokens, used=used, budget=budget)

        if not record["warned"] and used >= self.policy.warn_at(budget):
            await self.store.set_flag(session_id, "warned", True, ttl)
            decision.newly_warned = True
            self.audit.record(session_id, "warning", used=used, budget=budget,
                              percent_used=round(used / budget * 100, 2))

        if not record["exhausted"] and used >= budget:
            await self.store.set_flag(session_id, "exhausted", True, ttl)
            decision.newly_exhausted = True
            self.audit.record(session_id, "exhausted", used=used, budget=budget)

        return decision

    async def status(self, session_id: str) -> Dict[str, Any]:
        record = await self.store.read(session_id)
        if record is None:
            return {"error": "session_not_found", "session_id": session_id}
        used, budget = record["used"], record["budget"]
        state = "exhausted" if record["exhausted"] else "warning" if record["warned"] else "ok"
        return {
            "session_id": session_id,
            "state": state,
            "used": used,
            "budget": budget,
            "remaining": max(0, budget - used),
            "percent_used": round(used / budget * 100, 2) if budget else 100.0,
            "simulated_cost_usd": self.policy.cost_usd(used),
        }

    def record_event(self, session_id: str, event: str, **detail: Any) -> None:
        self.audit.record(session_id, event, **detail)
