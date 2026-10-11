"""
Phase 7: Guarded assistant composition.

Wires the orchestrator (routing, loop detection, agents) to the streaming
guard (PII masking, budget, judge) for one customer query:

    raw query
      -> input PII masking (FR-3.5)      agents never see raw PII
      -> orchestrator.route_query        multi-agent routing (FR-1, FR-2)
      -> tokenised agent answer          streamed via the guard pipeline
      -> output PII masking, budget, judge (FR-3.7, FR-5, FR-6)
"""

import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from src.guardrails.audit_logger import PIIAuditLogger
from src.guardrails.pii_detector import PIIDetector
from src.guardrails.pii_masker import PIIMasker

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\S+\s*")


def tokenize(text: str) -> List[str]:
    """Split text into word-level tokens. Concatenating the tokens reproduces the text."""
    return _TOKEN_RE.findall(text or "")


class GuardedAssistant:
    def __init__(
        self,
        orchestrator,
        detector: Optional[PIIDetector] = None,
        masker: Optional[PIIMasker] = None,
        pii_audit: Optional[PIIAuditLogger] = None,
    ):
        self.orchestrator = orchestrator
        self.detector = detector or PIIDetector(use_presidio=False)
        self.masker = masker or PIIMasker()
        self.pii_audit = pii_audit or PIIAuditLogger()
        self.sessions: Dict[str, Any] = {}
        self.last_session_id: Optional[str] = None

    def mask_input(self, query: str, user_id: str = "anonymous") -> str:
        """Mask PII in the customer's query before any agent receives it (FR-3.5)."""
        entities = self.detector.detect(query)
        if not entities:
            return query
        # One audit event per entity: PIIAuditLogger records only the primary type per call.
        for entity in entities:
            self.pii_audit.log_pii_detection(query, [entity], context="input_query", user_id=user_id)
        return self.masker.mask_text(query, entities)

    async def answer_tokens(
        self,
        query: str,
        user_id: str = "anonymous",
        org_id: str = "default",
    ) -> AsyncIterator[str]:
        """Route the (already masked) query and yield the final answer as tokens."""
        session = self.orchestrator.create_session(user_id, org_id)
        self.sessions[session.session_id] = session
        self.last_session_id = session.session_id

        response = await self.orchestrator.route_query(session.session_id, query)
        for token in tokenize(response.output_text):
            yield token

    def routing_path(self, session_id: str) -> List[str]:
        session = self.sessions.get(session_id)
        return list(session.routing_path) if session else []
