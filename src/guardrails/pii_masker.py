"""
PII (Personally Identifiable Information) Masking Engine

Real-time detection and masking of sensitive data in:
- User input queries
- Inter-agent communications
- Agent output responses

Uses Presidio (open-source) + custom regex patterns.

Implementation details in Phase 3.
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PIIEvent:
    """Log entry for PII detection."""

    timestamp: str
    user_id: str
    pii_type: str
    masking_rule: str
    context: str  # Brief context where PII was found


class PIIMasker:
    """
    Real-time PII masking engine.

    Detects and masks:
    - Credit card numbers (16-digit)
    - Indian Aadhaar numbers (12-digit)
    - API keys and secrets
    - Email addresses
    - Phone numbers
    - Names (via Presidio)
    """

    def __init__(self):
        """Initialize PII masker."""

        # Define PII patterns (regex-based)
        self.pii_patterns: Dict[str, str] = {
            "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # 16-digit card
            "AADHAAR": r"\b\d{4}\s\d{4}\s\d{4}\b",  # Indian Aadhaar
            "API_KEY": r"(?i)api[_-]?key[_-]?[a-zA-Z0-9]{32,}",  # API key pattern
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        }

        self.pii_events: List[PIIEvent] = []

    def mask_text(self, text: str, user_id: str = "unknown") -> str:
        """
        Mask all PII in text.

        Args:
            text: Input text to mask
            user_id: User ID for audit logging

        Returns:
            Text with all PII replaced with [REDACTED_<TYPE>]
        """
        if not text:
            return text

        masked_text = text
        detections: List[Tuple[str, str]] = []

        # Apply each PII pattern
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, masked_text, re.IGNORECASE)
            for match in matches:
                original = match.group()
                replacement = f"[REDACTED_{pii_type}]"
                masked_text = masked_text.replace(original, replacement)
                detections.append((pii_type, original))

        # Log detections
        for pii_type, original in detections:
            logger.warning(f"PII detected: {pii_type} in user query")

        return masked_text

    def mask_streaming_tokens(self, tokens: List[str], user_id: str = "unknown") -> List[str]:
        """
        Mask tokens in a streaming fashion.

        Process small batches of tokens (10-50) for real-time masking.

        Args:
            tokens: List of tokens to mask
            user_id: User ID for audit logging

        Returns:
            Masked tokens
        """
        # Join tokens into text, mask, split back
        text = "".join(tokens)
        masked_text = self.mask_text(text, user_id)

        # Split back into tokens (approximate)
        masked_tokens = masked_text.split()
        return masked_tokens

    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect all PII in text without masking.

        Args:
            text: Text to analyze

        Returns:
            List of detected PII entities with type and location
        """
        detections = []

        for pii_type, pattern in self.pii_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                detections.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })

        return detections

    def get_audit_log(self) -> List[PIIEvent]:
        """Get all recorded PII events."""
        return self.pii_events
