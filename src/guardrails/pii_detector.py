"""
Phase 3.1: PII Detection Engine

Uses Presidio for entity recognition with custom patterns for Affle-specific PII types.
Detects 20+ entity types with confidence scoring and real-time performance.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngine
    from presidio_analyzer.recognizer_registry import RecognizerRegistry
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PIIEntity:
    """Represents a detected PII entity."""

    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    source: str  # 'presidio' or 'custom'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entity_type": self.entity_type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source": self.source,
        }


class CustomPatternRegistry:
    """Registry of custom regex patterns for Affle-specific PII types."""

    # Affle-specific custom patterns
    CUSTOM_PATTERNS = {
        "AADHAAR_ID": {
            "pattern": r"\b\d{4}\s?\d{4}\s?\d{4}\b",  # 12-digit Aadhaar
            "description": "Indian Aadhaar identity number",
            "confidence": 0.95,
            "examples": ["1234 5678 9012", "123456789012"],
        },
        "API_KEY": {
            "pattern": r"(?i)api[_-]?key[=:]\s*[a-zA-Z0-9\-_.]{6,}",
            "description": "API key or token",
            "confidence": 0.95,
            "examples": ["api_key=abc123def456ghi789", "API-KEY: xyz789abc123"],
        },
        "TRANSACTION_ID": {
            "pattern": r"(?i)txn[_-]?id[=:]?\s*[A-Z0-9]{6,16}",
            "description": "Transaction identifier",
            "confidence": 0.90,
            "examples": ["TXN_ID=ABC123DEF456", "txn-id: XYZ789ABC"],
        },
        "CUSTOMER_ACCOUNT_ID": {
            "pattern": r"(?i)account[_-]?id[=:]?\s*[A-Z0-9]{8,20}",
            "description": "Customer account identifier",
            "confidence": 0.90,
            "examples": ["account_id=CUST123456789", "ACCOUNT-ID: ACC987654321"],
        },
        "SUPPORT_TICKET_ID": {
            "pattern": r"(?i)ticket[_-]?id[=:]?\s*[A-Z0-9]{6,12}",
            "description": "Support ticket number",
            "confidence": 0.90,
            "examples": ["TICKET-ID=TK123456", "ticket_id: SUP789012"],
        },
        "INTERNAL_USER_ID": {
            "pattern": r"(?i)user[_-]?id[=:]?\s*[A-Z0-9]{6,16}",
            "description": "Internal user identifier",
            "confidence": 0.85,
            "examples": ["user_id=USR123456789", "USER-ID: U987654321"],
        },
        "INTERNAL_ORG_ID": {
            "pattern": r"(?i)org[_-]?id[=:]?\s*[A-Z0-9]{6,16}",
            "description": "Internal organization identifier",
            "confidence": 0.85,
            "examples": ["org_id=ORG123456789", "ORG-ID: O987654321"],
        },
        "CREDIT_CARD_MASKED": {
            "pattern": r"(?i)card[_-]?number[=:]?\s*[0-9\*]{13,19}",
            "description": "Credit card number or masked card",
            "confidence": 0.95,
            "examples": ["card_number=4532123456789010", "CARD-NUMBER: ****1234"],
        },
    }

    @classmethod
    def get_pattern(cls, entity_type: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get pattern details for custom entity type."""
        if entity_type in cls.CUSTOM_PATTERNS:
            pattern_info = cls.CUSTOM_PATTERNS[entity_type]
            return pattern_info["pattern"], pattern_info
        return None

    @classmethod
    def list_patterns(cls) -> Dict[str, Dict[str, Any]]:
        """List all available custom patterns."""
        return cls.CUSTOM_PATTERNS.copy()


class PIIDetector:
    """
    Multi-layer PII Detection Engine

    Combines:
    1. Presidio analyzer (20+ standard PII types)
    2. Custom regex patterns (Affle-specific entities)
    3. Confidence scoring and deduplication
    """

    # Standard Presidio entity types we detect
    PRESIDIO_ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IBAN_CODE",
        "URL",
        "GPE",  # Geopolitical entity
        "MEDICAL_LICENSE",
        "US_PASSPORT",
        "DRIVER_LICENSE",
        "ROUTING_NUMBER",
        "US_BANK_ACCOUNT",
        "CRYPTO",
        "DATE_TIME",
        "LOCATION",
    ]

    def __init__(self, use_presidio: bool = True, confidence_threshold: float = 0.7):
        """
        Initialize PII detector.

        Args:
            use_presidio: Whether to use Presidio analyzer (requires library)
            confidence_threshold: Minimum confidence score to report entities (0.0-1.0)
        """
        self.use_presidio = use_presidio and PRESIDIO_AVAILABLE
        self.confidence_threshold = confidence_threshold
        self.analyzer = None
        self.pattern_cache: Dict[str, re.Pattern] = {}

        # Initialize Presidio if available
        if self.use_presidio:
            try:
                self.analyzer = AnalyzerEngine()
                logger.info("Presidio AnalyzerEngine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Presidio: {e}. Using custom patterns only.")
                self.use_presidio = False

        # Compile custom patterns
        self._compile_custom_patterns()

        logger.info(
            f"PIIDetector initialized: presidio={self.use_presidio}, "
            f"custom_patterns={len(self.pattern_cache)}, "
            f"threshold={confidence_threshold}"
        )

    def _compile_custom_patterns(self) -> None:
        """Compile and cache all custom regex patterns."""
        for entity_type, pattern_info in CustomPatternRegistry.CUSTOM_PATTERNS.items():
            try:
                self.pattern_cache[entity_type] = re.compile(
                    pattern_info["pattern"],
                    re.IGNORECASE
                )
            except re.error as e:
                logger.error(f"Invalid regex pattern for {entity_type}: {e}")

    def detect_presidio(self, text: str) -> List[PIIEntity]:
        """
        Detect PII using Presidio analyzer.

        Args:
            text: Text to analyze

        Returns:
            List of detected PIIEntity objects
        """
        if not self.use_presidio or not self.analyzer:
            return []

        try:
            results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=self.PRESIDIO_ENTITIES
            )

            entities = []
            for result in results:
                if result.score >= self.confidence_threshold:
                    entity = PIIEntity(
                        entity_type=result.entity_type,
                        text=text[result.start:result.end],
                        start=result.start,
                        end=result.end,
                        confidence=result.score,
                        source="presidio"
                    )
                    entities.append(entity)

            return entities
        except Exception as e:
            logger.error(f"Presidio detection failed: {e}")
            return []

    def detect_custom_patterns(self, text: str) -> List[PIIEntity]:
        """
        Detect PII using custom regex patterns.

        Args:
            text: Text to analyze

        Returns:
            List of detected PIIEntity objects
        """
        entities = []

        for entity_type, pattern in self.pattern_cache.items():
            pattern_info = CustomPatternRegistry.CUSTOM_PATTERNS.get(entity_type, {})
            confidence = pattern_info.get("confidence", 0.85)

            for match in pattern.finditer(text):
                # Skip if confidence below threshold
                if confidence < self.confidence_threshold:
                    continue

                entity = PIIEntity(
                    entity_type=entity_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    source="custom"
                )
                entities.append(entity)

        return entities

    def _deduplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """
        Remove overlapping entities, keeping highest confidence.

        Args:
            entities: List of detected entities

        Returns:
            Deduplicated entity list
        """
        if not entities:
            return []

        # Sort by start position, then by confidence (descending)
        sorted_entities = sorted(
            entities,
            key=lambda e: (e.start, -e.confidence)
        )

        # Remove overlapping entities
        deduped = []
        last_end = -1

        for entity in sorted_entities:
            # If this entity doesn't overlap with previous, keep it
            if entity.start >= last_end:
                deduped.append(entity)
                last_end = entity.end

        return deduped

    def detect(self, text: str) -> List[PIIEntity]:
        """
        Detect PII entities in text using all available methods.

        This is the main detection method that combines:
        1. Presidio analysis (if available)
        2. Custom pattern matching
        3. Confidence filtering
        4. Deduplication

        Args:
            text: Text to analyze for PII

        Returns:
            List of detected PIIEntity objects, sorted by position
        """
        if not text:
            return []

        # Collect detections from all sources
        all_entities = []

        # Presidio detection
        if self.use_presidio:
            all_entities.extend(self.detect_presidio(text))

        # Custom pattern detection
        all_entities.extend(self.detect_custom_patterns(text))

        # Deduplicate overlapping entities
        deduped = self._deduplicate_entities(all_entities)

        # Sort by position for consistent ordering
        deduped.sort(key=lambda e: e.start)

        logger.debug(
            f"PII Detection: found {len(deduped)} entities "
            f"(presidio={len([e for e in deduped if e.source=='presidio'])}, "
            f"custom={len([e for e in deduped if e.source=='custom'])})"
        )

        return deduped

    def detect_by_type(self, text: str, entity_types: List[str]) -> List[PIIEntity]:
        """
        Detect specific PII entity types.

        Args:
            text: Text to analyze
            entity_types: List of entity types to detect (e.g., ["PHONE_NUMBER", "EMAIL_ADDRESS"])

        Returns:
            List of detected PIIEntity objects matching requested types
        """
        all_entities = self.detect(text)
        return [e for e in all_entities if e.entity_type in entity_types]

    def get_statistics(self, entities: List[PIIEntity]) -> Dict[str, Any]:
        """
        Get statistics about detected entities.

        Args:
            entities: List of detected entities

        Returns:
            Statistics dictionary
        """
        if not entities:
            return {
                "total_entities": 0,
                "by_type": {},
                "by_source": {},
                "confidence_avg": 0.0,
                "confidence_min": 0.0,
                "confidence_max": 0.0,
            }

        by_type = {}
        by_source = {}
        confidences = []

        for entity in entities:
            # Count by type
            by_type[entity.entity_type] = by_type.get(entity.entity_type, 0) + 1

            # Count by source
            by_source[entity.source] = by_source.get(entity.source, 0) + 1

            # Collect confidences
            confidences.append(entity.confidence)

        return {
            "total_entities": len(entities),
            "by_type": by_type,
            "by_source": by_source,
            "confidence_avg": sum(confidences) / len(confidences) if confidences else 0.0,
            "confidence_min": min(confidences) if confidences else 0.0,
            "confidence_max": max(confidences) if confidences else 0.0,
        }
