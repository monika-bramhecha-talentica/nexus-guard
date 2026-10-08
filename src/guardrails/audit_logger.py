"""
Phase 3.3: PII Audit Logging

Comprehensive event logging for PII detection and masking operations.
Provides structured audit trails for compliance and forensic analysis.
"""

import json
import uuid
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta

from .pii_detector import PIIEntity

logger = logging.getLogger(__name__)


@dataclass
class PIIAuditEvent:
    """Single PII audit event with complete context."""

    event_id: str
    timestamp: str  # ISO 8601 format
    event_type: str  # "detection" or "masking"
    entity_type: str  # Type of PII detected
    entity_count: int  # Number of entities in event
    user_id: str  # Who triggered the event
    context: str  # Where/why (query, response, etc)
    masked_text: Optional[str]  # Masked version if applicable
    status: str  # "success" or "error"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class PIIAuditLogger:
    """
    Audit logger for PII detection and masking events.

    Provides:
    - Event logging with timestamps and context
    - User and action tracking
    - Query and filtering capabilities
    - Export to multiple formats
    - Data retention policy tracking
    """

    def __init__(
        self,
        storage_backend: str = "memory",
        retention_days: int = 90,
        max_events: int = 10000
    ):
        """
        Initialize audit logger.

        Args:
            storage_backend: "memory" or "file"
            retention_days: Days to retain events (default 90)
            max_events: Maximum events to keep in memory (default 10000)
        """
        self.storage_backend = storage_backend
        self.retention_days = retention_days
        self.max_events = max_events
        self.events: List[PIIAuditEvent] = []

        logger.info(
            f"PIIAuditLogger initialized: "
            f"backend={storage_backend}, retention={retention_days}d"
        )

    def log_pii_detection(
        self,
        text: str,
        entities: List[PIIEntity],
        context: str,
        user_id: str = "unknown"
    ) -> str:
        """
        Log PII detection event.

        Args:
            text: Text where PII was detected
            entities: List of detected entities
            context: Context where detection occurred
            user_id: User who triggered detection

        Returns:
            Event ID
        """
        if not entities:
            return ""

        event_id = f"evt_det_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Determine primary entity type
        primary_type = entities[0].entity_type if entities else "UNKNOWN"

        event = PIIAuditEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="detection",
            entity_type=primary_type,
            entity_count=len(entities),
            user_id=user_id,
            context=context,
            masked_text=None,
            status="success",
            details={
                "text_length": len(text),
                "confidence_scores": [e.confidence for e in entities],
                "detection_sources": list(set(e.source for e in entities)),
                "entity_types": list(set(e.entity_type for e in entities)),
            }
        )

        self._add_event(event)

        logger.info(
            f"PII detection logged: {len(entities)} entities "
            f"(user={user_id}, context={context})"
        )

        return event_id

    def log_pii_masking(
        self,
        original_text: str,
        masked_text: str,
        entities: List[PIIEntity],
        strategy: str,
        user_id: str = "unknown"
    ) -> str:
        """
        Log PII masking event.

        Args:
            original_text: Original text with PII
            masked_text: Text after masking
            entities: Entities that were masked
            strategy: Masking strategy used
            user_id: User who triggered masking

        Returns:
            Event ID
        """
        if not entities:
            return ""

        event_id = f"evt_mask_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Determine primary entity type
        primary_type = entities[0].entity_type if entities else "UNKNOWN"

        event = PIIAuditEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="masking",
            entity_type=primary_type,
            entity_count=len(entities),
            user_id=user_id,
            context="masking_operation",
            masked_text=masked_text,
            status="success",
            details={
                "strategy": strategy,
                "original_length": len(original_text),
                "masked_length": len(masked_text),
                "entity_types": list(set(e.entity_type for e in entities)),
                "confidence_scores": [e.confidence for e in entities],
            }
        )

        self._add_event(event)

        logger.info(
            f"PII masking logged: {len(entities)} entities "
            f"(strategy={strategy}, user={user_id})"
        )

        return event_id

    def _add_event(self, event: PIIAuditEvent) -> None:
        """
        Add event to audit log.

        Enforces max event limit.

        Args:
            event: Event to add
        """
        self.events.append(event)

        # Enforce max events limit
        if len(self.events) > self.max_events:
            # Remove oldest events (FIFO)
            self.events = self.events[-self.max_events:]
            logger.warning(f"Audit log limit reached, removed oldest events")

    def get_events(
        self,
        entity_type: Optional[str] = None,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[PIIAuditEvent]:
        """
        Query audit events with optional filters.

        Args:
            entity_type: Filter by entity type
            user_id: Filter by user ID
            event_type: Filter by event type ("detection" or "masking")
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of matching events
        """
        filtered = self.events

        # Apply filters
        if entity_type:
            filtered = [e for e in filtered if e.entity_type == entity_type]

        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]

        if start_date:
            start_iso = start_date.isoformat()
            filtered = [e for e in filtered if e.timestamp >= start_iso]

        if end_date:
            end_iso = end_date.isoformat()
            filtered = [e for e in filtered if e.timestamp <= end_iso]

        return filtered

    def get_events_by_user(self, user_id: str) -> List[PIIAuditEvent]:
        """Get all events for a specific user."""
        return self.get_events(user_id=user_id)

    def get_events_by_type(self, entity_type: str) -> List[PIIAuditEvent]:
        """Get all events for a specific entity type."""
        return self.get_events(entity_type=entity_type)

    def get_recent_events(self, hours: int = 24) -> List[PIIAuditEvent]:
        """Get events from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return self.get_events(start_date=cutoff)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about audit events.

        Returns:
            Dictionary with event statistics
        """
        if not self.events:
            return {
                "total_events": 0,
                "detection_events": 0,
                "masking_events": 0,
                "unique_users": 0,
                "unique_entity_types": 0,
                "total_entities_logged": 0,
            }

        detection_events = [e for e in self.events if e.event_type == "detection"]
        masking_events = [e for e in self.events if e.event_type == "masking"]

        unique_users = set(e.user_id for e in self.events)
        unique_types = set(e.entity_type for e in self.events)
        total_entities = sum(e.entity_count for e in self.events)

        return {
            "total_events": len(self.events),
            "detection_events": len(detection_events),
            "masking_events": len(masking_events),
            "unique_users": len(unique_users),
            "unique_entity_types": len(unique_types),
            "total_entities_logged": total_entities,
        }

    def check_retention_compliance(self) -> List[str]:
        """
        Check for events exceeding retention policy.

        Returns:
            List of event IDs that exceed retention
        """
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        cutoff_iso = cutoff.isoformat()

        expired = []
        for event in self.events:
            if event.timestamp < cutoff_iso:
                expired.append(event.event_id)

        return expired

    def cleanup_expired_events(self) -> int:
        """
        Remove events exceeding retention policy.

        Returns:
            Number of events removed
        """
        expired = self.check_retention_compliance()

        if expired:
            # Remove expired events
            self.events = [e for e in self.events if e.event_id not in expired]
            logger.info(f"Cleaned up {len(expired)} expired events")

        return len(expired)

    def export_events(
        self,
        filename: str,
        format: str = "jsonl",
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export audit events to file.

        Args:
            filename: Output filename
            format: Export format ("jsonl" or "json")
            filters: Optional filters to apply

        Returns:
            Filename
        """
        # Apply filters if provided
        events_to_export = self.events
        if filters:
            if "user_id" in filters:
                events_to_export = [
                    e for e in events_to_export if e.user_id == filters["user_id"]
                ]
            if "entity_type" in filters:
                events_to_export = [
                    e for e in events_to_export if e.entity_type == filters["entity_type"]
                ]
            if "event_type" in filters:
                events_to_export = [
                    e for e in events_to_export if e.event_type == filters["event_type"]
                ]

        # Export based on format
        if format == "jsonl":
            # JSON Lines format (one event per line)
            with open(filename, "w") as f:
                for event in events_to_export:
                    f.write(event.to_json() + "\n")
        else:  # json
            # JSON array format
            with open(filename, "w") as f:
                json.dump(
                    [e.to_dict() for e in events_to_export],
                    f,
                    default=str,
                    indent=2
                )

        logger.info(
            f"Exported {len(events_to_export)} events to {filename} "
            f"(format={format})"
        )

        return filename

    def get_event_count(self) -> int:
        """Get total number of logged events."""
        return len(self.events)

    def clear_events(self) -> int:
        """
        Clear all logged events.

        WARNING: This deletes all audit history!

        Returns:
            Number of events cleared
        """
        count = len(self.events)
        self.events = []
        logger.warning(f"Cleared {count} audit events")
        return count
