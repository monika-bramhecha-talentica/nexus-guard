"""
Phase 3.3: Compliance Reporting

Compliance report generation, data retention policy enforcement,
and regulatory export capabilities for PII audit events.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging

from .audit_logger import PIIAuditLogger, PIIAuditEvent

logger = logging.getLogger(__name__)


@dataclass
class ComplianceReport:
    """Compliance report with statistics and findings."""

    period_start: str
    period_end: str
    total_events: int
    total_pii_detected: int
    total_pii_masked: int
    events_by_type: Dict[str, int]
    detection_sources: Dict[str, int]
    masking_strategies: Dict[str, int]
    data_retention_violations: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)


class ComplianceReporter:
    """
    Compliance reporter for PII audit events.

    Provides:
    - Compliance report generation
    - Data retention policy enforcement
    - Breach and violation detection
    - Regulatory export capabilities
    """

    def __init__(
        self,
        logger_instance: PIIAuditLogger,
        retention_policy: Optional[Dict[str, int]] = None
    ):
        """
        Initialize compliance reporter.

        Args:
            logger_instance: PIIAuditLogger instance
            retention_policy: Dict mapping entity_type to retention days
                             Default: 90 days for all types
        """
        self.logger = logger_instance
        self.retention_policy = retention_policy or self._default_retention_policy()

    def _default_retention_policy(self) -> Dict[str, int]:
        """Get default retention policy."""
        return {
            "CREDIT_CARD": 365,
            "EMAIL_ADDRESS": 180,
            "PHONE_NUMBER": 180,
            "AADHAAR_ID": 365,
            "API_KEY": 90,
            "default": 90
        }

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> ComplianceReport:
        """
        Generate compliance report for date range.

        Args:
            start_date: Report period start
            end_date: Report period end

        Returns:
            ComplianceReport instance
        """
        events = self.logger.get_events(start_date=start_date, end_date=end_date)

        detection_events = [e for e in events if e.event_type == "detection"]
        masking_events = [e for e in events if e.event_type == "masking"]

        # Count by entity type
        events_by_type = {}
        for event in events:
            events_by_type[event.entity_type] = events_by_type.get(event.entity_type, 0) + 1

        # Detection sources from details
        detection_sources = {}
        for event in detection_events:
            if "detection_sources" in event.details:
                for source in event.details["detection_sources"]:
                    detection_sources[source] = detection_sources.get(source, 0) + 1

        # Masking strategies
        masking_strategies = {}
        for event in masking_events:
            if "strategy" in event.details:
                strategy = event.details["strategy"]
                masking_strategies[strategy] = masking_strategies.get(strategy, 0) + 1

        # Check retention compliance
        violations = self.check_data_retention_compliance()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            len(events),
            len(detection_events),
            len(masking_events),
            violations
        )

        return ComplianceReport(
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            total_events=len(events),
            total_pii_detected=len(detection_events),
            total_pii_masked=len(masking_events),
            events_by_type=events_by_type,
            detection_sources=detection_sources,
            masking_strategies=masking_strategies,
            data_retention_violations=violations,
            recommendations=recommendations
        )

    def check_data_retention_compliance(self) -> List[str]:
        """
        Check if any events exceed retention policy.

        Returns:
            List of event IDs that violate retention policy
        """
        violations = []
        now = datetime.utcnow()

        for event in self.logger.events:
            # Get retention days for entity type
            retention_days = self.retention_policy.get(
                event.entity_type,
                self.retention_policy.get("default", 90)
            )

            # Parse event timestamp
            try:
                # Remove 'Z' suffix and parse as naive datetime (stored as UTC)
                timestamp_str = event.timestamp.replace("Z", "")
                event_time = datetime.fromisoformat(timestamp_str)
                age_days = (now - event_time).days

                if age_days > retention_days:
                    violations.append(event.event_id)
            except (ValueError, AttributeError):
                # Skip events with invalid timestamps
                pass

        return violations

    def get_pii_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on PII detection and masking.

        Returns:
            Dictionary with PII statistics
        """
        events = self.logger.events

        if not events:
            return {
                "total_events": 0,
                "detection_events": 0,
                "masking_events": 0,
                "unique_users": 0,
                "unique_entity_types": 0,
                "total_entities": 0
            }

        detection_events = [e for e in events if e.event_type == "detection"]
        masking_events = [e for e in events if e.event_type == "masking"]

        return {
            "total_events": len(events),
            "detection_events": len(detection_events),
            "masking_events": len(masking_events),
            "unique_users": len(set(e.user_id for e in events)),
            "unique_entity_types": len(set(e.entity_type for e in events)),
            "total_entities": sum(e.entity_count for e in events)
        }

    def identify_unusual_activity(self, threshold: int = 100) -> List[PIIAuditEvent]:
        """
        Identify potential data exfiltration patterns.

        Flags users with unusually high PII detection/masking counts.

        Args:
            threshold: Event count threshold for flagging

        Returns:
            List of suspicious events
        """
        user_event_counts = {}
        suspicious_events = []

        for event in self.logger.events:
            user_event_counts[event.user_id] = user_event_counts.get(event.user_id, 0) + 1

        # Find users exceeding threshold
        suspicious_users = {
            user for user, count in user_event_counts.items()
            if count > threshold
        }

        # Return events from suspicious users
        if suspicious_users:
            suspicious_events = [
                e for e in self.logger.events
                if e.user_id in suspicious_users
            ]

        return suspicious_events

    def export_for_regulatory_review(self, filename: str) -> str:
        """
        Export audit trail in compliance format.

        Includes summary statistics and detailed event log.

        Args:
            filename: Output filename

        Returns:
            Filename
        """
        stats = self.get_pii_statistics()
        violations = self.check_data_retention_compliance()

        compliance_export = {
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "statistics": stats,
            "data_retention_violations": violations,
            "events": [e.to_dict() for e in self.logger.events]
        }

        with open(filename, "w") as f:
            json.dump(compliance_export, f, default=str, indent=2)

        logger.info(f"Exported compliance report to {filename}")
        return filename

    def _generate_recommendations(
        self,
        total: int,
        detections: int,
        maskings: int,
        violations: List[str]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []

        if not maskings and detections > 0:
            recommendations.append("PII detected but not masked - review masking policies")

        if violations:
            recommendations.append(f"Data retention violations found ({len(violations)} events) - cleanup recommended")

        if detections > 0:
            masking_rate = (maskings / detections * 100) if detections > 0 else 0
            if masking_rate < 80:
                recommendations.append(f"Masking rate is {masking_rate:.1f}% - target 100%")

        if not recommendations:
            recommendations.append("Compliance status: GOOD - no violations detected")

        return recommendations
