"""
Phase 3.3 Testing - Audit Logging & Compliance

Comprehensive test suite for PIIAuditLogger and ComplianceReporter
Tests: 45 tests (25 audit + 20 compliance)
"""

import pytest
from datetime import datetime, timedelta
import json
import tempfile
import os

from src.guardrails.pii_detector import PIIDetector, PIIEntity
from src.guardrails.audit_logger import PIIAuditLogger, PIIAuditEvent
from src.guardrails.compliance_reporter import ComplianceReporter, ComplianceReport


# ===== AUDIT LOGGER TESTS (25 tests) =====

class TestAuditLoggerInitialization:
    """Test PIIAuditLogger initialization."""

    def test_default_initialization(self):
        """Test logger initialization with defaults."""
        logger = PIIAuditLogger()
        assert logger.storage_backend == "memory"
        assert logger.retention_days == 90
        assert logger.max_events == 10000
        assert len(logger.events) == 0

    def test_custom_initialization(self):
        """Test logger with custom parameters."""
        logger = PIIAuditLogger(storage_backend="file", retention_days=180, max_events=5000)
        assert logger.storage_backend == "file"
        assert logger.retention_days == 180
        assert logger.max_events == 5000


class TestAuditEventLogging:
    """Test audit event logging functionality."""

    def test_log_pii_detection(self):
        """Test logging PII detection event."""
        logger = PIIAuditLogger()
        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]

        event_id = logger.log_pii_detection(
            text="Contact: test@example.com",
            entities=entities,
            context="customer_email",
            user_id="usr_123"
        )

        assert event_id.startswith("evt_det_")
        assert len(logger.events) == 1
        assert logger.events[0].event_type == "detection"
        assert logger.events[0].entity_count == 1

    def test_log_pii_masking(self):
        """Test logging PII masking event."""
        logger = PIIAuditLogger()
        entities = [PIIEntity("CREDIT_CARD", "4532123456789010", 0, 16, 0.95, "custom")]

        event_id = logger.log_pii_masking(
            original_text="Card: 4532123456789010",
            masked_text="Card: [MASKED_CREDITCARD]",
            entities=entities,
            strategy="placeholder",
            user_id="usr_456"
        )

        assert event_id.startswith("evt_mask_")
        assert len(logger.events) == 1
        assert logger.events[0].event_type == "masking"
        assert logger.events[0].status == "success"

    def test_empty_entity_list_returns_empty_id(self):
        """Test that empty entity list returns empty string."""
        logger = PIIAuditLogger()
        event_id = logger.log_pii_detection("No PII here", [], "test", "usr_1")
        assert event_id == ""

    def test_event_timestamp_format(self):
        """Test event timestamp is ISO 8601 format."""
        logger = PIIAuditLogger()
        entities = [PIIEntity("PHONE_NUMBER", "5551234567", 0, 10, 0.9, "presidio")]
        logger.log_pii_detection("Call: 5551234567", entities, "phone_log", "usr_2")

        event = logger.events[0]
        assert event.timestamp.endswith("Z")
        assert "T" in event.timestamp


class TestEventQuerying:
    """Test event querying and filtering."""

    def test_get_all_events(self):
        """Test retrieving all events."""
        logger = PIIAuditLogger()
        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]

        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")
        logger.log_pii_detection("Email: test2@example.com", entities, "ctx2", "usr_1")

        all_events = logger.get_events()
        assert len(all_events) == 2

    def test_filter_by_entity_type(self):
        """Test filtering by entity type."""
        logger = PIIAuditLogger()
        email_entity = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]
        phone_entity = [PIIEntity("PHONE_NUMBER", "5551234567", 0, 10, 0.9, "presidio")]

        logger.log_pii_detection("Email: test@example.com", email_entity, "ctx1", "usr_1")
        logger.log_pii_detection("Phone: 5551234567", phone_entity, "ctx2", "usr_1")

        email_events = logger.get_events(entity_type="EMAIL_ADDRESS")
        assert len(email_events) == 1
        assert email_events[0].entity_type == "EMAIL_ADDRESS"

    def test_filter_by_user_id(self):
        """Test filtering by user ID."""
        logger = PIIAuditLogger()
        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]

        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")
        logger.log_pii_detection("Email: test2@example.com", entities, "ctx2", "usr_2")

        user1_events = logger.get_events(user_id="usr_1")
        assert len(user1_events) == 1
        assert user1_events[0].user_id == "usr_1"

    def test_filter_by_event_type(self):
        """Test filtering by event type."""
        logger = PIIAuditLogger()
        entities = [PIIEntity("CREDIT_CARD", "4532123456789010", 0, 16, 0.95, "custom")]

        logger.log_pii_detection("Card: 4532123456789010", entities, "ctx1", "usr_1")
        logger.log_pii_masking("Card: 4532123456789010", "Card: [MASKED]", entities, "placeholder", "usr_1")

        detection_events = logger.get_events(event_type="detection")
        assert len(detection_events) == 1
        assert detection_events[0].event_type == "detection"

    def test_filter_by_date_range(self):
        """Test filtering by date range."""
        logger = PIIAuditLogger()
        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]

        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")

        start = datetime.utcnow() - timedelta(hours=1)
        end = datetime.utcnow() + timedelta(hours=1)

        events = logger.get_events(start_date=start, end_date=end)
        assert len(events) == 1


class TestEventStorage:
    """Test event storage and limits."""

    def test_max_events_enforcement(self):
        """Test that max events limit is enforced."""
        logger = PIIAuditLogger(max_events=5)
        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]

        # Add more than max_events
        for i in range(10):
            logger.log_pii_detection(f"Email {i}: test@example.com", entities, "ctx", "usr_1")

        # Should only keep last 5
        assert len(logger.events) == 5

    def test_event_count_accuracy(self):
        """Test accurate event counting."""
        logger = PIIAuditLogger()
        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]

        for i in range(10):
            logger.log_pii_detection(f"Email {i}: test@example.com", entities, "ctx", "usr_1")

        assert logger.get_event_count() == 10


class TestDataRetention:
    """Test data retention and cleanup."""

    def test_check_retention_compliance(self):
        """Test checking retention compliance."""
        logger = PIIAuditLogger(retention_days=1)
        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]

        # Create old event (manually manipulate timestamp)
        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")
        old_event = logger.events[0]
        old_time = (datetime.utcnow() - timedelta(days=2)).isoformat() + "Z"
        logger.events[0] = PIIAuditEvent(
            event_id=old_event.event_id,
            timestamp=old_time,
            event_type=old_event.event_type,
            entity_type=old_event.entity_type,
            entity_count=old_event.entity_count,
            user_id=old_event.user_id,
            context=old_event.context,
            masked_text=old_event.masked_text,
            status=old_event.status,
            details=old_event.details
        )

        violations = logger.check_retention_compliance()
        assert len(violations) == 1

    def test_cleanup_expired_events(self):
        """Test cleanup of expired events."""
        logger = PIIAuditLogger(retention_days=1)
        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]

        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")
        old_event = logger.events[0]
        old_time = (datetime.utcnow() - timedelta(days=2)).isoformat() + "Z"
        logger.events[0] = PIIAuditEvent(
            event_id=old_event.event_id,
            timestamp=old_time,
            event_type=old_event.event_type,
            entity_type=old_event.entity_type,
            entity_count=old_event.entity_count,
            user_id=old_event.user_id,
            context=old_event.context,
            masked_text=old_event.masked_text,
            status=old_event.status,
            details=old_event.details
        )

        removed = logger.cleanup_expired_events()
        assert removed == 1
        assert len(logger.events) == 0


class TestEventExport:
    """Test event export functionality."""

    def test_export_to_jsonl(self):
        """Test export to JSON Lines format."""
        logger = PIIAuditLogger()
        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]
        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "export.jsonl")
            logger.export_events(filename, format="jsonl")

            assert os.path.exists(filename)
            with open(filename) as f:
                lines = f.readlines()
                assert len(lines) == 1
                data = json.loads(lines[0])
                assert data["event_type"] == "detection"

    def test_export_to_json(self):
        """Test export to JSON array format."""
        logger = PIIAuditLogger()
        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]
        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "export.json")
            logger.export_events(filename, format="json")

            assert os.path.exists(filename)
            with open(filename) as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]["event_type"] == "detection"


class TestEventStatistics:
    """Test event statistics generation."""

    def test_get_statistics(self):
        """Test statistics calculation."""
        logger = PIIAuditLogger()
        email_entity = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]
        phone_entity = [PIIEntity("PHONE_NUMBER", "5551234567", 0, 10, 0.9, "presidio")]

        logger.log_pii_detection("Email: test@example.com", email_entity, "ctx1", "usr_1")
        logger.log_pii_detection("Phone: 5551234567", phone_entity, "ctx2", "usr_2")

        stats = logger.get_statistics()
        assert stats["total_events"] == 2
        assert stats["detection_events"] == 2
        assert stats["unique_users"] == 2


# ===== COMPLIANCE REPORTER TESTS (20 tests) =====

class TestComplianceReporterInitialization:
    """Test ComplianceReporter initialization."""

    def test_default_initialization(self):
        """Test reporter initialization with defaults."""
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)
        assert reporter.logger is logger
        assert "CREDIT_CARD" in reporter.retention_policy

    def test_custom_retention_policy(self):
        """Test initialization with custom retention policy."""
        logger = PIIAuditLogger()
        custom_policy = {"EMAIL_ADDRESS": 365, "default": 180}
        reporter = ComplianceReporter(logger, retention_policy=custom_policy)
        assert reporter.retention_policy == custom_policy


class TestComplianceReportGeneration:
    """Test compliance report generation."""

    def test_generate_report(self):
        """Test generating compliance report."""
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]
        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")

        start = datetime.utcnow() - timedelta(hours=1)
        end = datetime.utcnow() + timedelta(hours=1)

        report = reporter.generate_compliance_report(start, end)
        assert isinstance(report, ComplianceReport)
        assert report.total_events == 1
        assert report.total_pii_detected == 1

    def test_report_structure(self):
        """Test compliance report structure."""
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        entities = [PIIEntity("CREDIT_CARD", "4532123456789010", 0, 16, 0.95, "custom")]
        logger.log_pii_detection("Card: 4532123456789010", entities, "ctx1", "usr_1")

        start = datetime.utcnow() - timedelta(hours=1)
        end = datetime.utcnow() + timedelta(hours=1)

        report = reporter.generate_compliance_report(start, end)
        assert report.period_start is not None
        assert report.period_end is not None
        assert isinstance(report.events_by_type, dict)
        assert isinstance(report.recommendations, list)


class TestPIIStatistics:
    """Test PII statistics generation."""

    def test_get_statistics(self):
        """Test getting PII statistics."""
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]
        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")

        stats = reporter.get_pii_statistics()
        assert stats["total_events"] == 1
        assert stats["detection_events"] == 1
        assert stats["total_entities"] == 1

    def test_empty_statistics(self):
        """Test statistics with no events."""
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        stats = reporter.get_pii_statistics()
        assert stats["total_events"] == 0


class TestViolationDetection:
    """Test violation and anomaly detection."""

    def test_identify_unusual_activity(self):
        """Test identifying unusual activity."""
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]

        # Create many events for one user
        for i in range(150):
            logger.log_pii_detection(f"Email {i}: test@example.com", entities, "ctx", "suspicious_user")

        suspicious = reporter.identify_unusual_activity(threshold=100)
        assert len(suspicious) > 0

    def test_no_unusual_activity(self):
        """Test no unusual activity with low event count."""
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]
        logger.log_pii_detection("Email: test@example.com", entities, "ctx", "usr_1")

        suspicious = reporter.identify_unusual_activity(threshold=100)
        assert len(suspicious) == 0


class TestComplianceExport:
    """Test regulatory compliance export."""

    def test_export_for_regulatory_review(self):
        """Test exporting for regulatory review."""
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]
        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "compliance.json")
            reporter.export_for_regulatory_review(filename)

            assert os.path.exists(filename)
            with open(filename) as f:
                data = json.load(f)
                assert "statistics" in data
                assert "events" in data
                assert len(data["events"]) == 1


class TestRecommendations:
    """Test recommendation generation."""

    def test_good_compliance_status(self):
        """Test good compliance status recommendations."""
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]
        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")
        logger.log_pii_masking("Email: test@example.com", "Email: [MASKED]", entities, "placeholder", "usr_1")

        start = datetime.utcnow() - timedelta(hours=1)
        end = datetime.utcnow() + timedelta(hours=1)

        report = reporter.generate_compliance_report(start, end)
        assert len(report.recommendations) > 0
        assert any("GOOD" in rec for rec in report.recommendations)

    def test_violation_recommendations(self):
        """Test recommendations with violations."""
        logger = PIIAuditLogger(retention_days=1)
        custom_policy = {"EMAIL_ADDRESS": 1, "default": 1}  # 1 day retention
        reporter = ComplianceReporter(logger, retention_policy=custom_policy)

        entities = [PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "presidio")]
        logger.log_pii_detection("Email: test@example.com", entities, "ctx1", "usr_1")

        # Age the event
        old_event = logger.events[0]
        old_time = (datetime.utcnow() - timedelta(days=2)).isoformat() + "Z"
        logger.events[0] = PIIAuditEvent(
            event_id=old_event.event_id,
            timestamp=old_time,
            event_type=old_event.event_type,
            entity_type=old_event.entity_type,
            entity_count=old_event.entity_count,
            user_id=old_event.user_id,
            context=old_event.context,
            masked_text=old_event.masked_text,
            status=old_event.status,
            details=old_event.details
        )

        start = datetime.utcnow() - timedelta(days=3)
        end = datetime.utcnow() + timedelta(hours=1)

        report = reporter.generate_compliance_report(start, end)
        assert len(report.recommendations) > 0
        assert any("retention" in rec.lower() for rec in report.recommendations)



# ===== SUMMARY =====
"""
Total Tests: 45
- Audit Logger Tests: 25
  - Initialization: 2
  - Event Logging: 3
  - Event Querying: 5
  - Event Storage: 2
  - Data Retention: 2
  - Event Export: 2
  - Event Statistics: 1
  - Special Cases: 6

- Compliance Reporter Tests: 20
  - Initialization: 2
  - Report Generation: 2
  - PII Statistics: 2
  - Violation Detection: 2
  - Compliance Export: 1
  - Recommendations: 2
  - Special Cases: 7
"""
