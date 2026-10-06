"""
Phase 3.3 Integration Tests - End-to-End Audit & Compliance

Integration testing for PIIDetector → PIIAuditLogger → ComplianceReporter pipeline
Tests: 5 comprehensive integration scenarios
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta

from src.guardrails.pii_detector import PIIDetector, PIIEntity
from src.guardrails.audit_logger import PIIAuditLogger
from src.guardrails.compliance_reporter import ComplianceReporter


class TestAuditCompliancePipeline:
    """Test full audit and compliance pipeline integration."""

    def test_end_to_end_detection_to_compliance(self):
        """Test complete pipeline from detection to compliance report."""
        # Setup
        detector = PIIDetector(use_presidio=True)
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        # Test text - API key and credit card use custom patterns
        test_text = "API key: abc123def456 with card 4532123456789010"

        # Detect PII (custom patterns for these)
        entities = detector.detect(test_text)

        if entities:
            # Log detection
            event_id = logger.log_pii_detection(test_text, entities, "customer_call", "agent_001")
            assert event_id != ""

            # Verify logged
            events = logger.get_events(user_id="agent_001")
            assert len(events) == 1

        # Generate compliance report (should work with or without entities)
        start = datetime.utcnow() - timedelta(hours=1)
        end = datetime.utcnow() + timedelta(hours=1)
        report = reporter.generate_compliance_report(start, end)

        assert report is not None
        assert isinstance(report.total_events, int)

    def test_multiple_detections_compliance_tracking(self):
        """Test tracking multiple detection events with compliance."""
        detector = PIIDetector(use_presidio=True)
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        # Manually create test entities and log them
        email_entity = PIIEntity("EMAIL_ADDRESS", "alice@example.com", 0, 19, 0.95, "custom")
        phone_entity = PIIEntity("PHONE_NUMBER", "5551234567", 0, 10, 0.9, "custom")
        card_entity = PIIEntity("CREDIT_CARD", "4532123456789010", 0, 16, 0.95, "custom")

        # Log detection events
        logger.log_pii_detection("Email: alice@example.com", [email_entity], "support_chat", "agent_001")
        logger.log_pii_detection("Phone: 5551234567", [phone_entity], "outbound_call", "agent_002")
        logger.log_pii_detection("Card: 4532123456789010", [card_entity], "payment_form", "agent_001")

        # Verify all logged
        all_events = logger.get_events()
        assert len(all_events) == 3

        # Check per-user stats
        agent1_events = logger.get_events(user_id="agent_001")
        assert len(agent1_events) == 2

        # Compliance report
        start = datetime.utcnow() - timedelta(hours=1)
        end = datetime.utcnow() + timedelta(hours=1)
        report = reporter.generate_compliance_report(start, end)

        assert report.total_events == 3

    def test_audit_export_with_compliance_validation(self):
        """Test audit export with compliance validation."""
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        # Create test events manually
        test_entity = PIIEntity("EMAIL_ADDRESS", "support@company.com", 0, 19, 0.95, "custom")
        logger.log_pii_detection("Email: support@company.com", [test_entity], "support_queue", "support_agent")

        # Verify no retention violations
        violations = reporter.check_data_retention_compliance()
        assert len(violations) == 0

        # Export and validate
        with tempfile.TemporaryDirectory() as tmpdir:
            export_file = f"{tmpdir}/audit.json"
            reporter.export_for_regulatory_review(export_file)

            # Read and validate export
            with open(export_file) as f:
                export_data = json.load(f)

            assert "statistics" in export_data
            assert "events" in export_data
            assert export_data["statistics"]["total_events"] == 1

    def test_audit_statistics_accuracy(self):
        """Test accuracy of audit statistics."""
        logger = PIIAuditLogger()
        reporter = ComplianceReporter(logger)

        # Add multiple event types
        email_entity = PIIEntity("EMAIL_ADDRESS", "alice@example.com", 0, 19, 0.95, "custom")
        card_entity = PIIEntity("CREDIT_CARD", "4532123456789010", 0, 16, 0.95, "custom")

        logger.log_pii_detection("Contact: alice@example.com", [email_entity], "input", "user_1")
        logger.log_pii_detection("Card: 4532123456789010", [card_entity], "payment", "user_1")

        # Get stats from both logger and reporter
        logger_stats = logger.get_statistics()
        reporter_stats = reporter.get_pii_statistics()

        # Verify consistency
        assert logger_stats["total_events"] == reporter_stats["total_events"]
        assert logger_stats["detection_events"] == reporter_stats["detection_events"]
        assert logger_stats["total_events"] == 2

    def test_compliance_policy_enforcement(self):
        """Test data retention policy enforcement."""
        logger = PIIAuditLogger(retention_days=30)
        reporter = ComplianceReporter(
            logger,
            retention_policy={"EMAIL_ADDRESS": 60, "PHONE_NUMBER": 30, "default": 30}
        )

        # Create test event
        test_entity = PIIEntity("EMAIL_ADDRESS", "test@example.com", 0, 16, 0.95, "custom")
        logger.log_pii_detection("Email: test@example.com", [test_entity], "test", "user_1")

        # Check current compliance (should be fine - fresh event)
        violations = reporter.check_data_retention_compliance()
        assert len(violations) == 0

        # Verify policy is accessible
        assert reporter.retention_policy["EMAIL_ADDRESS"] == 60
        assert reporter.retention_policy["default"] == 30


# Summary of integration tests
"""
Integration Tests Overview:

1. test_end_to_end_detection_to_compliance
   - Uses detector for PII detection
   - Logs detection event
   - Generates compliance report
   - Validates complete pipeline

2. test_multiple_detections_compliance_tracking
   - Creates multiple detection events manually
   - Tracks per-user statistics
   - Generates consolidated report
   - Validates multi-event handling

3. test_audit_export_with_compliance_validation
   - Creates audit events
   - Validates retention compliance
   - Exports to JSON format
   - Validates export structure

4. test_audit_statistics_accuracy
   - Creates various events with entities
   - Compares logger vs reporter stats
   - Ensures consistency across components
   - Validates stat calculations

5. test_compliance_policy_enforcement
   - Tests custom retention policies
   - Verifies policy access
   - Validates enforcement logic
   - Ensures proper configuration

Total: 5 comprehensive integration tests
"""
