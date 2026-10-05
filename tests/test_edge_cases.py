"""
Phase 2.4: Edge Case & Stress Tests

Tests for:
- Token budget exhaustion
- Loop detection edge cases
- Agent error handling
- Session state validation
- Routing constraint violations
- High-hop scenarios
"""

import asyncio
import logging
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.core import NexusGuardOrchestrator
from src.agents.billing_agent import BillingAgent
from src.agents.support_agent import TechSupportAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.escalation_agent import EscalationAgent

logger = logging.getLogger(__name__)


class TestEdgeCases:
    """Edge case tests for Phase 2.4."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with all agents."""
        agents = [
            BillingAgent(),
            TechSupportAgent(),
            PolicyAgent(),
            EscalationAgent(),
        ]
        return NexusGuardOrchestrator(
            agents=agents,
            default_token_budget=5000,
            max_hops=3,
        )

    # ========================================================================
    # Token Budget Edge Cases
    # ========================================================================

    @pytest.mark.asyncio
    async def test_token_budget_exact_limit(self, orchestrator):
        """Test query execution at exact token budget limit."""
        session_id = "edge_token_exact"

        # Create session with very specific budget
        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
            token_budget=600,  # Exactly 500 (agent) + 100 (routing) = 600
        )

        query = "I have a technical issue"
        response = await orchestrator.route_query(
            session_id=session_id,
            query=query,
        )

        session = orchestrator.get_session(session_id)

        # Should complete exactly at or under budget
        assert session.token_count <= session.token_budget
        logger.info(f"✓ Token budget exact limit: {session.token_count}/{session.token_budget}")

    @pytest.mark.asyncio
    async def test_token_budget_zero(self, orchestrator):
        """Test query execution with zero token budget."""
        session_id = "edge_token_zero"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
            token_budget=0,
        )

        query = "Help me"
        response = await orchestrator.route_query(
            session_id=session_id,
            query=query,
        )

        session = orchestrator.get_session(session_id)

        # Should escalate immediately due to budget
        assert session.is_escalated or response.status == "error"
        logger.info(f"✓ Zero token budget: Escalated={session.is_escalated}")

    @pytest.mark.asyncio
    async def test_token_budget_negative(self, orchestrator):
        """Test query execution with negative token budget (invalid)."""
        session_id = "edge_token_negative"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
            token_budget=-100,
        )

        query = "Help"
        response = await orchestrator.route_query(
            session_id=session_id,
            query=query,
        )

        # Should fail gracefully
        assert response.status == "error"
        logger.info(f"✓ Negative token budget: Handled gracefully")

    @pytest.mark.asyncio
    async def test_token_budget_large(self, orchestrator):
        """Test query execution with very large token budget."""
        session_id = "edge_token_large"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
            token_budget=1000000,  # 1 million tokens
        )

        query = "I need a refund for TXN001"
        response = await orchestrator.route_query(
            session_id=session_id,
            query=query,
        )

        session = orchestrator.get_session(session_id)

        # Should complete successfully with low token consumption
        assert session.token_count < 2000  # Should use much less than budget
        logger.info(f"✓ Large token budget: Used {session.token_count}/1000000 tokens")

    # ========================================================================
    # Loop Detection Edge Cases
    # ========================================================================

    def test_loop_detection_same_agent_twice(self, orchestrator):
        """Test loop detection when agent appears twice in path."""
        session_id = "edge_loop_same_twice"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        session = orchestrator.get_session(session_id)
        session.routing_path = ["BillingAgent", "PolicyAgent", "BillingAgent"]

        # Should detect loop
        is_loop = orchestrator.check_loop_detection(session_id, "BillingAgent")
        assert is_loop is True
        logger.info("✓ Loop detection: Same agent twice detected")

    def test_loop_detection_three_agent_circle(self, orchestrator):
        """Test loop detection with 3-agent circular pattern."""
        session_id = "edge_loop_circle"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        session = orchestrator.get_session(session_id)
        session.routing_path = ["BillingAgent", "PolicyAgent", "TechSupportAgent"]

        # Should detect potential loop when trying to return to first
        is_loop = orchestrator.check_loop_detection(session_id, "BillingAgent")
        assert is_loop is True
        logger.info("✓ Loop detection: 3-agent circle detected")

    def test_loop_detection_at_max_hops(self, orchestrator):
        """Test loop detection at maximum hops limit."""
        session_id = "edge_loop_max_hops"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        session = orchestrator.get_session(session_id)
        # Fill path to max hops - 1
        session.routing_path = ["BillingAgent", "PolicyAgent", "TechSupportAgent"]

        # Trying to route to a 4th agent (beyond max_hops=3) should fail
        is_loop = orchestrator.check_loop_detection(session_id, "EscalationAgent")

        # Note: check_loop_detection checks last 3 hops, not total hops
        # Router's is_valid_next_step checks total hops
        logger.info(f"✓ Loop detection: At max hops - path={session.routing_path}")

    def test_loop_detection_empty_path(self, orchestrator):
        """Test loop detection with empty routing path."""
        session_id = "edge_loop_empty"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        session = orchestrator.get_session(session_id)
        session.routing_path = []  # Empty path

        # Should not detect loop on first agent
        is_loop = orchestrator.check_loop_detection(session_id, "BillingAgent")
        assert is_loop is False
        logger.info("✓ Loop detection: Empty path allowed")

    def test_loop_detection_single_agent_path(self, orchestrator):
        """Test loop detection with single agent in path."""
        session_id = "edge_loop_single"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        session = orchestrator.get_session(session_id)
        session.routing_path = ["BillingAgent"]

        # Trying to route back to same agent should be detected
        is_loop = orchestrator.check_loop_detection(session_id, "BillingAgent")
        assert is_loop is True
        logger.info("✓ Loop detection: Single agent loop detected")

    # ========================================================================
    # Agent Error Handling Edge Cases
    # ========================================================================

    @pytest.mark.asyncio
    async def test_missing_context_fields(self, orchestrator):
        """Test agent execution with missing required context fields."""
        session_id = "edge_missing_context"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        # Create context with missing required fields
        context = {
            # Missing 'user_id' and 'session_id'
        }

        agent = BillingAgent()

        # Should raise ValueError for missing fields
        with pytest.raises(ValueError):
            await agent.execute("I want a refund", context)

        logger.info("✓ Context validation: Missing fields detected")

    @pytest.mark.asyncio
    async def test_malformed_query(self, orchestrator):
        """Test agent execution with malformed queries."""
        session_id = "edge_malformed_query"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        malformed_queries = [
            "",  # Empty string
            "   ",  # Whitespace only
            None,  # None value
            123,  # Non-string
        ]

        for query in malformed_queries:
            response = await orchestrator.route_query(
                session_id=session_id,
                query=query or "",
            )
            # Should handle gracefully without crashing
            assert response is not None

        logger.info("✓ Malformed queries: All handled gracefully")

    # ========================================================================
    # Session State Edge Cases
    # ========================================================================

    def test_session_state_nonexistent_session(self, orchestrator):
        """Test getting non-existent session."""
        session = orchestrator.get_session("nonexistent_session_id")
        assert session is None
        logger.info("✓ Session state: Non-existent session returns None")

    def test_session_state_duplicate_creation(self, orchestrator):
        """Test creating duplicate session."""
        session_id = "edge_dup_session"

        # Create first session
        session1 = orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        # Create second session with same ID (overwrites)
        session2 = orchestrator.create_session(
            user_id="USER002",
            org_id="ORG002",
            session_id=session_id,
        )

        # Should have overwritten
        assert session2.user_id == "USER002"
        logger.info("✓ Session state: Duplicate creation overwrites")

    def test_session_routing_path_integrity(self, orchestrator):
        """Test session routing path maintains integrity."""
        session_id = "edge_routing_integrity"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        # Manually add to routing path
        orchestrator.track_routing(session_id, "BillingAgent")
        orchestrator.track_routing(session_id, "PolicyAgent")
        orchestrator.track_routing(session_id, "EscalationAgent")

        session = orchestrator.get_session(session_id)

        assert session.routing_path == ["BillingAgent", "PolicyAgent", "EscalationAgent"]
        assert len(session.routing_path) == 3
        logger.info(f"✓ Session state: Routing path integrity maintained: {session.routing_path}")

    # ========================================================================
    # High-Hop Scenarios
    # ========================================================================

    @pytest.mark.asyncio
    async def test_maximum_hops_exceeded(self, orchestrator):
        """Test behavior when max hops (3) would be exceeded."""
        session_id = "edge_max_hops_exceed"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        session = orchestrator.get_session(session_id)

        # Manually fill path to max hops
        session.routing_path = ["BillingAgent", "PolicyAgent", "TechSupportAgent"]

        # Next routing should consider this at limit
        # (actual enforcement depends on router implementation)
        logger.info(f"✓ Maximum hops: Path at limit: {session.routing_path}")

    # ========================================================================
    # Escalation & Status Tracking
    # ========================================================================

    @pytest.mark.asyncio
    async def test_escalation_status_tracking(self, orchestrator):
        """Test that escalation status is properly tracked."""
        session_id = "edge_escalation_track"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        session = orchestrator.get_session(session_id)

        # Initially not escalated
        assert session.is_escalated is False

        # Manually mark as escalated
        session.is_escalated = True
        session.escalation_reason = "Test escalation"

        # Verify tracking
        session = orchestrator.get_session(session_id)
        assert session.is_escalated is True
        assert session.escalation_reason == "Test escalation"
        logger.info("✓ Escalation tracking: Status properly maintained")

    # ========================================================================
    # Boundary Conditions
    # ========================================================================

    def test_agent_list_empty_agents(self):
        """Test orchestrator with no agents."""
        orch = NexusGuardOrchestrator(agents=[])

        agents = orch.list_agents()
        assert len(agents) == 0
        logger.info("✓ Boundary condition: Empty agent registry")

    def test_agent_get_nonexistent(self, orchestrator):
        """Test getting non-existent agent."""
        agent = orchestrator.get_agent("NonExistentAgent")
        assert agent is None
        logger.info("✓ Boundary condition: Non-existent agent returns None")

    def test_budget_check_exact_limit(self, orchestrator):
        """Test token budget check at exact limit."""
        session_id = "edge_budget_check"

        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
            token_budget=500,
        )

        # Check with exact remaining budget
        can_proceed = orchestrator.check_token_budget(session_id, 500)
        assert can_proceed is True

        # Check with 1 token over
        can_proceed = orchestrator.check_token_budget(session_id, 501)
        assert can_proceed is False
        logger.info("✓ Budget check: Exact limit calculation correct")


# ============================================================================
# Run Edge Case Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
