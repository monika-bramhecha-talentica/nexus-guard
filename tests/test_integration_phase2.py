"""
Phase 2 Integration Tests: Router + Agents + Orchestrator

Tests end-to-end routing with LLM-based agent selection and full agent business logic.
Includes 3 example customer queries demonstrating multi-agent orchestration.
"""

import asyncio
import logging
import pytest
from typing import List

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.core import NexusGuardOrchestrator, SessionState
from src.orchestrator.router import DeterministicRouter
from src.agents.base import AgentResponse
from src.agents.billing_agent import BillingAgent
from src.agents.support_agent import TechSupportAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.escalation_agent import EscalationAgent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestIntegrationPhase2:
    """Integration tests for Phase 2: Router + Agents + Orchestrator"""

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

    @pytest.fixture
    def session_state(self):
        """Create a test session."""
        return SessionState(
            session_id="test_session_001",
            user_id="USER001",
            org_id="ORG001",
        )

    # ========================================================================
    # EXAMPLE QUERY 1: Refund Request Flow
    # ========================================================================
    # Expected flow: BillingAgent -> PolicyAgent -> END (approve) or EscalationAgent
    # ========================================================================

    @pytest.mark.asyncio
    async def test_example_1_refund_request(self, orchestrator, session_state):
        """
        Example Query 1: Customer requests refund for transaction TXN001

        Flow:
        1. Router selects BillingAgent (keyword "refund")
        2. BillingAgent validates transaction and calculates refund
        3. Routes to PolicyAgent for policy validation
        4. PolicyAgent approves and returns to customer
        """
        query = (
            "I want a refund for transaction TXN001 (Product was defective and "
            "I received it broken). The amount was $150."
        )

        logger.info("=" * 80)
        logger.info("EXAMPLE QUERY 1: Refund Request")
        logger.info("=" * 80)
        logger.info(f"Query: {query}\n")

        # Create session
        orchestrator.create_session(
            user_id=session_state.user_id,
            org_id=session_state.org_id,
            session_id=session_state.session_id,
        )

        # Route query through orchestrator
        response = await orchestrator.route_query(
            session_id=session_state.session_id,
            query=query,
        )

        # Verify routing path
        session = orchestrator.get_session(session_state.session_id)
        logger.info(f"Routing path: {' → '.join(session.routing_path)}")
        logger.info(f"Final response:\n{response.output_text}\n")
        logger.info(f"Confidence: {response.confidence}")
        logger.info(f"Status: {response.status}\n")

        # Assertions
        assert response.status == "success"
        assert "BillingAgent" in session.routing_path  # First agent
        assert len(session.routing_path) >= 1
        assert response.confidence > 0.5

    # ========================================================================
    # EXAMPLE QUERY 2: Technical Support Issue (Non-Escalation)
    # ========================================================================
    # Expected flow: TechSupportAgent -> END (with troubleshooting steps)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_example_2_technical_support(self, orchestrator):
        """
        Example Query 2: Customer reports slow performance

        Flow:
        1. Router selects TechSupportAgent (keyword "slow")
        2. TechSupportAgent diagnoses performance issue
        3. Provides troubleshooting steps
        4. Returns to customer without escalation
        """
        query = (
            "The dashboard is loading very slowly this morning. "
            "Takes 30+ seconds to load my data. Is there an outage?"
        )

        logger.info("=" * 80)
        logger.info("EXAMPLE QUERY 2: Technical Support - Non-Critical")
        logger.info("=" * 80)
        logger.info(f"Query: {query}\n")

        # Create session
        session_id = "test_session_002"
        orchestrator.create_session(
            user_id="USER002",
            org_id="ORG001",
            session_id=session_id,
        )

        # Route query
        response = await orchestrator.route_query(
            session_id=session_id,
            query=query,
        )

        # Verify routing
        session = orchestrator.get_session(session_id)
        logger.info(f"Routing path: {' → '.join(session.routing_path)}")
        logger.info(f"Final response:\n{response.output_text}\n")
        logger.info(f"Status: {response.status}\n")

        # Assertions
        assert response.status == "success"
        assert "TechSupportAgent" in session.routing_path
        assert response.next_agent_hint == "END"  # Should not escalate
        assert "Troubleshooting Steps" in response.output_text or "performance" in response.output_text.lower()

    # ========================================================================
    # EXAMPLE QUERY 3: Complex Refund With Escalation
    # ========================================================================
    # Expected flow: BillingAgent -> PolicyAgent -> EscalationAgent
    # ========================================================================

    @pytest.mark.asyncio
    async def test_example_3_complex_refund_escalation(self, orchestrator):
        """
        Example Query 3: Large refund request that triggers escalation

        Flow:
        1. Router selects BillingAgent (keyword "refund")
        2. BillingAgent finds transaction TXN003 ($300, 60 days old)
        3. Routes to PolicyAgent for validation
        4. PolicyAgent detects high-value refund -> escalates to EscalationAgent
        5. EscalationAgent creates support ticket and returns confirmation
        """
        query = (
            "I need to request a refund for my annual plan (TXN003 - $300). "
            "I purchased it 60 days ago but haven't used it. Can I get my money back?"
        )

        logger.info("=" * 80)
        logger.info("EXAMPLE QUERY 3: Complex Refund With Escalation")
        logger.info("=" * 80)
        logger.info(f"Query: {query}\n")

        # Create session
        session_id = "test_session_003"
        orchestrator.create_session(
            user_id="USER003",
            org_id="ORG001",
            session_id=session_id,
        )

        # Route query
        response = await orchestrator.route_query(
            session_id=session_id,
            query=query,
        )

        # Verify routing
        session = orchestrator.get_session(session_id)
        logger.info(f"Routing path: {' → '.join(session.routing_path)}")
        logger.info(f"Final response:\n{response.output_text}\n")
        logger.info(f"Status: {response.status}\n")
        logger.info(f"Token count: {session.token_count}/{session.token_budget}\n")

        # Assertions
        assert response.status == "success"
        assert "BillingAgent" in session.routing_path  # First agent
        # May or may not hit PolicyAgent depending on routing, but will show final response

    # ========================================================================
    # Test Suite Utilities & Validation
    # ========================================================================

    @pytest.mark.asyncio
    async def test_router_deterministic(self, orchestrator):
        """Test that router provides deterministic routing decisions."""
        query = "I have a billing issue with my recent charge"

        # Create session
        session_id = "test_router_determinism"
        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        # First routing
        response1 = await orchestrator.route_query(
            session_id=session_id,
            query=query,
        )
        path1 = orchestrator.get_session(session_id).routing_path

        # Reset session
        orchestrator.sessions[session_id].routing_path = []

        # Second routing (should match first)
        response2 = await orchestrator.route_query(
            session_id=session_id,
            query=query,
        )
        path2 = orchestrator.get_session(session_id).routing_path

        logger.info(f"Path 1: {path1}")
        logger.info(f"Path 2: {path2}")

        # Assertions: First agent selection should be consistent
        assert len(path1) > 0 and len(path2) > 0
        assert path1[0] == path2[0]  # First agent selection should match

    @pytest.mark.asyncio
    async def test_session_state_tracking(self, orchestrator):
        """Test that session state is properly tracked through routing."""
        query = "I have a technical issue"

        # Create session
        session_id = "test_session_state"
        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
            token_budget=5000,
        )

        # Initial state
        initial_session = orchestrator.get_session(session_id)
        assert initial_session.token_count == 0
        assert len(initial_session.routing_path) == 0

        # Route query
        await orchestrator.route_query(
            session_id=session_id,
            query=query,
        )

        # Final state
        final_session = orchestrator.get_session(session_id)
        assert final_session.token_count > 0  # Tokens consumed
        assert len(final_session.routing_path) > 0  # Routing tracked
        assert final_session.token_count <= final_session.token_budget

        logger.info(f"Routing path: {final_session.routing_path}")
        logger.info(f"Tokens consumed: {final_session.token_count}")

    @pytest.mark.asyncio
    async def test_token_budget_enforcement(self, orchestrator):
        """Test that token budget is enforced during routing."""
        query = "I need help with my account"

        # Create session with very small budget
        session_id = "test_token_budget"
        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
            token_budget=50,  # Very small budget
        )

        # Route query - should hit budget limit
        response = await orchestrator.route_query(
            session_id=session_id,
            query=query,
        )

        session = orchestrator.get_session(session_id)

        logger.info(f"Response status: {response.status}")
        logger.info(f"Tokens used: {session.token_count}/{session.token_budget}")
        logger.info(f"Escalated: {session.is_escalated}")

        # Assertion: Should escalate or complete within budget
        assert session.token_count <= session.token_budget

    @pytest.mark.asyncio
    async def test_loop_detection(self, orchestrator):
        """Test that loop detection prevents infinite routing cycles."""
        query = "I need help"

        # Create session
        session_id = "test_loop_detection"
        orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        # Manually create a routing loop
        session = orchestrator.get_session(session_id)
        session.routing_path = ["BillingAgent", "TechSupportAgent", "BillingAgent"]

        # Check loop detection
        is_loop = orchestrator.check_loop_detection(session_id, "BillingAgent")

        logger.info(f"Routing path: {session.routing_path}")
        logger.info(f"Loop detected: {is_loop}")

        # Assertion: Should detect loop
        assert is_loop is True

    def test_agent_info_metadata(self, orchestrator):
        """Test that agent metadata is properly accessible."""
        agents_info = orchestrator.list_agents()

        logger.info("Available agents:")
        for agent_info in agents_info:
            logger.info(f"  - {agent_info['name']}: {agent_info['description']}")

        # Assertions
        assert len(agents_info) == 4
        agent_names = [a['name'] for a in agents_info]
        assert "BillingAgent" in agent_names
        assert "TechSupportAgent" in agent_names
        assert "PolicyAgent" in agent_names
        assert "EscalationAgent" in agent_names


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
