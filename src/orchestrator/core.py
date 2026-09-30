"""
Core Nexus Guard Orchestrator

Central coordinator that:
- Maintains agent registry
- Orchestrates agent execution
- Coordinates with guardrails (loop detection, PII masking, judge)
- Manages session state and token budgets
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from ..agents.base import BaseAgent, AgentResponse


logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Track state for a single user session."""

    session_id: str
    user_id: str
    org_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    routing_path: List[str] = field(default_factory=list)  # [Agent1, Agent2, ...]
    token_count: int = 0
    token_budget: int = 5000
    pii_events: List[Dict[str, Any]] = field(default_factory=list)
    loop_detection_events: List[Dict[str, Any]] = field(default_factory=list)
    judge_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    is_escalated: bool = False
    escalation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session state."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "created_at": self.created_at.isoformat(),
            "routing_path": self.routing_path,
            "token_count": self.token_count,
            "token_budget": self.token_budget,
            "pii_events": self.pii_events,
            "loop_detection_events": self.loop_detection_events,
            "judge_evaluations": self.judge_evaluations,
            "is_escalated": self.is_escalated,
            "escalation_reason": self.escalation_reason,
        }


class NexusGuardOrchestrator:
    """
    Main orchestrator for Nexus Guard.

    Responsibilities:
    1. Maintain registry of available agents
    2. Coordinate agent execution
    3. Manage session state
    4. Coordinate guardrails (loop detection, PII masking, judge)
    5. Track token budgets
    6. Maintain audit trails
    """

    def __init__(
        self,
        agents: Optional[List[BaseAgent]] = None,
        default_token_budget: int = 5000,
        max_hops: int = 3,
    ):
        """
        Initialize orchestrator.

        Args:
            agents: List of available agents
            default_token_budget: Default token limit per session
            max_hops: Maximum agent hops before escalation
        """
        self.agents: Dict[str, BaseAgent] = {}
        self.default_token_budget = default_token_budget
        self.max_hops = max_hops

        # Active sessions
        self.sessions: Dict[str, SessionState] = {}

        # Register initial agents
        if agents:
            for agent in agents:
                self.register_agent(agent)

        logger.info(f"NexusGuardOrchestrator initialized with {len(self.agents)} agents")

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the orchestrator.

        Args:
            agent: Agent to register
        """
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self.agents.get(agent_name)

    def list_agents(self) -> List[Dict[str, Any]]:
        """Get list of all registered agents."""
        return [agent.get_agent_info() for agent in self.agents.values()]

    def create_session(
        self,
        user_id: str,
        org_id: str,
        session_id: Optional[str] = None,
        token_budget: Optional[int] = None,
    ) -> SessionState:
        """
        Create a new user session.

        Args:
            user_id: User identifier
            org_id: Organization identifier
            session_id: Optional session ID (auto-generated if not provided)
            token_budget: Optional custom token budget (uses default if not provided)

        Returns:
            SessionState object
        """
        if not session_id:
            session_id = f"{user_id}_{org_id}_{datetime.utcnow().timestamp()}"

        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            token_budget=token_budget or self.default_token_budget,
        )

        self.sessions[session_id] = session
        logger.info(f"Created session: {session_id} for user {user_id}")
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def track_routing(self, session_id: str, agent_name: str) -> None:
        """
        Track that an agent was executed in this session.

        Args:
            session_id: Session ID
            agent_name: Agent name
        """
        session = self.get_session(session_id)
        if session:
            session.routing_path.append(agent_name)
            logger.info(f"Session {session_id} routing path: {session.routing_path}")

    def check_loop_detection(self, session_id: str, current_agent: str) -> bool:
        """
        Check if current agent execution would create a loop.

        Returns:
            True if loop detected, False otherwise
        """
        session = self.get_session(session_id)
        if not session:
            return False

        path = session.routing_path

        # Direct loop: current agent in last 2 hops
        if len(path) >= 2 and current_agent in path[-2:]:
            return True

        # Circular loop: A -> B -> C -> A pattern
        if len(path) >= 3:
            # Check if any agent appears twice in last 3 hops
            if len(set(path[-3:])) < 3:  # Duplicate found
                return True

        return False

    def record_loop_detection(self, session_id: str, loop_path: List[str], resolution: str) -> None:
        """
        Record a loop detection event.

        Args:
            session_id: Session ID
            loop_path: Routing path that created the loop
            resolution: How loop was resolved (e.g., "escalated")
        """
        session = self.get_session(session_id)
        if session:
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "loop_path": loop_path,
                "resolution": resolution,
            }
            session.loop_detection_events.append(event)
            logger.warning(f"Loop detected in session {session_id}: {loop_path}")

    def check_token_budget(self, session_id: str, tokens_to_add: int) -> bool:
        """
        Check if adding tokens would exceed budget.

        Args:
            session_id: Session ID
            tokens_to_add: Number of tokens to consume

        Returns:
            True if budget allows, False if would exceed
        """
        session = self.get_session(session_id)
        if not session:
            return False

        return (session.token_count + tokens_to_add) <= session.token_budget

    def consume_tokens(self, session_id: str, token_count: int) -> None:
        """
        Consume tokens from session budget.

        Args:
            session_id: Session ID
            token_count: Tokens to consume
        """
        session = self.get_session(session_id)
        if session:
            session.token_count += token_count
            if session.token_count >= session.token_budget:
                logger.warning(f"Session {session_id} exceeded token budget")

    def get_budget_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current budget status for session.

        Returns:
            {used, budget, remaining, percent_used}
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        return {
            "used": session.token_count,
            "budget": session.token_budget,
            "remaining": session.token_budget - session.token_count,
            "percent_used": (session.token_count / session.token_budget) * 100,
        }

    async def route_query(
        self,
        session_id: str,
        query: str,
        starting_agent: Optional[str] = None,
    ) -> AgentResponse:
        """
        Route a query through the agent network.

        This is a placeholder implementation. Full routing logic will be in Phase 1.

        Args:
            session_id: Session ID
            query: User query
            starting_agent: Starting agent (auto-select if not provided)

        Returns:
            Final AgentResponse
        """
        session = self.get_session(session_id)
        if not session:
            return AgentResponse(
                agent_id="Orchestrator",
                output_text="Session not found",
                confidence=0.0,
                status="error",
            )

        # Placeholder: route to BillingAgent as example
        agent = self.get_agent(starting_agent or "BillingAgent")
        if not agent:
            return AgentResponse(
                agent_id="Orchestrator",
                output_text=f"Agent {starting_agent} not found",
                confidence=0.0,
                status="error",
            )

        return await agent.execute(query, {"session_id": session_id, "user_id": session.user_id})

    def get_audit_log(self, session_id: str) -> Dict[str, Any]:
        """
        Get full audit trail for a session.

        Returns:
            Complete session state with all events
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        return session.to_dict()
