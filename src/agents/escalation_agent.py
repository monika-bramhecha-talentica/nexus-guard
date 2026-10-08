"""
Escalation Agent - Routes complex issues to human support

Business Logic:
- Create support tickets for escalated issues
- Determine priority based on issue type, customer value, and context
- Send notifications to appropriate support team
- Provide ticket tracking information
- Handle corrections from judge feedback loops
"""

import uuid
import logging
from typing import Any, Dict, Optional
from datetime import datetime
from .base import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class EscalationAgent(BaseAgent):
    """
    Specialized agent for escalations.

    Handles:
    - Complex queries that require human judgment
    - Loop detection recovery
    - Budget exhaustion handling
    - VIP/high-value customer escalations
    - Support ticket creation and routing
    """

    # Priority matrix
    PRIORITY_MATRIX = {
        "critical": {
            "sla_hours": 1,
            "team": "critical_support",
            "emoji": "🔴",
        },
        "high": {
            "sla_hours": 4,
            "team": "senior_support",
            "emoji": "🟠",
        },
        "normal": {
            "sla_hours": 24,
            "team": "standard_support",
            "emoji": "🟡",
        },
        "low": {
            "sla_hours": 72,
            "team": "general_support",
            "emoji": "🟢",
        },
    }

    # Mock support teams
    SUPPORT_TEAMS = {
        "critical_support": {"agents_available": 3, "avg_wait_time_min": 5},
        "senior_support": {"agents_available": 5, "avg_wait_time_min": 15},
        "standard_support": {"agents_available": 10, "avg_wait_time_min": 30},
        "general_support": {"agents_available": 15, "avg_wait_time_min": 45},
    }

    # Mock ticket database
    MOCK_TICKETS = {}

    def __init__(self):
        super().__init__(
            name="EscalationAgent",
            description="Routes issues to human support team for manual review",
            required_fields=["user_id", "session_id"],
        )

    def _determine_priority(self, context: Dict[str, Any]) -> str:
        """
        Determine priority based on escalation context.

        Returns:
            Priority level: critical, high, normal, or low
        """
        escalation_reason = context.get("escalation_reason", "").lower()
        is_vip = context.get("metadata", {}).get("vip_customer", False)
        refund_amount = context.get("metadata", {}).get("refund_amount", 0.0)
        fraud_score = context.get("metadata", {}).get("fraud_score", 0.0)

        # Critical: loop detection, token exhaustion, service down
        if any(reason in escalation_reason for reason in [
            "loop detected", "token budget exceeded", "service down", "critical"
        ]):
            return "critical"

        # High: large refunds, fraud risk, VIP customers
        if refund_amount > 5000 or fraud_score > 0.4 or is_vip:
            return "high"

        # Normal: standard escalations
        if any(reason in escalation_reason for reason in [
            "policy", "refund", "unable", "agent failed"
        ]):
            return "normal"

        # Low: general inquiries
        return "low"

    def _create_support_ticket(
        self,
        user_id: str,
        priority: str,
        escalation_reason: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Create a support ticket and return ticket ID.

        Returns:
            Ticket ID
        """
        ticket_id = f"TICKET-{uuid.uuid4().hex[:8].upper()}"

        ticket = {
            "id": ticket_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "priority": priority,
            "status": "open",
            "escalation_reason": escalation_reason,
            "metadata": context.get("metadata", {}),
            "routing_path": context.get("routing_path", []),
            "assigned_team": self.PRIORITY_MATRIX[priority]["team"],
            "sla_hours": self.PRIORITY_MATRIX[priority]["sla_hours"],
        }

        self.MOCK_TICKETS[ticket_id] = ticket
        logger.info(f"Created support ticket: {ticket_id} (Priority: {priority})")

        return ticket_id

    def _get_team_info(self, team_name: str) -> Dict[str, Any]:
        """Get team availability info."""
        return self.SUPPORT_TEAMS.get(team_name, {})

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        attempt: int = 1,
        feedback: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process escalation with full ticket creation and routing.

        Creates support ticket, assigns priority and team, and provides
        tracking information to customer.
        """
        self.validate_context(context)

        user_id = context.get("user_id")
        session_id = context.get("session_id")
        escalation_reason = context.get("escalation_reason", "Complex query requiring human review")

        # Handle judge feedback from previous attempt
        if attempt > 1 and feedback:
            logger.info(f"EscalationAgent processing feedback: {feedback}")
            # In a real system, would update ticket with new information
            return AgentResponse(
                agent_id=self.name,
                output_text=f"Updated escalation with new information: {feedback}\n\n"
                           "Your case has been forwarded to our specialist team.",
                confidence=0.9,
                next_agent_hint="END",
                metadata={
                    "user_id": user_id,
                    "attempt": attempt,
                    "feedback_applied": True,
                },
            )

        # Determine priority
        priority = self._determine_priority(context)

        # Create support ticket
        ticket_id = self._create_support_ticket(
            user_id, priority, escalation_reason, context
        )

        # Get team information
        team_name = self.PRIORITY_MATRIX[priority]["team"]
        team_info = self._get_team_info(team_name)

        # Build response
        priority_emoji = self.PRIORITY_MATRIX[priority]["emoji"]
        sla_hours = self.PRIORITY_MATRIX[priority]["sla_hours"]
        wait_time = team_info.get("avg_wait_time_min", 30)

        output = f"{priority_emoji} **ESCALATED TO HUMAN SUPPORT**\n\n"
        output += f"**Ticket ID**: {ticket_id}\n"
        output += f"**Priority**: {priority.upper()}\n"
        output += f"**Reason**: {escalation_reason}\n"
        output += f"**SLA Response Time**: {sla_hours} hour(s)\n"
        output += f"**Average Wait Time**: ~{wait_time} minutes\n\n"
        output += f"Your case has been assigned to our {team_name.replace('_', ' ').title()}.\n"
        output += f"A specialist will contact you shortly at your registered email/phone.\n\n"
        output += f"**Next Steps**:\n"
        output += f"1. Check your email for ticket confirmation\n"
        output += f"2. Reference your ticket ID: {ticket_id}\n"
        output += f"3. Our team will investigate and contact you within {sla_hours} hour(s)\n\n"
        output += f"Thank you for your patience. We appreciate your business."

        return AgentResponse(
            agent_id=self.name,
            output_text=output,
            confidence=1.0,
            next_agent_hint="END",
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "ticket_id": ticket_id,
                "priority": priority,
                "assigned_team": team_name,
                "escalation_reason": escalation_reason,
                "sla_hours": sla_hours,
                "status": "escalated_to_human",
            },
        )
