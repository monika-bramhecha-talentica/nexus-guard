"""
Tech Support Agent - Handles service failures, troubleshooting, escalation

Business Logic:
- Categorize technical issues (service down, performance, feature request, etc.)
- Provide troubleshooting steps based on issue category
- Check incident status in mock database
- Route to EscalationAgent if critical or requires engineering
- Handle corrections from judge feedback loops
"""

import re
import logging
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from .base import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class TechSupportAgent(BaseAgent):
    """
    Specialized agent for technical support.

    Handles:
    - Service failure diagnosis and categorization
    - Troubleshooting steps based on issue type
    - Incident status queries
    - Escalation determination
    """

    # Issue categories and severity
    ISSUE_CATEGORIES = {
        "service_down": {
            "severity": "critical",
            "keywords": ["down", "offline", "not working", "unavailable"],
            "escalate": True,
        },
        "performance": {
            "severity": "high",
            "keywords": ["slow", "lag", "delay", "timeout", "performance"],
            "escalate": False,
        },
        "feature_request": {
            "severity": "low",
            "keywords": ["feature", "request", "new", "add", "implement"],
            "escalate": False,
        },
        "bug": {
            "severity": "high",
            "keywords": ["bug", "error", "crash", "broken", "issue"],
            "escalate": False,
        },
        "login_issue": {
            "severity": "high",
            "keywords": ["login", "password", "authentication", "access", "locked"],
            "escalate": False,
        },
    }

    # Troubleshooting steps by category
    TROUBLESHOOTING_STEPS = {
        "service_down": [
            "Check service status page at status.example.com",
            "Try accessing from a different device or network",
            "Clear browser cache and cookies",
            "If issue persists, we'll escalate to our engineering team",
        ],
        "performance": [
            "Check your internet connection speed",
            "Close other applications consuming bandwidth",
            "Try using a different browser",
            "Clear browser cache",
            "If issue persists, contact support with timing details",
        ],
        "feature_request": [
            "We appreciate your suggestion!",
            "Feature requests can be submitted in our feedback portal",
            "Our product team reviews requests regularly",
        ],
        "bug": [
            "Thank you for reporting this issue",
            "Please provide steps to reproduce the bug",
            "Note your browser version and OS",
            "Our engineering team will investigate",
        ],
        "login_issue": [
            "Verify your username/email is correct",
            "Use 'Forgot Password' to reset your password",
            "Check for browser extensions that might interfere",
            "Try clearing browser cookies",
            "If you're still locked out, we can help reset your account",
        ],
    }

    # Mock incident database
    MOCK_INCIDENTS = {
        "INC001": {
            "title": "API service degradation",
            "status": "investigating",
            "severity": "critical",
            "start_time": datetime.utcnow() - timedelta(hours=2),
            "updates": ["Identified database performance issue", "Scaling resources"],
        },
        "INC002": {
            "title": "Dashboard slow loading",
            "status": "monitoring",
            "severity": "high",
            "start_time": datetime.utcnow() - timedelta(days=1),
            "updates": ["Fixed query optimization", "Monitoring performance"],
        },
    }

    def __init__(self):
        super().__init__(
            name="TechSupportAgent",
            description="Handles technical support, troubleshooting, and service failure diagnosis",
            required_fields=["user_id", "session_id"],
        )

    def _categorize_issue(self, query: str) -> Tuple[str, str, int]:
        """
        Categorize the technical issue.

        Returns:
            (category, severity, escalate_flag)
        """
        query_lower = query.lower()

        # Match against category keywords
        for category, details in self.ISSUE_CATEGORIES.items():
            for keyword in details["keywords"]:
                if keyword in query_lower:
                    return category, details["severity"], 1 if details["escalate"] else 0

        # Default to bug category if no match
        return "bug", "high", 0

    def _get_troubleshooting_steps(self, category: str) -> List[str]:
        """Get troubleshooting steps for issue category."""
        return self.TROUBLESHOOTING_STEPS.get(category, [
            "Please provide more details about your issue",
            "Our support team will help you resolve this",
        ])

    def _lookup_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Look up incident status."""
        return self.MOCK_INCIDENTS.get(incident_id)

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        attempt: int = 1,
        feedback: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process technical support query with full diagnostic logic.

        Categorizes issue, provides troubleshooting steps, and determines if escalation
        to engineering is needed.
        """
        self.validate_context(context)

        user_id = context.get("user_id")
        session_id = context.get("session_id")

        # Handle judge feedback from previous attempt
        if attempt > 1 and feedback:
            logger.info(f"TechSupportAgent correcting based on feedback: {feedback}")
            return AgentResponse(
                agent_id=self.name,
                output_text=f"Adjusting troubleshooting approach: {feedback}\n\n"
                           "Please try these updated steps and let us know if the issue persists.",
                confidence=0.7,
                next_agent_hint="END",
                metadata={
                    "user_id": user_id,
                    "attempt": attempt,
                    "feedback_applied": True,
                },
            )

        # Check for incident ID reference
        incident_match = re.search(r'(INC[-_]?\d{3,})', query, re.IGNORECASE)
        if incident_match:
            incident_id = incident_match.group(1).upper()
            incident = self._lookup_incident(incident_id)

            if incident:
                status_text = f"**Incident {incident_id}**: {incident['title']}\n"
                status_text += f"Status: {incident['status'].upper()}\n"
                status_text += f"Severity: {incident['severity'].upper()}\n"
                status_text += f"Duration: {(datetime.utcnow() - incident['start_time']).total_seconds() / 3600:.1f} hours\n"
                status_text += f"Recent updates:\n"
                for update in incident["updates"]:
                    status_text += f"  - {update}\n"

                return AgentResponse(
                    agent_id=self.name,
                    output_text=status_text,
                    confidence=0.9,
                    next_agent_hint="END",
                    metadata={
                        "user_id": user_id,
                        "incident_id": incident_id,
                        "incident_status": incident["status"],
                        "type": "incident_status",
                    },
                )

        # Categorize the issue
        category, severity, should_escalate = self._categorize_issue(query)

        # Get troubleshooting steps
        steps = self._get_troubleshooting_steps(category)

        # Build response
        output = f"**Issue Category**: {category.replace('_', ' ').title()}\n"
        output += f"**Severity**: {severity.upper()}\n\n"
        output += "**Troubleshooting Steps**:\n"
        for i, step in enumerate(steps, 1):
            output += f"{i}. {step}\n"

        # Determine next agent
        if should_escalate or severity == "critical":
            output += "\n⚠️ This appears to be a critical issue. Escalating to engineering team."
            next_agent = "EscalationAgent"
            confidence = 0.9
        else:
            output += "\n\nPlease try these steps and let us know if the issue persists."
            next_agent = "END"
            confidence = 0.8

        return AgentResponse(
            agent_id=self.name,
            output_text=output,
            confidence=confidence,
            next_agent_hint=next_agent,
            metadata={
                "user_id": user_id,
                "category": category,
                "severity": severity,
                "escalate": should_escalate,
                "status": "diagnosed",
            },
        )
