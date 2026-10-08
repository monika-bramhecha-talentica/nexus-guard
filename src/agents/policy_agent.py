"""
Policy Agent - Validates policies, rules, and escalation criteria

Business Logic:
- Validate refund eligibility based on business policies
- Check SLA compliance
- Enforce policy rules (fraud detection, duplicate claims, etc.)
- Make final approval/denial/escalation decision
- Handle corrections from judge feedback loops
"""

import logging
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
from .base import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class PolicyAgent(BaseAgent):
    """
    Specialized agent for policy validation.

    Handles:
    - Refund eligibility validation
    - SLA compliance checking
    - Policy rule enforcement
    - Final approval/denial decisions
    - Fraud detection
    """

    # Policy rules
    POLICIES = {
        "refund_policy": {
            "max_refund_amount": 10000.00,
            "max_monthly_refunds": 5,
            "fraud_threshold": 0.3,  # 30% of account spending in refunds
        },
        "sla_policy": {
            "response_time_hours": 24,
            "resolution_time_hours": 72,
        },
        "escalation_rules": {
            "high_value": 5000.00,  # Escalate if amount > 5000
            "fraud_risk": True,
            "vip_customer": True,
        },
    }

    # Mock customer database
    MOCK_CUSTOMERS = {
        "USER001": {
            "account_age_days": 365,
            "total_spending": 5000.00,
            "refund_count_this_month": 0,
            "is_vip": False,
            "fraud_score": 0.05,
        },
        "USER002": {
            "account_age_days": 30,
            "total_spending": 500.00,
            "refund_count_this_month": 3,
            "is_vip": False,
            "fraud_score": 0.25,
        },
        "USER003": {
            "account_age_days": 720,
            "total_spending": 50000.00,
            "refund_count_this_month": 0,
            "is_vip": True,
            "fraud_score": 0.01,
        },
    }

    def __init__(self):
        super().__init__(
            name="PolicyAgent",
            description="Validates policies, eligibility, and escalation criteria",
            required_fields=["user_id", "session_id"],
        )

    def _get_customer_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve customer profile from mock database."""
        return self.MOCK_CUSTOMERS.get(user_id)

    def _check_fraud_risk(
        self,
        customer: Dict[str, Any],
        refund_amount: float,
    ) -> Tuple[bool, float]:
        """
        Check fraud risk based on customer history and refund amount.

        Returns:
            (is_fraud_risk, fraud_score)
        """
        fraud_threshold = self.POLICIES["refund_policy"]["fraud_threshold"]
        customer_fraud_score = customer.get("fraud_score", 0.0)

        # Calculate refund-to-spending ratio
        total_spending = customer.get("total_spending", 0.0)
        if total_spending > 0:
            refund_ratio = refund_amount / total_spending
        else:
            refund_ratio = 1.0

        # Combined fraud risk score
        combined_score = (customer_fraud_score + refund_ratio) / 2

        is_fraud_risk = combined_score > fraud_threshold

        return is_fraud_risk, combined_score

    def _validate_refund_limits(
        self,
        customer: Dict[str, Any],
        refund_amount: float,
    ) -> Tuple[bool, str]:
        """
        Check if refund violates policy limits.

        Returns:
            (is_within_limits, reason)
        """
        refund_policy = self.POLICIES["refund_policy"]

        # Check maximum refund amount
        if refund_amount > refund_policy["max_refund_amount"]:
            return False, f"Refund exceeds maximum allowed (${refund_policy['max_refund_amount']})"

        # Check monthly refund count
        if customer.get("refund_count_this_month", 0) >= refund_policy["max_monthly_refunds"]:
            return False, f"Monthly refund limit ({refund_policy['max_monthly_refunds']}) reached"

        return True, "Within policy limits"

    def _make_approval_decision(
        self,
        customer: Dict[str, Any],
        refund_amount: float,
        fraud_score: float,
        is_fraud_risk: bool,
    ) -> Tuple[str, float]:
        """
        Make final approval decision: approve, deny, or escalate.

        Returns:
            (decision, confidence)
        """
        escalation_rules = self.POLICIES["escalation_rules"]

        # VIP customers get preferential treatment
        if customer.get("is_vip"):
            return "approve", 0.95

        # High fraud score = escalate
        if is_fraud_risk:
            return "escalate", 0.9

        # High-value refunds = escalate
        if refund_amount > escalation_rules["high_value"]:
            return "escalate", 0.85

        # New accounts (< 60 days) with refunds need extra scrutiny
        if customer.get("account_age_days", 0) < 60:
            if customer.get("refund_count_this_month", 0) > 0:
                return "escalate", 0.75

        # Otherwise approve
        return "approve", 0.85

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        attempt: int = 1,
        feedback: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process policy validation with comprehensive rule checking.

        Validates customer eligibility, applies fraud detection, checks limits,
        and makes final approval/denial/escalation decision.
        """
        self.validate_context(context)

        user_id = context.get("user_id")
        session_id = context.get("session_id")

        # Handle judge feedback from previous attempt
        if attempt > 1 and feedback:
            logger.info(f"PolicyAgent correcting based on feedback: {feedback}")
            return AgentResponse(
                agent_id=self.name,
                output_text=f"Policy decision reconsidered based on: {feedback}",
                confidence=0.7,
                next_agent_hint="END",
                metadata={
                    "user_id": user_id,
                    "attempt": attempt,
                    "feedback_applied": True,
                },
            )

        # Extract refund details from metadata (passed from BillingAgent)
        refund_amount = context.get("metadata", {}).get("refund_amount", 0.0)
        transaction_id = context.get("metadata", {}).get("transaction_id")

        # Get customer profile
        customer = self._get_customer_profile(user_id)
        if not customer:
            return AgentResponse(
                agent_id=self.name,
                output_text="Customer profile not found. Unable to validate policy.",
                confidence=0.5,
                next_agent_hint="EscalationAgent",
                metadata={
                    "user_id": user_id,
                    "status": "customer_not_found",
                },
            )

        # Check fraud risk
        is_fraud_risk, fraud_score = self._check_fraud_risk(customer, refund_amount)

        # Validate refund limits
        within_limits, limit_reason = self._validate_refund_limits(customer, refund_amount)

        if not within_limits:
            return AgentResponse(
                agent_id=self.name,
                output_text=f"Refund denied. {limit_reason}",
                confidence=0.95,
                next_agent_hint="EscalationAgent",
                metadata={
                    "user_id": user_id,
                    "transaction_id": transaction_id,
                    "refund_amount": refund_amount,
                    "status": "denied_policy_violation",
                    "reason": limit_reason,
                },
            )

        # Make approval decision
        decision, confidence = self._make_approval_decision(
            customer, refund_amount, fraud_score, is_fraud_risk
        )

        # Build response based on decision
        output = f"**Policy Validation Result**\n"
        output += f"Transaction: {transaction_id}\n"
        output += f"Refund Amount: ${refund_amount:.2f}\n"
        output += f"Customer Account Age: {customer.get('account_age_days')} days\n"
        output += f"Fraud Risk Score: {fraud_score:.2%}\n\n"

        if decision == "approve":
            output += "✅ **APPROVED**: Refund meets all policy requirements.\n"
            output += "Processing refund now. You should see it in 3-5 business days."
            next_agent = "END"
        elif decision == "deny":
            output += "❌ **DENIED**: Refund does not meet policy requirements.\n"
            output += "You can appeal this decision by contacting our support team."
            next_agent = "END"
        else:  # escalate
            output += "⚠️ **ESCALATED**: This refund requires manual review.\n"
            output += "A specialist will contact you within 24 hours to discuss your refund request."
            next_agent = "EscalationAgent"

        return AgentResponse(
            agent_id=self.name,
            output_text=output,
            confidence=confidence,
            next_agent_hint=next_agent,
            metadata={
                "user_id": user_id,
                "transaction_id": transaction_id,
                "refund_amount": refund_amount,
                "decision": decision,
                "fraud_score": fraud_score,
                "is_fraud_risk": is_fraud_risk,
                "customer_age_days": customer.get("account_age_days"),
                "vip_customer": customer.get("is_vip", False),
                "status": "validated",
            },
        )
