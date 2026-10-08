"""
Billing Agent - Handles refund requests, transaction queries, payment status

Business Logic:
- Extract refund amount, reason, and transaction details from query
- Check transaction status in mock database
- Calculate refund eligibility based on business rules
- Route to PolicyAgent for policy validation before processing
- Handle corrections from judge feedback loops
"""

import re
import logging
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
from .base import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class BillingAgent(BaseAgent):
    """
    Specialized agent for billing operations.

    Handles:
    - Refund requests and processing
    - Transaction history queries
    - Payment status checks
    - Invoice generation
    - Determines if refund should be escalated to PolicyAgent
    """

    # Mock transaction database
    MOCK_TRANSACTIONS = {
        "TXN001": {
            "amount": 150.00,
            "date": datetime.utcnow() - timedelta(days=5),
            "status": "completed",
            "method": "credit_card",
            "description": "Premium subscription"
        },
        "TXN002": {
            "amount": 45.50,
            "date": datetime.utcnow() - timedelta(days=30),
            "status": "completed",
            "method": "paypal",
            "description": "One-time purchase"
        },
        "TXN003": {
            "amount": 300.00,
            "date": datetime.utcnow() - timedelta(days=60),
            "status": "completed",
            "method": "credit_card",
            "description": "Annual plan"
        },
    }

    # Refund policy rules
    REFUND_POLICIES = {
        "standard": {"days": 30, "percentage": 100},  # Full refund within 30 days
        "extended": {"days": 60, "percentage": 75},   # 75% refund within 60 days
        "limited": {"days": 90, "percentage": 50},    # 50% refund within 90 days
    }

    def __init__(self):
        super().__init__(
            name="BillingAgent",
            description="Handles billing inquiries, refunds, and transaction queries",
            required_fields=["user_id", "session_id"],
        )

    def _extract_refund_request(self, query: str) -> Tuple[Optional[str], Optional[float], str]:
        """
        Extract transaction ID, refund amount, and reason from query.

        Returns:
            (transaction_id, amount, reason)
        """
        # Look for transaction ID patterns (TXN001, TXN-001, etc.)
        txn_match = re.search(r'(TXN[-_]?\d{3,})', query, re.IGNORECASE)
        transaction_id = txn_match.group(1).upper() if txn_match else None

        # Look for amount patterns ($50, 50 dollars, etc.)
        amount_match = re.search(r'\$?\d+\.?\d*', query)
        amount = float(amount_match.group(0).replace('$', '')) if amount_match else None

        # Extract reason (look for keywords)
        reason = "Not specified"
        reason_keywords = {
            "duplicate": "Duplicate charge",
            "defective": "Defective product",
            "unsatisfied": "Product unsatisfactory",
            "wrong": "Wrong item received",
            "missing": "Missing item",
            "broken": "Item broken on arrival",
        }
        for keyword, reason_text in reason_keywords.items():
            if keyword in query.lower():
                reason = reason_text
                break

        return transaction_id, amount, reason

    def _lookup_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Look up transaction in mock database."""
        return self.MOCK_TRANSACTIONS.get(transaction_id)

    def _calculate_refund_eligibility(
        self,
        transaction: Dict[str, Any],
        refund_amount: Optional[float] = None,
    ) -> Tuple[bool, str, float]:
        """
        Calculate refund eligibility based on transaction date and policy.

        Returns:
            (is_eligible, policy_applied, refund_amount)
        """
        txn_date = transaction["date"]
        days_since_txn = (datetime.utcnow() - txn_date).days
        original_amount = transaction["amount"]

        # Determine applicable policy
        if days_since_txn <= 30:
            policy = self.REFUND_POLICIES["standard"]
            policy_name = "Standard 30-day"
        elif days_since_txn <= 60:
            policy = self.REFUND_POLICIES["extended"]
            policy_name = "Extended 60-day"
        elif days_since_txn <= 90:
            policy = self.REFUND_POLICIES["limited"]
            policy_name = "Limited 90-day"
        else:
            return False, "Outside refund window", 0.0

        # Calculate refund amount
        calculated_refund = original_amount * (policy["percentage"] / 100)

        # If specific amount requested, use that if <= calculated_refund
        if refund_amount and refund_amount <= calculated_refund:
            final_refund = refund_amount
        else:
            final_refund = calculated_refund

        is_eligible = True
        return is_eligible, policy_name, final_refund

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        attempt: int = 1,
        feedback: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process billing query with full business logic.

        Extracts refund request details, validates transaction, checks eligibility,
        and routes to PolicyAgent for further validation or approves refund.
        """
        # Validate context
        self.validate_context(context)

        user_id = context.get("user_id")
        session_id = context.get("session_id")

        # Handle judge feedback from previous attempt
        if attempt > 1 and feedback:
            logger.info(f"BillingAgent correcting based on feedback: {feedback}")
            # In a real system, this would adjust logic based on feedback
            # For now, we'll return with adjusted confidence
            return AgentResponse(
                agent_id=self.name,
                output_text=f"Reprocessing refund request with corrections: {feedback}",
                confidence=0.7,
                next_agent_hint="PolicyAgent",
                metadata={
                    "user_id": user_id,
                    "attempt": attempt,
                    "feedback_applied": True,
                },
            )

        # Extract refund request details
        transaction_id, requested_amount, reason = self._extract_refund_request(query)

        # If no transaction ID found, request more information
        if not transaction_id:
            return AgentResponse(
                agent_id=self.name,
                output_text="I need more details to process your refund. Please provide your transaction ID (e.g., TXN001) and the refund amount.",
                confidence=0.5,
                next_agent_hint="END",
                metadata={
                    "user_id": user_id,
                    "status": "need_more_info",
                    "reason": "No transaction ID found",
                },
            )

        # Look up transaction
        transaction = self._lookup_transaction(transaction_id)
        if not transaction:
            return AgentResponse(
                agent_id=self.name,
                output_text=f"Transaction {transaction_id} not found in our system. Please verify the transaction ID and try again.",
                confidence=0.9,
                next_agent_hint="END",
                metadata={
                    "user_id": user_id,
                    "transaction_id": transaction_id,
                    "status": "transaction_not_found",
                },
            )

        # Check refund eligibility
        is_eligible, policy_name, refund_amount = self._calculate_refund_eligibility(
            transaction, requested_amount
        )

        if not is_eligible:
            return AgentResponse(
                agent_id=self.name,
                output_text=f"Unfortunately, this transaction ({transaction_id}) is outside our refund window. "
                           f"Transaction was {(datetime.utcnow() - transaction['date']).days} days ago. "
                           f"Standard refunds are available within 30 days.",
                confidence=0.95,
                next_agent_hint="EscalationAgent",
                metadata={
                    "user_id": user_id,
                    "transaction_id": transaction_id,
                    "status": "ineligible",
                    "reason": policy_name,
                },
            )

        # Eligible for refund - route to PolicyAgent for validation
        output = (
            f"Refund request received for transaction {transaction_id}.\n"
            f"Original amount: ${transaction['amount']:.2f}\n"
            f"Requested refund: ${refund_amount:.2f}\n"
            f"Policy applied: {policy_name}\n"
            f"Reason: {reason}\n\n"
            f"Routing to Policy validation before processing."
        )

        return AgentResponse(
            agent_id=self.name,
            output_text=output,
            confidence=0.85,
            next_agent_hint="PolicyAgent",  # Route to policy validation
            metadata={
                "user_id": user_id,
                "transaction_id": transaction_id,
                "original_amount": transaction["amount"],
                "refund_amount": refund_amount,
                "policy": policy_name,
                "reason": reason,
                "status": "eligible_for_validation",
            },
        )
