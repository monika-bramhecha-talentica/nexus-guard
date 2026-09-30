"""
Agent Interface & Specialized Agents Module

This module defines the base agent interface that all specialized agents must implement,
as well as concrete implementations of domain-specific agents (Billing, Tech Support, Policy, etc.)
"""

from .base import BaseAgent, AgentResponse
from .billing_agent import BillingAgent
from .support_agent import TechSupportAgent
from .policy_agent import PolicyAgent
from .escalation_agent import EscalationAgent

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "BillingAgent",
    "TechSupportAgent",
    "PolicyAgent",
    "EscalationAgent",
]
