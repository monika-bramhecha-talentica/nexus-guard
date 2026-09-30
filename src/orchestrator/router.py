"""
Deterministic Router for Agent Routing

Decides which agent should handle a query based on:
- Query content
- Current execution context
- Agent capabilities
- Routing history

Implementation to be completed in Phase 2.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DeterministicRouter:
    """
    Deterministic router for multi-agent orchestration.

    Makes routing decisions based on structured context, not LLM randomness.
    Enables reproducible, auditable agent paths.
    """

    def __init__(self):
        """Initialize router."""
        self.routing_rules: Dict[str, Any] = {}

    def add_routing_rule(self, pattern: str, target_agent: str, priority: int = 0) -> None:
        """
        Add a routing rule.

        Args:
            pattern: Regex pattern to match against query
            target_agent: Agent to route to if pattern matches
            priority: Higher priority evaluated first
        """
        self.routing_rules[pattern] = {
            "target_agent": target_agent,
            "priority": priority,
        }

    def decide_next_agent(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Decide next agent to route to.

        Args:
            query: Current query
            context: Execution context (routing history, etc.)

        Returns:
            Name of next agent, or "END" to return to user
        """
        # Placeholder implementation
        # Full LLM-based routing with determinism in Phase 2

        routing_history = context.get("routing_path", [])

        # Simple heuristic: cycle through agents
        if len(routing_history) == 0:
            return "BillingAgent"
        elif "BillingAgent" in routing_history and "PolicyAgent" not in routing_history:
            return "PolicyAgent"
        else:
            return "END"

    def is_valid_next_step(
        self,
        current_agent: str,
        next_agent: str,
        routing_path: list,
    ) -> bool:
        """
        Validate that routing from current agent to next agent is valid.

        Prevents loops and enforces routing constraints.
        """
        # Check for direct loop
        if next_agent == current_agent:
            return False

        # Check for circular pattern
        if len(routing_path) >= 2 and next_agent in routing_path[-2:]:
            return False

        return True
