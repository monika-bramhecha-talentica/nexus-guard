"""
Deterministic Router for Agent Routing

LLM-based agent routing using Ollama + Mistral 7B.
Decides which agent should handle a query based on:
- Query content and context
- Agent capabilities and specializations
- Routing history and constraints
- Safety and loop prevention

All routing decisions are deterministic and auditable.
"""

import json
import logging
import re
import requests
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import config

logger = logging.getLogger(__name__)


class AgentCapability:
    """Defines what an agent can handle."""

    def __init__(
        self,
        name: str,
        description: str,
        keywords: List[str],
        capabilities: List[str],
        max_hops: int = 3,
    ):
        """
        Initialize agent capability.

        Args:
            name: Agent identifier (e.g., "BillingAgent")
            description: What this agent does
            keywords: Keywords that trigger this agent
            capabilities: What this agent can accomplish
            max_hops: Maximum routing hops for this agent
        """
        self.name = name
        self.description = description
        self.keywords = keywords
        self.capabilities = capabilities
        self.max_hops = max_hops


class DeterministicRouter:
    """
    Deterministic router using LLM for intelligent agent selection.

    Makes routing decisions based on:
    1. Query analysis (intent, domain)
    2. Agent capabilities and specialization
    3. Routing history and constraints
    4. Safety checks (loop prevention, max hops)

    Uses Ollama + Mistral 7B for reproducible routing.
    """

    # Define agent capabilities
    AGENT_REGISTRY: Dict[str, AgentCapability] = {
        "BillingAgent": AgentCapability(
            name="BillingAgent",
            description="Handles billing, refunds, transactions, and payment issues",
            keywords=["refund", "billing", "payment", "transaction", "charge", "invoice", "credit"],
            capabilities=[
                "Process refunds",
                "Check transaction status",
                "Verify payment history",
                "Escalate billing disputes",
            ],
        ),
        "TechSupportAgent": AgentCapability(
            name="TechSupportAgent",
            description="Handles technical issues, service failures, and troubleshooting",
            keywords=["error", "bug", "issue", "not working", "broken", "technical", "support"],
            capabilities=[
                "Diagnose technical issues",
                "Provide troubleshooting steps",
                "Check service status",
                "Escalate critical outages",
            ],
        ),
        "PolicyAgent": AgentCapability(
            name="PolicyAgent",
            description="Handles policy questions, eligibility, coverage, and compliance",
            keywords=["policy", "eligible", "coverage", "compliance", "terms", "conditions"],
            capabilities=[
                "Verify policy eligibility",
                "Explain coverage details",
                "Check compliance requirements",
                "Validate policy terms",
            ],
        ),
        "EscalationAgent": AgentCapability(
            name="EscalationAgent",
            description="Routes complex issues to human support",
            keywords=["escalate", "human", "urgent", "critical", "complex"],
            capabilities=[
                "Route to human support",
                "Handle complex escalations",
                "Create support tickets",
                "Notify support team",
            ],
        ),
    }

    def __init__(self, ollama_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize router with Ollama integration.

        Args:
            ollama_url: Ollama API endpoint (default: from config)
            model: LLM model name (default: from config)
        """
        self.ollama_url = ollama_url or config.OLLAMA_HOST
        self.model = model or config.OLLAMA_MODEL
        self.routing_history: Dict[str, List[str]] = {}

        logger.info(f"Router initialized: {self.ollama_url} | model: {self.model}")

    def _call_ollama(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Call Ollama API for LLM routing decision.

        Args:
            prompt: Routing prompt for the LLM
            max_tokens: Maximum tokens in response

        Returns:
            LLM response text
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,  # Low temperature for deterministic routing
                    "num_predict": max_tokens,
                },
                timeout=config.JUDGE_TIMEOUT_SEC,
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"LLM routing failed: {str(e)}")

    def _build_routing_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        available_agents: List[str],
    ) -> str:
        """
        Build structured prompt for LLM routing decision.

        Args:
            query: User query
            context: Execution context
            available_agents: List of agents available for routing

        Returns:
            Formatted prompt for LLM
        """
        routing_path = context.get("routing_path", [])
        attempt = context.get("attempt", 1)

        # Build agent options
        agent_options = []
        for agent_name in available_agents:
            if agent_name in self.AGENT_REGISTRY:
                agent = self.AGENT_REGISTRY[agent_name]
                agent_options.append(
                    f"- {agent_name}: {agent.description}\n"
                    f"  Capabilities: {', '.join(agent.capabilities)}"
                )

        routing_history_str = " → ".join(routing_path) if routing_path else "None"

        prompt = f"""You are a routing AI for a multi-agent customer support system.
Your task is to determine which agent should handle the next step of this customer request.

CURRENT QUERY: {query}

ROUTING HISTORY: {routing_history_str}
ATTEMPT NUMBER: {attempt}

AVAILABLE AGENTS:
{chr(10).join(agent_options)}

CONSTRAINTS:
1. Do NOT route to an agent already in the path (to prevent loops)
2. Consider the context and routing history
3. Choose the most appropriate agent for the current situation
4. If the query is resolved, respond "END" to return to the user
5. If unable to route, respond "EscalationAgent"

Based on the query and routing history, which agent should handle this?
Respond with ONLY the agent name (e.g., "BillingAgent", "TechSupportAgent", "PolicyAgent", "EscalationAgent", or "END").
Do not include any explanation."""

        return prompt

    def decide_next_agent(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Decide next agent using LLM-based routing.

        Args:
            query: Current user query
            context: Execution context (routing_path, attempt, etc.)

        Returns:
            Name of next agent, or "END" to return to user

        Raises:
            RuntimeError: If LLM routing fails
        """
        routing_path = context.get("routing_path", [])
        current_agent = context.get("current_agent", None)

        # Build list of available agents (exclude already visited)
        available_agents = [
            agent
            for agent in self.AGENT_REGISTRY.keys()
            if agent not in routing_path
        ]

        # If no agents left, escalate
        if not available_agents:
            logger.warning("No available agents - escalating to human support")
            return "EscalationAgent"

        # Build and execute routing prompt
        prompt = self._build_routing_prompt(query, context, available_agents)

        try:
            response = self._call_ollama(prompt)
            next_agent = response.strip()

            # Extract agent name (handle potential LLM verbosity)
            for agent_name in list(self.AGENT_REGISTRY.keys()) + ["END"]:
                if agent_name in next_agent:
                    next_agent = agent_name
                    break

            # Validate the decision
            if next_agent not in self.AGENT_REGISTRY and next_agent != "END":
                logger.warning(f"Invalid routing decision '{next_agent}' - defaulting to END")
                return "END"

            logger.info(
                f"Routing decision: {current_agent or 'START'} → {next_agent} "
                f"(path: {' → '.join(routing_path)})"
            )

            return next_agent

        except Exception as e:
            logger.error(f"Routing error: {e} - defaulting to EscalationAgent")
            return "EscalationAgent"

    def is_valid_next_step(
        self,
        current_agent: str,
        next_agent: str,
        routing_path: List[str],
    ) -> bool:
        """
        Validate that routing from current agent to next agent is safe.

        Prevents loops and enforces routing constraints.

        Args:
            current_agent: Agent making the suggestion
            next_agent: Suggested next agent
            routing_path: Full path taken so far

        Returns:
            True if valid, False if violates constraints
        """
        # Allow END always
        if next_agent == "END":
            return True

        # Check if agent exists
        if next_agent not in self.AGENT_REGISTRY:
            logger.warning(f"Invalid agent: {next_agent}")
            return False

        # Check for direct loop (self-routing)
        if next_agent == current_agent:
            logger.warning(f"Direct loop detected: {current_agent} → {current_agent}")
            return False

        # Check for immediate circular pattern
        if len(routing_path) >= 2 and next_agent in routing_path[-2:]:
            logger.warning(f"Circular pattern detected: {routing_path[-2:]} → {next_agent}")
            return False

        # Check for exceeding max hops
        max_hops = self.AGENT_REGISTRY[next_agent].max_hops
        if len(routing_path) >= max_hops:
            logger.warning(f"Max hops exceeded for {next_agent}: {len(routing_path)} >= {max_hops}")
            return False

        # Check if agent is already in path
        if next_agent in routing_path:
            logger.warning(f"Agent {next_agent} already in path: {routing_path}")
            return False

        return True

    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """
        Get information about an agent.

        Args:
            agent_name: Name of agent

        Returns:
            Agent metadata dictionary
        """
        if agent_name not in self.AGENT_REGISTRY:
            return {}

        agent = self.AGENT_REGISTRY[agent_name]
        return {
            "name": agent.name,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "keywords": agent.keywords,
            "max_hops": agent.max_hops,
        }

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all available agents and their capabilities.

        Returns:
            List of agent metadata dictionaries
        """
        return [self.get_agent_info(name) for name in self.AGENT_REGISTRY.keys()]
