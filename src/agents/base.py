"""
Base Agent Interface for Nexus Guard

Defines the contract that all specialized agents must implement.
All agents inherit from BaseAgent and implement the execute() method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class AgentResponse:
    """
    Standard response format for all agents.

    Attributes:
        agent_id (str): Unique identifier of the agent that generated this response
        output_text (str): Main response content to send to user or next agent
        confidence (float): Confidence score (0.0-1.0) of agent's answer
        next_agent_hint (Optional[str]): Suggested next agent to route to (or "END" for user response)
        metadata (Dict[str, Any]): Additional context (cost, reasoning, etc.)
        status (str): "success" or "error"
        error_message (Optional[str]): Error description if status == "error"
        timestamp (str): ISO 8601 timestamp
    """

    agent_id: str
    output_text: str
    confidence: float
    next_agent_hint: Optional[str] = "END"
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "output_text": self.output_text,
            "confidence": self.confidence,
            "next_agent_hint": self.next_agent_hint,
            "metadata": self.metadata,
            "status": self.status,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents in Nexus Guard.

    All concrete agents (BillingAgent, TechSupportAgent, etc.) inherit from this class
    and implement the execute() method.

    Attributes:
        name (str): Unique agent identifier (e.g., "BillingAgent")
        description (str): Human-readable description of agent's purpose
        required_fields (List[str]): Fields that must be present in agent context
    """

    def __init__(self, name: str, description: str, required_fields: Optional[List[str]] = None):
        """
        Initialize base agent.

        Args:
            name: Unique agent name
            description: What this agent does
            required_fields: Fields required in execution context
        """
        self.name = name
        self.description = description
        self.required_fields = required_fields or []

    @abstractmethod
    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        attempt: int = 1,
        feedback: Optional[str] = None,
    ) -> AgentResponse:
        """
        Execute agent logic to process query and return response.

        Args:
            query: User query or routed task
            context: Execution context (session_id, user_id, routing history, etc.)
            attempt: Which execution attempt (1 = first, 2 = after judge correction)
            feedback: Optional feedback from judge for correction loops

        Returns:
            AgentResponse with output text, confidence, and routing hint

        Raises:
            ValueError: If required fields are missing
            Exception: Any domain-specific errors
        """
        pass

    def validate_context(self, context: Dict[str, Any]) -> bool:
        """
        Validate that context contains all required fields.

        Args:
            context: Execution context to validate

        Returns:
            True if valid, raises ValueError otherwise

        Raises:
            ValueError: If required fields missing
        """
        missing = [field for field in self.required_fields if field not in context]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return True

    def get_agent_info(self) -> Dict[str, Any]:
        """Return metadata about this agent."""
        return {
            "name": self.name,
            "description": self.description,
            "required_fields": self.required_fields,
        }

    async def correct(
        self,
        query: str,
        context: Dict[str, Any],
        feedback: str,
    ) -> AgentResponse:
        """
        Attempt to correct previous response based on judge feedback.
        Default implementation delegates to execute() with attempt=2 and feedback.

        Args:
            query: Original query
            context: Execution context
            feedback: Judge's feedback on why previous response failed

        Returns:
            Corrected AgentResponse
        """
        return await self.execute(
            query=query,
            context=context,
            attempt=2,
            feedback=feedback,
        )
