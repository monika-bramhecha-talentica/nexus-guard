"""
Loop Detection Engine

Detects infinite routing loops in agent networks.
Enforces maximum hop limit (default: 3 hops).

Implementation details in Phase 2.
"""

import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LoopEvent:
    """Record of a detected loop."""

    timestamp: str
    routing_path: List[str]
    loop_type: str  # "direct", "circular", or "max_hop"
    resolution: str


class LoopDetector:
    """
    Detects and prevents infinite routing loops.

    Loop Types:
    1. Direct Loop: Agent A → B → A (detected in last 2 hops)
    2. Circular Loop: A → B → C → A (detected in last 3 hops)
    3. Max Hop Exceeded: >3 agent hops (enforced limit)
    """

    def __init__(self, max_hops: int = 3):
        """
        Initialize loop detector.

        Args:
            max_hops: Maximum allowed hops before escalation
        """
        self.max_hops = max_hops
        self.loop_events: List[LoopEvent] = []

    def detect_loop(self, routing_path: List[str]) -> Tuple[bool, str]:
        """
        Detect if current routing path contains a loop.

        Args:
            routing_path: List of agent names in order visited

        Returns:
            (is_loop_detected, loop_type)
        """
        if not routing_path:
            return False, "no_loop"

        # Check: max hops exceeded
        if len(routing_path) > self.max_hops:
            return True, "max_hop"

        # Check: direct loop (last 2 hops)
        if len(routing_path) >= 2:
            if routing_path[-1] == routing_path[-2]:
                return True, "direct"

        # Check: agent appears multiple times in last N hops
        if len(routing_path) >= 3:
            # Check for any agent appearing twice in last 3
            last_three = routing_path[-3:]
            if len(set(last_three)) < len(last_three):  # Duplicate exists
                return True, "circular"

        return False, "no_loop"

    def is_valid_next_agent(self, current_path: List[str], next_agent: str) -> bool:
        """
        Check if routing to next_agent is valid (doesn't create loop).

        Args:
            current_path: Routing path so far
            next_agent: Agent we want to route to

        Returns:
            True if valid, False if would create loop
        """
        test_path = current_path + [next_agent]
        is_loop, _ = self.detect_loop(test_path)
        return not is_loop

    def get_loop_prevention_reason(self, routing_path: List[str]) -> str:
        """
        Get human-readable reason why a path is invalid.

        Args:
            routing_path: Path to check

        Returns:
            Explanation message
        """
        is_loop, loop_type = self.detect_loop(routing_path)

        if loop_type == "direct":
            return f"Direct loop detected: agent routed back to itself"
        elif loop_type == "circular":
            return f"Circular loop detected: agents cycling {routing_path[-3:]}"
        elif loop_type == "max_hop":
            return f"Maximum hop limit ({self.max_hops}) exceeded. Escalating to human support."
        else:
            return "No loop detected"

    def record_loop_event(
        self,
        routing_path: List[str],
        loop_type: str,
        resolution: str,
    ) -> None:
        """
        Record a loop detection event for audit trail.

        Args:
            routing_path: Path that triggered loop
            loop_type: Type of loop detected
            resolution: How loop was resolved
        """
        from datetime import datetime

        event = LoopEvent(
            timestamp=datetime.utcnow().isoformat(),
            routing_path=routing_path,
            loop_type=loop_type,
            resolution=resolution,
        )
        self.loop_events.append(event)
        logger.warning(f"Loop detected: {loop_type} in path {routing_path}")

    def get_audit_log(self) -> List[LoopEvent]:
        """Get all recorded loop detection events."""
        return self.loop_events
