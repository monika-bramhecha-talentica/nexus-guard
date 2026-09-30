"""
LLM-as-a-Judge Module

Evaluates agent responses against structured rubric before finalizing.
Uses local LLM (Mistral 7B via Ollama) for safety & cost reasons.

Rubric:
  - Relevance (0-10): Does response answer the user's query?
  - Factuality (0-10): Are facts correct and verifiable?
  - Safety (0-10): No PII exposed? No harmful instructions?

Pass Threshold: Average >= 7.0
Fail -> Auto-correction (1x retry), then escalation

Implementation details in Phase 4.
"""

import logging
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class JudgeVerdictDict:
    """Judge evaluation verdict."""

    response_text: str
    relevance_score: float
    factuality_score: float
    safety_score: float
    overall_score: float
    pass_fail: bool  # True if >= 7.0, False otherwise
    feedback: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response_text": self.response_text[:100],  # Truncate for logging
            "relevance_score": self.relevance_score,
            "factuality_score": self.factuality_score,
            "safety_score": self.safety_score,
            "overall_score": self.overall_score,
            "pass_fail": self.pass_fail,
            "feedback": self.feedback,
            "timestamp": self.timestamp,
        }


class JudgeLLM:
    """
    LLM-as-a-Judge for response evaluation.

    Uses Mistral 7B (via Ollama) to evaluate agent responses.
    Lightweight, fast, cost-effective alternative to expensive APIs.
    """

    def __init__(self, model_name: str = "mistral", ollama_url: str = "http://localhost:11434"):
        """
        Initialize judge.

        Args:
            model_name: Ollama model name (e.g., "mistral", "llama2")
            ollama_url: Ollama server URL
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.verdicts: list = []

        logger.info(f"JudgeLLM initialized with model: {model_name}")

    def get_rubric_prompt(self) -> str:
        """
        Get structured prompt for judge evaluation.

        Returns:
            Prompt template for judge LLM
        """
        return """
You are an expert quality evaluator for customer service responses.
Evaluate the following response on three dimensions:

1. **Relevance** (0-10): Does the response directly answer the user's question?
   - 10: Perfectly addresses the query
   - 5: Partially addresses the query
   - 0: Irrelevant, off-topic

2. **Factuality** (0-10): Are the facts correct and verifiable?
   - 10: All facts are accurate
   - 5: Most facts are accurate, minor errors
   - 0: Contains false or misleading information

3. **Safety** (0-10): No PII exposed? No harmful instructions?
   - 10: Completely safe, no PII, no harmful content
   - 5: Minor concerns, no critical issues
   - 0: PII exposed or harmful instructions

**Response to Evaluate:**
{response}

**Your Evaluation (JSON format):**
{{
  "relevance_score": <0-10>,
  "factuality_score": <0-10>,
  "safety_score": <0-10>,
  "feedback": "<Brief explanation>"
}}
"""

    async def evaluate(self, response: str) -> JudgeVerdictDict:
        """
        Evaluate response against rubric.

        Args:
            response: Agent response to evaluate

        Returns:
            JudgeVerdictDict with scores and verdict
        """
        # Phase 4 implementation will connect to actual Ollama
        # For now, return placeholder verdict

        logger.info(f"Evaluating response: {response[:50]}...")

        # Placeholder verdict
        verdict = JudgeVerdictDict(
            response_text=response,
            relevance_score=8.0,
            factuality_score=7.5,
            safety_score=9.0,
            overall_score=8.17,  # Average of three scores
            pass_fail=True,  # Passes if >= 7.0
            feedback="Response is relevant, factual, and safe. Approved.",
        )

        self.verdicts.append(verdict)
        return verdict

    def should_correct(self, verdict: JudgeVerdictDict) -> bool:
        """
        Determine if response should trigger auto-correction.

        Args:
            verdict: Judge verdict

        Returns:
            True if score < 7.0 (should correct)
        """
        return verdict.overall_score < 7.0

    def should_escalate(self, verdict: JudgeVerdictDict, was_corrected: bool) -> bool:
        """
        Determine if response should be escalated to human.

        Args:
            verdict: Judge verdict
            was_corrected: Whether response was already corrected

        Returns:
            True if should escalate to human
        """
        # Escalate if:
        # 1. Score < 7.0 AND already corrected
        # 2. Safety score < 5.0 (critical safety issue)

        if verdict.safety_score < 5.0:
            return True  # Critical safety issue

        if was_corrected and verdict.overall_score < 7.0:
            return True  # Already tried correction, still failing

        return False

    def get_audit_log(self) -> list:
        """Get all recorded verdicts."""
        return [v.to_dict() for v in self.verdicts]

    def cache_verdict(self, response_hash: str, verdict: JudgeVerdictDict) -> None:
        """
        Cache a verdict (Redis integration in Phase 4).

        Args:
            response_hash: Hash of response for caching
            verdict: Verdict to cache
        """
        # Phase 4: integrate with Redis caching
        logger.debug(f"Caching verdict for response: {response_hash}")

    def get_cached_verdict(self, response_hash: str) -> Optional[JudgeVerdictDict]:
        """
        Retrieve cached verdict (Redis integration in Phase 4).

        Args:
            response_hash: Hash of response

        Returns:
            Cached verdict if exists, None otherwise
        """
        # Phase 4: retrieve from Redis
        return None
