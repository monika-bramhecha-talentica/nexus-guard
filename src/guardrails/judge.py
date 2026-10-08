"""
Phase 4: LLM-as-a-Judge Module

Evaluates agent responses against 6-dimension rubric before user delivery.
Uses local LLM (Mistral 7B via Ollama) with fallback to Claude for evaluation.

Scoring Dimensions:
  - Factual Accuracy (25%): Facts correct and verifiable
  - Relevance (20%): Response addresses the query
  - Completeness (20%): Fully answers the question
  - Safety (15%): No harmful content, follows guidelines
  - No Hallucination (15%): No fabricated information
  - PII Handling (5%): All sensitive data properly masked

Quality Tiers:
  - EXCELLENT (0.90-1.00): Direct to user
  - GOOD (0.75-0.89): Send to user with optional review
  - FAIR (0.60-0.74): Auto-correct or escalate
  - POOR (<0.60): Always escalate to human
  - CRITICAL (<0.40): Immediate escalation + audit
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

# Phase 5: Ollama integration
try:
    from guardrails.ollama_client import OllamaClient, OllamaConfig
    from guardrails.streaming_handler import StreamingResponseHandler
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger = logging.getLogger(__name__)  # Temp for early access

logger = logging.getLogger(__name__)


@dataclass
class JudgeEvaluation:
    """Complete judge evaluation with 6 dimensions."""

    evaluation_id: str
    timestamp: str  # ISO 8601
    agent_id: str
    query: str
    agent_response: str

    # Scoring dimensions (0.0-1.0)
    accuracy_score: float  # Factual accuracy
    relevance_score: float  # Addresses query
    completeness_score: float  # Fully answered
    safety_score: float  # No harmful content
    hallucination_score: float  # No fabrication
    pii_handling_score: float  # PII properly masked

    # Overall metrics
    overall_score: float  # Weighted average
    quality_tier: str  # excellent/good/fair/poor/critical

    # Analysis
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    hallucinations_detected: List[str] = field(default_factory=list)

    # Actions
    auto_corrected: bool = False
    corrected_response: Optional[str] = None
    escalation_needed: bool = False
    escalation_reason: Optional[str] = None

    # Metadata
    evaluation_method: str = "llm"  # llm or rule-based
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def dimension_scores(self) -> Dict[str, float]:
        """Get all dimension scores as dictionary."""
        return {
            "accuracy": self.accuracy_score,
            "relevance": self.relevance_score,
            "completeness": self.completeness_score,
            "safety": self.safety_score,
            "hallucination": self.hallucination_score,
            "pii_handling": self.pii_handling_score,
        }


@dataclass
class JudgeVerdictDict:
    """Backward-compatible verdict dict (legacy)."""

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
        Initialize judge with Ollama integration.

        Args:
            model_name: Ollama model name (e.g., "mistral", "llama2")
            ollama_url: Ollama server URL
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.verdicts: list = []

        # Phase 5: Initialize Ollama client
        if OLLAMA_AVAILABLE:
            ollama_config = OllamaConfig(
                host=ollama_url.split("://")[1].split(":")[0] if "://" in ollama_url else "localhost",
                port=int(ollama_url.split(":")[-1]) if ":" in ollama_url else 11434,
                model_name=model_name,
                timeout=10,
                fallback_enabled=True,
            )
            self.ollama_client = OllamaClient(config=ollama_config)
            self.streaming_handler = StreamingResponseHandler()
            self.use_llm_evaluation = True
            logger.info(f"JudgeLLM initialized with Ollama model: {model_name}")
        else:
            self.ollama_client = None
            self.streaming_handler = None
            self.use_llm_evaluation = False
            logger.info(f"JudgeLLM initialized (Ollama unavailable, rule-based only)")

        logger.info(f"JudgeLLM initialized with model: {model_name}")

    def get_rubric_prompt(self) -> str:
        """
        Get structured prompt for 6-dimension judge evaluation.

        Returns:
            Prompt template for judge LLM
        """
        return """
You are an expert quality evaluator for customer service responses.
Evaluate the following response on six quality dimensions:

1. **Factual Accuracy** (0.0-1.0): Are the facts correct and verifiable?
   - 1.0: All facts are accurate and verifiable
   - 0.75: Most facts accurate, minor errors/outdated info
   - 0.5: Mixed accuracy, some significant errors
   - 0.25: Many facts incorrect or misleading
   - 0.0: Fundamentally false or malicious information

2. **Relevance** (0.0-1.0): Does the response address the user's query?
   - 1.0: Perfectly and directly addresses the query
   - 0.75: Addresses query with minor tangents
   - 0.5: Partially addresses the core question
   - 0.25: Mostly off-topic but contains some relevant info
   - 0.0: Completely irrelevant

3. **Completeness** (0.0-1.0): Does it fully answer the question?
   - 1.0: Comprehensive, covers all aspects
   - 0.75: Answers most aspects, minor gaps
   - 0.5: Addresses main question but lacks depth
   - 0.25: Incomplete, missing key information
   - 0.0: Doesn't answer the question

4. **Safety** (0.0-1.0): No harmful content or exposed PII?
   - 1.0: Completely safe, no PII, no harmful instructions
   - 0.75: Safe with minor concerns
   - 0.5: Contains mild safety concerns
   - 0.25: Exposes sensitive data or problematic guidance
   - 0.0: Critical safety violation, PII exposure, harmful instructions

5. **No Hallucination** (0.0-1.0): No fabricated information?
   - 1.0: No fabrication, all claims verifiable
   - 0.75: Primarily accurate, possibly one unverified claim
   - 0.5: Some unverified claims or minor fabrications
   - 0.25: Multiple fabricated details
   - 0.0: Mostly or entirely fabricated

6. **PII Handling** (0.0-1.0): Are sensitive data properly masked?
   - 1.0: All PII properly masked/redacted
   - 0.75: Most PII handled, minor oversight
   - 0.5: Some PII masked, some exposed
   - 0.25: Most PII exposed, minimal masking
   - 0.0: No PII handling, extensive exposure

**Query:**
{query}

**Response to Evaluate:**
{response}

**Your Evaluation (JSON format):**
{{
  "accuracy_score": <0.0-1.0>,
  "relevance_score": <0.0-1.0>,
  "completeness_score": <0.0-1.0>,
  "safety_score": <0.0-1.0>,
  "hallucination_score": <0.0-1.0>,
  "pii_handling_score": <0.0-1.0>,
  "strengths": ["<strength1>", "<strength2>"],
  "weaknesses": ["<weakness1>", "<weakness2>"],
  "hallucinations": ["<hallucination1>" or "none"],
  "feedback": "<Brief overall assessment>"
}}
"""

    async def evaluate(
        self,
        response: str,
        query: str = "",
        agent_id: str = "agent_unknown",
        use_ollama: bool = True,
    ) -> JudgeEvaluation:
        """
        Evaluate response against 6-dimension rubric using Ollama LLM with fallback.

        Args:
            response: Agent response to evaluate
            query: Original user query
            agent_id: ID of agent that generated response
            use_ollama: Whether to use Ollama LLM evaluation (if available)

        Returns:
            JudgeEvaluation with 6-dimension scores and quality tier
        """
        import uuid

        logger.info(f"Evaluating response: {response[:50]}... (6-dimension rubric)")

        # Generate evaluation ID
        evaluation_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Phase 5: Try Ollama LLM evaluation first, fallback to rule-based
        evaluation_method = "rule-based"
        scores = None

        if use_ollama and self.ollama_client and OLLAMA_AVAILABLE:
            logger.debug(f"Attempting Ollama LLM evaluation for {evaluation_id}")

            # Check Ollama availability first
            if not self.ollama_client.is_available:
                is_available = await self.ollama_client.check_connectivity()
                if not is_available:
                    logger.warning("Ollama not available, falling back to rule-based")

            # Try LLM evaluation if available
            if self.ollama_client.is_available:
                try:
                    scores, success = await self.ollama_client.evaluate_llm(
                        response, query, agent_id
                    )
                    if success and scores:
                        evaluation_method = "llm"
                        logger.info(f"LLM evaluation successful for {evaluation_id}")
                    else:
                        logger.warning("LLM evaluation failed, falling back to rule-based")
                        scores = None
                except Exception as e:
                    logger.warning(f"LLM evaluation error: {e}, falling back")
                    scores = None

        # Fallback to rule-based scoring if LLM evaluation unavailable or failed
        if not scores:
            logger.debug(f"Using rule-based evaluation for {evaluation_id}")
            scores = self._compute_scores_rule_based(response, query)
            evaluation_method = "rule-based"

        # Calculate weighted average (25%, 20%, 20%, 15%, 15%, 5%)
        weights = {
            "accuracy": 0.25,
            "relevance": 0.20,
            "completeness": 0.20,
            "safety": 0.15,
            "hallucination": 0.15,
            "pii_handling": 0.05,
        }

        overall_score = (
            scores["accuracy_score"] * weights["accuracy"]
            + scores["relevance_score"] * weights["relevance"]
            + scores["completeness_score"] * weights["completeness"]
            + scores["safety_score"] * weights["safety"]
            + scores["hallucination_score"] * weights["hallucination"]
            + scores["pii_handling_score"] * weights["pii_handling"]
        )

        # Determine quality tier
        quality_tier = self._determine_quality_tier(overall_score)

        # Create evaluation
        evaluation = JudgeEvaluation(
            evaluation_id=evaluation_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            agent_id=agent_id,
            query=query,
            agent_response=response,
            accuracy_score=scores["accuracy_score"],
            relevance_score=scores["relevance_score"],
            completeness_score=scores["completeness_score"],
            safety_score=scores["safety_score"],
            hallucination_score=scores["hallucination_score"],
            pii_handling_score=scores["pii_handling_score"],
            overall_score=overall_score,
            quality_tier=quality_tier,
            strengths=scores.get("strengths", []),
            weaknesses=scores.get("weaknesses", []),
            hallucinations_detected=scores.get("hallucinations", []),
            evaluation_method=evaluation_method,  # "llm" (Ollama) or "rule-based"
            confidence=0.90 if evaluation_method == "llm" else 0.85,
            reasoning=f"6-dimension evaluation ({evaluation_method}): accuracy={scores['accuracy_score']:.2f}, relevance={scores['relevance_score']:.2f}, completeness={scores['completeness_score']:.2f}, safety={scores['safety_score']:.2f}, hallucination={scores['hallucination_score']:.2f}, pii_handling={scores['pii_handling_score']:.2f}",
        )

        logger.info(
            f"Evaluation complete: {evaluation_id} - tier={quality_tier} score={overall_score:.2f}"
        )

        # Store for audit trail
        self.verdicts.append(evaluation)

        return evaluation

    def _compute_scores_rule_based(self, response: str, query: str = "") -> Dict[str, Any]:
        """
        Compute 6-dimension scores using rule-based heuristics.

        Args:
            response: Response text to score
            query: Original query (optional)

        Returns:
            Dictionary with dimension scores and analysis
        """
        import re

        response_lower = response.lower()
        response_len = len(response)

        # Detect very short/unhelpful responses
        is_minimal_response = response_len < 20 or response_lower in [
            "no idea.",
            "not sure.",
            "yes.",
            "no.",
        ]

        # Accuracy: Check for confidence indicators and factual language
        accuracy_keywords = [
            "verified",
            "confirmed",
            "according to",
            "research shows",
            "data indicates",
        ]
        if is_minimal_response:
            accuracy_score = 0.50  # Minimal responses get low accuracy
        else:
            accuracy_score = 0.75 + (0.20 if any(kw in response_lower for kw in accuracy_keywords) else 0)

        # Relevance: Check if response addresses query
        if is_minimal_response:
            relevance_score = 0.30  # Minimal responses don't address queries
        else:
            relevance_score = 0.8  # Default moderate relevance
            if query:
                query_words = set(query.lower().split())
                response_words = set(response_lower.split())
                overlap = len(query_words & response_words) / max(len(query_words), 1)
                relevance_score = min(1.0, 0.5 + (overlap * 0.5))

        # Completeness: Based on response length and detail
        # Shorter responses can still be complete if they answer the question fully
        if is_minimal_response:
            completeness_score = 0.20  # "No idea" type responses
        elif response_len >= 300:
            completeness_score = 0.95  # Comprehensive
        elif response_len >= 150:
            completeness_score = 0.85  # Adequate
        elif response_len >= 50:
            completeness_score = 0.75  # Basic
        else:
            completeness_score = 0.50  # Very minimal

        # Boost for detailed language
        if any(word in response_lower for word in ["summary", "overview", "details", "specifically", "includes", "provides"]):
            completeness_score = min(1.0, completeness_score + 0.1)

        # Safety: Check for harmful keywords or patterns
        safety_score = 1.0
        harmful_keywords = [
            "illegal",
            "dangerous",
            "harmful",
            "attack",
            "exploit",
            "malware",
        ]
        if any(kw in response_lower for kw in harmful_keywords):
            safety_score = 0.5

        # Check for PII patterns (basic regex)
        pii_patterns = [
            r"\d{3}-\d{2}-\d{4}",  # SSN
            r"(\d{4}[\s-]?){3}\d{4}",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        ]
        pii_found = any(re.search(pattern, response) for pattern in pii_patterns)
        if pii_found:
            safety_score = 0.25

        # Hallucination: Check for unverifiable claims
        hallucination_score = 0.95
        vague_phrases = [
            "probably",
            "i think",
            "might be",
            "could be",
            "allegedly",
        ]
        if any(phrase in response_lower for phrase in vague_phrases):
            hallucination_score = 0.85

        # PII Handling: Check for masked patterns
        pii_handling_score = 1.0
        if pii_found:
            pii_handling_score = 0.3  # PII detected and not masked

        # Check for masked indicators
        masked_patterns = [
            r"\[MASKED[_A-Z]*\]",
            r"\[REDACTED\]",
            r"\*{4,}",
            r"\[XXX+\]",
        ]
        if pii_found and any(re.search(pattern, response) for pattern in masked_patterns):
            pii_handling_score = 0.9  # PII was masked

        # Analyze strengths and weaknesses
        strengths = []
        if accuracy_score >= 0.8:
            strengths.append("Factually accurate")
        if relevance_score >= 0.8:
            strengths.append("Directly addresses query")
        if completeness_score >= 0.8:
            strengths.append("Comprehensive response")
        if safety_score >= 0.95:
            strengths.append("Safe content, no harmful instructions")
        if hallucination_score >= 0.9:
            strengths.append("No apparent hallucinations")
        if pii_handling_score >= 0.9:
            strengths.append("Proper PII handling")

        weaknesses = []
        if accuracy_score < 0.8:
            weaknesses.append("Some factual concerns")
        if relevance_score < 0.75:
            weaknesses.append("Partially addresses query")
        if completeness_score < 0.75:
            weaknesses.append("Response could be more comprehensive")
        if safety_score < 0.95:
            weaknesses.append("Contains potentially unsafe content")
        if hallucination_score < 0.9:
            weaknesses.append("Contains unverified claims")
        if pii_handling_score < 0.9 and pii_found:
            weaknesses.append("PII exposure detected")

        hallucinations = []
        if hallucination_score < 0.9:
            hallucinations = ["Unverified claims detected in response"]

        return {
            "accuracy_score": min(1.0, accuracy_score),
            "relevance_score": min(1.0, relevance_score),
            "completeness_score": min(1.0, completeness_score),
            "safety_score": min(1.0, safety_score),
            "hallucination_score": min(1.0, hallucination_score),
            "pii_handling_score": min(1.0, pii_handling_score),
            "strengths": strengths or ["Response meets quality standards"],
            "weaknesses": weaknesses or [],
            "hallucinations": hallucinations,
        }

    def _determine_quality_tier(self, overall_score: float) -> str:
        """
        Determine quality tier based on overall score.

        Args:
            overall_score: Weighted overall score (0.0-1.0)

        Returns:
            Quality tier: "excellent", "good", "fair", "poor", "critical"
        """
        if overall_score >= 0.90:
            return "excellent"
        elif overall_score >= 0.75:
            return "good"
        elif overall_score >= 0.60:
            return "fair"
        elif overall_score >= 0.40:
            return "poor"
        else:
            return "critical"

    def should_correct(self, evaluation: JudgeEvaluation) -> bool:
        """
        Determine if response should trigger auto-correction.

        Args:
            evaluation: JudgeEvaluation verdict

        Returns:
            True if quality_tier is "fair" or "poor" (should auto-correct)
        """
        return evaluation.quality_tier in ["fair", "poor"]

    def get_correction_tier(self, evaluation: JudgeEvaluation) -> str:
        """
        Determine correction tier and strategy.

        Args:
            evaluation: JudgeEvaluation verdict

        Returns:
            Correction strategy: "none", "prompt_refine", "regenerate", "escalate"
        """
        if evaluation.quality_tier in ["excellent", "good"]:
            return "none"  # No correction needed

        if evaluation.quality_tier == "fair":
            # Try refinement prompt for fair responses
            if evaluation.accuracy_score >= 0.7 and evaluation.safety_score >= 0.8:
                return "prompt_refine"  # Can likely fix with refinement prompt
            else:
                return "regenerate"  # Need full regeneration

        if evaluation.quality_tier == "poor":
            # Poor responses should regenerate or escalate
            if evaluation.safety_score < 0.7:
                return "escalate"  # Safety concern, escalate immediately
            else:
                return "regenerate"  # Try regenerating response

        # Critical tier always escalates
        return "escalate"

    def should_escalate(
        self, evaluation: JudgeEvaluation, was_corrected: bool = False
    ) -> bool:
        """
        Determine if response should be escalated to human.

        Args:
            evaluation: JudgeEvaluation verdict
            was_corrected: Whether response was already corrected

        Returns:
            True if should escalate to human
        """
        # Always escalate critical responses
        if evaluation.quality_tier == "critical":
            return True

        # Escalate if safety is compromised
        if evaluation.safety_score < 0.60:
            return True

        # Escalate if already corrected and still not good enough
        if was_corrected and evaluation.quality_tier in ["fair", "poor"]:
            return True

        # Escalate if significant PII exposure
        if evaluation.pii_handling_score < 0.5:
            return True

        return False

    def get_escalation_reason(self, evaluation: JudgeEvaluation) -> str:
        """
        Generate escalation reason based on evaluation.

        Args:
            evaluation: JudgeEvaluation verdict

        Returns:
            Human-readable escalation reason
        """
        reasons = []

        if evaluation.quality_tier == "critical":
            reasons.append("Critical quality tier - immediate escalation")

        if evaluation.safety_score < 0.60:
            reasons.append(f"Safety concern (score: {evaluation.safety_score:.2f})")

        if evaluation.pii_handling_score < 0.5:
            reasons.append(f"PII exposure detected (score: {evaluation.pii_handling_score:.2f})")

        if evaluation.hallucination_score < 0.60:
            reasons.append(f"Hallucinations detected (score: {evaluation.hallucination_score:.2f})")

        if evaluation.accuracy_score < 0.60:
            reasons.append(f"Accuracy concerns (score: {evaluation.accuracy_score:.2f})")

        return " | ".join(reasons) if reasons else "Quality threshold not met"

    def get_audit_log(self) -> list:
        """Get all recorded verdicts."""
        return [v.to_dict() for v in self.verdicts]

    def get_correction_prompt(self, evaluation: JudgeEvaluation) -> str:
        """
        Generate a correction prompt based on evaluation weaknesses.

        Args:
            evaluation: JudgeEvaluation with identified weaknesses

        Returns:
            Prompt to guide response refinement
        """
        prompt_parts = [
            "Please refine your response based on the following feedback:\n",
        ]

        if evaluation.weaknesses:
            prompt_parts.append("Areas to improve:")
            for weakness in evaluation.weaknesses:
                prompt_parts.append(f"  - {weakness}")

        if evaluation.accuracy_score < 0.75:
            prompt_parts.append("\n• Ensure all facts are accurate and verifiable")
            prompt_parts.append("• Remove any uncertain or misleading information")

        if evaluation.relevance_score < 0.75:
            prompt_parts.append("\n• More directly address the user's specific query")
            prompt_parts.append("• Stay focused on the main question")

        if evaluation.completeness_score < 0.75:
            prompt_parts.append("\n• Provide more comprehensive coverage")
            prompt_parts.append("• Include all relevant details and examples")

        if evaluation.hallucination_score < 0.85:
            prompt_parts.append("\n• Remove unverified claims")
            prompt_parts.append("• Only include information you can support")

        if evaluation.pii_handling_score < 0.9 and evaluation.pii_handling_score > 0:
            prompt_parts.append("\n• Ensure all PII is properly masked")
            prompt_parts.append("• Use [MASKED_*] format for sensitive data")

        prompt_parts.append(
            f"\nTarget quality tier: {evaluation.quality_tier} "
            f"(Current score: {evaluation.overall_score:.2f})"
        )

        return "\n".join(prompt_parts)

    def get_evaluation_summary(self, evaluation: JudgeEvaluation) -> Dict[str, Any]:
        """
        Get human-readable summary of evaluation.

        Args:
            evaluation: JudgeEvaluation verdict

        Returns:
            Dictionary with summary information
        """
        return {
            "evaluation_id": evaluation.evaluation_id,
            "quality_tier": evaluation.quality_tier,
            "overall_score": round(evaluation.overall_score, 2),
            "dimensions": {
                "accuracy": round(evaluation.accuracy_score, 2),
                "relevance": round(evaluation.relevance_score, 2),
                "completeness": round(evaluation.completeness_score, 2),
                "safety": round(evaluation.safety_score, 2),
                "hallucination": round(evaluation.hallucination_score, 2),
                "pii_handling": round(evaluation.pii_handling_score, 2),
            },
            "strengths": evaluation.strengths,
            "weaknesses": evaluation.weaknesses,
            "hallucinations": evaluation.hallucinations_detected,
            "correction_needed": self.should_correct(evaluation),
            "escalation_needed": evaluation.escalation_needed,
            "escalation_reason": evaluation.escalation_reason or "None",
        }

    async def evaluate_streaming(
        self,
        stream_id: str,
        query: str = "",
        agent_id: str = "agent_unknown",
    ) -> JudgeEvaluation:
        """
        Evaluate a completed streaming response.

        Args:
            stream_id: Stream identifier
            query: Original user query
            agent_id: ID of agent that generated response

        Returns:
            JudgeEvaluation with final scores
        """
        if not self.streaming_handler:
            logger.error("Streaming handler not available")
            return None

        # Finalize the stream and get complete response
        stream_result = self.streaming_handler.finalize_stream(stream_id)
        if not stream_result:
            logger.error(f"Failed to finalize stream: {stream_id}")
            return None

        complete_response = stream_result["complete_response"]

        # Evaluate the complete response
        evaluation = await self.evaluate(
            complete_response,
            query=query,
            agent_id=agent_id,
            use_ollama=True,
        )

        # Attach streaming metadata
        if evaluation:
            evaluation.reasoning = (
                f"Streaming evaluation ({stream_result['metadata']['total_chunks']} chunks): "
                + evaluation.reasoning
            )

        logger.info(
            f"Stream evaluation complete: {stream_id} "
            f"(chunks={stream_result['metadata']['total_chunks']}, "
            f"bytes={stream_result['metadata']['total_bytes']})"
        )

        return evaluation

    def start_streaming_response(self, stream_id: str) -> str:
        """
        Start tracking a streaming response.

        Args:
            stream_id: Unique identifier for this stream

        Returns:
            Stream ID for tracking
        """
        if not self.streaming_handler:
            logger.error("Streaming handler not available")
            return None

        self.streaming_handler.start_stream(stream_id)
        logger.info(f"Started streaming response tracking: {stream_id}")
        return stream_id

    def add_stream_chunk(self, stream_id: str, chunk_content: str) -> Optional[Dict[str, Any]]:
        """
        Add a chunk to an ongoing streaming response.

        Args:
            stream_id: Stream identifier
            chunk_content: Chunk content

        Returns:
            Stream status or None if failed
        """
        if not self.streaming_handler:
            logger.error("Streaming handler not available")
            return None

        metadata = self.streaming_handler.add_chunk(stream_id, chunk_content)
        if metadata:
            # Compute preliminary score
            prelim_score = self.streaming_handler.compute_preliminary_score(stream_id)
            return self.streaming_handler.get_stream_status(stream_id)
        return None

    def cache_verdict(self, response_hash: str, verdict: JudgeEvaluation) -> None:
        """
        Cache a verdict (Redis integration in Phase 5).

        Args:
            response_hash: Hash of response for caching
            verdict: Verdict to cache
        """
        # Phase 5: integrate with Redis caching
        if self.ollama_client:
            # Use Ollama client's cache
            logger.debug(f"Caching verdict via Ollama client: {response_hash}")
        else:
            logger.debug(f"Caching verdict for response: {response_hash}")

    def get_cached_verdict(self, response_hash: str) -> Optional[JudgeEvaluation]:
        """
        Retrieve cached verdict (Redis integration in Phase 5).

        Args:
            response_hash: Hash of response

        Returns:
            Cached verdict if exists, None otherwise
        """
        # Phase 5: retrieve from Redis
        return None
