"""
Phase 4: LLM-as-a-Judge Testing Module

Unit tests for 6-dimension evaluation framework:
- JudgeEvaluation dataclass validation
- 6-dimension scoring (accuracy, relevance, completeness, safety, hallucination, PII)
- Quality tier classification
- Auto-correction logic
- Escalation routing
- Rule-based scoring heuristics
"""

import pytest
from datetime import datetime
from src.guardrails.judge import JudgeEvaluation, JudgeLLM


class TestJudgeEvaluation:
    """Test JudgeEvaluation dataclass."""

    def test_judge_evaluation_creation(self):
        """Test creating a JudgeEvaluation instance."""
        eval = JudgeEvaluation(
            evaluation_id="eval_001",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="billing_agent",
            query="What's my account balance?",
            agent_response="Your balance is $500.",
            accuracy_score=0.95,
            relevance_score=1.0,
            completeness_score=0.85,
            safety_score=1.0,
            hallucination_score=0.95,
            pii_handling_score=1.0,
            overall_score=0.94,
            quality_tier="excellent",
        )

        assert eval.evaluation_id == "eval_001"
        assert eval.quality_tier == "excellent"
        assert eval.overall_score == 0.94
        assert eval.agent_id == "billing_agent"

    def test_judge_evaluation_to_dict(self):
        """Test converting JudgeEvaluation to dictionary."""
        eval = JudgeEvaluation(
            evaluation_id="eval_002",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="support_agent",
            query="How do I reset my password?",
            agent_response="Click the reset button in settings.",
            accuracy_score=0.9,
            relevance_score=0.95,
            completeness_score=0.8,
            safety_score=1.0,
            hallucination_score=1.0,
            pii_handling_score=1.0,
            overall_score=0.93,
            quality_tier="excellent",
        )

        eval_dict = eval.to_dict()

        assert isinstance(eval_dict, dict)
        assert eval_dict["evaluation_id"] == "eval_002"
        assert eval_dict["quality_tier"] == "excellent"

    def test_judge_evaluation_dimension_scores(self):
        """Test dimension_scores method."""
        eval = JudgeEvaluation(
            evaluation_id="eval_003",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="test_agent",
            query="Test query",
            agent_response="Test response",
            accuracy_score=0.85,
            relevance_score=0.90,
            completeness_score=0.88,
            safety_score=0.92,
            hallucination_score=0.89,
            pii_handling_score=0.95,
            overall_score=0.90,
            quality_tier="excellent",
        )

        scores = eval.dimension_scores()

        assert scores["accuracy"] == 0.85
        assert scores["relevance"] == 0.90
        assert scores["completeness"] == 0.88
        assert scores["safety"] == 0.92
        assert scores["hallucination"] == 0.89
        assert scores["pii_handling"] == 0.95


class TestJudgeLLMEvaluation:
    """Test JudgeLLM evaluation methods."""

    def test_judge_llm_initialization(self):
        """Test JudgeLLM initialization."""
        judge = JudgeLLM(model_name="mistral", ollama_url="http://localhost:11434")

        assert judge.model_name == "mistral"
        assert judge.ollama_url == "http://localhost:11434"
        assert judge.verdicts == []

    @pytest.mark.asyncio
    async def test_evaluate_excellent_response(self):
        """Test evaluation of excellent response."""
        judge = JudgeLLM()

        response = (
            "Your account balance is $1,234.56. This includes all transactions "
            "up to today. Your credit card is in good standing. For more details, "
            "please visit your account dashboard."
        )
        query = "What is my account balance?"

        evaluation = await judge.evaluate(response, query, "billing_agent")

        # Good quality response - meets or exceeds targets
        assert evaluation.quality_tier in ["excellent", "good"]
        assert evaluation.overall_score >= 0.75
        assert evaluation.accuracy_score >= 0.75
        assert evaluation.safety_score >= 0.95

    @pytest.mark.asyncio
    async def test_evaluate_good_response(self):
        """Test evaluation of good quality response."""
        judge = JudgeLLM()

        response = "The process takes approximately 5-7 business days to complete. Contact support for expedited options."
        query = "How long does processing take?"

        evaluation = await judge.evaluate(response, query, "support_agent")

        assert evaluation.quality_tier in ["excellent", "good"]
        assert evaluation.overall_score >= 0.75

    @pytest.mark.asyncio
    async def test_evaluate_fair_response(self):
        """Test evaluation of fair quality response."""
        judge = JudgeLLM()

        response = "It probably depends on various factors and might take some time."
        query = "How long will this take?"

        evaluation = await judge.evaluate(response, query, "generic_agent")

        assert evaluation.quality_tier in ["fair", "good"]
        # Fair responses have lower completeness scores
        assert evaluation.completeness_score < 0.85

    @pytest.mark.asyncio
    async def test_evaluate_poor_response(self):
        """Test evaluation of poor quality response."""
        judge = JudgeLLM()

        response = "No idea."
        query = "What is the status of my order?"

        evaluation = await judge.evaluate(response, query, "support_agent")

        assert evaluation.quality_tier in ["poor", "fair"]
        assert evaluation.completeness_score < 0.7


class TestScoringHeuristics:
    """Test rule-based scoring heuristics."""

    def test_score_accuracy_with_confidence_indicators(self):
        """Test accuracy scoring with confidence indicators."""
        judge = JudgeLLM()

        response_weak = "I think the answer might be around $500."
        response_strong = (
            "According to recent data, the confirmed amount is exactly $500.50, "
            "verified through three independent sources."
        )

        scores_weak = judge._compute_scores_rule_based(response_weak)
        scores_strong = judge._compute_scores_rule_based(response_strong)

        assert scores_strong["accuracy_score"] >= scores_weak["accuracy_score"]

    def test_score_relevance_with_query_overlap(self):
        """Test relevance scoring based on query-response overlap."""
        judge = JudgeLLM()

        query = "What is the refund policy?"
        response_relevant = (
            "Our refund policy allows returns within 30 days of purchase. "
            "The refund process typically takes 5-7 business days."
        )
        response_irrelevant = "We have many products available for purchase today."

        scores_relevant = judge._compute_scores_rule_based(response_relevant, query)
        scores_irrelevant = judge._compute_scores_rule_based(response_irrelevant, query)

        assert scores_relevant["relevance_score"] > scores_irrelevant["relevance_score"]

    def test_score_completeness_by_length(self):
        """Test completeness scoring based on response length."""
        judge = JudgeLLM()

        short_response = "Yes."
        long_response = (
            "Yes, this feature is available. We support it across all platforms including "
            "mobile, desktop, and web browsers. The feature was introduced in version 3.0 "
            "and has been continuously improved based on user feedback. For detailed setup "
            "instructions, please refer to our documentation."
        )

        scores_short = judge._compute_scores_rule_based(short_response)
        scores_long = judge._compute_scores_rule_based(long_response)

        assert scores_long["completeness_score"] > scores_short["completeness_score"]

    def test_score_safety_detects_harmful_content(self):
        """Test safety scoring detects harmful keywords."""
        judge = JudgeLLM()

        safe_response = "Here's the safest approach to solve this problem."
        unsafe_response = "You could try an illegal method to bypass the system."

        scores_safe = judge._compute_scores_rule_based(safe_response)
        scores_unsafe = judge._compute_scores_rule_based(unsafe_response)

        assert scores_safe["safety_score"] >= 0.95
        assert scores_unsafe["safety_score"] <= 0.5

    def test_score_pii_detection(self):
        """Test PII detection in responses."""
        judge = JudgeLLM()

        safe_response = "Your account information is [MASKED_EMAIL_ADDRESS]"
        unsafe_response = "Your SSN is 123-45-6789 and email is john@example.com"

        scores_safe = judge._compute_scores_rule_based(safe_response)
        scores_unsafe = judge._compute_scores_rule_based(unsafe_response)

        # Safe response should have good PII handling
        assert scores_safe["pii_handling_score"] >= 0.8
        # Unsafe response exposes PII
        assert scores_unsafe["pii_handling_score"] < 0.5

    def test_score_hallucination_with_vague_language(self):
        """Test hallucination scoring detects unverified claims."""
        judge = JudgeLLM()

        verified_response = "According to our records, the feature was launched on October 1, 2026."
        unverified_response = (
            "I think probably the feature might have been released, "
            "allegedly around that time."
        )

        scores_verified = judge._compute_scores_rule_based(verified_response)
        scores_unverified = judge._compute_scores_rule_based(unverified_response)

        assert scores_verified["hallucination_score"] >= 0.9
        assert scores_unverified["hallucination_score"] < 0.9


class TestQualityTierClassification:
    """Test quality tier determination."""

    def test_excellent_tier(self):
        """Test excellent quality tier classification."""
        judge = JudgeLLM()

        # Overall score >= 0.90 = excellent
        assert judge._determine_quality_tier(0.95) == "excellent"
        assert judge._determine_quality_tier(0.90) == "excellent"

    def test_good_tier(self):
        """Test good quality tier classification."""
        judge = JudgeLLM()

        # 0.75 <= overall_score < 0.90 = good
        assert judge._determine_quality_tier(0.85) == "good"
        assert judge._determine_quality_tier(0.75) == "good"
        assert judge._determine_quality_tier(0.89) == "good"

    def test_fair_tier(self):
        """Test fair quality tier classification."""
        judge = JudgeLLM()

        # 0.60 <= overall_score < 0.75 = fair
        assert judge._determine_quality_tier(0.70) == "fair"
        assert judge._determine_quality_tier(0.60) == "fair"
        assert judge._determine_quality_tier(0.74) == "fair"

    def test_poor_tier(self):
        """Test poor quality tier classification."""
        judge = JudgeLLM()

        # 0.40 <= overall_score < 0.60 = poor
        assert judge._determine_quality_tier(0.50) == "poor"
        assert judge._determine_quality_tier(0.40) == "poor"
        assert judge._determine_quality_tier(0.59) == "poor"

    def test_critical_tier(self):
        """Test critical quality tier classification."""
        judge = JudgeLLM()

        # overall_score < 0.40 = critical
        assert judge._determine_quality_tier(0.30) == "critical"
        assert judge._determine_quality_tier(0.0) == "critical"


class TestAutoCorrection:
    """Test auto-correction logic."""

    def test_should_correct_fair_tier(self):
        """Test should_correct returns True for fair tier."""
        judge = JudgeLLM()

        evaluation = JudgeEvaluation(
            evaluation_id="eval_fair",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="test_agent",
            query="Test",
            agent_response="Test response",
            accuracy_score=0.70,
            relevance_score=0.72,
            completeness_score=0.68,
            safety_score=0.90,
            hallucination_score=0.75,
            pii_handling_score=0.90,
            overall_score=0.75,
            quality_tier="fair",
        )

        assert judge.should_correct(evaluation) is True

    def test_should_not_correct_excellent_tier(self):
        """Test should_correct returns False for excellent tier."""
        judge = JudgeLLM()

        evaluation = JudgeEvaluation(
            evaluation_id="eval_excellent",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="test_agent",
            query="Test",
            agent_response="Test response",
            accuracy_score=0.95,
            relevance_score=0.98,
            completeness_score=0.92,
            safety_score=1.0,
            hallucination_score=0.98,
            pii_handling_score=1.0,
            overall_score=0.96,
            quality_tier="excellent",
        )

        assert judge.should_correct(evaluation) is False


class TestCorrectionTier:
    """Test correction tier determination."""

    def test_correction_tier_none(self):
        """Test no correction needed for excellent/good tiers."""
        judge = JudgeLLM()

        eval_excellent = JudgeEvaluation(
            evaluation_id="eval_001",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="agent",
            query="q",
            agent_response="r",
            accuracy_score=0.95,
            relevance_score=0.95,
            completeness_score=0.95,
            safety_score=0.95,
            hallucination_score=0.95,
            pii_handling_score=0.95,
            overall_score=0.95,
            quality_tier="excellent",
        )

        assert judge.get_correction_tier(eval_excellent) == "none"

    def test_correction_tier_prompt_refine(self):
        """Test prompt refinement for fair tier with good accuracy/safety."""
        judge = JudgeLLM()

        eval_fair = JudgeEvaluation(
            evaluation_id="eval_002",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="agent",
            query="q",
            agent_response="r",
            accuracy_score=0.80,  # Good
            relevance_score=0.65,  # Fair
            completeness_score=0.68,  # Fair
            safety_score=0.85,  # Good
            hallucination_score=0.80,  # Good
            pii_handling_score=0.90,  # Good
            overall_score=0.68,
            quality_tier="fair",
        )

        tier = judge.get_correction_tier(eval_fair)
        # Should prompt refine due to good accuracy/safety but fair overall
        assert tier in ["prompt_refine", "regenerate"]

    def test_correction_tier_escalate_safety_issue(self):
        """Test escalation for poor tier with safety issues."""
        judge = JudgeLLM()

        eval_poor_safety = JudgeEvaluation(
            evaluation_id="eval_003",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="agent",
            query="q",
            agent_response="r",
            accuracy_score=0.60,
            relevance_score=0.55,
            completeness_score=0.50,
            safety_score=0.50,  # Low safety
            hallucination_score=0.60,
            pii_handling_score=0.40,
            overall_score=0.48,
            quality_tier="poor",
        )

        tier = judge.get_correction_tier(eval_poor_safety)
        assert tier == "escalate"


class TestEscalation:
    """Test escalation logic."""

    def test_should_escalate_critical_tier(self):
        """Test escalation for critical tier."""
        judge = JudgeLLM()

        evaluation = JudgeEvaluation(
            evaluation_id="eval_critical",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="agent",
            query="q",
            agent_response="r",
            accuracy_score=0.30,
            relevance_score=0.20,
            completeness_score=0.10,
            safety_score=0.20,
            hallucination_score=0.25,
            pii_handling_score=0.15,
            overall_score=0.20,
            quality_tier="critical",
        )

        assert judge.should_escalate(evaluation) is True

    def test_should_escalate_safety_issue(self):
        """Test escalation for safety issues."""
        judge = JudgeLLM()

        evaluation = JudgeEvaluation(
            evaluation_id="eval_unsafe",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="agent",
            query="q",
            agent_response="r",
            accuracy_score=0.80,
            relevance_score=0.80,
            completeness_score=0.80,
            safety_score=0.50,  # Safety issue
            hallucination_score=0.80,
            pii_handling_score=0.80,
            overall_score=0.75,
            quality_tier="good",
        )

        assert judge.should_escalate(evaluation) is True

    def test_should_escalate_pii_exposure(self):
        """Test escalation for PII exposure."""
        judge = JudgeLLM()

        evaluation = JudgeEvaluation(
            evaluation_id="eval_pii",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="agent",
            query="q",
            agent_response="r",
            accuracy_score=0.80,
            relevance_score=0.80,
            completeness_score=0.80,
            safety_score=0.90,
            hallucination_score=0.80,
            pii_handling_score=0.30,  # PII exposure
            overall_score=0.75,
            quality_tier="good",
        )

        assert judge.should_escalate(evaluation) is True

    def test_should_not_escalate_excellent(self):
        """Test no escalation for excellent tier."""
        judge = JudgeLLM()

        evaluation = JudgeEvaluation(
            evaluation_id="eval_excellent",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="agent",
            query="q",
            agent_response="r",
            accuracy_score=0.95,
            relevance_score=0.95,
            completeness_score=0.95,
            safety_score=0.95,
            hallucination_score=0.95,
            pii_handling_score=0.95,
            overall_score=0.95,
            quality_tier="excellent",
        )

        assert judge.should_escalate(evaluation) is False


class TestEscalationReason:
    """Test escalation reason generation."""

    def test_escalation_reason_critical(self):
        """Test escalation reason for critical tier."""
        judge = JudgeLLM()

        evaluation = JudgeEvaluation(
            evaluation_id="eval_001",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="agent",
            query="q",
            agent_response="r",
            accuracy_score=0.30,
            relevance_score=0.20,
            completeness_score=0.10,
            safety_score=0.20,
            hallucination_score=0.25,
            pii_handling_score=0.15,
            overall_score=0.20,
            quality_tier="critical",
        )

        reason = judge.get_escalation_reason(evaluation)
        assert "critical" in reason.lower()

    def test_escalation_reason_multiple_issues(self):
        """Test escalation reason with multiple issues."""
        judge = JudgeLLM()

        evaluation = JudgeEvaluation(
            evaluation_id="eval_002",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="agent",
            query="q",
            agent_response="r",
            accuracy_score=0.50,
            relevance_score=0.45,
            completeness_score=0.40,
            safety_score=0.50,  # Issue
            hallucination_score=0.50,  # Issue
            pii_handling_score=0.30,  # Issue
            overall_score=0.45,
            quality_tier="poor",
        )

        reason = judge.get_escalation_reason(evaluation)
        # Should mention multiple issues
        assert "|" in reason  # Multiple reasons separated by |


class TestCorrectionPrompt:
    """Test correction prompt generation."""

    def test_correction_prompt_generation(self):
        """Test that correction prompts are generated appropriately."""
        judge = JudgeLLM()

        evaluation = JudgeEvaluation(
            evaluation_id="eval_001",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="agent",
            query="What are the steps?",
            agent_response="Do steps.",
            accuracy_score=0.70,
            relevance_score=0.65,
            completeness_score=0.60,
            safety_score=0.85,
            hallucination_score=0.75,
            pii_handling_score=0.90,
            overall_score=0.70,
            quality_tier="fair",
            weaknesses=[
                "Response could be more comprehensive",
                "Partially addresses query",
            ],
        )

        prompt = judge.get_correction_prompt(evaluation)

        assert "refine" in prompt.lower() or "improve" in prompt.lower()
        assert "fair" in prompt.lower()


class TestEvaluationSummary:
    """Test evaluation summary generation."""

    def test_evaluation_summary(self):
        """Test evaluation summary contains all required fields."""
        judge = JudgeLLM()

        evaluation = JudgeEvaluation(
            evaluation_id="eval_001",
            timestamp="2026-10-07T12:00:00Z",
            agent_id="billing_agent",
            query="What is my balance?",
            agent_response="Your balance is $500.",
            accuracy_score=0.95,
            relevance_score=0.98,
            completeness_score=0.90,
            safety_score=1.0,
            hallucination_score=0.95,
            pii_handling_score=1.0,
            overall_score=0.96,
            quality_tier="excellent",
            strengths=["Accurate", "Safe"],
            escalation_needed=False,
        )

        summary = judge.get_evaluation_summary(evaluation)

        assert summary["evaluation_id"] == "eval_001"
        assert summary["quality_tier"] == "excellent"
        assert summary["overall_score"] == 0.96
        assert "dimensions" in summary
        assert summary["dimensions"]["accuracy"] == 0.95
        assert summary["correction_needed"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
