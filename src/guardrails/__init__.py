"""
Guardrails Module - Safety mechanisms for agentic outputs

Three core guardrails:
1. Loop Detection: Prevent infinite agent routing
2. PII Masking: Real-time data privacy enforcement
3. LLM-as-a-Judge: Quality evaluation and auto-correction

Phase 3 Components:
- PIIDetector: Multi-layer PII detection (Presidio + custom patterns)
- PIIMasker: 4-strategy masking engine (placeholder, partial, hash, replacement)
- StreamPIIProcessor: Async token-level stream processor with sub-50ms latency
"""

from .loop_detector import LoopDetector
from .pii_masker import PIIMasker
from .judge import JudgeLLM
from .pii_detector import PIIDetector, PIIEntity, CustomPatternRegistry
from .stream_processor import StreamPIIProcessor, StreamStats

__all__ = [
    "LoopDetector",
    "PIIMasker",
    "JudgeLLM",
    "PIIDetector",
    "PIIEntity",
    "CustomPatternRegistry",
    "StreamPIIProcessor",
    "StreamStats",
]
