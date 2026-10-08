"""
Phase 3.2: Stream PII Processor

Async token-level streaming processor for real-time PII masking.
Targets sub-50ms latency for 100 tokens with buffering and batching.
"""

import asyncio
import logging
import time
from typing import List, AsyncGenerator, Optional
from dataclasses import dataclass, field

from .pii_detector import PIIDetector, PIIEntity
from .pii_masker import PIIMasker

logger = logging.getLogger(__name__)


@dataclass
class StreamStats:
    """Track stream processing metrics."""

    total_tokens: int = 0
    tokens_processed: int = 0
    entities_detected: int = 0
    entities_masked: int = 0
    latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0


class StreamPIIProcessor:
    """
    Async stream processor for PII detection and masking.

    Features:
    - Token-level buffering (configurable, default 512)
    - Batch processing for efficiency
    - Real-time latency tracking
    - Sub-50ms latency target for 100 tokens
    """

    def __init__(
        self,
        detector: PIIDetector,
        masker: PIIMasker,
        buffer_size: int = 512,
        batch_size: int = 100
    ):
        """
        Initialize stream processor.

        Args:
            detector: PIIDetector instance for entity detection
            masker: PIIMasker instance for entity masking
            buffer_size: Number of tokens to buffer before processing
            batch_size: Batch size for internal processing
        """
        self.detector = detector
        self.masker = masker
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer: List[str] = []
        self.stats = StreamStats()

        logger.info(
            f"StreamPIIProcessor initialized: "
            f"buffer_size={buffer_size}, batch_size={batch_size}"
        )

    async def process_stream(
        self,
        token_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """
        Process streaming tokens with PII detection and masking.

        Algorithm:
        1. Accumulate tokens in buffer until buffer_size reached
        2. Process buffer (detect + mask)
        3. Yield masked tokens
        4. Continue until stream ends

        Performance target: <50ms per 100 tokens

        Args:
            token_stream: Async generator yielding tokens

        Yields:
            Masked tokens one at a time
        """
        start_time = time.perf_counter()

        try:
            async for token in token_stream:
                self.buffer.append(token)
                self.stats.total_tokens += 1

                # Process when buffer reaches desired size
                if len(self.buffer) >= self.buffer_size:
                    masked_batch = await self.process_batch(self.buffer)
                    for masked_token in masked_batch:
                        yield masked_token
                    self.buffer = []

            # Process remaining tokens in buffer
            if self.buffer:
                masked_batch = await self.process_batch(self.buffer)
                for masked_token in masked_batch:
                    yield masked_token
                self.buffer = []

        finally:
            # Calculate final statistics
            elapsed = time.perf_counter() - start_time
            self.stats.latency_ms = elapsed * 1000
            if self.stats.total_tokens > 0:
                self.stats.throughput_tokens_per_sec = (
                    self.stats.total_tokens / elapsed
                )
            logger.info(
                f"Stream processing complete: "
                f"tokens={self.stats.total_tokens}, "
                f"latency={self.stats.latency_ms:.2f}ms, "
                f"throughput={self.stats.throughput_tokens_per_sec:.0f} tok/s"
            )

    async def process_batch(self, tokens: List[str]) -> List[str]:
        """
        Process a batch of tokens efficiently.

        Steps:
        1. Join tokens into text
        2. Detect PII entities
        3. Mask entities
        4. Split back into tokens

        Args:
            tokens: List of tokens to process

        Returns:
            List of masked tokens
        """
        if not tokens:
            return []

        # Join tokens into text (simple space join)
        text = " ".join(tokens)

        # Detect PII entities
        batch_start = time.perf_counter()
        entities = self.detector.detect(text)
        self.stats.entities_detected += len(entities)

        # Mask entities
        masked_text = self.masker.mask_text(text, entities)
        self.stats.entities_masked += len(entities)

        # Split back into tokens
        masked_tokens = masked_text.split()
        self.stats.tokens_processed += len(tokens)

        latency = (time.perf_counter() - batch_start) * 1000
        logger.debug(
            f"Batch processed: {len(tokens)} tokens, "
            f"{len(entities)} entities, {latency:.2f}ms"
        )

        return masked_tokens

    def get_latency_estimate(self) -> float:
        """
        Return estimated latency in ms for current configuration.

        Based on buffer size and batch processing overhead.

        Returns:
            Estimated latency in milliseconds
        """
        # Rough estimate: 0.05ms per token + 1ms overhead
        return (self.buffer_size * 0.05) + 1.0

    def reset_buffer(self) -> None:
        """
        Clear buffer and reset for new stream.

        Use this at session boundaries or to start fresh.
        """
        self.buffer.clear()
        logger.debug("Buffer reset")

    def reset_stats(self) -> None:
        """Reset statistics for new measurement period."""
        self.stats = StreamStats()
        logger.debug("Statistics reset")

    def get_stats(self) -> StreamStats:
        """
        Get current stream statistics.

        Returns:
            StreamStats dataclass with metrics
        """
        return self.stats

    def get_stats_dict(self) -> dict:
        """
        Get statistics as dictionary.

        Returns:
            Dictionary representation of StreamStats
        """
        return {
            "total_tokens": self.stats.total_tokens,
            "tokens_processed": self.stats.tokens_processed,
            "entities_detected": self.stats.entities_detected,
            "entities_masked": self.stats.entities_masked,
            "latency_ms": self.stats.latency_ms,
            "throughput_tokens_per_sec": self.stats.throughput_tokens_per_sec,
        }


async def create_token_stream(tokens: List[str]) -> AsyncGenerator[str, None]:
    """
    Helper function to create an async token stream from a list.

    Simulates streaming by yielding tokens with minimal delay.

    Args:
        tokens: List of tokens to stream

    Yields:
        Tokens one at a time
    """
    for token in tokens:
        yield token
        # Minimal delay to simulate async behavior
        await asyncio.sleep(0)
