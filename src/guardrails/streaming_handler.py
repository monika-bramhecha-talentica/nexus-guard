"""
Phase 5: Streaming Response Handler

Processes streaming responses in chunks for real-time evaluation.
Accumulates partial responses, computes preliminary scores, and finalizes on completion.

Features:
  - Buffer accumulation for partial responses
  - Chunk-by-chunk evaluation capability
  - Final evaluation on complete response
  - Metadata tracking (chunk count, streaming time)
  - Incremental scoring (preliminary and final)
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class StreamingChunk:
    """A single chunk from a streaming response."""

    chunk_id: int
    content: str
    timestamp: str
    chunk_size: int = 0

    def __post_init__(self):
        self.chunk_size = len(self.content)


@dataclass
class StreamingMetadata:
    """Metadata about streaming response."""

    stream_id: str
    start_time: str
    total_chunks: int = 0
    total_bytes: int = 0
    streaming_complete: bool = False
    end_time: Optional[str] = None
    total_streaming_time_ms: float = 0.0
    chunks: List[StreamingChunk] = field(default_factory=list)

    def add_chunk(self, chunk: StreamingChunk) -> None:
        """Add a chunk to streaming metadata."""
        self.chunks.append(chunk)
        self.total_chunks += 1
        self.total_bytes += chunk.chunk_size

    def finalize(self) -> None:
        """Mark streaming as complete and calculate duration."""
        self.streaming_complete = True
        self.end_time = datetime.utcnow().isoformat() + "Z"

        if self.chunks:
            start = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.end_time.replace("Z", "+00:00"))
            self.total_streaming_time_ms = (end - start).total_seconds() * 1000


class StreamingResponseHandler:
    """
    Handles evaluation of streaming responses.

    Accumulates partial responses, computes preliminary scores on chunks,
    and provides final evaluation when streaming completes.
    """

    def __init__(self):
        """Initialize streaming handler."""
        self.active_streams: Dict[str, StreamingMetadata] = {}
        self.stream_buffers: Dict[str, str] = {}
        self.chunk_scores: Dict[str, List[Dict[str, float]]] = {}

        logger.info("StreamingResponseHandler initialized")

    def start_stream(self, stream_id: str) -> StreamingMetadata:
        """
        Start a new streaming response.

        Args:
            stream_id: Unique identifier for this stream

        Returns:
            StreamingMetadata object for tracking
        """
        metadata = StreamingMetadata(
            stream_id=stream_id,
            start_time=datetime.utcnow().isoformat() + "Z",
        )

        self.active_streams[stream_id] = metadata
        self.stream_buffers[stream_id] = ""
        self.chunk_scores[stream_id] = []

        logger.info(f"Stream started: {stream_id}")
        return metadata

    def add_chunk(
        self, stream_id: str, chunk_content: str
    ) -> Optional[StreamingMetadata]:
        """
        Add a chunk to the streaming response.

        Args:
            stream_id: Stream identifier
            chunk_content: Chunk content text

        Returns:
            Updated StreamingMetadata or None if stream doesn't exist
        """
        if stream_id not in self.active_streams:
            logger.warning(f"Stream not found: {stream_id}")
            return None

        metadata = self.active_streams[stream_id]
        chunk_id = metadata.total_chunks

        # Create chunk object
        chunk = StreamingChunk(
            chunk_id=chunk_id,
            content=chunk_content,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        # Add to metadata
        metadata.add_chunk(chunk)

        # Accumulate in buffer
        self.stream_buffers[stream_id] += chunk_content

        logger.debug(
            f"Chunk {chunk_id} added to stream {stream_id} "
            f"({chunk.chunk_size} bytes, total: {metadata.total_bytes})"
        )

        return metadata

    def compute_preliminary_score(self, stream_id: str) -> Optional[Dict[str, float]]:
        """
        Compute preliminary evaluation score for accumulated chunks.

        Uses rule-based heuristics on accumulated buffer.

        Args:
            stream_id: Stream identifier

        Returns:
            Dictionary with preliminary dimension scores or None if stream invalid
        """
        if stream_id not in self.stream_buffers:
            logger.warning(f"Stream buffer not found: {stream_id}")
            return None

        accumulated_response = self.stream_buffers[stream_id]

        # Use simple heuristics for preliminary scoring
        # This will be enhanced with LLM evaluation in Phase 5
        preliminary_scores = self._compute_preliminary_heuristics(accumulated_response)

        # Store for later reference
        self.chunk_scores[stream_id].append(preliminary_scores)

        logger.debug(
            f"Preliminary score computed for stream {stream_id}: "
            f"accuracy={preliminary_scores.get('accuracy_score', 0):.2f}"
        )

        return preliminary_scores

    def finalize_stream(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """
        Finalize streaming response and compute final scores.

        Args:
            stream_id: Stream identifier

        Returns:
            Dictionary with final evaluation and metadata or None if stream invalid
        """
        if stream_id not in self.active_streams:
            logger.warning(f"Stream not found: {stream_id}")
            return None

        metadata = self.active_streams[stream_id]
        metadata.finalize()

        complete_response = self.stream_buffers[stream_id]

        # Compute final scores on complete response
        final_scores = self._compute_final_scores(complete_response)

        # Compute confidence adjustment based on completion
        confidence = self._adjust_confidence_for_streaming(metadata)

        result = {
            "stream_id": stream_id,
            "complete_response": complete_response,
            "metadata": {
                "total_chunks": metadata.total_chunks,
                "total_bytes": metadata.total_bytes,
                "streaming_time_ms": metadata.total_streaming_time_ms,
                "start_time": metadata.start_time,
                "end_time": metadata.end_time,
            },
            "final_scores": final_scores,
            "confidence": confidence,
            "chunk_count": len(metadata.chunks),
            "chunk_progression": [c.chunk_size for c in metadata.chunks],
        }

        logger.info(
            f"Stream finalized: {stream_id} "
            f"({metadata.total_chunks} chunks, {metadata.total_bytes} bytes)"
        )

        return result

    def get_stream_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a streaming response.

        Args:
            stream_id: Stream identifier

        Returns:
            Status dictionary with metadata and scores or None
        """
        if stream_id not in self.active_streams:
            return None

        metadata = self.active_streams[stream_id]
        accumulated = self.stream_buffers[stream_id]

        return {
            "stream_id": stream_id,
            "is_complete": metadata.streaming_complete,
            "chunks_received": metadata.total_chunks,
            "bytes_received": metadata.total_bytes,
            "accumulated_length": len(accumulated),
            "elapsed_time_ms": self._get_elapsed_time_ms(metadata),
        }

    def cleanup_stream(self, stream_id: str) -> None:
        """
        Clean up resources for a completed stream.

        Args:
            stream_id: Stream identifier
        """
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
        if stream_id in self.stream_buffers:
            del self.stream_buffers[stream_id]
        if stream_id in self.chunk_scores:
            del self.chunk_scores[stream_id]

        logger.debug(f"Stream cleaned up: {stream_id}")

    def _compute_preliminary_heuristics(
        self, accumulated_response: str
    ) -> Dict[str, float]:
        """
        Compute preliminary scores using rule-based heuristics.

        Args:
            accumulated_response: Accumulated response text so far

        Returns:
            Dictionary with preliminary dimension scores
        """
        response_lower = accumulated_response.lower()
        response_len = len(accumulated_response)

        # Preliminary accuracy (based on length and confidence indicators)
        accuracy_keywords = [
            "verified",
            "confirmed",
            "according to",
            "research shows",
        ]
        accuracy_score = (
            0.75 + (0.15 if any(kw in response_lower for kw in accuracy_keywords) else 0)
        )

        # Preliminary relevance (assume good if has substantial content)
        relevance_score = 0.75 if response_len > 100 else 0.60

        # Preliminary completeness (based on length)
        if response_len >= 300:
            completeness_score = 0.85
        elif response_len >= 100:
            completeness_score = 0.70
        else:
            completeness_score = 0.50  # Incomplete preliminary

        # Preliminary safety
        safety_score = 1.0
        if any(kw in response_lower for kw in ["illegal", "attack", "exploit"]):
            safety_score = 0.5

        # Preliminary hallucination
        hallucination_score = (
            0.85 if any(p in response_lower for p in ["probably", "might be"]) else 0.95
        )

        # Preliminary PII handling
        pii_handling_score = 0.95

        return {
            "accuracy_score": min(1.0, accuracy_score),
            "relevance_score": min(1.0, relevance_score),
            "completeness_score": min(1.0, completeness_score),
            "safety_score": min(1.0, safety_score),
            "hallucination_score": min(1.0, hallucination_score),
            "pii_handling_score": min(1.0, pii_handling_score),
        }

    def _compute_final_scores(self, complete_response: str) -> Dict[str, float]:
        """
        Compute final evaluation scores on complete response.

        Args:
            complete_response: Complete response text

        Returns:
            Dictionary with final dimension scores
        """
        # For now, uses same heuristics as preliminary
        # Phase 5 will integrate Ollama LLM evaluation here
        return self._compute_preliminary_heuristics(complete_response)

    def _adjust_confidence_for_streaming(self, metadata: StreamingMetadata) -> float:
        """
        Adjust confidence based on streaming characteristics.

        Args:
            metadata: StreamingMetadata with chunk information

        Returns:
            Confidence score (0.0-1.0)
        """
        base_confidence = 0.85

        # Boost confidence for larger responses
        if metadata.total_bytes > 1000:
            base_confidence = min(1.0, base_confidence + 0.10)

        # Reduce confidence for very fragmented responses
        if metadata.total_chunks > 20 and metadata.total_bytes < 500:
            base_confidence = max(0.5, base_confidence - 0.15)

        return base_confidence

    def _get_elapsed_time_ms(self, metadata: StreamingMetadata) -> float:
        """
        Get elapsed time since stream started.

        Args:
            metadata: StreamingMetadata

        Returns:
            Elapsed time in milliseconds
        """
        try:
            start = datetime.fromisoformat(metadata.start_time.replace("Z", "+00:00"))
            now = datetime.utcnow().replace(tzinfo=start.tzinfo)
            return (now - start).total_seconds() * 1000
        except Exception:
            return 0.0

    def get_streaming_stats(self) -> Dict[str, Any]:
        """
        Get overall streaming statistics.

        Returns:
            Dictionary with streaming stats
        """
        return {
            "active_streams": len(self.active_streams),
            "total_chunks_processed": sum(
                m.total_chunks for m in self.active_streams.values()
            ),
            "total_bytes_processed": sum(
                m.total_bytes for m in self.active_streams.values()
            ),
        }
