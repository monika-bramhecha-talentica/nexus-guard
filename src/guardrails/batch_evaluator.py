"""
Phase 5 Day 2: Batch Evaluation Capability

Enables efficient evaluation of multiple responses in parallel.

Features:
  - Async batch processing with configurable concurrency
  - Per-item error handling and recovery
  - Aggregated results with success/failure tracking
  - Progress tracking for large batches
  - Timeout and retry management
  - Result deduplication for identical responses
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BatchEvaluationItem:
    """Single item in batch evaluation request."""

    item_id: str
    response: str
    query: str = ""
    agent_id: str = "agent_unknown"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchEvaluationResult:
    """Result for single batch item."""

    item_id: str
    success: bool
    evaluation: Optional[Any] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    from_cache: bool = False
    retry_count: int = 0


class BatchEvaluator:
    """
    Async batch evaluator for multiple responses.

    Provides efficient parallel evaluation with error handling.
    """

    def __init__(
        self,
        evaluate_fn,
        max_concurrent: int = 10,
        timeout_seconds: int = 30,
        max_retries: int = 2,
    ):
        """
        Initialize batch evaluator.

        Args:
            evaluate_fn: Async evaluation function (response, query, agent_id) -> evaluation
            max_concurrent: Max concurrent evaluations
            timeout_seconds: Timeout per evaluation
            max_retries: Max retries on timeout/error
        """
        self.evaluate_fn = evaluate_fn
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        self.stats = {
            "total_items": 0,
            "successful": 0,
            "failed": 0,
            "from_cache": 0,
            "total_elapsed_ms": 0.0,
            "retries_used": 0,
        }

        logger.info(
            f"BatchEvaluator initialized "
            f"(concurrent={max_concurrent}, timeout={timeout_seconds}s, retries={max_retries})"
        )

    async def evaluate_batch(
        self, items: List[BatchEvaluationItem]
    ) -> Tuple[List[BatchEvaluationResult], Dict[str, Any]]:
        """
        Evaluate batch of items concurrently.

        Args:
            items: List of evaluation items

        Returns:
            Tuple of (results, stats)
        """
        self.stats["total_items"] = len(items)

        logger.info(f"Starting batch evaluation of {len(items)} items")

        # Create tasks with semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._evaluate_item_with_semaphore(semaphore, item) for item in items
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Update stats
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        from_cache = sum(1 for r in results if r.from_cache)

        self.stats["successful"] = successful
        self.stats["failed"] = failed
        self.stats["from_cache"] = from_cache
        self.stats["total_elapsed_ms"] = sum(r.elapsed_ms for r in results)

        logger.info(
            f"Batch evaluation complete: "
            f"{successful} success, {failed} failed, {from_cache} from cache"
        )

        return results, self._get_stats()

    async def _evaluate_item_with_semaphore(
        self, semaphore: asyncio.Semaphore, item: BatchEvaluationItem
    ) -> BatchEvaluationResult:
        """Evaluate single item with concurrency control."""
        async with semaphore:
            return await self._evaluate_item_with_retry(item)

    async def _evaluate_item_with_retry(
        self, item: BatchEvaluationItem
    ) -> BatchEvaluationResult:
        """Evaluate single item with retry logic."""
        result = BatchEvaluationResult(
            item_id=item.item_id,
            success=False,
        )

        start_time = datetime.utcnow()

        for attempt in range(self.max_retries):
            try:
                # Call evaluation function with timeout
                evaluation = await asyncio.wait_for(
                    self.evaluate_fn(item.response, item.query, item.agent_id),
                    timeout=self.timeout_seconds,
                )

                # Check if result is a tuple (success, eval_dict) from LLM
                if isinstance(evaluation, tuple):
                    success, eval_dict = evaluation
                    if success:
                        result.success = True
                        result.evaluation = eval_dict
                        # Check if this was from cache (metadata in eval)
                        result.from_cache = eval_dict.get("_from_cache", False)
                        break
                    else:
                        # LLM eval failed, try next attempt
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(0.1 * (attempt + 1))
                            continue
                else:
                    # Direct result from rule-based evaluation
                    result.success = True
                    result.evaluation = evaluation
                    break

            except asyncio.TimeoutError:
                logger.warning(
                    f"Evaluation timeout for item {item.item_id} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                result.error = f"Timeout after {self.timeout_seconds}s"

            except Exception as e:
                logger.warning(
                    f"Evaluation error for item {item.item_id}: {e} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                result.error = str(e)

        # Calculate elapsed time
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        result.elapsed_ms = elapsed
        result.retry_count = self.max_retries - 1 if not result.success else 0

        if not result.success:
            self.stats["retries_used"] += result.retry_count

        return result

    def _get_stats(self) -> Dict[str, Any]:
        """Get batch statistics."""
        successful = self.stats["successful"]
        total = self.stats["total_items"]

        return {
            "total_items": total,
            "successful": successful,
            "failed": self.stats["failed"],
            "success_rate": successful / total if total > 0 else 0.0,
            "from_cache": self.stats["from_cache"],
            "cache_hit_rate": (
                self.stats["from_cache"] / total if total > 0 else 0.0
            ),
            "total_elapsed_ms": self.stats["total_elapsed_ms"],
            "avg_latency_ms": (
                self.stats["total_elapsed_ms"] / total if total > 0 else 0.0
            ),
            "retries_used": self.stats["retries_used"],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        return self._get_stats()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "total_items": 0,
            "successful": 0,
            "failed": 0,
            "from_cache": 0,
            "total_elapsed_ms": 0.0,
            "retries_used": 0,
        }


class SmartBatchEvaluator(BatchEvaluator):
    """
    Smart batch evaluator with deduplication and optimization.

    Detects identical responses and evaluates them once.
    """

    async def evaluate_batch_smart(
        self, items: List[BatchEvaluationItem]
    ) -> Tuple[List[BatchEvaluationResult], Dict[str, Any]]:
        """
        Evaluate batch with deduplication.

        Items with identical responses share the same evaluation result.

        Args:
            items: List of evaluation items

        Returns:
            Tuple of (results, stats)
        """
        # Group items by response
        response_groups: Dict[str, List[BatchEvaluationItem]] = {}
        for item in items:
            if item.response not in response_groups:
                response_groups[item.response] = []
            response_groups[item.response].append(item)

        logger.info(
            f"Smart batch: {len(items)} items, "
            f"{len(response_groups)} unique responses"
        )

        # Evaluate only unique responses
        unique_items = [
            items_list[0] for items_list in response_groups.values()
        ]
        unique_results, stats = await self.evaluate_batch(unique_items)

        # Map results back to all items
        unique_result_map = {r.item_id: r for r in unique_results}
        all_results = []

        for item in items:
            # Find result for same response
            representative = response_groups[item.response][0]
            result = unique_result_map[representative.item_id]

            # Create result for this item (copy with new item_id)
            item_result = BatchEvaluationResult(
                item_id=item.item_id,
                success=result.success,
                evaluation=result.evaluation,
                error=result.error,
                elapsed_ms=result.elapsed_ms,
                from_cache=result.from_cache,
                retry_count=result.retry_count,
            )
            all_results.append(item_result)

        # Update stats for deduplication
        dedup_stats = stats.copy()
        dedup_stats["unique_responses"] = len(response_groups)
        dedup_stats["deduplication_savings"] = len(items) - len(response_groups)

        logger.info(f"Smart batch complete: saved {len(items) - len(response_groups)} evaluations")

        return all_results, dedup_stats
