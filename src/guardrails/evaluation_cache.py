"""
Phase 5 Day 2: Enhanced Evaluation Result Caching

Implements multi-layer caching strategy:
  - L1: In-memory cache (fast, per-process)
  - L2: Optional Redis cache (distributed, persistent)

Features:
  - TTL-based expiration
  - Cache key strategies (response hash, query hash, combined)
  - Cache statistics and monitoring
  - Graceful Redis fallback
  - Cache warming capability
  - Per-dimension caching for partial evaluations
"""

import logging
import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    key: str
    value: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: int = 3600  # 1 hour default
    hits: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        age = time.time() - self.timestamp
        return age > self.ttl_seconds

    def access(self) -> None:
        """Record access for LRU tracking."""
        self.hits += 1
        self.last_accessed = time.time()


class InMemoryEvaluationCache:
    """
    Fast in-memory evaluation cache with LRU eviction.

    Primary cache layer - high performance, per-process storage.
    """

    def __init__(self, max_entries: int = 1000):
        """
        Initialize in-memory cache.

        Args:
            max_entries: Maximum cache entries before LRU eviction
        """
        self.max_entries = max_entries
        self.cache: Dict[str, CacheEntry] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "max_size_bytes": 0,
        }

        logger.info(f"InMemoryEvaluationCache initialized (max={max_entries})")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached evaluation.

        Args:
            key: Cache key

        Returns:
            Cached evaluation dict or None if miss/expired
        """
        if key not in self.cache:
            self.stats["misses"] += 1
            return None

        entry = self.cache[key]

        # Check expiration
        if entry.is_expired():
            del self.cache[key]
            self.stats["misses"] += 1
            logger.debug(f"Cache entry expired: {key}")
            return None

        # Record access and move to end (LRU)
        entry.access()
        self.cache.move_to_end(key)
        self.stats["hits"] += 1

        logger.debug(
            f"Cache hit: {key} (hits={entry.hits}, age={time.time() - entry.timestamp:.1f}s)"
        )
        return entry.value

    def set(
        self, key: str, value: Dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        """
        Set cache entry.

        Args:
            key: Cache key
            value: Evaluation dictionary
            ttl_seconds: Time to live in seconds
        """
        # Remove if exists to update position
        if key in self.cache:
            del self.cache[key]

        # Create new entry
        entry = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl_seconds,
        )

        # Add to cache
        self.cache[key] = entry

        # Evict LRU if full
        if len(self.cache) > self.max_entries:
            evicted_key, evicted_entry = self.cache.popitem(last=False)
            self.stats["evictions"] += 1
            logger.debug(f"LRU eviction: {evicted_key} (hits={evicted_entry.hits})")

        # Update stats
        self._update_stats()

    def delete(self, key: str) -> None:
        """Delete cache entry."""
        if key in self.cache:
            del self.cache[key]
            self._update_stats()

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        hit_rate = (
            self.stats["hits"]
            / (self.stats["hits"] + self.stats["misses"])
            if (self.stats["hits"] + self.stats["misses"]) > 0
            else 0.0
        )

        return {
            "entries": total_entries,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self.stats["evictions"],
            "max_size_bytes": self.stats["max_size_bytes"],
            "avg_entry_size": (
                self.stats["max_size_bytes"] / total_entries
                if total_entries > 0
                else 0
            ),
        }

    def _update_stats(self) -> None:
        """Update cache size statistics."""
        total_bytes = sum(len(json.dumps(e.value)) for e in self.cache.values())
        self.stats["max_size_bytes"] = total_bytes


class EvaluationCacheKeyBuilder:
    """Build cache keys from evaluation parameters."""

    @staticmethod
    def hash_text(text: str) -> str:
        """Generate SHA256 hash of text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def response_only_key(response: str) -> str:
        """Key based on response alone (reuse across queries)."""
        return f"resp_{EvaluationCacheKeyBuilder.hash_text(response)}"

    @staticmethod
    def response_query_key(response: str, query: str) -> str:
        """Key based on response + query (context-aware)."""
        resp_hash = EvaluationCacheKeyBuilder.hash_text(response)
        query_hash = EvaluationCacheKeyBuilder.hash_text(query)
        return f"eval_{resp_hash}_{query_hash}"

    @staticmethod
    def agent_context_key(response: str, query: str, agent_id: str) -> str:
        """Key based on response + query + agent (fully qualified)."""
        resp_hash = EvaluationCacheKeyBuilder.hash_text(response)
        query_hash = EvaluationCacheKeyBuilder.hash_text(query)
        agent_hash = EvaluationCacheKeyBuilder.hash_text(agent_id)
        return f"eval_{resp_hash}_{query_hash}_{agent_hash}"


class LayeredEvaluationCache:
    """
    Multi-layer evaluation cache with L1 (memory) and optional L2 (Redis).

    Provides transparent caching with graceful fallback.
    """

    def __init__(
        self,
        max_memory_entries: int = 1000,
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
    ):
        """
        Initialize layered cache.

        Args:
            max_memory_entries: Max entries in L1 cache
            use_redis: Enable Redis L2 cache
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
        """
        self.l1_cache = InMemoryEvaluationCache(max_memory_entries)
        self.use_redis = use_redis
        self.redis_client = None
        self.redis_available = False
        self.default_ttl = default_ttl

        # Try to initialize Redis if enabled
        if use_redis:
            self._init_redis(redis_url)

        logger.info(
            f"LayeredEvaluationCache initialized "
            f"(L1={max_memory_entries}, L2_redis={self.redis_available})"
        )

    def _init_redis(self, redis_url: str) -> None:
        """Initialize Redis connection."""
        try:
            import redis

            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis L2 cache initialized")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}. Continuing with L1 only.")
            self.redis_available = False

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached evaluation (L1 -> L2 fallback).

        Args:
            key: Cache key

        Returns:
            Cached evaluation or None
        """
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            logger.debug(f"L1 cache hit: {key}")
            return value

        # Try L2 Redis if available
        if self.redis_available:
            try:
                redis_value = self.redis_client.get(key)
                if redis_value:
                    value = json.loads(redis_value)
                    # Promote to L1 cache
                    self.l1_cache.set(key, value, self.default_ttl)
                    logger.debug(f"L2 cache hit, promoted to L1: {key}")
                    return value
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        return None

    def set(self, key: str, value: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        """
        Set cache entry (L1 + optional L2).

        Args:
            key: Cache key
            value: Evaluation dict
            ttl_seconds: TTL override
        """
        ttl = ttl_seconds or self.default_ttl

        # Always set in L1
        self.l1_cache.set(key, value, ttl)

        # Also set in Redis if available
        if self.redis_available:
            try:
                self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value),
                )
                logger.debug(f"Set in L1+L2: {key}")
            except Exception as e:
                logger.warning(f"Redis set failed: {e}. L1 only.")

    def delete(self, key: str) -> None:
        """Delete from both cache layers."""
        self.l1_cache.delete(key)
        if self.redis_available:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")

    def clear(self) -> None:
        """Clear both cache layers."""
        self.l1_cache.clear()
        if self.redis_available:
            try:
                # Clear only keys matching our pattern
                pattern = "eval_*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared Redis cache ({len(keys)} keys)")
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics from both layers."""
        stats = {
            "l1": self.l1_cache.get_stats(),
            "l2_available": self.redis_available,
        }

        if self.redis_available:
            try:
                info = self.redis_client.info("memory")
                stats["l2"] = {
                    "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                    "memory_peak_mb": info.get("used_memory_peak", 0) / (1024 * 1024),
                }
            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")
                stats["l2"] = {"error": str(e)}

        return stats

    def warm_cache(self, evaluations: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        """
        Pre-populate cache with evaluations.

        Args:
            evaluations: List of (response, query, eval_dict) tuples
        """
        count = 0
        for response, query, eval_dict in evaluations:
            key = EvaluationCacheKeyBuilder.response_query_key(response, query)
            self.set(key, eval_dict, self.default_ttl)
            count += 1

        logger.info(f"Cache warmed: {count} entries")


# Global cache instance
_global_cache: Optional[LayeredEvaluationCache] = None


def get_evaluation_cache() -> LayeredEvaluationCache:
    """Get or create global evaluation cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = LayeredEvaluationCache(
            max_memory_entries=1000,
            use_redis=False,  # Disabled by default, enable with env var
        )
    return _global_cache


def reset_evaluation_cache() -> None:
    """Reset global cache (for testing)."""
    global _global_cache
    _global_cache = None
