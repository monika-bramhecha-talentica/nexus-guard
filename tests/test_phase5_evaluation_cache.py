"""
Phase 5 Day 2: Evaluation Cache Tests

Tests for multi-layer caching system:
  - In-memory cache (L1)
  - Optional Redis cache (L2)
  - Cache key generation
  - TTL and expiration
  - LRU eviction
  - Cache statistics
  - Cache warming
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock

from src.guardrails.evaluation_cache import (
    CacheEntry,
    InMemoryEvaluationCache,
    EvaluationCacheKeyBuilder,
    LayeredEvaluationCache,
    get_evaluation_cache,
    reset_evaluation_cache,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_initialization(self):
        """Test cache entry creation."""
        value = {"accuracy": 0.85, "relevance": 0.90}
        entry = CacheEntry(key="test_key", value=value)

        assert entry.key == "test_key"
        assert entry.value == value
        assert entry.hits == 0
        assert entry.ttl_seconds == 3600

    def test_cache_entry_expiration(self):
        """Test TTL-based expiration."""
        entry = CacheEntry(key="test", value={}, ttl_seconds=1)

        assert entry.is_expired() is False

        # Simulate time passage
        entry.timestamp = time.time() - 2
        assert entry.is_expired() is True

    def test_cache_entry_access_tracking(self):
        """Test access count tracking."""
        entry = CacheEntry(key="test", value={})

        assert entry.hits == 0
        entry.access()
        assert entry.hits == 1
        entry.access()
        assert entry.hits == 2


class TestInMemoryEvaluationCache:
    """Tests for in-memory cache."""

    def test_cache_initialization(self):
        """Test cache creation."""
        cache = InMemoryEvaluationCache(max_entries=500)
        assert len(cache.cache) == 0
        assert cache.max_entries == 500

    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        cache = InMemoryEvaluationCache()
        value = {"accuracy": 0.85}

        cache.set("key1", value)
        result = cache.get("key1")

        assert result == value
        assert cache.stats["hits"] == 1

    def test_cache_miss(self):
        """Test cache miss."""
        cache = InMemoryEvaluationCache()

        result = cache.get("nonexistent")
        assert result is None
        assert cache.stats["misses"] == 1

    def test_cache_expiration(self):
        """Test TTL expiration."""
        cache = InMemoryEvaluationCache()
        value = {"score": 0.5}

        cache.set("exp_key", value, ttl_seconds=1)
        assert cache.get("exp_key") is not None

        # Simulate expiration
        cache.cache["exp_key"].timestamp = time.time() - 2
        result = cache.get("exp_key")
        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when full."""
        cache = InMemoryEvaluationCache(max_entries=3)

        cache.set("key1", {"v": 1})
        cache.set("key2", {"v": 2})
        cache.set("key3", {"v": 3})

        assert len(cache.cache) == 3

        # Add one more, should evict key1 (LRU)
        cache.set("key4", {"v": 4})

        assert len(cache.cache) == 3
        assert "key1" not in cache.cache
        assert cache.stats["evictions"] == 1

    def test_cache_delete(self):
        """Test cache deletion."""
        cache = InMemoryEvaluationCache()
        cache.set("del_key", {"v": 1})

        assert cache.get("del_key") is not None
        cache.delete("del_key")
        assert cache.get("del_key") is None

    def test_cache_clear(self):
        """Test clearing entire cache."""
        cache = InMemoryEvaluationCache()

        cache.set("k1", {})
        cache.set("k2", {})
        cache.set("k3", {})

        assert len(cache.cache) == 3
        cache.clear()
        assert len(cache.cache) == 0

    def test_cache_statistics(self):
        """Test cache statistics."""
        cache = InMemoryEvaluationCache()

        cache.set("k1", {"accuracy": 0.85})
        cache.get("k1")  # Hit
        cache.get("k1")  # Hit
        cache.get("k2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["entries"] == 1
        assert 0.5 < stats["hit_rate"] < 0.7  # 2 hits, 3 total

    def test_cache_lru_ordering(self):
        """Test LRU ordering with access."""
        cache = InMemoryEvaluationCache(max_entries=2)

        cache.set("old", {"v": 1})
        cache.set("new", {"v": 2})

        # Access old key to update LRU order
        cache.get("old")

        # Add another, should evict 'new' not 'old'
        cache.set("newer", {"v": 3})

        assert "old" in cache.cache
        assert "new" not in cache.cache
        assert "newer" in cache.cache


class TestEvaluationCacheKeyBuilder:
    """Tests for cache key generation."""

    def test_response_only_key(self):
        """Test response-only key generation."""
        response = "This is a response"
        key = EvaluationCacheKeyBuilder.response_only_key(response)

        assert key.startswith("resp_")
        assert len(key) > 10

    def test_response_query_key(self):
        """Test response+query key generation."""
        response = "This is a response"
        query = "What is something?"

        key = EvaluationCacheKeyBuilder.response_query_key(response, query)

        assert key.startswith("eval_")
        assert len(key) > 20

    def test_agent_context_key(self):
        """Test fully qualified key generation."""
        response = "Response"
        query = "Query"
        agent_id = "agent_123"

        key = EvaluationCacheKeyBuilder.agent_context_key(response, query, agent_id)

        assert key.startswith("eval_")
        assert len(key) > 30

    def test_key_determinism(self):
        """Test that same inputs produce same keys."""
        response = "Deterministic response"
        query = "Same query"

        key1 = EvaluationCacheKeyBuilder.response_query_key(response, query)
        key2 = EvaluationCacheKeyBuilder.response_query_key(response, query)

        assert key1 == key2

    def test_key_uniqueness(self):
        """Test that different inputs produce different keys."""
        key1 = EvaluationCacheKeyBuilder.response_query_key("response1", "query")
        key2 = EvaluationCacheKeyBuilder.response_query_key("response2", "query")

        assert key1 != key2


class TestLayeredEvaluationCache:
    """Tests for multi-layer cache."""

    def test_layered_cache_initialization(self):
        """Test cache creation without Redis."""
        cache = LayeredEvaluationCache(
            max_memory_entries=500,
            use_redis=False,
        )

        assert cache.l1_cache is not None
        assert cache.redis_available is False

    def test_layered_cache_set_get_l1_only(self):
        """Test L1-only cache operations."""
        cache = LayeredEvaluationCache(use_redis=False)
        value = {"accuracy": 0.85, "relevance": 0.90}

        cache.set("key1", value)
        result = cache.get("key1")

        assert result == value

    def test_layered_cache_l1_fallback(self):
        """Test fallback from L1 to L2 (simulated)."""
        cache = LayeredEvaluationCache(use_redis=False)
        value = {"score": 0.75}

        cache.set("key", value)
        result = cache.get("key")

        assert result == value

    def test_layered_cache_delete(self):
        """Test deletion from layered cache."""
        cache = LayeredEvaluationCache(use_redis=False)

        cache.set("del_key", {"v": 1})
        assert cache.get("del_key") is not None

        cache.delete("del_key")
        assert cache.get("del_key") is None

    def test_layered_cache_clear(self):
        """Test clearing layered cache."""
        cache = LayeredEvaluationCache(use_redis=False)

        cache.set("k1", {})
        cache.set("k2", {})

        cache.clear()

        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_layered_cache_statistics(self):
        """Test cache statistics."""
        cache = LayeredEvaluationCache(use_redis=False)

        cache.set("k1", {"accuracy": 0.85})
        cache.get("k1")
        cache.get("k2")  # Miss

        stats = cache.get_stats()
        assert "l1" in stats
        assert stats["l2_available"] is False
        assert stats["l1"]["hits"] == 1
        assert stats["l1"]["misses"] == 1

    def test_layered_cache_ttl_override(self):
        """Test TTL override."""
        cache = LayeredEvaluationCache(use_redis=False, default_ttl=3600)

        cache.set("key", {"v": 1}, ttl_seconds=1)

        # Verify the entry was set with the override TTL
        assert "key" in cache.l1_cache.cache
        assert cache.l1_cache.cache["key"].ttl_seconds == 1

    def test_cache_warming(self):
        """Test cache warming."""
        cache = LayeredEvaluationCache(use_redis=False)

        evaluations = [
            ("response1", "query1", {"accuracy": 0.85}),
            ("response2", "query2", {"accuracy": 0.90}),
        ]

        cache.warm_cache(evaluations)

        # Verify entries were cached
        key1 = EvaluationCacheKeyBuilder.response_query_key("response1", "query1")
        key2 = EvaluationCacheKeyBuilder.response_query_key("response2", "query2")

        assert cache.get(key1) is not None
        assert cache.get(key2) is not None


class TestGlobalCacheInstance:
    """Tests for global cache singleton."""

    def test_global_cache_creation(self):
        """Test global cache instance."""
        reset_evaluation_cache()
        cache1 = get_evaluation_cache()
        cache2 = get_evaluation_cache()

        assert cache1 is cache2

    def test_global_cache_usage(self):
        """Test using global cache."""
        reset_evaluation_cache()
        cache = get_evaluation_cache()

        cache.set("global_key", {"v": 1})
        result = get_evaluation_cache().get("global_key")

        assert result == {"v": 1}

    def test_global_cache_reset(self):
        """Test resetting global cache."""
        cache1 = get_evaluation_cache()
        cache1.set("test_key", {})

        reset_evaluation_cache()
        cache2 = get_evaluation_cache()

        assert cache1 is not cache2
        assert cache2.get("test_key") is None


class TestCachePerformance:
    """Tests for cache performance characteristics."""

    def test_cache_hit_performance(self):
        """Test cache hit latency."""
        cache = InMemoryEvaluationCache()
        value = {"accuracy": 0.85}

        cache.set("perf_key", value)

        # Measure hit time
        start = time.time()
        for _ in range(1000):
            cache.get("perf_key")
        elapsed = time.time() - start

        # Should be very fast (< 10ms per 1000 ops)
        assert elapsed < 0.01

    def test_cache_miss_performance(self):
        """Test cache miss latency."""
        cache = InMemoryEvaluationCache()

        # Measure miss time
        start = time.time()
        for _ in range(1000):
            cache.get("nonexistent")
        elapsed = time.time() - start

        # Should be very fast (< 10ms per 1000 ops)
        assert elapsed < 0.01

    def test_cache_set_performance(self):
        """Test cache write latency."""
        cache = InMemoryEvaluationCache()

        # Measure set time
        start = time.time()
        for i in range(100):
            cache.set(f"key_{i}", {"accuracy": 0.85})
        elapsed = time.time() - start

        # Should be reasonably fast
        assert elapsed < 0.1


# Test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
