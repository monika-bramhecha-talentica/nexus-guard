"""
Phase 2.4: Performance Benchmarking Tests

Measures latency for:
- Router decision latency
- Agent execution time
- Loop detection overhead
- End-to-end query response time
"""

import asyncio
import logging
import time
from typing import Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.core import NexusGuardOrchestrator, SessionState
from src.orchestrator.router import DeterministicRouter
from src.agents.billing_agent import BillingAgent
from src.agents.support_agent import TechSupportAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.escalation_agent import EscalationAgent


logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark suite for Nexus Guard components."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.results: Dict[str, List[float]] = {}
        self.orchestrator = NexusGuardOrchestrator(
            agents=[
                BillingAgent(),
                TechSupportAgent(),
                PolicyAgent(),
                EscalationAgent(),
            ]
        )

    def _record_metric(self, name: str, value_ms: float):
        """Record a performance metric."""
        if name not in self.results:
            self.results[name] = []
        self.results[name].append(value_ms)

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a metric."""
        if not values:
            return {}

        sorted_vals = sorted(values)
        return {
            "min_ms": min(sorted_vals),
            "max_ms": max(sorted_vals),
            "avg_ms": sum(sorted_vals) / len(sorted_vals),
            "p50_ms": sorted_vals[len(sorted_vals) // 2],
            "p95_ms": sorted_vals[int(len(sorted_vals) * 0.95)],
            "p99_ms": sorted_vals[int(len(sorted_vals) * 0.99)],
            "count": len(sorted_vals),
        }

    def benchmark_router_decision(self, iterations: int = 10):
        """Benchmark router decision latency."""
        logger.info(f"\n{'='*80}")
        logger.info("BENCHMARK: Router Decision Latency")
        logger.info(f"{'='*80}")

        query = "I need help with my billing"
        context = {
            "routing_path": [],
            "attempt": 1,
            "current_agent": None,
        }

        for i in range(iterations):
            start = time.time()

            # Reset context for each iteration
            context["routing_path"] = []

            try:
                # Note: This will fail if Ollama is not running
                # In CI/CD, we'll mock this
                next_agent = self.orchestrator.router.decide_next_agent(query, context)
                elapsed_ms = (time.time() - start) * 1000
                self._record_metric("router_decision_latency", elapsed_ms)
                logger.info(f"  Iteration {i+1}: {elapsed_ms:.2f}ms → {next_agent}")
            except Exception as e:
                logger.warning(f"  Iteration {i+1}: Router failed (Ollama not running?) - {str(e)[:50]}")
                # Record high value to indicate failure
                self._record_metric("router_decision_latency", 5000.0)

    def benchmark_agent_execution(self, iterations: int = 5):
        """Benchmark individual agent execution time."""
        logger.info(f"\n{'='*80}")
        logger.info("BENCHMARK: Agent Execution Time")
        logger.info(f"{'='*80}")

        agents = [
            ("BillingAgent", BillingAgent()),
            ("TechSupportAgent", TechSupportAgent()),
            ("PolicyAgent", PolicyAgent()),
            ("EscalationAgent", EscalationAgent()),
        ]

        test_queries = {
            "BillingAgent": "I want a refund for transaction TXN001",
            "TechSupportAgent": "The dashboard is loading slowly",
            "PolicyAgent": "Validate refund eligibility",
            "EscalationAgent": "Escalate this issue to human support",
        }

        for agent_name, agent in agents:
            logger.info(f"\n  {agent_name}:")
            query = test_queries.get(agent_name, "Help me with this")
            context = {
                "user_id": "USER001",
                "session_id": "bench_session",
                "routing_path": [],
            }

            for i in range(iterations):
                start = time.time()

                try:
                    # Run agent synchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        agent.execute(query, context)
                    )
                    loop.close()

                    elapsed_ms = (time.time() - start) * 1000
                    self._record_metric(f"{agent_name}_execution_time", elapsed_ms)
                    logger.info(f"    Iteration {i+1}: {elapsed_ms:.2f}ms")
                except Exception as e:
                    logger.warning(f"    Iteration {i+1}: Failed - {str(e)[:50]}")
                    self._record_metric(f"{agent_name}_execution_time", 1000.0)

    def benchmark_loop_detection(self, iterations: int = 100):
        """Benchmark loop detection overhead."""
        logger.info(f"\n{'='*80}")
        logger.info("BENCHMARK: Loop Detection Overhead")
        logger.info(f"{'='*80}")

        session_id = "bench_loop_session"
        self.orchestrator.create_session(
            user_id="USER001",
            org_id="ORG001",
            session_id=session_id,
        )

        session = self.orchestrator.get_session(session_id)

        # Build different routing paths for testing
        test_paths = [
            ["BillingAgent"],
            ["BillingAgent", "PolicyAgent"],
            ["BillingAgent", "PolicyAgent", "BillingAgent"],  # Loop
            ["BillingAgent", "TechSupportAgent", "PolicyAgent", "BillingAgent"],  # Circular
        ]

        for path_idx, path in enumerate(test_paths):
            session.routing_path = path.copy()
            test_agent = "BillingAgent"  # Try to route back to first agent

            logger.info(f"\n  Path {path_idx + 1}: {' → '.join(path)} → {test_agent}")

            for i in range(iterations):
                start = time.time()
                is_loop = self.orchestrator.check_loop_detection(session_id, test_agent)
                elapsed_ms = (time.time() - start) * 1000
                self._record_metric("loop_detection_latency", elapsed_ms)

            stats = self._calculate_stats(self.results["loop_detection_latency"])
            logger.info(f"    Results: avg={stats['avg_ms']:.3f}ms, p95={stats['p95_ms']:.3f}ms")

    async def benchmark_end_to_end(self, iterations: int = 3):
        """Benchmark end-to-end query routing."""
        logger.info(f"\n{'='*80}")
        logger.info("BENCHMARK: End-to-End Query Routing")
        logger.info(f"{'='*80}")

        queries = [
            ("Refund", "I want a refund for transaction TXN001 for $150"),
            ("Support", "The dashboard is loading very slowly"),
            ("Policy", "Can I get a refund for TXN003 ($300, 60 days ago)?"),
        ]

        for query_type, query in queries:
            logger.info(f"\n  Query Type: {query_type}")
            logger.info(f"  Query: {query[:50]}...")

            for i in range(iterations):
                session_id = f"bench_e2e_{query_type.lower()}_{i}"
                self.orchestrator.create_session(
                    user_id="USER001",
                    org_id="ORG001",
                    session_id=session_id,
                )

                start = time.time()

                try:
                    response = await self.orchestrator.route_query(
                        session_id=session_id,
                        query=query,
                    )
                    elapsed_ms = (time.time() - start) * 1000
                    self._record_metric(f"e2e_{query_type}_latency", elapsed_ms)

                    session = self.orchestrator.get_session(session_id)
                    logger.info(
                        f"    Iteration {i+1}: {elapsed_ms:.2f}ms "
                        f"(path: {' → '.join(session.routing_path)})"
                    )
                except Exception as e:
                    logger.warning(f"    Iteration {i+1}: Failed - {str(e)[:50]}")
                    self._record_metric(f"e2e_{query_type}_latency", 5000.0)

    def print_report(self):
        """Print benchmark report."""
        logger.info(f"\n\n{'='*80}")
        logger.info("PERFORMANCE BENCHMARK REPORT")
        logger.info(f"{'='*80}\n")

        performance_targets = {
            "router_decision_latency": 100,  # <100ms target
            "loop_detection_latency": 50,    # <50ms target
            "e2e_Refund_latency": 1000,      # <1s target
            "e2e_Support_latency": 1000,     # <1s target
            "e2e_Policy_latency": 1000,      # <1s target
        }

        for metric_name, values in sorted(self.results.items()):
            stats = self._calculate_stats(values)
            target = performance_targets.get(metric_name)

            logger.info(f"{metric_name}:")
            logger.info(f"  Count:   {stats.get('count', 0)} samples")
            logger.info(f"  Min:     {stats.get('min_ms', 0):.2f}ms")
            logger.info(f"  Avg:     {stats.get('avg_ms', 0):.2f}ms")
            logger.info(f"  P50:     {stats.get('p50_ms', 0):.2f}ms")
            logger.info(f"  P95:     {stats.get('p95_ms', 0):.2f}ms")
            logger.info(f"  P99:     {stats.get('p99_ms', 0):.2f}ms")
            logger.info(f"  Max:     {stats.get('max_ms', 0):.2f}ms")

            if target:
                p95_value = stats.get('p95_ms', 0)
                status = "✅ PASS" if p95_value <= target else "⚠️  MISS"
                logger.info(f"  Target:  {target}ms → {status}")

            logger.info("")

        return self.results


async def run_benchmarks():
    """Run all benchmarks."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    benchmark = PerformanceBenchmark()

    logger.info("\n" + "="*80)
    logger.info("NEXUS GUARD - PHASE 2.4 PERFORMANCE BENCHMARKS")
    logger.info("="*80)

    # Run benchmarks
    logger.info("\n[1/4] Router Decision Latency Benchmark...")
    benchmark.benchmark_router_decision(iterations=5)

    logger.info("\n[2/4] Agent Execution Time Benchmark...")
    benchmark.benchmark_agent_execution(iterations=3)

    logger.info("\n[3/4] Loop Detection Overhead Benchmark...")
    benchmark.benchmark_loop_detection(iterations=50)

    logger.info("\n[4/4] End-to-End Query Routing Benchmark...")
    await benchmark.benchmark_end_to_end(iterations=2)

    # Print report
    benchmark.print_report()

    logger.info("\n" + "="*80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
