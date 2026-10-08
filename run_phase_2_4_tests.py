#!/usr/bin/env python3
"""
Phase 2.4 Comprehensive Test Runner
Executes all testing workstreams and generates a completion report.

Workstreams:
1. Integration tests - validate 3 example customer queries end-to-end
2. Benchmark tests - measure router latency, agent execution, loop detection
3. Edge case tests - validate system robustness under stress
"""

import asyncio
import logging
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase24TestRunner:
    """Orchestrates Phase 2.4 testing across all workstreams."""

    def __init__(self):
        """Initialize test runner."""
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.results = {}
        self.start_time = None
        self.end_time = None

    def print_header(self, title: str):
        """Print formatted header."""
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")

    def run_test_suite(self, test_file: str, suite_name: str) -> Tuple[int, str]:
        """Run a pytest test suite and capture output."""
        logger.info(f"Running {suite_name}...")

        test_path = self.tests_dir / test_file
        if not test_path.exists():
            logger.error(f"Test file not found: {test_path}")
            return 1, f"Test file not found: {test_path}"

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path),
                 "-v", "-s", "--tb=short", "--color=yes"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(self.project_root)
            )

            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            error_msg = f"{suite_name} timed out after 5 minutes"
            logger.error(error_msg)
            return 1, error_msg
        except Exception as e:
            error_msg = f"{suite_name} execution failed: {str(e)}"
            logger.error(error_msg)
            return 1, error_msg

    def parse_pytest_output(self, output: str) -> Dict:
        """Parse pytest output to extract key metrics."""
        metrics = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": 0.0
        }

        # Count test results
        for line in output.split('\n'):
            if ' passed' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed':
                        try:
                            metrics["passed"] = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
            if ' failed' in line and 'passed' not in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'failed':
                        try:
                            metrics["failed"] = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
            if ' skipped' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'skipped':
                        try:
                            metrics["skipped"] = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
            if 's' in line and '==' in line:
                try:
                    # Extract duration from "== 12.34s =="
                    parts = line.split()
                    for part in parts:
                        if part.endswith('s'):
                            metrics["duration"] = float(part[:-1])
                except (ValueError, IndexError):
                    pass

        return metrics

    async def run_all_tests(self) -> bool:
        """Run all Phase 2.4 test suites."""
        self.start_time = datetime.now()
        self.print_header("PHASE 2.4: COMPREHENSIVE TEST EXECUTION")
        logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Tests directory: {self.tests_dir}\n")

        all_passed = True
        test_suites = [
            ("test_integration_phase2.py", "Integration Tests (3 Example Queries)"),
            ("test_benchmarks.py", "Performance Benchmarking Suite"),
            ("test_edge_cases.py", "Edge Case & Stress Testing Suite"),
        ]

        for test_file, suite_name in test_suites:
            self.print_header(f"[1/3] {suite_name}")

            returncode, output = self.run_test_suite(test_file, suite_name)
            metrics = self.parse_pytest_output(output)

            self.results[suite_name] = {
                "returncode": returncode,
                "metrics": metrics,
                "output": output,
                "status": "✅ PASS" if returncode == 0 else "❌ FAIL"
            }

            # Print output excerpt
            print(output)

            if returncode != 0:
                all_passed = False
                logger.error(f"{suite_name} FAILED (exit code: {returncode})")
            else:
                logger.info(f"{suite_name} PASSED")
                logger.info(f"  - Passed: {metrics['passed']}")
                logger.info(f"  - Failed: {metrics['failed']}")
                logger.info(f"  - Skipped: {metrics['skipped']}")
                logger.info(f"  - Duration: {metrics['duration']:.2f}s")

        self.end_time = datetime.now()
        return all_passed

    def generate_report(self) -> str:
        """Generate Phase 2.4 completion report."""
        duration = (self.end_time - self.start_time).total_seconds()

        report = f"""
{'='*80}
PHASE 2.4 TEST EXECUTION REPORT
{'='*80}

Execution Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} to {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}
Total Duration: {duration:.2f}s

{'='*80}
TEST RESULTS SUMMARY
{'='*80}

"""

        total_passed = 0
        total_failed = 0
        total_skipped = 0

        for suite_name, result in self.results.items():
            metrics = result["metrics"]
            status = result["status"]

            report += f"\n{suite_name}\n"
            report += f"  Status:   {status}\n"
            report += f"  Passed:   {metrics['passed']}\n"
            report += f"  Failed:   {metrics['failed']}\n"
            report += f"  Skipped:  {metrics['skipped']}\n"
            report += f"  Duration: {metrics['duration']:.2f}s\n"

            total_passed += metrics['passed']
            total_failed += metrics['failed']
            total_skipped += metrics['skipped']

        report += f"\n{'='*80}"
        report += f"\nAGGREGATE RESULTS\n"
        report += f"{'='*80}\n"
        report += f"Total Tests:  {total_passed + total_failed + total_skipped}\n"
        report += f"Passed:       {total_passed}\n"
        report += f"Failed:       {total_failed}\n"
        report += f"Skipped:      {total_skipped}\n"
        report += f"Success Rate: {(total_passed / (total_passed + total_failed + total_skipped) * 100):.1f}%\n"

        if total_failed == 0:
            report += f"\n✅ ALL PHASE 2.4 TESTS PASSED\n"
        else:
            report += f"\n⚠️  {total_failed} TEST(S) FAILED\n"

        report += f"\n{'='*80}\n"
        report += f"WORKSTREAM COMPLETION STATUS\n"
        report += f"{'='*80}\n\n"
        report += f"[1/4] Integration Tests (3 Example Queries)\n"
        report += f"      Status: {self.results.get('Integration Tests (3 Example Queries)', {}).get('status', '⏳ PENDING')}\n"
        report += f"      Coverage: BillingAgent → PolicyAgent, TechSupportAgent, Multi-hop routing\n\n"
        report += f"[2/4] Performance Benchmarking\n"
        report += f"      Status: {self.results.get('Performance Benchmarking Suite', {}).get('status', '⏳ PENDING')}\n"
        report += f"      Metrics: Router latency, Agent execution, Loop detection, E2E response\n\n"
        report += f"[3/4] Edge Case & Stress Testing\n"
        report += f"      Status: {self.results.get('Edge Case & Stress Testing Suite', {}).get('status', '⏳ PENDING')}\n"
        report += f"      Coverage: Token budget, Loop detection, Error handling, Session state\n\n"
        report += f"[4/4] Test Report & Documentation\n"
        report += f"      Status: ✅ GENERATED (this report)\n"
        report += f"      Output: /mnt/user-data/outputs/PHASE_2_4_TEST_REPORT.md\n"

        report += f"\n{'='*80}\n"
        report += f"NEXT STEPS\n"
        report += f"{'='*80}\n\n"
        report += f"Phase 3: PII Masking Engine Implementation\n"
        report += f"  - Presidio integration for entity recognition\n"
        report += f"  - Custom pattern definitions (Aadhaar, API keys, etc.)\n"
        report += f"  - Streaming masking with token-level buffering\n"
        report += f"  - Audit logging for PII detection events\n"
        report += f"  - Real-time performance targeting <50ms latency\n\n"

        return report

    async def main(self):
        """Main entry point."""
        try:
            all_passed = await self.run_all_tests()

            self.print_header("PHASE 2.4 COMPLETION REPORT")
            report = self.generate_report()
            print(report)

            # Save report
            report_path = Path("/mnt/user-data/outputs/PHASE_2_4_TEST_REPORT.md")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report)
            logger.info(f"Report saved to: {report_path}")

            return 0 if all_passed else 1

        except Exception as e:
            logger.exception(f"Test runner failed: {e}")
            return 1


async def main():
    """Entry point."""
    runner = Phase24TestRunner()
    exit_code = await runner.main()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
