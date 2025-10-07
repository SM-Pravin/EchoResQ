#!/usr/bin/env python3
"""
Enhanced Test Runner for Emergency AI Phase 3
Comprehensive testing with detailed reporting and Phase 3 component validation.
"""

import os
import sys
import time
import json
import pytest
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "WORKING_FILES"))

# Color output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


@dataclass
class TestSuiteResult:
    """Result of running a test suite."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    details: Dict[str, Any]
    issues: List[str]


class EnhancedTestRunner:
    """Enhanced test runner with Phase 3 validation."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.working_files = PROJECT_ROOT / "WORKING_FILES"
        self.test_results: List[TestSuiteResult] = []
    
    def print_header(self, text: str, color: str = Colors.BLUE):
        """Print formatted header."""
        print(f"\n{color}{Colors.BOLD}{'='*60}")
        print(f"{text.center(60)}")
        print(f"{'='*60}{Colors.END}")
    
    def print_status(self, text: str, status: str = "info"):
        """Print colored status message."""
        colors = {
            "success": Colors.GREEN,
            "error": Colors.RED,
            "warning": Colors.YELLOW,
            "info": Colors.BLUE
        }
        color = colors.get(status, "")
        prefix = {
            "success": "[OK]",
            "error": "[ERROR]",
            "warning": "[WARNING]", 
            "info": "[INFO]"
        }.get(status, "•")
        
        print(f"{color}{prefix} {text}{Colors.END}")
    
    def run_stress_tests(self) -> TestSuiteResult:
        """Run comprehensive stress tests."""
        self.print_status("Running stress test suite...", "info")
        start_time = time.time()
        
        try:
            from tests.stress_test_suite import StressTestSuite, run_stress_tests
            
            # Run the stress test suite
            suite = StressTestSuite()
            
            # First run quick validation
            quick_validation = suite.run_quick_validation()
            if not quick_validation:
                return TestSuiteResult(
                    suite_name="stress_tests",
                    total_tests=1,
                    passed=0,
                    failed=1,
                    skipped=0,
                    duration_seconds=time.time() - start_time,
                    details={"quick_validation": False},
                    issues=["Quick validation failed"]
                )
            
            # Run full stress tests
            benchmark_result = suite.run_all_stress_tests()
            
            duration = time.time() - start_time
            
            return TestSuiteResult(
                suite_name="stress_tests",
                total_tests=benchmark_result.total_tests,
                passed=benchmark_result.passed,
                failed=benchmark_result.failed,
                skipped=0,
                duration_seconds=duration,
                details={
                    "avg_processing_time_ms": benchmark_result.avg_processing_time_ms,
                    "avg_confidence": benchmark_result.avg_confidence,
                    "quick_validation": quick_validation
                },
                issues=[]
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name="stress_tests",
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration_seconds=time.time() - start_time,
                details={"error": str(e)},
                issues=[f"Stress test suite failed: {e}"]
            )
    
    def run_edge_case_tests(self) -> TestSuiteResult:
        """Run edge case tests."""
        self.print_status("Running edge case test suite...", "info")
        start_time = time.time()
        
        try:
            from tests.edge_case_tests import EdgeCaseTestSuite
            
            suite = EdgeCaseTestSuite()
            results = suite.run_all_edge_case_tests()
            
            passed = sum(1 for r in results if r.success)
            failed = len(results) - passed
            
            duration = time.time() - start_time
            
            issues = []
            for result in results:
                if not result.success and result.error_message:
                    issues.append(f"{result.test_name}: {result.error_message}")
            
            return TestSuiteResult(
                suite_name="edge_case_tests",
                total_tests=len(results),
                passed=passed,
                failed=failed,
                skipped=0,
                duration_seconds=duration,
                details={"test_results": [asdict(r) for r in results[:5]]},  # First 5 results
                issues=issues[:10]  # First 10 issues
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name="edge_case_tests",
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration_seconds=time.time() - start_time,
                details={"error": str(e)},
                issues=[f"Edge case test suite failed: {e}"]
            )
    
    def run_pytest_tests(self) -> TestSuiteResult:
        """Run pytest-based tests."""
        self.print_status("Running pytest test suite...", "info")
        start_time = time.time()
        
        try:
            test_dir = self.working_files / "tests"
            
            # Run pytest programmatically
            import pytest
            
            # Collect test results
            class TestResultCollector:
                def __init__(self):
                    self.results = {"passed": 0, "failed": 0, "skipped": 0, "errors": []}
                
                def pytest_runtest_logreport(self, report):
                    if report.when == "call":
                        if report.outcome == "passed":
                            self.results["passed"] += 1
                        elif report.outcome == "failed":
                            self.results["failed"] += 1
                            self.results["errors"].append(f"{report.nodeid}: {report.longrepr}")
                        elif report.outcome == "skipped":
                            self.results["skipped"] += 1
            
            collector = TestResultCollector()
            
            # Run pytest with custom plugin
            pytest_args = [
                str(test_dir),
                "-v",
                "-x",  # Stop on first failure
                "--tb=short"
            ]
            
            exit_code = pytest.main(pytest_args + ["-p", "no:warnings"], plugins=[collector])
            
            duration = time.time() - start_time
            total_tests = collector.results["passed"] + collector.results["failed"] + collector.results["skipped"]
            
            return TestSuiteResult(
                suite_name="pytest_tests",
                total_tests=total_tests,
                passed=collector.results["passed"],
                failed=collector.results["failed"],
                skipped=collector.results["skipped"],
                duration_seconds=duration,
                details={"exit_code": exit_code},
                issues=collector.results["errors"][:5]  # First 5 errors
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name="pytest_tests",
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration_seconds=time.time() - start_time,
                details={"error": str(e)},
                issues=[f"Pytest execution failed: {e}"]
            )
    
    def run_performance_benchmarks(self) -> TestSuiteResult:
        """Run performance benchmarks."""
        self.print_status("Running performance benchmarks...", "info")
        start_time = time.time()
        
        try:
            from benchmarks.performance_profiler import run_performance_benchmarks
            
            # Look for sample audio files
            audio_samples = []
            sample_dir = self.working_files / "audio_samples"
            if sample_dir.exists():
                audio_samples = list(sample_dir.glob("*.wav"))
            
            if not audio_samples:
                # Create a test audio file
                import numpy as np
                import soundfile as sf
                import tempfile
                
                # Generate test tone
                duration = 5.0
                sample_rate = 16000
                t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                audio_data = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, audio_data, sample_rate)
                    audio_samples = [tmp.name]
            
            # Run benchmarks
            results = run_performance_benchmarks([str(f) for f in audio_samples])
            
            duration = time.time() - start_time
            
            # Clean up temporary files
            for sample in audio_samples:
                if str(sample).startswith(tempfile.gettempdir()):
                    try:
                        os.unlink(sample)
                    except:
                        pass
            
            return TestSuiteResult(
                suite_name="performance_benchmarks",
                total_tests=len(audio_samples),
                passed=len(audio_samples) if results else 0,
                failed=0 if results else len(audio_samples),
                skipped=0,
                duration_seconds=duration,
                details=results if results else {},
                issues=[] if results else ["Benchmark execution failed"]
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name="performance_benchmarks",
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration_seconds=time.time() - start_time,
                details={"error": str(e)},
                issues=[f"Performance benchmarks failed: {e}"]
            )
    
    def run_phase3_validation(self) -> TestSuiteResult:
        """Run Phase 3 component validation."""
        self.print_status("Running Phase 3 validation...", "info")
        start_time = time.time()
        
        try:
            # Import and run the Phase 3 validator
            sys.path.insert(0, str(self.project_root))
            from phase3_validator import Phase3Validator
            
            validator = Phase3Validator()
            validation_result = validator.run_comprehensive_validation()
            
            duration = time.time() - start_time
            
            # Count components
            total_components = len(validation_result.components)
            functional_components = sum(1 for c in validation_result.components.values() if c.functional)
            
            issues = validation_result.critical_issues.copy()
            for component in validation_result.components.values():
                issues.extend(component.issues[:2])  # First 2 issues per component
            
            return TestSuiteResult(
                suite_name="phase3_validation",
                total_tests=total_components,
                passed=functional_components,
                failed=total_components - functional_components,
                skipped=0,
                duration_seconds=duration,
                details={
                    "overall_score": validation_result.overall_score,
                    "phase3_complete": validation_result.phase3_complete,
                    "recommendations": validation_result.recommendations[:5]  # First 5
                },
                issues=issues[:10]  # First 10 issues
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name="phase3_validation",
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration_seconds=time.time() - start_time,
                details={"error": str(e)},
                issues=[f"Phase 3 validation failed: {e}"]
            )
    
    def run_all_tests(self) -> List[TestSuiteResult]:
        """Run all test suites."""
        self.print_header("EMERGENCY AI - COMPREHENSIVE TEST SUITE")
        
        test_suites = [
            ("Phase 3 Validation", self.run_phase3_validation),
            ("Stress Tests", self.run_stress_tests),
            ("Edge Case Tests", self.run_edge_case_tests),
            ("Performance Benchmarks", self.run_performance_benchmarks),
            ("Pytest Tests", self.run_pytest_tests)
        ]
        
        for suite_name, test_function in test_suites:
            try:
                result = test_function()
                self.test_results.append(result)
                
                # Print immediate feedback
                success_rate = (result.passed / max(result.total_tests, 1)) * 100
                status = "success" if success_rate >= 80 else "warning" if success_rate >= 50 else "error"
                self.print_status(f"{suite_name}: {result.passed}/{result.total_tests} passed ({success_rate:.1f}%)", status)
                
            except Exception as e:
                self.print_status(f"{suite_name}: Failed with error: {e}", "error")
                traceback.print_exc()
        
        return self.test_results
    
    def print_summary(self):
        """Print comprehensive test summary."""
        self.print_header("TEST EXECUTION SUMMARY", Colors.GREEN)
        
        total_tests = sum(r.total_tests for r in self.test_results)
        total_passed = sum(r.passed for r in self.test_results)
        total_failed = sum(r.failed for r in self.test_results)
        total_skipped = sum(r.skipped for r in self.test_results)
        total_duration = sum(r.duration_seconds for r in self.test_results)
        
        overall_success_rate = (total_passed / max(total_tests, 1)) * 100
        
        print(f"\n{Colors.BOLD}Overall Results:{Colors.END}")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {Colors.GREEN}{total_passed}{Colors.END}")
        print(f"  Failed: {Colors.RED}{total_failed}{Colors.END}")
        print(f"  Skipped: {Colors.YELLOW}{total_skipped}{Colors.END}")
        print(f"  Success Rate: {overall_success_rate:.1f}%")
        print(f"  Total Duration: {total_duration:.2f}s")
        
        print(f"\n{Colors.BOLD}Suite Breakdown:{Colors.END}")
        for result in self.test_results:
            success_rate = (result.passed / max(result.total_tests, 1)) * 100
            status_icon = "[OK]" if success_rate >= 80 else "[WARNING]" if success_rate >= 50 else "[ERROR]"
            print(f"  {status_icon} {result.suite_name}: {result.passed}/{result.total_tests} ({success_rate:.1f}%)")
        
        # Show critical issues
        all_issues = []
        for result in self.test_results:
            all_issues.extend(result.issues)
        
        if all_issues:
            print(f"\n{Colors.RED}{Colors.BOLD}Critical Issues Found:{Colors.END}")
            for i, issue in enumerate(all_issues[:10], 1):  # Show first 10
                print(f"  {i}. {issue}")
            if len(all_issues) > 10:
                print(f"  ... and {len(all_issues) - 10} more issues")
        
        # Phase 3 status
        phase3_result = next((r for r in self.test_results if r.suite_name == "phase3_validation"), None)
        if phase3_result:
            phase3_complete = phase3_result.details.get("phase3_complete", False)
            phase3_score = phase3_result.details.get("overall_score", 0)
            
            print(f"\n{Colors.BOLD}Phase 3 Status:{Colors.END}")
            status_text = "COMPLETE" if phase3_complete else "INCOMPLETE"
            status_color = Colors.GREEN if phase3_complete else Colors.RED
            print(f"  Status: {status_color}{status_text}{Colors.END}")
            print(f"  Score: {phase3_score:.1f}/100")
        
        return overall_success_rate >= 75  # Consider 75% success rate as passing
    
    def export_results(self, output_path: Optional[Path] = None):
        """Export detailed test results."""
        if output_path is None:
            output_path = self.project_root / "test_results.json"
        
        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": sum(r.total_tests for r in self.test_results),
                "total_passed": sum(r.passed for r in self.test_results),
                "total_failed": sum(r.failed for r in self.test_results),
                "total_skipped": sum(r.skipped for r in self.test_results),
                "overall_success_rate": (sum(r.passed for r in self.test_results) / max(sum(r.total_tests for r in self.test_results), 1)) * 100
            },
            "suites": [asdict(result) for result in self.test_results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.print_status(f"Test results exported to: {output_path}", "success")


def main():
    """Main test runner function."""
    runner = EnhancedTestRunner()
    
    try:
        # Run all tests
        results = runner.run_all_tests()
        
        # Print summary
        success = runner.print_summary()
        
        # Export results
        runner.export_results()
        
        return 0 if success else 1
        
    except Exception as e:
        runner.print_status(f"Test execution failed: {e}", "error")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())