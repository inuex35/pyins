#!/usr/bin/env python3
"""
Run all PyINS examples and verify they work correctly
"""

import sys
import os
import subprocess
import time
from pathlib import Path


def run_example(example_path, timeout=30):
    """Run a single example with timeout"""
    print(f"\n{'='*70}")
    print(f"Running: {example_path}")
    print('='*70)
    
    try:
        start = time.time()
        result = subprocess.run(
            [sys.executable, example_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"✓ {os.path.basename(example_path)} completed successfully ({elapsed:.2f}s)")
            return True, elapsed
        else:
            print(f"✗ {os.path.basename(example_path)} failed with return code {result.returncode}")
            if result.stderr:
                print("\nError output:")
                print("-" * 40)
                print(result.stderr[:500])
            return False, elapsed
            
    except subprocess.TimeoutExpired:
        print(f"⏱ {os.path.basename(example_path)} timed out after {timeout}s")
        return False, timeout
    except Exception as e:
        print(f"⚠ Error running {os.path.basename(example_path)}: {e}")
        return False, 0


def main():
    """Run all examples"""
    print("=" * 70)
    print("PyINS Example Test Runner")
    print("=" * 70)
    
    # Define working examples
    examples = [
        "gnss/example_gnss_processing.py",
        "gnss/example_observables.py",
        "gnss/example_satellite_positions.py",
        "rtk/example_double_difference.py",
        "rtk/example_lambda.py",
    ]
    
    examples_dir = Path(__file__).parent
    
    print(f"\nFound {len(examples)} examples to run")
    
    # Run each example
    results = []
    total_time = 0
    
    for example in examples:
        example_path = examples_dir / example
        if example_path.exists():
            success, elapsed = run_example(str(example_path))
            results.append((example, success, elapsed))
            total_time += elapsed
        else:
            print(f"\n⚠ Example not found: {example}")
            results.append((example, False, 0))
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for _, success, _ in results if success)
    fail_count = len(results) - success_count
    
    print(f"\nTotal examples: {len(results)}")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed:  {fail_count}")
    print(f"\nTotal execution time: {total_time:.2f}s")
    
    print("\n" + "-" * 70)
    print("Detailed Results:")
    print("-" * 70)
    print(f"{'Example':<50} {'Status':<10} {'Time (s)':<10}")
    print("-" * 70)
    
    for example, success, elapsed in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{example:<50} {status:<10} {elapsed:>8.2f}")
    
    # Exit with appropriate code
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()