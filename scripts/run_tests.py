#!/usr/bin/env python3
"""
Script to run different types of tests.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    print(f"Success: {result.stdout}")
    return True


def main():
    """Main test runner."""
    print("BestBeta Test Runner")
    print("=" * 30)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("Error: pyproject.toml not found. Run this script from the project root.")
        sys.exit(1)

    print("\nAvailable test options:")
    print("1. Regular tests (fast)")
    print("2. Hypothesis tests (comprehensive)")
    print("3. All tests")
    print("4. Install dev dependencies first")
    print("5. Install all dependencies (dev + UI)")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        # Regular tests
        cmd = "python -m pytest tests/ -v -m 'not hypothesis'"
        if not run_command(cmd, "Running regular tests"):
            sys.exit(1)

    elif choice == "2":
        # Hypothesis tests
        cmd = "python -m pytest tests/ -v -m hypothesis"
        if not run_command(cmd, "Running Hypothesis tests"):
            sys.exit(1)

    elif choice == "3":
        # All tests
        cmd = "python -m pytest tests/ -v"
        if not run_command(cmd, "Running all tests"):
            sys.exit(1)

    elif choice == "4":
        # Install dev dependencies
        cmd = "pip install -e .[dev]"
        if not run_command(cmd, "Installing dev dependencies"):
            sys.exit(1)
        print("\nDev dependencies installed. You can now run Hypothesis tests.")

    elif choice == "5":
        # Install all dependencies
        cmd = "pip install -e .[all]"
        if not run_command(cmd, "Installing all dependencies"):
            sys.exit(1)
        print("\nAll dependencies installed. You can now run all tests and use the UI.")

    else:
        print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        sys.exit(1)


if __name__ == "__main__":
    main()
