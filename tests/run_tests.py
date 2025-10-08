#!/usr/bin/env python3
"""
Test runner script for STT system.
"""
import sys
import subprocess
import os


def run_tests():
    """Run all tests with pytest."""
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # Run pytest with appropriate options
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/',
        '-v',
        '--tb=short',
        '--disable-warnings'
    ]
    
    # Add markers for different test types
    if len(sys.argv) > 1:
        if sys.argv[1] == '--unit':
            cmd.extend(['-m', 'unit'])
        elif sys.argv[1] == '--integration':
            cmd.extend(['-m', 'integration'])
        elif sys.argv[1] == '--slow':
            cmd.extend(['-m', 'slow'])
        elif sys.argv[1] == '--api':
            cmd.extend(['-m', 'requires_api'])
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
