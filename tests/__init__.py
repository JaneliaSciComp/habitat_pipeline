"""
KilosortData test suite.

Critical tests for neural electrophysiology data loading and analysis.
"""

__version__ = "1.0.0"

# Test categories
LOADING_TESTS = "test_kilosort_data_loading.py"
ANALYSIS_TESTS = "test_kilosort_data_analysis.py" 
INTEGRATION_TESTS = "test_integration.py"

# Quick test runners for interactive use
def run_loading_tests():
    """Run data loading validation tests."""
    import subprocess
    import sys
    from pathlib import Path
    
    test_dir = Path(__file__).parent
    cmd = [sys.executable, "-m", "pytest", str(test_dir / LOADING_TESTS), "-v"]
    subprocess.run(cmd)

def run_analysis_tests():
    """Run analysis method tests."""
    import subprocess
    import sys
    from pathlib import Path
    
    test_dir = Path(__file__).parent
    cmd = [sys.executable, "-m", "pytest", str(test_dir / ANALYSIS_TESTS), "-v"]
    subprocess.run(cmd)

def run_all_tests():
    """Run complete test suite."""
    from .run_tests import run_tests
    return run_tests()