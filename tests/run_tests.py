"""
Simple test runner for KilosortData tests.

Run with: python run_tests.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run the KilosortData test suite."""
    
    # Get the test directory
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    # Add project root to Python path so we can import modules
    sys.path.insert(0, str(project_root))
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not installed. Install with: pip install pytest")
        return False
    
    print("🧪 Running KilosortData Test Suite (Updated for actual implementation)")
    print("=" * 60)
    
    # Test categories to run
    test_files = [
        "test_kilosort_data_loading.py",
        "test_kilosort_data_analysis.py",
        "test_integration.py",
        "test_multi_animal_session.py",
        "test_multi_animal_tracking.py",
        "test_social_spatial_fields.py",
        "test_inter_brain_dynamics.py",
        "test_behavior_features.py",
        "test_inter_brain_plots.py",
        "test_run_inter_brain.py",
        "test_gui_inter_brain.py",
    ]
    
    all_passed = True
    
    for test_file in test_files:
        test_path = test_dir / test_file
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            continue
            
        print(f"\n📁 Running {test_file}")
        print("-" * 30)
        
        # Run pytest for this file
        cmd = ["python", "-m", "pytest", str(test_path), "-v", "--tb=short"]
        
        try:
            result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {test_file} - All tests passed")
            else:
                print(f"❌ {test_file} - Some tests failed")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                all_passed = False
                
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed successfully!")
    else:
        print("💥 Some tests failed. Check output above for details.")
    
    return all_passed


def run_specific_category(category: str):
    """Run tests for a specific category."""
    
    categories = {
        "loading": "test_kilosort_data_loading.py",
        "analysis": "test_kilosort_data_analysis.py",
        "integration": "test_integration.py",
        "multi_animal": "test_multi_animal_session.py",
        "inter_brain": "test_inter_brain_dynamics.py",
        "behavior_features": "test_behavior_features.py",
        "inter_brain_plots": "test_inter_brain_plots.py",
        "run_inter_brain": "test_run_inter_brain.py",
        "gui_inter_brain": "test_gui_inter_brain.py",
        "all": None  # Run all tests
    }
    
    if category not in categories:
        print(f"Unknown category: {category}")
        print(f"Available categories: {list(categories.keys())}")
        return False
    
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    sys.path.insert(0, str(project_root))
    
    if category == "all":
        return run_tests()
    
    test_file = categories[category]
    test_path = test_dir / test_file
    
    print(f"🧪 Running {category} tests")
    print("=" * 30)
    
    cmd = ["python", "-m", "pytest", str(test_path), "-v", "--tb=short"]
    
    try:
        result = subprocess.run(cmd, cwd=str(project_root))
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        category = sys.argv[1]
        success = run_specific_category(category)
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)