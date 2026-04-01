"""
Configuration file for storing path settings and other constants
"""
from pathlib import Path

# Get the base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# Define paths relative to the base directory
CASE_DIR = BASE_DIR / "cases" / "exercise" / "with_sweating" / "case1"

# Journal file paths
PARAMS_JOURNAL = CASE_DIR / "change_parameter.log"
RUN_JOURNAL = CASE_DIR / "steady_and_transient.log"

# Case file paths
CASE_FILE = CASE_DIR / "Exercise_Case_SteadyState.cas.h5" 