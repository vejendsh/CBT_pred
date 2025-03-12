"""
Configuration settings for the project.
"""

from pathlib import Path

# Get the base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# Directory for storing run data
RUN_DIR = BASE_DIR / r"runs"

# Define paths relative to the base directory
CASE_DIR = BASE_DIR / r"cases" / r"exercise" / r"with_sweating" / r"case_1"

# Case file paths
CASE_FILE = CASE_DIR / r"Case1.cas.h5"  

# Profile generation settings
PROFILE_DURATION = 3600  # Duration in seconds (1 hour)
STEP_SIZE = 60  # Step size in seconds (1 minute)

# Fourier sampler settings
u_max_freq = 10  # Maximum frequency for Fourier sampler
nu_range = [0, 1]  # Default range for normalized values




# # Journal file paths
# PARAMS_JOURNAL = CASE_DIR / r"journals" / r"params.log"
# RUN_JOURNAL = CASE_DIR / r"journals" / r"run.log"


