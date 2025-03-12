"""
This script uses Ansys Fluent solver to solve a given case for different values of case parameters.
The case parameters are defined in the utils.parameters file.
This implementation uses PyFluent API directly instead of journal files.
"""

import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Importing necessary libraries
import ansys.fluent.core as pyfluent  # Python API for Ansys Fluent
import time  # For timing operations
import utils.parameters as parameters  # Custom module for parameter handling
from src.config import CASE_FILE, CASE_DIR  # Import path configurations


def set_input_parameters(solver_session, parameters_dict):
    """
    Sets input parameters in Ansys Fluent using PyFluent API.
    
    Args:
        solver_session: The active PyFluent solver session
        parameters_dict: Dictionary containing parameter names and their values
    """
    # Access the parameters in the Fluent session
    fluent_params = solver_session.parameters
    
    # Set each parameter value
    for param_name, param_value in parameters_dict.items():
        # Set the parameter value
        fluent_params.input_parameters[param_name].value = param_value
        print(f"Set parameter '{param_name}' to '{param_value}'")


def run_simulation(solver_session, case_number):
    """
    Runs a steady-state simulation followed by a transient simulation in Ansys Fluent.
    
    Args:
        solver_session: The active PyFluent solver session
        case_number: The case number for saving the results
    """
    # Set up steady-state simulation
    print("Setting up steady-state simulation...")
    solver_session.setup.general.solver.time = "Steady"
    
    # Initialize the solution
    print("Initializing solution...")
    solver_session.solution.initialization.initialize()
    
    # Patch temperature and velocity values
    print("Patching initial values...")
    # Patch temperature
    solver_session.solution.initialization.patch.variable = "temperature"
    solver_session.solution.initialization.patch.zones = ["fluid", "solid-1", "solid-2"]  # Replace with your actual zone names
    solver_session.solution.initialization.patch.patch()
    
    # Patch velocity
    solver_session.solution.initialization.patch.variable = "velocity"
    solver_session.solution.initialization.patch.patch()
    
    # Run steady-state calculation
    print("Running steady-state calculation...")
    solver_session.solution.run_calculation.calculate()
    
    # Switch to transient simulation
    print("Switching to transient simulation...")
    solver_session.setup.general.solver.time = "Transient"
    
    # Run transient calculation
    print("Running transient calculation...")
    solver_session.solution.run_calculation.calculate()
    
    # Save the case and data
    output_file = CASE_DIR / f"Exercise_Case_SteadyState_{case_number}.cas.h5"
    print(f"Saving results to {output_file}...")
    solver_session.file.write(file_name=rf"{output_file}")


# Start timing the execution
start_time = time.time()

# Launch Ansys Fluent
solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
                                processor_count=4,
                                ui_mode="gui",
                                py=True)

# Read the initial case file
solver_session.file.read_case(file_name=rf"{CASE_FILE}")

# Set parameters and run the initial case
set_input_parameters(solver_session, parameters.dataset_dict[-1])
run_simulation(solver_session, 0)

# Exit Fluent
solver_session.exit()

# Run the case for all the parameters
for i in range(parameters.size_data-1):
    solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
                                            processor_count=4,
                                            ui_mode="gui",
                                            py=True)

    # Read the previous case file
    case_file = CASE_DIR / f"Exercise_Case_SteadyState_{i}.cas.h5"
    solver_session.file.read_case(file_name=rf"{case_file}")
    
    # Set parameters and run the simulation
    set_input_parameters(solver_session, parameters.dataset_dict[i])
    run_simulation(solver_session, i + 1)
    
    # Exit Fluent
    solver_session.exit()

# Calculate and print the total execution time
end_time = time.time()
time_taken = end_time - start_time
print(f"Total time taken: {time_taken:.2f} seconds")
