"""
This script uses PyFluent API to set parameters and run Ansys Fluent simulations.
It replaces the functionality of both params.log and run.log journal files.
"""

import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import ansys.fluent.core as pyfluent
import time
import utils.parameters as parameters
from src.config import CASE_FILE, CASE_DIR


def set_input_parameters(solver_session, parameters_dict):
    """
    Sets input parameters in Ansys Fluent using PyFluent API.
    
    Args:
        solver_session: The active PyFluent solver session
        parameters_dict: Dictionary containing parameter names and their values
                        Example: {"heat_generation_muscle": "500 [W/m^3]",
                                 "heat_generation_fat": "100 [W/m^3]",
                                 "heat_generation_skin": "200 [W/m^3]",
                                 "ambient_temperature": "25 [C]",
                                 "heat_transfer_coefficient": "10 [W/m^2 K]"}
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
    solver_session.file.write(file_name=str(output_file))


def run_case(case_number, params_dict):
    """
    Runs a complete case with the given parameters.
    
    Args:
        case_number: The case number for saving the results
        params_dict: Dictionary containing parameter names and their values
    """
    # Launch Fluent
    print(f"Starting case {case_number}...")
    solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, 
                                           dimension=pyfluent.Dimension.THREE,
                                           processor_count=4,
                                           ui_mode="gui",
                                           py=True)
    
    try:
        # Read the case file
        if case_number == 0:
            # For the first case, read the initial case file
            solver_session.file.read_case(file_name=str(CASE_FILE))
        else:
            # For subsequent cases, read the previous case file
            prev_case_file = CASE_DIR / f"Exercise_Case_SteadyState_{case_number-1}.cas.h5"
            solver_session.file.read_case(file_name=str(prev_case_file))
        
        # Set the parameters
        set_input_parameters(solver_session, params_dict)
        
        # Run the simulation
        run_simulation(solver_session, case_number)
        
        print(f"Case {case_number} completed successfully.")
    except Exception as e:
        print(f"Error in case {case_number}: {str(e)}")
    finally:
        # Exit Fluent
        solver_session.exit()


def main():
    """
    Main function to run all cases.
    """
    start_time = time.time()
    
    # Run the initial case
    run_case(0, parameters.dataset_dict[-1])
    
    # Run all other cases
    for i in range(parameters.size_data-1):
        run_case(i+1, parameters.dataset_dict[i])
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Total time taken: {time_taken:.2f} seconds")


if __name__ == "__main__":
    main() 