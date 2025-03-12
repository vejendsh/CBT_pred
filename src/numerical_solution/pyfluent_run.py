"""
This script uses PyFluent API to run Ansys Fluent simulations.
It replaces the functionality of the run.log journal file.
"""

import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import ansys.fluent.core as pyfluent
from src.config import CASE_FILE, CASE_DIR


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


def main():
    """
    Example usage of the run_simulation function.
    """
    # Launch Fluent
    solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, 
                                           dimension=pyfluent.Dimension.THREE,
                                           processor_count=4,
                                           ui_mode="gui",
                                           py=True)
    
    # Read the case file
    solver_session.file.read_case(file_name=rf"{CASE_FILE}")
    
    # Run the simulation
    run_simulation(solver_session, case_number=0)
    
    # Exit Fluent when done
    solver_session.exit()


if __name__ == "__main__":
    main() 