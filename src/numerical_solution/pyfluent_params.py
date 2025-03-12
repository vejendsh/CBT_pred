"""
This script uses PyFluent API to set input parameters in Ansys Fluent.
It replaces the functionality of the params.log journal file.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
import re
from src.config import RUN_DIR
from src.config import CASE_FILE
from src.utils.utils import generate_profile

import ansys.fluent.core as pyfluent


def set_input_parameters(solver_session, scalar_params, profile_params):
    """
    Sets input parameters in Ansys Fluent using PyFluent API.
    
    Args:
        solver_session: The active PyFluent solver session
        scalar_params: Dictionary containing scalar parameter names and their ranges
                      Example: {"metab_muscle": [500, 1000],
                               "metab_organ": [100, 200]}
        profile_params: Dictionary containing profile parameter names and their ranges
                       Example: {"HR": [60, 120]}
    """
    # Set each parameter value using named expressions
    named_exp = solver_session.settings.setup.named_expressions

    for param_name, param_range in scalar_params.items():
        random_scalar = round(np.random.uniform(param_range[0], param_range[1]), 3)
        param = named_exp[param_name]
        units = param_range[-1]
        param.definition.set_state(f"{random_scalar} [{units}]")
        param_value = param.definition.get_state()
        print(f"Set {param_name} to {param_value}")

    for profile_name, profile_range in profile_params.items():
        # Generate a profile file using the utility function
        random_profile = generate_profile(profile_name, profile_range, RUN_DIR)
        # Read the profile into Fluent
        solver_session.file.read_profile(file_name=rf"{RUN_DIR}\profiles\{random_profile}")
        # Set the named expression to use the profile
        units = profile_range[-1]
        named_exp[profile_name].definition.set_state(f"profile('random_{profile_name.lower()}_profile', '{profile_name.lower()}')")
        print(f"Set {profile_name} to use profile: {random_profile}") 


def main():
    """
    Example usage of the set_input_parameters function.
    """
    # Define parameter ranges
    scalar_params = {
        "metab_muscle": [500, 1000, "W/m^3"],
        "metab_organ": [100, 200, "W/m^3"],
        "metab_head": [200, 300, "W/m^3"],
        "T_amb": [20, 30, "C"],
        "h": [10, 20, "W/m^2/K"]
    }
    profile_params = {
        "HR" : [3600, 7200, "s^-1"]
    }

    # Launch Fluent
    solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, 
                                           dimension=pyfluent.Dimension.THREE,
                                           processor_count=4,
                                           py=True)
    
    # Read the case file
    solver_session.file.read_case(file_name=rf"{CASE_FILE}")
    named_exp = solver_session.settings.setup.named_expressions

    print("\nExisting named expressions:")
    param_names = list(scalar_params) + list(profile_params)
    print(f"param_names_before: {param_names}")
    named_exp.compute(names=param_names)

    # Set the parameters
    set_input_parameters(solver_session, scalar_params, profile_params)
    
    # List updated named expressions
    print("\nUpdated named expressions:")
    print(f"param_names_after: {param_names}")
    print(named_exp.get_object_names())
    named_exp.compute(names=param_names)
    solver_session.file.write(file_type="case", file_name=CASE_FILE)
 

    # Exit Fluent when done
    solver_session.exit()

if __name__ == "__main__":
    main() 