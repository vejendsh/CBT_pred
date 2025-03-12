"""
This script uses Ansys Fluent solver to solve a given case for different values of case parameters.
The case parameters are defined in the utils.parameters file.
Journal files are used to update the case parameters and run the case.
The solved case files are saved in the case_folder_path.
"""

# Importing necessary libraries
import ansys.fluent.core as pyfluent  # Python API for Ansys Fluent
import re  # Regular expressions for text substitution
import utils.parameters as parameters  # Custom module for parameter handling
import time  # For timing operations
from pathlib import Path  # For platform-independent path handling
from src.config import PARAMS_JOURNAL, RUN_JOURNAL, CASE_DIR, CASE_FILE  # Import path configurations
print("Hello World")

def update_journal_files(params_journal_path, run_journal_path, new_parameters, file_number):
    """
    Updates the Ansys Fluent journal files with new case parameters.
    
    Args:
        params_journal_path: Path to the journal file containing case parameters
        run_journal_path: Path to the journal file containing run instructions for the current case
        new_parameters: A Dictionary of parameters to update (old_value: new_value pairs)
        file_number: Case number for the new case file
    """
    # Open the journal file containing case parameters 
    with open(params_journal_path, 'r') as file:
        file_contents = file.read()

    # Replace all old parameter values with new values
    for old_value, new_value in new_parameters.items():
        file_contents = re.sub(re.escape(old_value), new_value, file_contents)

    # Write the updated contents back to the parameters journal file
    with open(params_journal_path, 'w') as file:
        file.write(file_contents)

    # Open the run journal file
    with open(run_journal_path, 'r') as file:
        file_contents = file.read()

    # Update the case filename in the run journal
    new_filename = f"Exercise_Case_SteadyState_{file_number}.cas.h5"
    file_contents = file_contents.replace(f"Exercise_Case_SteadyState_{file_number-1}.cas.h5", new_filename)

    # Write the updated contents back to the run journal file
    with open(run_journal_path, 'w') as file:
        file.write(file_contents)

# start_time = time.time()

# Launch Ansys Fluent
solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
                                processor_count=4,
                                ui_mode="gui",
                                py=True)

# Read the initial case file
solver_session.file.read_case(file_name=str(CASE_FILE))

# Update the journal files with new case parameters
update_journal_files(PARAMS_JOURNAL, RUN_JOURNAL, parameters.dataset_dict[-1], 0)

# Read the updated journal files
solver_session.file.read_journal(file_name_list=[str(PARAMS_JOURNAL), str(RUN_JOURNAL)])

# Exit Ansys Fluent
solver_session.exit()

# end_time = time.time()
# time_taken = end_time - start_time
# print(f"time_taken: {time_taken}")

# Run the case for all the parameters
for i in range(parameters.size_data-1):
    solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
                                            processor_count=4,
                                            ui_mode="gui",
                                            py=True)

    case_file = CASE_DIR / f"Exercise_Case_SteadyState_{i}.cas.h5"
    solver_session.file.read_case(file_name=str(case_file))

    update_journal_files(PARAMS_JOURNAL, RUN_JOURNAL, parameters.dataset_dict[i], i + 1)
    solver_session.file.read_journal(file_name_list=[str(PARAMS_JOURNAL), str(RUN_JOURNAL)])
    solver_session.exit()
