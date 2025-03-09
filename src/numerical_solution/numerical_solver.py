# Importing necessary libraries
import ansys.fluent.core as pyfluent  # Python API for Ansys Fluent
import re  # Regular expressions for text substitution
import utils.parameters as parameters  # Custom module for parameter handling
import time  # For timing operations



def update_journal_files(params_journal_path, run_journal_path, new_parameters, file_number):
    """
    Updates the Ansys Fluent journal files with new case parameters.
    
    Args:
        params_journal_path: Path to the journal file containing case parameters
        run_journal_path: Path to the journal file containing case run instructions 
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


params_journal_path = r"C:\Users\vejendsh\OneDrive - University of Cincinnati\Swarup_MS_Thesis_2015\Wholebodymodel\Exercise\With_sweating\Case1\change_parameter.log"
run_journal_path = r"C:\Users\vejendsh\OneDrive - University of Cincinnati\Swarup_MS_Thesis_2015\Wholebodymodel\Exercise\With_sweating\Case1\steady_and_transient.log"
case_folder_path = r"C:\Users\vejendsh\OneDrive - University of Cincinnati\Swarup_MS_Thesis_2015\Wholebodymodel\Exercise\With_sweating\Case1"

# start_time = time.time()
solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
                                processor_count=4,
                                ui_mode="gui",
                                py=True)

solver_session.file.read_case(file_name=case_file_path + r"\Exercise_Case_SteadyState.cas.h5")

update_file(journal_path_1, journal_path_2, parameters.dataset_dict[-1], 0)

solver_session.file.read_journal(file_name_list=[journal_path_1, journal_path_2])

solver_session.exit()
# end_time = time.time()
# time_taken = end_time - start_time
# print(f"time_taken: {time_taken}")


for i in range(parameters.size_data-1):
    solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
                                            processor_count=4,
                                            ui_mode="gui",
                                            py=True)

    solver_session.file.read_case(
        file_name=case_file_path + fr"\Exercise_Case_SteadyState_{i}.cas.h5")

    update_file(journal_path_1, journal_path_2, parameters.dataset_dict[i], i + 1)
    solver_session.file.read_journal(file_name_list=[journal_path_1, journal_path_2])
    solver_session.exit()
