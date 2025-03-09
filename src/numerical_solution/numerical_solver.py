import ansys.fluent.core as pyfluent
import re
import parameters
import time

# Function to update the file
def update_file(file_path_1, file_path_2, new_values, n):

    with open(file_path_1, 'r') as file:
        file_contents = file.read()

    for old_value, new_value in new_values.items():
        file_contents = re.sub(re.escape(old_value), new_value, file_contents)

    with open(file_path_1, 'w') as file:
        file.write(file_contents)

    with open(file_path_2, 'r') as file:
        file_contents = file.read()

    new_filename = f"Exercise_Case_SteadyState_{n}.cas.h5"
    file_contents = file_contents.replace(f"Exercise_Case_SteadyState_{n-1}.cas.h5", new_filename)

    with open(file_path_2, 'w') as file:
        file.write(file_contents)


journal_path_1 = r"C:\Users\vejendsh\OneDrive - University of Cincinnati\Swarup_MS_Thesis_2015\Wholebodymodel\Exercise\With_sweating\Case1\change_parameter.log"
journal_path_2 = r"C:\Users\vejendsh\OneDrive - University of Cincinnati\Swarup_MS_Thesis_2015\Wholebodymodel\Exercise\With_sweating\Case1\steady_and_transient.log"
case_file_path = r"C:\Users\vejendsh\OneDrive - University of Cincinnati\Swarup_MS_Thesis_2015\Wholebodymodel\Exercise\With_sweating\Case1"

start_time = time.time()
solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
                                processor_count=4,
                                ui_mode="gui",
                                py=True)

solver_session.file.read_case(file_name=case_file_path + r"\Exercise_Case_SteadyState.cas.h5")

update_file(journal_path_1, journal_path_2, parameters.dataset_dict[-1], 0)

solver_session.file.read_journal(file_name_list=[journal_path_1, journal_path_2])

solver_session.exit()
end_time = time.time()
time_taken = end_time - start_time
print(f"time_taken: {time_taken}")

#
# for i in range(parameters.size_data-1):
#     solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
#                                             processor_count=4,
#                                             ui_mode="gui",
#                                             py=True)
#
#     solver_session.file.read_case(
#         file_name=case_file_path + fr"\Exercise_Case_SteadyState_{i}.cas.h5")
#
#     update_file(journal_path_1, journal_path_2, parameters.dataset_dict[i], i + 1)
#     solver_session.file.read_journal(file_name_list=[journal_path_1, journal_path_2])
#     solver_session.exit()
