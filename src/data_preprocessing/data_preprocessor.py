# h
# T_amb
# metab_head
# metab_muscle
# metab_organ


import ansys.fluent.core as pyfluent
import glob
import os
from ansys.fluent.core.filereader.case_file import CaseFile
import torch
from tqdm import tqdm
import src.utils.utils as utils
from src.config import BASE_DIR

# Runs from ``organize_run``: ``data/raw/Run_n/{case_files, core_temp, ...}``
data_path = BASE_DIR / "data" / "raw"
run_dirs = sorted(glob.glob(os.path.join(data_path, "Run_*")), key=utils.sort_run_dir)
input_data_list = []

for folder_path in run_dirs:
    case_glob = os.path.join(folder_path, "case_files", "Exercise_Case_SteadyState_*.cas.h5")
    for file_path in tqdm(sorted(glob.glob(case_glob), key=utils.sort_case_file), desc="Processing Files", leave=False):
        print(file_path)
        reader = CaseFile(case_file_name=file_path)
        input_data_list.append(torch.tensor([float(p.value.split()[0]) for p in reader.input_parameters()[-5:]]))

input_data = torch.stack(input_data_list, dim=0) 
utils.save_tensor_to_file(input_data, "input_data")
print(input_data.shape)

output_data_list = []
for folder_path in run_dirs:
    core_glob = os.path.join(folder_path, "core_temp", "*.out")
    for file_path in tqdm(sorted(glob.glob(core_glob), key=utils.sort_core_temp_file), desc="Processing Core temp Files"):
        with open(file_path, "r") as f:
            lines = f.readlines()
            if "Time Step" in lines[1]:
                instance_list = []
                for line in lines[3:]:
                    instance_list.append(torch.tensor([float(v) for v in line.strip().split()]))
                output_data_list.append(torch.stack(instance_list, dim=0))

output_data = torch.stack(output_data_list, dim=0)
utils.save_tensor_to_file(output_data, "output")

print(output_data.shape)







# reader = CaseFile(case_file_name=case_file_path)
# data = [p.name for p in reader.input_parameters()]
# print(data)