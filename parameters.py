
# Input parameters:
# - Metabolic rate of head
# - Metabolic rate of Organs
# - Initial Metabolic Rate of Muscle
# - Ambient Temperature
# - Heat transfer coefficient

import torch
size_data = 4000
metab_head_min = 7527.6
metab_head_max = 10922.4
metab_mus_min = 451.6
metab_mus_max = 655.3
metab_org_min = 1143.6
metab_org_max = 1659.4
T_amb_min = -15
T_amb_max = 55
h_min = 1
h_max = 10

metab_head_data = torch.empty(size_data).uniform_(metab_head_min, metab_head_max)
metab_mus_data = torch.empty(size_data).uniform_(metab_mus_min, metab_mus_max)
metab_org_data = torch.empty(size_data).uniform_(metab_org_min, metab_org_max)
T_amb_data = torch.empty(size_data).uniform_(T_amb_min, T_amb_max)
h_data = torch.empty(size_data).uniform_(h_min, h_max)

dataset = torch.stack([metab_head_data, metab_mus_data, metab_org_data, T_amb_data, h_data])

dataset_dict = {n: {str(dataset[i, n].item()): str(dataset[i, n + 1].item()) for i in range(dataset.shape[0])} for n in range(dataset.shape[1] - 1)}
dict_append = {-1: {"000": str(dataset[0, 0].item()), "001": str(dataset[1, 0].item()), "002": str(dataset[2, 0].item()), "003": str(dataset[3, 0].item()), "004": str(dataset[4, 0].item())}}
dataset_dict.update(dict_append)
