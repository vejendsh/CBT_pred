
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


def fmt_param(j: int, val) -> str:
    """Format parameter j for Fluent's expression Definition line (units + spacing match GUI)."""
    x = float(val)
    s = f"{x:.15g}"
    if j == 0:
        return f"{s}[W/m^3]"
    if j in (1, 2):
        return f"{s} [W/m^3]"
    if j == 3:
        return f"{s} [C]"
    if j == 4:
        return f"{s} [W/m^2 K]"
    raise IndexError(f"Parameter index j must be 0..4, got {j}")


def expression_strings_for_sample(sample_idx: int) -> list[str]:
    """The five Definition strings for dataset column ``sample_idx``."""
    return [fmt_param(j, dataset[j, sample_idx].item()) for j in range(5)]
