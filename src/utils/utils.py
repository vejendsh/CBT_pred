import os
import time
import torch
import re
from src.config import BASE_DIR, CASE_DIR
import glob
import shutil


def _remove_path_retry(path, attempts=20, delay_s=0.5):
    """Remove a file; retry on Windows lock (Fluent may hold .trn briefly after exit)."""
    last_err = None
    for _ in range(attempts):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError as e:
            last_err = e
            time.sleep(delay_s)
    if last_err is not None:
        print(f"Warning: could not delete {path!r} (still in use). Remove it manually.")
    return False


# Function to save tensor data to a file
def save_tensor_to_file(tensor, file_name):
    if ".pt" in file_name:
        pass
    else:
        file_name = file_name + ".pt"

    # Check if file exists
    if os.path.exists(file_name):
        print(f"File {file_name} already exists.")
        user_input = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if user_input == 'y':
            torch.save(tensor, file_name)
            print(f"File '{file_name}' has been overwritten.")
        else:
            # Optionally, create a new file with a unique name
            new_file_path = file_name
            counter = 1
            while os.path.exists(new_file_path):
                base, ext = os.path.splitext(file_name)
                new_file_path = f"{base}_{counter}{ext}"
                counter += 1
            torch.save(tensor, new_file_path)
            print(f"File '{new_file_path}' has been saved instead.")
    else:
        torch.save(tensor, file_name)
        print(f"File '{file_name}' has been saved.")


def min_max_normalize(tensor):

    # Min-Max Normalization
    min_vals = tensor.min(dim=0, keepdim=True).values  # Minimum value per column
    max_vals = tensor.max(dim=0, keepdim=True).values  # Maximum value per column

    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals)

    return normalized_tensor

def z_score_normalize(tensor):

    # Compute mean and standard deviation along each feature (dim=0)
    mean = torch.mean(tensor, dim=0)
    std = torch.std(tensor, dim=0)

    # Prevent division by zero (if std is zero)
    std[std == 0] = 1e-8

    # Z-score normalization
    z_score_normalized = (tensor - mean) / std

    return z_score_normalized


def _is_steady_state_scrap_out(basename: str) -> bool:
    """
    True for files with _<even>_1.out in the name
    """
    m = re.search(r'_(\d+)_1', basename)
    if m:
        return int(m.group(1)) % 2 == 0
    return re.search(r'\d', basename) is None



_RFILE_MARKER = "-rfile"


def sort_run_dir(path: str) -> int:
    """Sort key for ``Run_n`` directories under ``data/raw``."""
    m = re.fullmatch(r"Run_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def sort_case_file(path: str) -> int:
    """Sort key for ``Exercise_Case_SteadyState_{i}.cas.h5`` in a run's ``case_files`` folder."""
    m = re.search(r"Exercise_Case_SteadyState_(\d+)\.cas\.h5$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def sort_core_temp_file(path: str) -> int:
    """Sort key for ``core_temp_{k}_1.out`` produced by ``_organize_run_dir_by_rfile_prefix``."""
    m = re.search(r"core_temp_(\d+)_1\.out$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _least_unused_rfile_index(sub: str, prefix: str) -> int:
    """Smallest positive n such that ``{prefix}_{n}_1.out`` does not exist under ``sub``."""
    n = 1
    while os.path.exists(os.path.join(sub, f"{prefix}_{n}_1.out")):
        n += 1
    return n


def _organize_run_dir_by_rfile_prefix(run_dir: str) -> None:
    """Move each *-rfile*.out in run_dir into run_dir/<prefix>/ as ``<prefix>_{n}_1.out`` (least unused n)."""
    for name in list(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue
        if _RFILE_MARKER not in name:
            continue
        prefix = name.split(_RFILE_MARKER, 1)[0]
        if not prefix:
            continue
        sub = os.path.join(run_dir, prefix)
        os.makedirs(sub, exist_ok=True)
        n = _least_unused_rfile_index(sub, prefix)
        new_name = f"{prefix}_{n}_1.out"
        shutil.move(path, os.path.join(sub, new_name))


class CustomLoss(torch.nn.Module):
    def __init__(self, avg_penalty=1):
        super(CustomLoss, self).__init__()
        self.avg_penalty = avg_penalty

    def forward(self, preds, targets):
        loss = torch.mean((preds - targets) ** 2) + 10*torch.sum((preds[:, :3] - targets[:, :3]) ** 2)

        return loss


def organize_run(run_number):
    # delete .trn files and .bat files in base directory
    for file in os.listdir(BASE_DIR):
        if file.endswith(".trn"):
            _remove_path_retry(os.path.join(BASE_DIR, file))
        if file.endswith(".bat"):
            _remove_path_retry(os.path.join(BASE_DIR, file))

    # DELETE other steady-state .out FILES 
    for name in os.listdir(BASE_DIR):
        if not name.endswith(".out"):
            continue
        if _is_steady_state_scrap_out(name):
            _remove_path_retry(os.path.join(BASE_DIR, name))

    # archive remaining .out files under data/raw/Run_{run_number}
    run_dir = os.path.join(BASE_DIR, "data", "raw", f"Run_{run_number}")
    os.makedirs(run_dir, exist_ok=True)
    for f in glob.glob(os.path.join(BASE_DIR, "*.out")):
        shutil.move(f, os.path.join(run_dir, os.path.basename(f)))
    _organize_run_dir_by_rfile_prefix(run_dir)
    case_files_dir = os.path.join(run_dir, "case_files")
    os.makedirs(case_files_dir, exist_ok=True)

    # Fluent writes .cas.h5 / .dat.h5 to BASE_DIR; move them into run_dir / case_files folder
    for file in glob.glob(os.path.join(BASE_DIR, "*.h5")):
        shutil.move(file, os.path.join(case_files_dir, os.path.basename(file)))


