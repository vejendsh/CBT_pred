import os
import torch
import re
import numpy as np
from pathlib import Path
import csv

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


# Function to extract numbers from the filename
def sort_case_file(file_path):
    # Extract the first sequence of digits from the filename
    match = re.search(r'\d+', os.path.basename(file_path))
    return int(match.group()) if match else 0  # Default to 0 if no number is found


def sort_core_temp_file(file_path):
    # Extract the first sequence of digits from the filename
    match = re.search(r'_(\d+)_1', os.path.basename(file_path))
    return int(match.group(1)) if match else 0  # Default to 0 if no number is found


class CustomLoss(torch.nn.Module):
    def __init__(self, avg_penalty=1):
        super(CustomLoss, self).__init__()
        self.avg_penalty = avg_penalty

    def forward(self, preds, targets):
        loss = torch.mean((preds - targets) ** 2) + 10*torch.sum((preds[:, :3] - targets[:, :3]) ** 2)

        return loss


class FunctionSampler:
    def __init__(self, x):
        self.x = x

    def sample(self, *args, **kwargs):
        pass


class Fourier(FunctionSampler):
    def __init__(self, x):
        super().__init__(x)
        self.nx = len(self.x)

    def sample(self, n, max_freq=None, range=None):
        # Use config value if not provided
        if max_freq is None:
            max_freq = config.u_max_freq
        if range is None:
            range = config.nu_range
            
        num_terms = np.random.randint(1, max_freq+1,  size=(n, 1, 1))
        amp_cos = np.random.uniform(-5, 5, size=(n, 1, max_freq))
        amp_sin = np.random.uniform(-5, 5, size=(n, 1, max_freq))
        phase_cos = np.random.uniform(-np.pi, np.pi, size=(n, 1, max_freq))
        phase_sin = np.random.uniform(-np.pi, np.pi, size=(n, 1, max_freq))

        freq = np.arange(1, max_freq+1, 1).reshape(-1, 1)

        functions_cos = np.real(amp_cos*(np.e**(1j * (np.expand_dims((2 * np.pi * self.x * freq).T, 0) + phase_cos))))
        functions_sin = np.real(amp_sin*(np.e**(1j * (np.expand_dims((2 * np.pi * self.x * freq).T, 0) + phase_sin))))
        functions = functions_cos + functions_sin

        mask = np.broadcast_to(num_terms, (n, self.nx, max_freq)) >= np.broadcast_to(freq.reshape(1, 1, -1), (n, self.nx, max_freq))
        functions = np.sum(functions*mask, axis=-1)
        max = np.max(functions)
        min = np.min(functions)
        mean = np.mean(functions)
        if range is not None:
            functions = range[0] + (((range[1]-range[0])/(max-min))*(functions-min))
        return functions


def generate_profile(profile_name, profile_range, RUN_DIR=None, profile_duration=None, step_size=None):
    """
    Generates a profile CSV file for use in Ansys Fluent.
    
    Args:
        profile_name (str): Name of the profile (e.g., "HR", "temperature")
        profile_range (list): Range of values [min, max] for the profile
        RUN_DIR (str, optional): Directory where profiles will be stored. Defaults to config.RUN_DIR.
        profile_duration (int, optional): Duration of the profile in seconds. Defaults to config.PROFILE_DURATION.
        step_size (int, optional): Time step size in seconds. Defaults to config.STEP_SIZE.
        
    Returns:
        str: Filename of the generated profile
    """
    # Import config here to avoid circular imports
    from src.config import RUN_DIR as DEFAULT_RUN_DIR
    from src.config import PROFILE_DURATION as DEFAULT_PROFILE_DURATION
    from src.config import STEP_SIZE as DEFAULT_STEP_SIZE
    
    # Use default values from config if not provided
    if RUN_DIR is None:
        RUN_DIR = DEFAULT_RUN_DIR
    if profile_duration is None:
        profile_duration = DEFAULT_PROFILE_DURATION
    if step_size is None:
        step_size = DEFAULT_STEP_SIZE
    
    # Create profiles directory if it doesn't exist
    profiles_dir = Path(RUN_DIR) / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate time points
    time_points = np.arange(0, profile_duration + step_size, step_size)
    
    # Normalize time points to [0, 1] for the Fourier sampler
    normalized_time = time_points / profile_duration
    
    # Create a Fourier sampler
    sampler = Fourier(normalized_time)
    
    # Generate a random profile within the specified range
    profile_values = sampler.sample(1, max_freq=5, range=profile_range)[0]
    
    # Create filename
    filename = f"random_{profile_name}_profile.csv"
    filepath = profiles_dir / filename
    
    # Write the profile to a CSV file
    with open(filepath, 'w', newline='') as csvfile:
        # Write header
        csvfile.write("[Name]\n")
        csvfile.write(f"random_{profile_name}_profile\n")
        csvfile.write("[Data]\n")
        csvfile.write(f"time,{profile_name}\n")
        
        # Write data
        for t, val in zip(time_points, profile_values):
            csvfile.write(f"{t},{val}\n")
    
    print(f"Generated profile: {filename}")
    return filename


