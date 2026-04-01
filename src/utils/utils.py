import os
import torch
import re

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


