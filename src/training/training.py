import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch import optim
from model import NeuralNetwork
import wandb
import utils
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CBTDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

input_data =  torch.load("input_data_3.pt")
print(input_data[11])
output_data = torch.load("output_3.pt")[:, :, 1]
input_data_norm = utils.min_max_normalize(input_data).to(device)
output_data = torch.fft.rfft(output_data, dim=1).to(device)
real = output_data.real
imag = output_data.imag
output_data_norm = torch.nan_to_num(utils.min_max_normalize(torch.stack([real, imag], dim=-1).view(970, -1)), nan=0)
output_min = torch.min((torch.stack([real, imag], dim=-1).view(970, -1)), dim=0).values
output_max = torch.max((torch.stack([real, imag], dim=-1).view(970, -1)), dim=0).values


train_size = int((9/10)*(input_data.shape[0]))
test_size = int((1/10)*(input_data.shape[0]))
dataset_size = int(train_size + test_size)

train_dataset = CBTDataset(input_data_norm[:train_size], output_data_norm[:train_size])
test_dataset = CBTDataset(input_data_norm[train_size:], output_data_norm[train_size:])

batch_size = 128

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="CBT_pred_exercise",
# )
#

def run_epoch(dataloader, epoch, model, loss_fn, optimizer, scheduler, mode="train"):

    model.train()
    loss_accum = 0
    step = 0

    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for batch, (X, y) in enumerate(tqdmDataLoader):

            X, y = X.to(device), y.to(device)
            # Compute prediction and loss
            optimizer.zero_grad()
            pred = model(X)
            loss_node = loss_fn(pred, y)
            loss_accum += loss_node
            step += 1
            loss = loss_accum/step

            if mode == "train":
                # Backpropagation
                loss_node.backward()
                optimizer.step()
                scheduler.step(loss_node)

                # wandb.log({"train_epoch": epoch+1, "train_loss": loss.item(), "train_loss_node": loss_node.item(), "LR": optimizer.state_dict()['param_groups'][0]["lr"]})

                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "train_epoch": epoch + 1,
                        "train_loss": loss.item(),
                        "train_loss_node: ": loss_node.item(),
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )
            # else:
                # wandb.log({"test_epoch": epoch+1, "test_loss": loss, "test_loss_node": loss_node.item()})

            del loss_node

import time
def train_model(epochs):
    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss(reduction="sum")
    # loss_fn = utils.CustomLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    reducePScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1000, cooldown=1000)
    lambda_func = lambda epoch: 1.0
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_func)
    start_time = time.time()
    for epoch in range(epochs):
        run_epoch(train_dataloader, epoch, model, loss_fn, optimizer, reducePScheduler, mode="train")
        run_epoch(test_dataloader, epoch, model, loss_fn, optimizer, reducePScheduler, mode="test")
    end_time = time.time()
    total_time = end_time-start_time

    print("Done!")
    print(f"time_taken: {total_time}")

    return model


num_epochs = 4000
# trained_model = train_model(num_epochs)
# torch.save(trained_model, 'model.pth')


from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.offsetbox import AnchoredText


def compare(n=80):

    for i in range(n):
        model = torch.load('model.pth')
        model.eval()
        input, output = test_dataset[i]
        pred = model(input)

        pred_unnorm = output_min + ((output_max - output_min)*pred)
        output_unnorm = output_min + ((output_max - output_min)*output)

        pred_real_part = pred_unnorm[::2]
        pred_imag_part = pred_unnorm[1::2]
        pred_fft = torch.complex(pred_real_part, pred_imag_part)

        output_real_part = output_unnorm[::2]
        output_imag_part = output_unnorm[1::2]
        output_fft = torch.complex(output_real_part, output_imag_part)

        pred_final = torch.fft.irfft(pred_fft, 100)
        output_final = torch.fft.irfft(output_fft, 100)

        x = np.linspace(0, 30, 100)

        mpl.rcParams["font.size"] = 19
        plt.plot(x, output_final.detach().numpy() - 273.15, "-", lw=1.5, label="Calculated Core Body Temperature (Computational Model)")
        plt.plot(x, pred_final.detach().numpy() - 273.15, "--", lw=2, label="Predicted Core Body Temperature (Surrogate Model)")
        plt.xlabel("Time (min)")
        plt.ylabel("Core Body Temperature ($^\circ$C)")
        plt.grid(visible=True, which='major', linestyle='--', linewidth=0.7)
        # plt.grid(visible=True, which='minor', linestyle=':', linewidth=0.5)
        plt.minorticks_on()
        plt.legend()
        plt.show()
        plt.pause(1)
        plt.close()


# compare()
x = np.linspace(0, 30, 100)
# Computation of MSE in testing dataset
def mse():
    model = torch.load('model.pth')
    model.eval()
    input, output = test_dataset.input_data, test_dataset.output_data
    start_time = time.time()
    pred = model(input)
    end_time = time.time()
    time_taken = end_time - start_time
    pred_unnorm = output_min + ((output_max - output_min) * pred)
    output_unnorm = output_min + ((output_max - output_min) * output)

    pred_real_part = pred_unnorm[:, ::2]
    pred_imag_part = pred_unnorm[:, 1::2]
    pred_fft = torch.complex(pred_real_part, pred_imag_part)

    output_real_part = output_unnorm[:, ::2]
    output_imag_part = output_unnorm[:, 1::2]
    output_fft = torch.complex(output_real_part, output_imag_part)

    pred_final = torch.fft.irfft(pred_fft, 100, axis=-1)
    output_final = torch.fft.irfft(output_fft, 100, axis=-1)
    # print(pred_final)
    # print(output_final)
    mse = torch.mean((output_final-pred_final)**2)
    mae = torch.mean(torch.abs(output_final-pred_final))
    corr = 0
    for i in range(pred_final.shape[0]):
        corr += torch.corrcoef(torch.stack((pred_final[i], output_final[i])))[0, 1]
    corr_mean = corr/pred_final.shape[0]
    # Flatten the tensors
    output_final_flat = output_final.view(-1)  # Shape: (9700,)
    pred_final_flat = pred_final.view(-1)  # Shape: (9700,)

    # Compute mean of output_final
    y_mean = torch.mean(output_final_flat)

    # Calculate SS_res and SS_tot
    ss_res = torch.sum((output_final_flat - pred_final_flat) ** 2)  # Residual sum of squares
    ss_tot = torch.sum((output_final_flat - y_mean) ** 2)  # Total sum of squares

    # Calculate R2
    r2 = 1 - (ss_res / ss_tot)

    print(f"R-squared: {r2.item()}")

    print(f"mse: {mse}")
    print(f"mae: {mae}")
    print(f"pearson r coefficient: {corr_mean}")
    print(f"time_taken: {time_taken}")
    # print(f"pearson r coefficient:{corr_matrix}")



# mse()