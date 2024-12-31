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
output_data = torch.load("output_3.pt")[:, :, 1]
input_data_norm = utils.min_max_normalize(input_data).to(device)
output_data = torch.fft.rfft(output_data, dim=1).to(device)
real = output_data.real
imag = output_data.imag
output_data_norm = torch.nan_to_num(utils.min_max_normalize(torch.stack([real, imag], dim=-1).view(970, -1)), nan=0)
output_min = torch.min((torch.stack([real, imag], dim=-1).view(970, -1)), dim=0).values
output_max = torch.max((torch.stack([real, imag], dim=-1).view(970, -1)), dim=0).values
print(output_data_norm)
print(output_data_norm.shape)

train_size = int((60/61)*(input_data.shape[0]))
test_size = int((1/61)*(input_data.shape[0]))
dataset_size = int(train_size + test_size)

train_dataset = CBTDataset(input_data_norm[:train_size], output_data_norm[:train_size])
test_dataset = CBTDataset(input_data_norm[train_size:], output_data_norm[train_size:])

batch_size = 128

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

wandb.init(
    # set the wandb project where this run will be logged
    project="CBT_pred_exercise",
)


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

                wandb.log({"train_epoch": epoch+1, "train_loss": loss.item(), "train_loss_node": loss_node.item(), "LR": optimizer.state_dict()['param_groups'][0]["lr"]})

                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "train_epoch": epoch + 1,
                        "train_loss": loss.item(),
                        "train_loss_node: ": loss_node.item(),
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )
            else:
                wandb.log({"test_epoch": epoch+1, "test_loss": loss, "test_loss_node": loss_node.item()})

            del loss_node


def train_model(epochs):
    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss(reduction="sum")
    # loss_fn = utils.CustomLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    reducePScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1000, cooldown=1000)
    lambda_func = lambda epoch: 1.0
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_func)
    for epoch in range(epochs):
        run_epoch(train_dataloader, epoch, model, loss_fn, optimizer, reducePScheduler, mode="train")
        run_epoch(test_dataloader, epoch, model, loss_fn, optimizer, reducePScheduler, mode="test")

    print("Done!")

    return model


num_epochs = 4000
trained_model = train_model(num_epochs)
torch.save(trained_model, 'model.pth')


from matplotlib import pyplot as plt
import numpy as np


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

        x = np.linspace(0, 1, 100)
        plt.plot(x, output_final.detach().numpy())
        plt.plot(x, pred_final.detach().numpy())
        plt.show()
        plt.pause(1)
        plt.close()


compare()

