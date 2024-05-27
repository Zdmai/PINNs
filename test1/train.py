import math
import os

import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from visdom import Visdom

from .net import EPNet
from .utils import build_loader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


dir = "./datas/"
electron = "Electron.hdf5"
photon = "Photon.hdf5"

electron_url = "https://cernbox.cern.ch/remote.php/dav/public-files/FbXw3V4XNyYB3oA/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5"
photon_url = "https://cernbox.cern.ch/remote.php/dav/public-files/AtBT8y4MiQYFcgc/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5"


# configration of the model
config = {
    "seed": 5201314,  # Your seed number, you can pick your lucky number. :)
    "select_all": True,  # Whether to use all features.
    "valid_ratio": 0.2,  # validation_size = train_size * valid_ratio
    "n_epochs": 300,  # Number of epochs.
    "n_classes": 3,
    "train_root": "./dataset/train/",
    "valild_root": "./dataset/val/",
    "base_channels": 3,
    "input_channels": 1,
    "input_shape": (1, 1, 150, 150),
    "depth": 4,
    "block_type": "basic",
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "early_stop": 400,  # If model has not improved for this many consecutive epochs, stop training.
    "save_path": "./models/model.ckpt",  # Your model will be saved here.
}


# test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

train_loader, valid_loader = build_loader(config)

#
model = EPNet(config)


# Train
# model = test()
def train(model, train_loader, valid_loader, config, device):
    # ll = nn.MSELoss()
    ll = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.8)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

    # writer = SummaryWriter()  # Writer of tensorboard.
    vis = Visdom(server="http://localhost", port=6006, env="main")

    if not os.path.isdir("./models"):
        os.mkdir("./models")

    n_epochs, best_loss, step, early_stop_count = config["n_epochs"], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        model.to(device)
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = ll(pred, y)
            # print("pred argmax:", torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)))

            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
            train_pbar.set_postfix({"loss": loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)

        # writer.add_scalar("Loss/train", mean_train_loss, step)
        vis.line(mean_train_loss, step, win="train_loss", update="append")
        lr_scheduler.step()

        #####################

        # Valid

        #####################
        model.eval()  # Set your model to evaluation mode.

        right = []
        loss_record = []
        for x, y in valid_loader:

            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                # print(f'pred.shape = {pred.shape} y.shape = {y.shape}')
                loss = ll(pred, y)
            right.append(torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)))

            train_pbar.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
            train_pbar.set_postfix({"acc": right[-1] / config["batch_size"]})

            loss_record.append(loss.item())

        mean_valid_acc = sum(right) / (len(right) * config["batch_size"])
        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(
            f"Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}, Accary: {mean_valid_acc:4f}%"
        )

        # writer.add_scalar("Acc/valid", mean_valid_acc, step)
        vis.line(mean_valid_acc, step, win="valid_acc", update="append")
        # writer.add_scalar("Loss/valid", mean_valid_loss, step)
        vis.line(mean_valid_loss, step, win="valid_loss", update="append")

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config["save_path"])  # Save your best model
            print("Saving model with loss {:.3f}...".format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config["early_stop"]:
            print("\nModel is not improving, so we halt the training session.")
            return


if __name__ == "__main__":
    train(model, train_loader, valid_loader, config, device)
    print("\x1b[31mrun tendorboard.....\033[m")
    os.system("tensorboard --logdir=./runs/")
