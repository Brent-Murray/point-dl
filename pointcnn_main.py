import logging
import os
import sys

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.tools import IOStream, PointCloudsInFiles, _init_
from models.pointcnn import PointCNN


train_dataset_path = r"D:\MurrayBrent\git\point-dl\input\train"
test_dataset_path = r"D:\MurrayBrent\git\point-dl\input\test"
model_name = "PointCNN_10000"
use_columns = ["intensity"]

def test_one_epoch(device, model, test_loader):
    model.eval()  # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    test_loss = 0.0
    pred = 0.0
    count = 0

    for i, data in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
        data.to(device)

        # Call model
        output = model(data)

        # Define validation loss using negative log likelihood loss and softmax
        loss_val = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(output, dim=1),
            target=data.y,
        )

        # Update test_lost and count
        test_loss += loss_val.item()
        count += output.size(0)

        # Update pred
        _, pred1 = output.max(dim=1)
        ag = pred1 == data.y
        am = ag.sum()
        pred += am.item()

    # Calculate test_loss and accuracy
    test_loss = float(test_loss) / count
    accuracy = float(pred) / count

    return test_loss, accuracy


def test(device, model, test_loader, textio):
    test_loss, test_accuracy = test_one_epoch(device, model, test_loader)
    textio.cprint(
        "Validation Loss: %f & Validation Accuracy: %f" % (test_loss, test_accuracy)
    )


def train_one_epoch(device, model, train_loader, optimizer, epoch_number):
    model.train()
    train_loss = 0.0
    pred = 0.0
    count = 0

    for i, data in enumerate(
        tqdm(train_loader, desc="Epoch: " + str(epoch_number), leave=False)
    ):
        # Send data to device
        data.to(device)

        # Call model
        output = model(data)

        # Define validation loss using negative log likelihood loss and softmax
        loss_val = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(output, dim=1),
            target=data.y,
        )

        # Forward + backward + optimize
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Update train_loss and count
        train_loss += loss_val.item()
        count += output.size(0)

        # Update pred
        _, pred1 = output.max(dim=1)
        ag = pred1 == data.y
        am = ag.sum()
        pred += am.item()

    # Calculate train_loss and accuracy
    train_loss = float(train_loss) / count
    accuracy = float(pred) / count

    return train_loss, accuracy


def train(
    device,
    model,
    train_loader,
    test_loader,
    boardio,
    textio,
    checkpoint,
    model_name,
    optimizer="Adam",
    start_epoch=0,
    epochs=200,
):
    # Set up optimizer
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer == "Adam":  # Adam optimizer
        optimizer = torch.optim.Adam(learnable_params)
    else:  # SGD optimizer
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    # Set up checkpoint
    if checkpoint is not None:
        min_loss = checkpoint["min_loss"]
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Define best_test_loss
    best_test_loss = np.inf

    for epoch in tqdm(range(start_epoch, epochs), desc= "Total", leave=False):
        # Train Model
        train_loss, train_accuracy = train_one_epoch(
            device, model, train_loader, optimizer, epoch + 1
        )

        # Test Model
        test_loss, test_accuracy = test_one_epoch(device, model, test_loader)

        # Save Best Model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            snap = {
                # state_dict: https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "min_loss": best_test_loss,
                "optimizer": optimizer.state_dict,
            }
            torch.save(snap, f"checkpoints/{model_name}/models/best_model_snap.t7")
            torch.save(
                model.state_dict, f"checkpoints/{model_name}/models/best_model.t7"
            )

        # Save model
        torch.save(snap, f"checkpoints/{model_name}/models/model_snap.t7")
        torch.save(model.state_dict, f"checkpoints/{model_name}/models/model.t7")

        boardio.add_scalar("Train Loss", train_loss, epoch + 1)
        boardio.add_scalar("Test Loss", test_loss, epoch + 1)
        boardio.add_scalar("Best Test Loss", best_test_loss, epoch + 1)
        boardio.add_scalar("Train Accuracy", train_accuracy, epoch + 1)
        boardio.add_scalar("Test Accuracy", test_accuracy, epoch + 1)

        textio.cprint(
            "EPOCH:: %d, Training Loss: %f, Testing Loss: %f, Best Loss: %f"
            % (epoch + 1, train_loss, test_loss, best_test_loss)
        )
        textio.cprint(
            "EPOCH:: %d, Training Accuracy: %f Testing Accuracy: %f"
            % (epoch + 1, train_accuracy, test_accuracy)
        )
        
def main():

    boardio = SummaryWriter(log_dir="checkpoints/" + model_name)
    _init_(model_name)

    textio = IOStream("checkpoints/" + model_name + "/run.log")
    textio.cprint(model_name)

    # Get training and test datasets
    trainset = PointCloudsInFiles(
        train_dataset_path, "*.laz", "Class", max_points=10_000, use_columns=use_columns
    )
    testset = PointCloudsInFiles(
        test_dataset_path, "*.laz", "Class", max_points=10_000, use_columns=use_columns
    )

    # Load training and test datasets
    train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    
    model = PointCNN(numfeatures=len(use_columns))

    checkpoint = None

    model.to(device)

    train(
        device=device,
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        test_loader=test_loader,
        boardio=boardio,
        textio=textio,
        checkpoint=checkpoint,
    )
    
if __name__ == "__main__":
    main()