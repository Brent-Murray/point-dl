# Adapted from https://github.com/WangYueFt/dgcnn/tree/master/pytorch

# Notes
# line 68/98 in trian see if loss is correct

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.dgcnn import DGCNN
from models.dgcnn_extended import DGCNNEx
from utils.tools import create_comp_csv, delete_files


def train(args, io, train_loader, test_loader, trial=None):
    # Set Device
    device = torch.device("cuda" if args["cuda"] else "cpu")
    exp_name = args["exp_name"]

    # Load Model
    if args["model"] == "dgcnn":
        model = DGCNN(args, len(args["classes"])).to(device)
    elif args["model"] == "dgcnn_extended":
        model = DGCNNEx(args, len(args["classes"])).to(device)
    else:
        raise Exception("Model Not Implemented")
        
    # Data Parallel
    model = nn.DataParallel(model, device_ids=list(range(0, args["n_gpus"])))

    # Set Up optimizer
    if args["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args["lr"],
            momentum=args["momentum"],
            weight_decay=1e-4,
        )
    elif args["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=1e-4)
    else:
        raise Exception("Optimizer Not Implemented")

    # Set Up Adaptive Learning
    if args["adaptive_lr"] is True:
        scheduler1 = ReduceLROnPlateau(optimizer, "min", patience=args["patience"])
        scheduler2 = StepLR(optimizer, step_size=args["step_size"], gamma=0.1)
        change = 0

    # Set initial best test loss
    best_test_loss = np.inf

    # Set initial triggertimes
    triggertimes = 0

    for epoch in tqdm(
        range(args["epochs"]), desc="Model Total: ", leave=False, colour="red"
    ):
        # Set up trianing
        model.train()
        train_loss = 0.0
        count = 0

        # Training
        for data, label in tqdm(
            train_loader, desc="Training Total: ", leave=False, colour="cyan"
        ):
            # Get data, labels & batch size
            data, label = (
                data.to(device),
                label.to(device).squeeze(),
            )
            data = data.permute(0, 2, 1)  # Check that this is the right order
            batch_size = data.size()[0]

            # Zero gradients
            optimizer.zero_grad()

            # Run Model
            output = model(data)

            # Calculate loss
            loss = F.mse_loss(
                F.softmax(output, dim=1), target=label
            )  # get the right loss fuction

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Update count and train_loss
            count += batch_size
            train_loss += loss.item() * batch_size  # see if this is correct

        # Get average train loss
        train_loss = float(train_loss) / count

        # # print and save epoch + training loss
        # io.cprint(f"Epoch: {epoch + 1}, Training Loss: {train_loss}")

        # Set up Validation
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            count = 0
            test_pred = []
            test_true = []

            # Validation
            for data, label in tqdm(
                test_loader, desc="Validation Total: ", leave=False, colour="green"
            ):
                # Get data, labels, & batch size
                data, label = (
                    data.to(device),
                    label.to(device).squeeze(),
                )
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]

                # Run Model
                output = model(data)

                # Calculate loss
                loss = F.mse_loss(
                    F.softmax(output, dim=1), target=label
                )  # get the right loss function

                # Update count and test_loss
                count += batch_size
                test_loss += loss.item() * batch_size  # see if this is correct

                # Append true/pred
                label_np = label.cpu().numpy()
                if label_np.ndim == 2:
                    test_true.append(label_np)
                else:
                    label_np = label_np[np.newaxis, :]
                    test_true.append(label_np)
                pred_np = F.softmax(output, dim=1)
                pred_np = pred_np.detach().cpu().numpy()
                if pred_np.ndim == 2:
                    test_pred.append(pred_np)
                else:
                    pred_np = pred_np[np.newaxis,:]
                    test_pred.append(pred_np)

            # Concatenate true/pred
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)

            # Calculate R2
            r2 = r2_score(test_true.flatten(), test_pred.flatten().round(2))

            # Get average test loss
            test_loss = float(test_loss) / count

        # print and save epoch + train loss + test loss
        io.cprint(
            f"Epoch: {epoch + 1}, Training Loss: {train_loss}, Test Loss: {test_loss}, R2: {r2}"
        )

        # Save Best Model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(
                model.state_dict(), f"checkpoints/{exp_name}/models/best_model.t7"
            )

            # delete old file
            delete_files(f"checkpoints/{exp_name}/output", "*.csv")

            # Create CSV of output
            create_comp_csv(
                test_true.flatten(),
                test_pred.round(2).flatten(),
                args["classes"],
                f"checkpoints/{exp_name}/output/outputs_epoch{epoch+1}.csv",
            )

        if args["adaptive_lr"] is True:
            if test_loss > best_test_loss:
                triggertimes += 1
                if triggertimes >= args["patience"]:
                    change = 1
            else:
                triggertimes = 0
            if change == 0:
                scheduler1.step(test_loss)
                io.cprint(
                    f"LR: {scheduler1.optimizer.param_groups[0]['lr']}, Trigger Times: {triggertimes}, Scheduler: Plateau"
                )
            else:
                scheduler2.step()
                io.cprint(
                    f"LR: {scheduler2.optimizer.param_groups[0]['lr']}, Scheduler: Step"
                )
                
                
def test(args, io, test_loader):
    device = torch.device("cuda" if args["cuda"] else "cpu")

    # Load Model
    if args["model"] == "dgcnn":
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not Implemented")

    # Data Parallel
    model = nn.DataParallel(model, device_ids=list(range(0, args["n_gpus"])))

    # Load Pretrained Model
    model.load_state_dict(torch.load(args["model_path"]))

    # Setup for Testing
    model = model.eval()
    test_true = []
    test_pred = []

    # Testing
    for data, label in tqdm(
        test_loader, desc="Validation Total: ", leave=False, colour="green"
    ):
        # Get data, labels, & batch size
        data, label = (
            data.to(device),
            label.to(device).squeeze(),
        )
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]

        # Run Model
        output = model(data)

        # Append true/pred
        test_true.append(label.cpu().numpy())
        test_pred.append(pred.detach().cpu().numpy())

    # Concatenate true/pred
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    # Calculate R2
    r2 = r2_score(test_true, test_pred.round(2))

    io.cprint(f"R2: {r2}")