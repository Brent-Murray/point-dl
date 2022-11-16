import os
import warnings


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from augment.augmentor import Augmentor
from models.dgcnn import DGCNN
from common import loss_utils
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.tools import create_comp_csv, delete_files

warnings.filterwarnings("ignore")


# Notes
# Adapt augment to fit shape of data
# Change noise to augment pointclouds instead of random
# Fix loss functions
#    - make sure to include original and augmented point clouds not just output tensors
#    - make output dimensions suitable for dgcnn


def train(params, io, train_loader, test_loader):
    # Run model
    device = torch.device("cuda" if params["cuda"] else "cpu")
    exp_name = params["exp_name"]

    # Classifier
    if params["model"] == "dgcnn":
        classifier = DGCNN(params, len(params["classes"])).to(device).cuda()
    else:
        raise Exception("Model Not Implemented")

    # Augmentor
    augmentor = Augmentor().cuda()

    # Run in Parallel
    if params["n_gpus"] > 1:
        classifier = nn.DataParallel(
            classifier.cuda(), device_ids=list(range(0, params["n_gpus"]))
        )
        augmentor = nn.DataParallel(
            augmentor.cuda(), device_ids=list(range(0, params["n_gpus"]))
        )

    # Set up optimizers
    if params["optimizer_a"] == "sgd":
        optimizer_a = optim.SGD(
            augmentor.parameters(),
            lr=params["lr_a"],
            momentum=params["momentum"],
            weight_decay=1e-4,
        )
    elif params["optimizer_a"] == "adam":
        optimizer_a = optim.Adam(
            augmentor.parameters(), lr=params["lr_a"], betas=(0.9, 0.999), eps=1e-08
        )
    else:
        raise Exception("Optimizer Not Implemented")

    if params["optimizer_c"] == "sgd":
        optimizer_c = optim.SGD(
            classifier.parameters(),
            lr=params["lr_c"],
            momentum=params["momentum"],
            weight_decay=1e-4,
        )
    elif params["optimizer_c"] == "adam":
        optimizer_c = optim.Adam(
            classifier.parameters(), lr=params["lr_c"], betas=(0.9, 0.999), eps=1e-08
        )
    else:
        raise Exception("Optimizer Not Implemented")

    # Adaptive Learning
    if params["adaptive_lr"] is True:
        scheduler1 = ReduceLROnPlateau(optimizer_c, "min", patience=params["patience"])
        scheduler2 = StepLR(optimizer_c, step_size=params["step_size"], gamma=0.1)
        change = 0

    # Set initial best test loss
    best_test_loss = np.inf

    # Set initial triggertimes
    triggertimes = 0

    # Iterate through number of epochs
    for epoch in tqdm(
        range(params["epochs"]), desc="Model Total: ", leave=False, colour="red"
    ):
        # augmentor.train()
        # classifier.train()
        train_loss_a = 0.0
        train_loss_c = 0.0
        count = 0

        for data, label in tqdm(
            train_loader, desc="Training Total: ", leave=False, colour="cyan"
        ):
            # Get data and label
            data, label = (data.to(device), label.to(device).squeeze())

            # Permute data into correct shape
            data = data.permute(0, 2, 1)  # adapt augmentor to fit with this permutation

            # Get batch size
            batch_size = data.size()[0]

            # Augment
            noise = (0.02 * torch.randn(batch_size, 1024))
            noise = noise.to(device)
            
            augmentor.train()
            classifier.train()
            optimizer_a.zero_grad()  # zero gradients
            group = (data, noise)
            aug_pc = augmentor(group)

            # Classify
            out_true = classifier(data)  # classify truth
            out_aug = classifier(aug_pc)  # classify augmented
            
            # Augmentor Loss
            aug_loss = loss_utils.g_loss(label, out_true, out_aug, data, aug_pc)

            # Backward + Optimizer Augmentor
            aug_loss.backward(retain_graph=True)
            # aug_loss.backward()
            # optimizer_a.step()
           
            # Classifier Loss
            optimizer_c.zero_grad()  # zero gradients
            cls_loss = loss_utils.d_loss(label, out_true, out_aug)

            # Backward + Optimizer Classifier
            # cls_loss.backward(retain_graph=True)
            cls_loss.backward()
            optimizer_a.step()
            optimizer_c.step()

            # Update loss' and count
            train_loss_a += aug_loss.item()
            train_loss_c += cls_loss.item()
            count = batch_size

        # Get average loss'
        train_loss_a = float(train_loss_a) / count
        train_loss_c = float(train_loss_c) / count

        # Set up Validation
        classifier.eval()
        with torch.no_grad():
            test_loss = 0.0
            count = 0
            test_pred = []
            test_true = []

            # Validation
            for data, label in tqdm(
                test_loader, desc="Validation Total: ", leave=False, colour="green"
            ):
                # Get data and label
                data, label = (data.to(device), label.to(device).squeeze())

                # Permute data into correct shape
                data = data.permute(0, 2, 1)

                # Get batch size
                batch_size = data.size()[0]

                # Run model
                output = classifier(data)

                # Calculate loss
                loss = F.mse_loss(F.softmax(output, dim=1), target=label)

                # Update count and test_loss
                count += batch_size
                test_loss += loss.item()

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
                    pred_np = pred_np[np.newaxis, :]
                    test_pred.append(pred_np)

            # Concatenate true/pred
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)

            # Calculate R2
            r2 = r2_score(test_true.flatten(), test_pred.flatten().round(2))

            # get average test loss
            test_loss = float(test_loss) / count

        # print and save losses and r2
        io.cprint(
            f"Epoch: {epoch + 1}, Training - Augmentor Loss: {train_loss_a}, Training - Classifier Loss: {train_loss_c}, Validation Loss: {test_loss}, R2: {r2}"
        )

        # Save Best Model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(
                classifier.state_dict(), f"checkpoints/{exp_name}/models/best_mode.t7"
            )

            # delete old files
            delete_files(f"checkpoints/{exp_name}/output", "*.csv")

            # Create CSV of best model output
            create_comp_csv(
                test_true.flatten(),
                test_pred.round(2).flatten(),
                params["classes"],
                f"checkpoints/{exp_name}/output/outputs_epoch{epoch+1}.csv",
            )

        # Apply addaptive learning
        if params["adaptive_lr"] is True:
            if test_loss > best_test_loss:
                triggertimes += 1
                if triggertimes >= params["patience"]:
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
                

def test(params, io, test_loader):
    device = torch.device("cuda" if params["cuda"] else "cpu")

    # Load model
    if params["model"] == "dgcnn":
        model = DGCNN(params, len(params["classes"])).to(device)
    else:
        raise Exception("Model Not Implemented")
        
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
        test_loader, desc="Testing Total: ", leave=False, colour="green"
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