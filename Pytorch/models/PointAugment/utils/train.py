import os
import random
import warnings


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from augment.augmentor import Augmentor
from models.dgcnn import DGCNN
from models.pointnet2 import PointNet2
from common import loss_utils
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils.tools import create_comp_csv, delete_files, variable_df, write_las, plot_3d, plot_2d
from utils.send_telegram import send_telegram

warnings.filterwarnings("ignore")


def train(params, io, trainset, testset):
    # Run model
    device = torch.device("cuda" if params["cuda"] else "cpu")
    exp_name = params["exp_name"]

    # Classifier
    if params["model"] == "dgcnn":
        classifier = DGCNN(params, len(params["classes"])).to(device).cuda()
    elif params["model"] == "pn2":
        classifier = PointNet2(len(params["classes"])).to(device).cuda()
    else:
        raise Exception("Model Not Implemented")

    # Augmentor
    if params["augmentor"]:
        augmentor = Augmentor().cuda()

    # Run in Parallel
    if params["n_gpus"] > 1:
        classifier = nn.DataParallel(
            classifier.cuda(), device_ids=list(range(0, params["n_gpus"]))
        )
        if params["augmentor"]:
            augmentor = nn.DataParallel(
                augmentor.cuda(), device_ids=list(range(0, params["n_gpus"]))
            )

    # Set up optimizers
    if params["augmentor"]:
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
        scheduler1_c = ReduceLROnPlateau(optimizer_c, "min", patience=params["patience"])
        scheduler2_c = StepLR(optimizer_c, step_size=params["step_size"], gamma=0.1)
        if params["augmentor"]:
            scheduler1_a = ReduceLROnPlateau(optimizer_a, "min", patience=params["patience"])
            scheduler2_a = StepLR(optimizer_a, step_size=params["step_size"], gamma=0.1)
        change = 0

    # Set initial best test loss
    best_test_loss = np.inf

    # Set initial triggertimes
    triggertimes = 0

    epoch_list = []
    training_losses_a = []
    training_losses_c = []
    training_r2s = []
    validation_losses = []
    validation_r2s = []
    
    weights = params["train_weights"]
    weights = torch.Tensor(np.array(weights)).to(device)
    # Iterate through number of epochs
    
    for epoch in tqdm(
        range(params["epochs"]), desc="Model Total: ", leave=False, colour="red"
    ):
        # augmentor.train()
        # classifier.train()
        send_telegram(f"Epoch: {epoch}")
        trainset_idx = list(range(len(trainset)))
        random.shuffle(trainset_idx)
        rem = len(trainset_idx) % params["batch_size"]
        if rem <= 3:
            trainset_idx = trainset_idx[: len(trainset_idx) - rem]
            trainset = Subset(trainset, trainset_idx)
        train_loader = DataLoader(trainset, batch_size=params["batch_size"], shuffle=True, pin_memory=True)
        test_loader = DataLoader(testset, batch_size=params["batch_size"], shuffle=False, pin_memory=True)
        train_loss_a = 0.0
        train_loss_c = 0.0
        count = 0
        true_pred = []
        aug_pred = []
        train_true = []
        j=0
        
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
            if params["augmentor"]:
                noise = (0.02 * torch.randn(batch_size, 1024))
                noise = noise.to(device)
                augmentor.train()
                
            classifier.train()
            if params["augmentor"]:
                optimizer_a.zero_grad()  # zero gradients
                group = (data, noise)
                aug_pc = augmentor(group)

            # Classify
            out_true = classifier(data)  # classify truth
            if params["augmentor"]:
                out_aug = classifier(aug_pc)  # classify augmented
            
                # Augmentor Loss
                aug_loss = loss_utils.g_loss(label, out_true, out_aug, data, aug_pc, weights)

                # Backward + Optimizer Augmentor
                aug_loss.backward(retain_graph=True)
            # aug_loss.backward()
            # optimizer_a.step()
           
            # Classifier Loss
            optimizer_c.zero_grad()  # zero gradients
            if params["augmentor"]:
                cls_loss = loss_utils.d_loss(label, out_true, out_aug, weights)
            else:
                cls_loss = loss_utils.calc_loss(label, out_true, weights)

            # Backward + Optimizer Classifier
            # cls_loss.backward(retain_graph=True)
            cls_loss.backward()
            if params["augmentor"]:
                optimizer_a.step()
            optimizer_c.step()

            # Update loss' and count
            if params["augmentor"]:
                train_loss_a += aug_loss.item()
            train_loss_c += cls_loss.item()
            count = batch_size
            
            # Append true/pred
            label_np = label.cpu().numpy()
            if label_np.ndim == 2:
                train_true.append(label_np)
            else:
                label_np = label_np[np.newaxis, :]
                train_true.append(label_np)
            
            if params["augmentor"]:
                aug_np = F.softmax(out_aug, dim=1)
                aug_np = aug_np.detach().cpu().numpy()
                if aug_np.ndim == 2:
                    aug_pred.append(aug_np)
                else:
                    aug_np = aug_np[np.newaxis, :]
                    aug_pred.append(aug_np)
            
            true_np = F.softmax(out_true, dim=1)
            true_np = true_np.detach().cpu().numpy()
            if true_np.ndim ==2:
                true_pred.append(true_np)
            else:
                true_np = true_np[np.newaxis, :]
                true_pred.append(true_np)
            if params["augmentor"]:
                if epoch + 1 in [1, 50, 100, 150, 200, 250, 300]:
                    if random.random() > 0.99:
                        aug_pc_np = aug_pc.detach().cpu().numpy()
                        true_pc_np = data.detach().cpu().numpy()
                        try:
                            write_las(aug_pc_np[1], f"checkpoints/{exp_name}/output/laz/epoch{epoch + 1}_pc{j}_aug.laz")
                            write_las(true_pc_np[1], f"checkpoints/{exp_name}/output/laz/epoch{epoch + 1}_pc{j}_true.laz")
                            j+=1
                        except:
                            j+=1

        # Concatenate true/pred
        train_true = np.concatenate(train_true)
        if params["augmentor"]:
            aug_pred = np.concatenate(aug_pred)
        true_pred = np.concatenate(true_pred)
        
        # Calculate R2's
        if params["augmentor"]:
            aug_r2 = r2_score(train_true.flatten(), aug_pred.flatten().round(2))
        true_r2 = r2_score(train_true.flatten(), true_pred.flatten().round(2))
        
        if params["augmentor"]:
            train_r2 = float(aug_r2 + true_r2) / 2
        else:
            train_r2 = true_r2
        
        # Get average loss'
        if params["augmentor"]:
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
            val_r2 = r2_score(test_true.flatten(), test_pred.flatten().round(2))

            # get average test loss
            test_loss = float(test_loss) / count

        # print and save losses and r2
        if params["augmentor"]:
            io.cprint(
                f"Epoch: {epoch + 1}, Training - Augmentor Loss: {train_loss_a}, Training - Classifier Loss: {train_loss_c}, Training R2: {train_r2}, Validation Loss: {test_loss}, R2: {val_r2}"
            )

            # Create output Dataframe
            out_dict = {"epoch": [epoch + 1],
                        "aug_loss": [train_loss_a],
                        "class_loss": [train_loss_c],
                        "train_r2": [train_r2],
                        "val_loss": [test_loss],
                        "val_r2": [val_r2]}
        else:
            io.cprint(
                f"Epoch: {epoch + 1}, Training - Classifier Loss: {train_loss_c}, Training R2: {train_r2}, Validation Loss: {test_loss}, R2: {val_r2}"
            )

            # Create output Dataframe
            out_dict = {"epoch": [epoch + 1],
                        "class_loss": [train_loss_c],
                        "train_r2": [train_r2],
                        "val_loss": [test_loss],
                        "val_r2": [val_r2]}
        out_df = pd.DataFrame.from_dict(out_dict)
        
        if epoch + 1 > 1:
            loss_r2_df = pd.read_csv(f"checkpoints/{exp_name}/loss_r2.csv")
            loss_r2_df = pd.concat([loss_r2_df, out_df])
            loss_r2_df.to_csv(f"checkpoints/{exp_name}/loss_r2.csv", index=False)
        else:
            out_df.to_csv(f"checkpoints/{exp_name}/loss_r2.csv", index=False)

        # Save Best Model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            if params["augmentor"]:
                send_telegram(f"Training - Augmentor Loss: {train_loss_a}, Training - Classifier Loss: {train_loss_c}, Training R2: {train_r2}, Validation Loss: {test_loss}, R2: {val_r2}")
            else:
                send_telegram(f"Training - Classifier Loss: {train_loss_c}, Training R2: {train_r2}, Validation Loss: {test_loss}, R2: {val_r2}")
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
                if triggertimes > params["patience"]:
                    change = 1
            else:
                triggertimes = 0
            if change == 0:
                if params["augmentor"]:
                    scheduler1_a.step(test_loss)
                scheduler1_c.step(test_loss)
                if params["augmentor"]:
                    io.cprint(
                        f"Augmentor LR: {scheduler1_a.optimizer.param_groups[0]['lr']}, Trigger Times: {triggertimes}, Scheduler: Plateau"
                    )
                io.cprint(
                    f"Classifier LR: {scheduler1_c.optimizer.param_groups[0]['lr']}, Trigger Times: {triggertimes}, Scheduler: Plateau"
                )
            else:
                if params["augmentor"]:
                    scheduler2_a.step()
                scheduler2_c.step()
                if params["augmentor"]:
                    io.cprint(
                        f"Augmentor LR: {scheduler2_a.optimizer.param_groups[0]['lr']}, Scheduler: Step"
                    )
                io.cprint(
                    f"Classifier LR: {scheduler2_c.optimizer.param_groups[0]['lr']}, Scheduler: Step"
                )                             

def test(params, io, testset):
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
    test_loader = DataLoader(testset, batch_size=params["batch_size"], shuffle=False, pin_memory=True)
                                 
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