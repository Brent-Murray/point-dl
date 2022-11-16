from __future__ import print_function

import os
import sys
import warnings

import torch
import optuna
import numpy as np
import pandas as pd
from optuna.trial import TrialState
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
from utils.train import train, test
from utils.tools import IOStream, PointCloudsInPickle, _init_, model_args
from utils.augmentation import AugmentPointCloudsInPickle

warnings.filterwarnings("ignore")

def objective(trial):
    params = {
        "exp_name": "dgcnn_extended_7168",  # experiment name
        "model": "dgcnn_extended",  # model
        "batch_size": 8,  # batch size
        "train_path": r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\train\trainingsets\fps",
        "train_pickle": r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\train\plot_comp.pkl",
        "test_path": r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\val\trainingsets\fps",
        "test_pickle": r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\val\plot_comp.pkl",
        "augment": True, # augment
        "n_augs": 1, # number of augmentations
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "SB"],  # classes
        "n_gpus": torch.cuda.device_count(),
        "epochs": 300,  # total epochs
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),  # optimizer
        "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),  # learning rate
        "adaptive_lr": True,  # adaptive learning rate
        "patience": trial.suggest_int("patience", low=2, high=20, step=2),  # patience
        "step_size": trial.suggest_int("step_size", low=1, high=20, step=2),  # step size
        "momentum": trial.suggest_float("momentum", 0.5, 0.9, log=True),  # sgd momentum
        "num_points": 7168,  # number of points
        "dropout": 0.5,  # dropout rate
        "emb_dims": 1024,  # dimension of embeddings
        "k": trial.suggest_int("k", low=5, high=20, step=1),  # k nearest points
        "model_path": "",  # pretrained model path
        "cuda": True,  # use cuda
        "eval": False,  # run testing
    }
    
    _init_(params["exp_name"])

    # initiate IOStream
    io = IOStream("checkpoints/" + params["exp_name"] + "/run.log")
    io.cprint(params["exp_name"])
    io.cprint(str(params))

    if params["cuda"]:
        io.cprint("Using GPU")
    else:
        io.cprint("Using CPU")

    # Load datasets
    train_data_path = os.path.join(params["train_path"], str(params["num_points"]))
    train_pickle = params["train_pickle"]
    trainset = PointCloudsInPickle(train_data_path, train_pickle)
    
    if params["augment"] == True:
        for i in range(params["n_augs"]):
            aug_trainset = AugmentPointCloudsInPickle(
                train_data_path,
                train_pickle,
            )
            
            trainset = torch.utils.data.ConcatDataset(
                [trainset, aug_trainset]
            )
        
    train_loader = DataLoader(trainset, batch_size=params["batch_size"], shuffle=True)

    test_data_path = os.path.join(params["test_path"], str(params["num_points"]))
    test_pickle = params["test_pickle"]
    testset = PointCloudsInPickle(test_data_path, test_pickle)
    test_loader = DataLoader(testset, batch_size=params["batch_size"], shuffle=False)

    # Set Device
    device = torch.device("cuda" if params["cuda"] else "cpu")
    exp_name = params["exp_name"]

    # Load Model
    if params["model"] == "dgcnn":
        model = DGCNN(params, len(params["classes"])).to(device)
    if params["model"] == "dgcnn_extended":
        model = DGCNNEx(params, len(params["classes"])).to(device)
    else:
        raise Exception("Not Implemented")
    
    model = nn.DataParallel(model, device_ids=list(range(0, params["n_gpus"])))
    
    # Set Up optimizer
    if params["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=params["lr"],
            momentum=params["momentum"],
            weight_decay=1e-4,
        )
    elif params["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    else:
        raise Exception("Not Implemented")
        
    # Set Up Adaptive Learning
    if params["adaptive_lr"] is True:
        scheduler1 = ReduceLROnPlateau(optimizer, "min", patience=params["patience"])
        scheduler2 = StepLR(optimizer, step_size=params["step_size"], gamma=0.1)
        change = 0
        
    # Set initial best test loss
    best_test_loss = np.inf

    # Set initial triggertimes
    triggertimes = 0

    for epoch in tqdm(
        range(params["epochs"]), desc="Model Total: ", leave=False, colour="red"
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
            train_loss = loss.item() * batch_size  # see if this is correct

        # Get average train loss
        train_loss = float(train_loss) / count

        # # print and save epoch + training loss
        # io.cprint(f"Epoch: {epoch + 1}, Training Loss: {train_loss}")

        # Set up Validation
        model.eval()
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
                label_np = label_np[np.newaxis,:]
                test_true.append(label_np)
            pred_np = F.softmax(output, dim=1)
            pred_np = pred_np.detach().cpu().numpy()
            if pred_np.ndim == 2:
                test_pred.append(pred_np)
            else:
                test_np = test_np[np.newaxis,:]
                tet_pred.append(pred_np)

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
            best_loss_r2 = r2

            # delete old file
            delete_files(f"checkpoints/{exp_name}/output", "*.csv")

            # Create CSV of output
            create_comp_csv(
                test_true.flatten(),
                test_pred.round(2).flatten(),
                params["classes"],
                f"checkpoints/{exp_name}/output/outputs_epoch{epoch+1}.csv",
            )

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
        trial.report(best_loss_r2, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return best_loss_r2

if __name__ == "__main__":
    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logio = IOStream(r"D:\MurrayBrent\scripts\point-dl\Pytorch\DGCNN\checkpoints\DGCNN_7168_tuning.log")
    
    logio.cprint("Study Statistics: ")
    logio.cprint(f"   Number of finished tirals: {len(study.trials)}")
    logio.cprint(f"   Number of pruned trials: {len(pruned_trials)}")
    logio.cprint(f"   Number of complete trials: {len(complete_trials)}")
    logio.cprint("Best Trial:")
    trial = study.best_trial
    logio.cprint(f"  Value: {trial.value}")
    logio.cprint("  Params: ")
    for key, value in trial.params.items():
        logio.cprint(f"    {key}: {value}")

    # Get Parameter Importance
    param_importance = optuna.importance.get_param_importance(study)
    names = list(param_importance.keys())
    values = list(param_importance.values())
    pi_df = pd.DataFrame({"Importance": values}, index=names)

    # Plot
    plt.rcParams["figure.figsize"] = 15, 8
    plt.rcParams["figure.autolayout"] = True  # auto layout
    plt.rcParams["figure.facecolor"] = "white"  # facecolor
    sns.heatmap(pi_df, annot=True, cmap="Blues", cbar=False)
    plt.savefig(r"D:\MurrayBrent\scripts\point-dl\Pytorch\DGCNN\checkpoints\DGCNN_7168_tuning_importance.png")
    