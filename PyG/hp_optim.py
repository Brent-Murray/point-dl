import glob
import logging
import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from models import classifier, dgcnn, pointcnn, pointnet2
from optuna.trial import TrialState
from sklearn.metrics import r2_score
from torch.nn import functional as F
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from tqdm import tqdm
from torch import nn
from utils.augmentation import AugmentPointCloudsInPickle
from utils.tools import (
    IOStream,
    PointCloudsInPickle,
    _init_,
    check_multi_gpu,
    delete_files,
    notifi,
    plot_stats,
)
from utils.train_comp import test, train

warnings.filterwarnings("ignore")

def objective(trial):
    hp = {
        "model": "DGCNN",  # one of "PointNet2", "PointCNN", "DGCNN"
        "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),  # learning rate *
        "adaptive_lr": True,  # use adaptive learning
        "early_stopping": False,  # use early stopping
        "n_gpus": torch.cuda.device_count(),
        "max_points": [7168],  # max points
        "use_columns": [],  # fileds to include from pointcloud
        "pretrained": "",  # pretrained model
        "augment": True,  # augment
        "num_augs": trial.suggest_int("num_augs", low=1, high=20, step=1),  # number of augmentations *
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "SB"],  # classes
        "batch_size": trial.suggest_int("batch_size", low=4, high=10, step=2),  # batch size *
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "SGD"]),  # optimizer *
        "start_epoch": 0,  # starting epoch
        "epochs": 100,  # number of epochs
        "patience": trial.suggest_int("patience", low=2, high=10, step=2),  # patience *
        "step_size": trial.suggest_int("step_size", low=1, high=20, step=2),  # step size *
    }

    for max_points in tqdm(
        hp["max_points"], desc="Total: ", leave=False, colour="blue"
    ):

        # Get Model Name
        model_name = hp["model"]
        model_name = f"{model_name}_{max_points}_tuning"

        # Check for muliple GPU's
        multi_gpu = check_multi_gpu()

        # Create Folder Structure
        _init_(model_name)

        # Set up logger
        textio = IOStream("checkpoints/" + model_name + "/run.log")
        textio.cprint(model_name)
        for key, value in hp.items():
            textio.cprint(f"    {key}: {value}")

        # Get training, validation and test datasets
        if train_dataset_pickle:  # training dataset
            train_data_path = os.path.join(train_dataset_path, str(max_points))
            trainset = PointCloudsInPickle(
                train_data_path,
                train_dataset_pickle,
                max_points=max_points,
                use_columns=hp["use_columns"],
            )

            # Augment training data
            if hp["augment"] is True:
                for i in range(hp["num_augs"]):
                    aug_trainset = AugmentPointCloudsInPickle(
                        train_data_path,
                        train_dataset_pickle,
                        max_points=max_points,
                        use_columns=hp["use_columns"],
                    )

                    # Concat training and augmented training datasets
                    trainset = torch.utils.data.ConcatDataset([trainset, aug_trainset])

            # Load training dataset
            if multi_gpu is True:  # loader into multi gpus
                train_loader = DataListLoader(
                    trainset,
                    batch_size=hp["batch_size"],
                    shuffle=True,
                    num_workers=0,
                )
            else:  # loader into single gpu/cpu
                train_loader = DataLoader(
                    trainset,
                    batch_size=hp["batch_size"],
                    shuffle=True,
                    num_workers=0,
                )

        if val_dataset_pickle:  # validation dataset
            val_data_path = os.path.join(val_dataset_path, str(max_points))
            valset = PointCloudsInPickle(
                val_data_path,
                val_dataset_pickle,
                max_points=max_points,
                use_columns=hp["use_columns"],
            )

            # Load validation dataset
            if multi_gpu is True:  # loader for multi gpus
                val_loader = DataListLoader(
                    valset,
                    batch_size=hp["batch_size"],
                    shuffle=False,
                    num_workers=0,
                )
            else:  # loader for single gpu/cpu
                val_loader = DataLoader(
                    valset,
                    batch_size=hp["batch_size"],
                    shuffle=False,
                    num_workers=0,
                )
        # Define device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define Model
        # PointNet++ model
        if hp["model"] == "PointNet2":
            model = pointnet2.Net(num_features=len(hp["use_columns"]))

        # PointCNN model
        if hp["model"] == "PointCNN":
            model = pointcnn.Net(num_features=len(hp["use_columns"]))

        # DGCNN model
        if hp["model"] == "DGCNN":
            model = dgcnn.Net(num_features=len(hp["use_columns"]))

        # My Model
        if hp["model"] == "MYModel":
            model = my_model.Net(num_features=len(hp["use_columns"]))

        # Classify
        model = classifier.Classifier(model=model, num_classes=len(hp["classes"]))

        # Send model to defined device
        model.to(device)

        # Set DataParallel if Multiple GPU's available
        if multi_gpu is True:
            model = DataParallel(
                model.cuda(), device_ids=list(range(0, hp["n_gpus"]))
            )

        # Training
        learnable_params = filter(lambda p: p.requires_grad, model.parameters())
        if hp["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(learnable_params, lr=hp["lr"])
        if hp["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(learnable_params, lr=hp["lr"])

        if hp["adaptive_lr"] is True:
            scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=hp["patience"]
            )  # reduce on plateu
            scheduler2 = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=hp["step_size"], gamma=0.1
            )  # reduce after 30 epochs
            change = 0
        tirggertimes = 0

        # Run for every epoch
        for epoch in tqdm(
            range(hp["start_epoch"], hp["epochs"]),
            desc="Model Total",
            leave=False,
            colour="red",
        ):
            model.train()
            train_loss = 0.0
            pred = 0.0
            count = 0
            for i, data in enumerate(
                tqdm(
                    train_loader,
                    desc=f"Epoch: {str(epoch + 1)}",
                    leave=False,
                    colour="cyan",
                )
            ):
                output = model(data)
                loss_val = F.mse_loss(
                    F.softmax(output, dim=1),
                    target=torch.stack([d.y for d in data]).to(output.device),
                )
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                pred = 0.0
                count = 0

                y_true = torch.tensor([], device=device)
                y_pred = torch.tensor([], device=device)

                for i, data in enumerate(
                    tqdm(val_loader, desc="Validation", leave=False, colour="green")
                ):
                    output = model(data)
                    loss_val = F.mse_loss(
                        F.softmax(output, dim=1),
                        target=torch.stack([d.y for d in data]).to(output.device),
                    )

                    test_loss += loss_val.item()
                    count += output.size(0)
                    y = torch.stack([d.y for d in data]).to(output.device)
                    y_true = torch.cat((y_true, y), 0)
                    y_pred = torch.cat((y_pred, F.softmax(output, dim=1)), 0)
                y_true = y_true.cpu().numpy()
                y_pred = y_pred.cpu().numpy()
                r2 = r2_score(y_true.flatten(), y_pred.flatten().round(2))

                trial.report(r2, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
    return r2


if __name__ == "__main__":
    
    # Path to datasets
    train_dataset_path = (
        r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\train\trainingsets\fps"
    )

    val_dataset_path = r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\val\trainingsets\fps"

    train_dataset_pickle = r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\train\plot_comp.pkl"
    val_dataset_pickle = r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\val\plot_comp.pkl"

    test_dataset_pickle = ""  # "" if no testing dataset
    
    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logio = IOStream(r"D:\MurrayBrent\scripts\point-dl\checkpoints\DGCNN_7168_tuning.log")
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
    plt.savefig(r"D:\MurrayBrent\scripts\point-dl\checkpoints\DGCNN_7168_tuning_importance.png")