import logging
import os
import sys
import warnings

import torch
from models import classifier, dgcnn, pointcnn, pointnet2
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from tqdm import tqdm
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

# Path to datasets
train_dataset_path = (
    r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\train\trainingsets\fps"
)

val_dataset_path = r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\val\trainingsets\fps"

train_dataset_pickle = r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\train\plot_comp.pkl"
val_dataset_pickle = r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\val\plot_comp.pkl"

test_dataset_pickle = ""  # "" if no testing dataset

# Load pretrained model ("" if training)
pretrained = ""

# Define Hyperparameters
hp = {
    "model": "DGCNN",  # one of "PointNet2", "PointCNN", "DGCNN", "MYModel
    "lr": 1e-6,  # learning rate
    "adaptive_lr": True,  # use adaptive learning
    "early_stopping": False,  # use early stopping
    "n_gpus": torch.cuda.device_count(),
    "max_points": [7168],  # max points
    "use_columns": [],  # fileds to include from pointcloud
    "pretrained": pretrained,  # pretrained model
    "augment": True,  # augment
    "num_augs": 10,  # number of augmentations
    "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "SB"],  # classes
    "batch_size": 10,  # batch size
    "optimizer": "Adam",  # optimizer
    "start_epoch": 0,  # starting epoch
    "epochs": 300,  # number of epochs
    "patience": 20,  # patience
    "step_size": 30,  # step size
}


def main(params):
    # Check for issues with hyperparameters
    if params["adaptive_lr"] == params["early_stopping"]:
        print("Cannot use Adaptive Learning and Early Stopping")
        exit()
    else:
        for max_points in tqdm(
            params["max_points"], desc="Total: ", leave=False, colour="blue"
        ):
            model_name = params["model"]
            model_name = f"{model_name}_{max_points}"

            # Check for muliple GPU's
            multi_gpu = check_multi_gpu()

            # Create Folder Structure
            _init_(model_name)

            # Set up logger
            textio = IOStream("checkpoints/" + model_name + "/run.log")
            textio.cprint(model_name)

            # Get training, validation and test datasets
            if train_dataset_pickle:  # training dataset
                train_data_path = os.path.join(train_dataset_path, str(max_points))
                trainset = PointCloudsInPickle(
                    train_data_path,
                    train_dataset_pickle,
                    max_points=max_points,
                    use_columns=params["use_columns"],
                )

                # Augment training data
                if params["augment"] is True:
                    for i in range(params["num_augs"]):
                        aug_trainset = AugmentPointCloudsInPickle(
                            train_data_path,
                            train_dataset_pickle,
                            max_points=max_points,
                            use_columns=params["use_columns"],
                        )

                        # Concat training and augmented training datasets
                        trainset = torch.utils.data.ConcatDataset(
                            [trainset, aug_trainset]
                        )

                # Load training dataset
                if multi_gpu is True:  # loader into multi gpus
                    train_loader = DataListLoader(
                        trainset,
                        batch_size=params["batch_size"],
                        shuffle=True,
                        num_workers=0,
                    )
                else:  # loader into single gpu/cpu
                    train_loader = DataLoader(
                        trainset,
                        batch_size=params["batch_size"],
                        shuffle=True,
                        num_workers=0,
                    )

            if val_dataset_pickle:  # validation dataset
                val_data_path = os.path.join(val_dataset_path, str(max_points))
                valset = PointCloudsInPickle(
                    val_data_path,
                    val_dataset_pickle,
                    max_points=max_points,
                    use_columns=params["use_columns"],
                )

                # Load validation dataset
                if multi_gpu is True:  # loader for multi gpus
                    val_loader = DataListLoader(
                        valset,
                        batch_size=params["batch_size"],
                        shuffle=False,
                        num_workers=0,
                    )
                else:  # loader for single gpu/cpu
                    val_loader = DataLoader(
                        valset,
                        batch_size=params["batch_size"],
                        shuffle=False,
                        num_workers=0,
                    )

            if test_dataset_pickle:  # test dataset
                test_data_path = os.path.join(test_dataset_path, str(max_points))
                testset = PointCloudsInPickle(
                    test_data_path,
                    test_dataset_pickle,
                    max_points=max_points,
                    use_columns=params["use_columns"],
                )

                # Load testing dataset
                if multi_gpu is True:  # loader for multi gpus
                    test_loader = DataListLoader(
                        testset,
                        batch_size=params["batch_size"],
                        shuffle=False,
                        num_workers=0,
                    )
                else:  # loader for single gpu/cpu
                    test_loader = DataLoader(
                        testset,
                        batch_size=params["batch_size"],
                        shuffle=False,
                        num_workers=0,
                    )

            # Define device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Define Model
            # PointNet++ model
            if params["model"] == "PointNet2":
                model = pointnet2.Net(num_features=len(params["use_columns"]))

            # PointCNN model
            if params["model"] == "PointCNN":
                model = pointcnn.Net(num_features=len(params["use_columns"]))

            # DGCNN model
            if params["model"] == "DGCNN":
                model = dgcnn.Net(num_features=len(params["use_columns"]))
            
            # My Model
            if params["model"] == "MYModel":
                model = my_model.Net(num_features=len(params["use_columns"]))

            # Classify
            model = classifier.Classifier(
                model=model, num_classes=len(params["classes"])
            )

            # Load existing model
            if pretrained:
                assert os.path.isfile(pretrained)
                model.load_state_dict(torch.load(pretrained, map_location="cpu"))

            # Send model to defined device
            model.to(device)

            # Set DataParallel if Multiple GPU's available
            if multi_gpu is True:
                model = DataParallel(
                    model.cuda(), device_ids=list(range(0, params["n_gpus"]))
                )

            if pretrained:
                finished = test(
                    device=device, model=model, test_loader=test_loader, texio=textio
                )
                return finished

            else:
                train(
                    device=device,
                    model=model,
                    model_name=model_name,
                    train_loader=train_loader,
                    test_loader=val_loader,
                    textio=textio,
                    classes=params["classes"],
                    adaptive_lr=params["adaptive_lr"],
                    lr=params["lr"],
                    early_stopping=params["early_stopping"],
                    optimizer=params["optimizer"],
                    start_epoch=params["start_epoch"],
                    epochs=params["epochs"],
                    patience=params["patience"],
                    step_size=params["step_size"],
                )

            # Empty Cache
            torch.cuda.empty_cache()


# Runtime
if __name__ == "__main__":
    main(params=hp)
    plot_stats(
        root_dir=r"D:\MurrayBrent\scripts\point-dl\checkpoints",
        point_densities=hp["max_points"],
        model=hp["model"],
        stats=["r2", "rmse"],
        save_csv=True,
        csv_name="stats.csv",
        save_fig=True,
    )
    notifi("Model is Complete", " ")