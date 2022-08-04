import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from models.pointnet2 import Net
from sklearn.metrics import classification_report, confusion_matrix
from tensorboardX import SummaryWriter
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataLoader, DataListLoader
from tqdm import tqdm
from utils.augmentation import AugmentPointCloudsInFiles
from utils.tools import (
    IOStream,
    PointCloudsInFiles,
    _init_,
    delete_files,
    make_confusion_matrix,
)
from utils.train import train, test

# Path to datasets
train_dataset_path = r"D:\MurrayBrent\data\RMF_ITD\PLOT_LAS\BUF_5M_SC\train"
val_dataset_path = r"D:\MurrayBrent\data\RMF_ITD\PLOT_LAS\BUF_5M_SC\val"
test_dataset_path = ""
# test_dataset_path = r"D:\MurrayBrent\data\RMF_ITD\PLOT_LAS\BUF_5M_SC\test"

# Load pretrained model ("" if training)
# pretrained = r"D:\MurrayBrent\git\point-dl\notebooks\checkpoints\PointCNN_2048_6\models\best_model.t7"
pretrained = ""

# Batch Size
batch_size=32

# Number of augmentations
num_augs = 10

# max_points
# 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240,
# 11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432, 19456, 20480
max_points_list = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240]

# Fields to include from pointcloud
use_columns = ["intensity"]
# use_columns = []

# Classes: must be in same order as in data
# classes = ["Con", "Dec"]
classes = [
    "Jack Pine",
    "White Spruce",
    "Black Spruce",
    # "Balsam Fir",
    # "Eastern White Cedar",
    "American Larch",
    "Paper Birch",
    "Trembling Aspen",
]


def main(pretrained="", augment=True, num_augs=num_augs):
    for max_points in tqdm(max_points_list, desc="Total: ", leave=False, colour="blue"):
        # Model Name
        model_name = f"PointNet2_{max_points}_{len(classes)}"
    
        # Set up TensorBoard summary writer
        boardio = SummaryWriter(log_dir="checkpoints/" + model_name)
        _init_(model_name)

        # Set up logger
        textio = IOStream("checkpoints/" + model_name + "/run.log")
        textio.cprint(model_name)

        # Check for multi GPU's
        if torch.cuda.device_count() > 1:
            multi_gpu = True
        else:
            multi_gpu = False


        # Get training, validation and test datasets
        if train_dataset_path:
            trainset = PointCloudsInFiles(
                train_dataset_path,
                "*.laz",
                "Class",
                max_points=max_points,
                use_columns=use_columns,
            )

            # Augment training data
            if augment is True:
                for i in range(num_augs):
                    aug_trainset = AugmentPointCloudsInFiles(
                        train_dataset_path,
                        "*.laz",
                        "Class",
                        max_points=max_points,
                        use_columns=use_columns,
                    )

                    # Concat training and augmented training datasets
                    trainset = torch.utils.data.ConcatDataset([trainset, aug_trainset])
            # Load training dataset
            if multi_gpu is True:
                train_loader = DataListLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
            else:
                train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)


        if val_dataset_path:
            valset = PointCloudsInFiles(
                val_dataset_path,
                "*.laz",
                "Class",
                max_points=max_points,
                use_columns=use_columns,
            )
            # Load validation dataset
            if multi_gpu is True:
                val_loader = DataListLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
            else:
                val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)


        if test_dataset_path:
            testset = PointCloudsInFiles(
                test_dataset_path,
                "*.laz",
                "Class",
                max_points=max_points,
                use_columns=use_columns,
            )
            # Load testing dataset
            if multi_gpu is True:
                test_loader = DataListLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
            else:
                test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


        # Define device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define model
        model = Net(num_classes=len(classes), num_features=len(use_columns))

        # Checkpoint
        checkpoint = None

        # Load existing model
        if pretrained:
            assert os.path.isfile(pretrained)
            model.load_state_dict(torch.load(pretrained, map_location="cpu"))

        # Send model to defined device
        model.to(device)
        
        # Set DataParallel if Multiple GPUs available
        if multi_gpu is True:
            print("Using Multiple GPUs")
            model = DataParallel(model.cuda(), device_ids=list(range(0,torch.cuda.device_count())))

        # Run testing
        if pretrained:
            finished = test(
                device=device, model=model, test_loader=test_loader, classes=classes, textio=textio,)
            return finished
        # Run training
        else:
            train(
                device=device,
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                test_loader=val_loader,
                classes=classes,
                boardio=boardio,
                textio=textio,
                checkpoint=checkpoint,
            )
        
        
# Runtime
if __name__ == "__main__":
    main()