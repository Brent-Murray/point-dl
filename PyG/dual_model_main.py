import logging
import os
import sys
import warnings 

import torch
from models import pointnet2, pointcnn, dual_model, classifier
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
    check_multi_gpu,
    notifi,
)
from utils.train import train, test

warnings.filterwarnings('ignore')

# Path to datasets
train_dataset_path = (
    r"D:\MurrayBrent\data\RMF_ITD\PLOT_LAS\BUF_5M_SC\train\trainingsets\fps"
)

val_dataset_path = (
    r"D:\MurrayBrent\data\RMF_ITD\PLOT_LAS\BUF_5M_SC\val\trainingsets\fps"
)

test_dataset_path = ""  # "" if no testing dataset

# Load pretrained model ("" if trianing)
pretrained = ""

# Batch Size
batch_size = 8

# Number of augmentations
num_augs = 10

# Max points
max_points_list = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240]

# Fields to include from pointcloud
use_columns = ["intensity"]  # [] if none

# Classes: Must be in same order as in data
classes = [
    "Jack Pine",
    "White Spruce",
    "Black Spruce",
    "American Larch",
    "Paper Birch",
    "Trembling Aspen",
]


def main(model_n, pretrained=pretrained, augment=True, num_augs=num_augs):
    # Iterate through max_points_list
    for max_points in tqdm(max_points_list, desc="Total: ", leave=False, colour="blue"):
        # Set model name
        model_name = model_n + f"_{max_points}_{len(classes)}"
        
        # Check for muliple GPU's
        multi_gpu = check_multi_gpu()

        # Set up TensorBoard summary writer
        # Change to something else
        boardio = SummaryWriter(log_dir="checkpoints/" + model_name)
        _init_(model_name)
        
        # Set up logger
        textio = IOStream("checkpoints/" + model_name + "/run.log")
        textio.cprint(model_name)

        # Get training, validation and test datasets
        if train_dataset_path:  # training dataset
            train_data_path = os.path.join(train_dataset_path, str(max_points))
            trainset = PointCloudsInFiles(
                train_data_path,
                "*.laz",
                "Class",
                max_points=max_points,
                use_columns=use_columns,
            )

            # Augment training data
            if augment is True:
                for i in range(num_augs):
                    aug_trainset = AugmentPointCloudsInFiles(
                        train_data_path,
                        "*.laz",
                        "Class",
                        max_points=max_points,
                        use_columns=use_columns,
                    )

                    # Concat training and augmented training datasets
                    trainset = torch.utils.data.ConcatDataset([trainset, aug_trainset])

            # Load training dataset
            if multi_gpu is True:  # loader into multi gpus
                train_loader = DataListLoader(
                    trainset, batch_size=batch_size, shuffle=True, num_workers=0
                )
            else:  # loader into single gpu/cpu
                train_loader = DataLoader(
                    trainset, batch_size=batch_size, shuffle=True, num_workers=0
                )

        if val_dataset_path:  # validation dataset
            val_data_path = os.path.join(val_dataset_path, str(max_points))
            valset = PointCloudsInFiles(
                val_data_path,
                "*.laz",
                "Class",
                max_points=max_points,
                use_columns=use_columns,
            )

            # Load validation dataset
            if multi_gpu is True:  # loader for multi gpus
                val_loader = DataListLoader(
                    valset, batch_size=batch_size, shuffle=False, num_workers=0
                )
            else:  # loader for single gpu/cpu
                val_loader = DataLoader(
                    valset, batch_size=batch_size, shuffle=False, num_workers=0
                )

        if test_dataset_path:  # test dataset
            test_data_path = os.path.join(test_dataset_path, str(max_points))
            testset = PointCloudsInFiles(
                test_data_path,
                "*.laz",
                "Class",
                max_points=max_points,
                use_columns=use_columns,
            )
            # Load testing dataset
            if multi_gpu is True:  # loader for multi gpus
                test_loader = DataListLoader(
                    testset, batch_size=batch_size, shuffle=False, num_workers=0
                )
            else:  # loader for single gpu/cpu
                test_loader = DataLoader(
                    testset, batch_size=batch_size, shuffle=False, num_workers=0
                )

        # Define device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define Models (can use any models with output size of 1024)
        model1 = pointnet2.Net(num_features=len(use_columns)) # PointNet++
        model2 = pointcnn.Net(num_features=len(use_columns)) # PointCNN
        
        # Combine Models and Classifiy
        model = dual_model.Concat(model1, model2) # dual model
        model = classifier.Classifier(model=model, num_classes=len(classes)) # classify

        # Load existing model
        if pretrained:
            assert os.path.isfile(pretrained)
            model.load_state_dict(torch.load(pretrained, map_loaction="cpu"))
        
        # Send model to defined device
        model.to(device)
        
        # Set DataParallel if Multiple GPUs available
        if multi_gpu is True:
            model = DataParallel(
                model.cuda(), device_ids=list(range(0, torch.cuda.device_count()))
            )
        
        # Run testing
        if pretrained:
            finished = test(
                device=device,
                model=model,
                test_loader=test_loader,
                classes=classes,
                textio=textio,
            )
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
            )
        
        # Empty Cache
        torch.cuda.empty_cache()
        
        
# Runtime
if __name__ == "__main__":
    main(model_n="Dual_PN2_PCNN")
    notifi("Model is Complete", " ")