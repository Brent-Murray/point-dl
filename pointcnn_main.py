import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from models.pointcnn import PointCNN
from sklearn.metrics import classification_report, confusion_matrix
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.augmentation import AugmentPointCloudsInFiles
from utils.tools import (
    IOStream,
    PointCloudsInFiles,
    _init_,
    delete_files,
    make_confusion_matrix,
)


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
num_augs = 15

# max_points
# 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240,
# 11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432, 19456, 20480
# 8192 points max with PointCNN
max_points = 1024

# Fields to include from pointcloud
use_columns = ["intensity"]
# use_columns = ["X","Y","Z"]

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

# Model Name
model_name = f"PointCNN_{max_points}_{len(classes)}"


def test_one_epoch(device, model, test_loader, testing=False):
    model.eval()  # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    test_loss = 0.0
    pred = 0.0
    count = 0
    y_pred = torch.tensor([], device=device)  # empty tensor
    y_true = torch.tensor([], device=device)  # empty tensor
    outs = torch.tensor([], device=device)  # empty tensor

    # Iterate through data in loader
    for i, data in enumerate(
        tqdm(test_loader, desc="Validation", leave=False, colour="green")
    ):
        # Send data to defined device
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

        # Update pred and true
        _, pred1 = output.max(dim=1)
        ag = pred1 == data.y
        am = ag.sum()
        pred += am.item()

        y_true = torch.cat((y_true, data.y), 0)  # concatentate true values
        y_pred = torch.cat((y_pred, pred1), 0)  # concatenate predicted values
        outs = torch.cat((outs, output), 0)  # concatentate output

    # Calculate test_loss and accuracy
    test_loss = float(test_loss) / count
    accuracy = float(pred) / count

    # For validation
    if testing is False:
        # Create confusion matrix and classification report
        y_true = y_true.cpu().numpy()  # convert to array and send to cpu
        y_pred = y_pred.cpu().numpy()  # convert to array and send to cpu
        conf_mat = confusion_matrix(y_true, y_pred)  # create confusion matrix
        cls_rpt = classification_report(  # create classification report
            y_true,
            y_pred,
            target_names=classes,
            labels=np.arange(len(classes)),
            output_dict=True,
            zero_division=1,
        )
        return test_loss, accuracy, conf_mat, cls_rpt

    # For testing
    else:
        # out = torch.nn.functional.log_softmax(output, dim=1)  # softmax of output
        out = torch.nn.functional.softmax(outs, dim=1)
        return test_loss, accuracy, out
    
    
def test(device, model, test_loader, textio):
    # Run test_one_epoch with testing as true
    test_loss, test_accuracy, out = test_one_epoch(
        device, model, test_loader, testing=True
    )

    # Print and save loss and accuracy
    textio.cprint(
        "Testing Loss: %f & Testing Accuracy: %f" % (test_loss, test_accuracy)
    )

    return out


def train_one_epoch(device, model, train_loader, optimizer, epoch_number):
    model.train()
    train_loss = 0.0
    pred = 0.0
    count = 0

    # Iterate through data in loader
    for i, data in enumerate(
        tqdm(
            train_loader, desc="Epoch: " + str(epoch_number), leave=False, colour="blue"
        )
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
    early_stopping = True,
    patience = 20, # Added patience for early stopping
):
    # Set up optimizer
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer == "Adam":  # Adam optimizer
        optimizer = torch.optim.Adam(learnable_params, lr=0.001)
    else:  # SGD optimizer
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    # Set up checkpoint
    if checkpoint is not None:
        min_loss = checkpoint["min_loss"]
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Define best_test_loss
    best_test_loss = np.inf
    triggertimes = 0 # Added triggertimes

    # Run for every epoch
    for epoch in tqdm(
        range(start_epoch, epochs), desc="Total", leave=False, colour="red"
    ):
        # Train Model
        train_loss, train_accuracy = train_one_epoch(
            device, model, train_loader, optimizer, epoch + 1
        )

        # Validate model: testing=False
        test_loss, test_accuracy, conf_mat, cls_rpt = test_one_epoch(
            device, model, test_loader, testing=False
        )

        # Save Best Model
        if test_loss < best_test_loss:
            best_test_loss = test_loss

            # Create snap dictionary
            snap = {
                # state_dict: https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "min_loss": best_test_loss,
                "optimizer": optimizer.state_dict,
            }
            # Save best snap dictionary
            torch.save(snap, f"checkpoints/{model_name}/models/best_model_snap.t7")

            # Save best model
            torch.save(
                model.state_dict(), f"checkpoints/{model_name}/models/best_model.t7"
            )

            # Make confusion matrix figure
            make_confusion_matrix(conf_mat, categories=classes)

            # Save best model confusion matrix
            delete_files(
                f"checkpoints/{model_name}/confusion_matrix/best", "*.png"
            )  # delete previous

            plt.savefig(
                f"checkpoints/{model_name}/confusion_matrix/best/confusion_matrix_epoch{epoch+1}.png",
                bbox_inches="tight",
                dpi=300,
            )  # save png

            plt.close()

            # Create classification report figure
            sns.heatmap(pd.DataFrame(cls_rpt).iloc[:-1, :].T, annot=True, cmap="Blues")

            # save best model classification report
            delete_files(
                f"checkpoints/{model_name}/classification_report/best", "*.png"
            )  # delete previous

            plt.savefig(
                f"checkpoints/{model_name}/classification_report/best/classification_report_epoch{epoch+1}.png",
                bbox_inches="tight",
                dpi=300,
            )  # save png

            plt.close()

        # Create confusion matrix figure
        make_confusion_matrix(conf_mat, categories=classes)
        plt.savefig(
            f"checkpoints/{model_name}/confusion_matrix/all/confusion_matrix_epoch{epoch+1}.png",
            bbox_inches="tight",
            dpi=300,
        )  # save .png

        plt.close()

        # Create classification report figure
        sns.heatmap(pd.DataFrame(cls_rpt).iloc[:-1, :].T, annot=True, cmap="Blues")
        plt.savefig(
            f"checkpoints/{model_name}/classification_report/all/classification_report_epoch{epoch+1}.png",
            bbox_inches="tight",
            dpi=300,
        )  # save .png
        plt.close()

        # Save most recent model
        torch.save(snap, f"checkpoints/{model_name}/models/model_snap.t7")
        torch.save(model.state_dict(), f"checkpoints/{model_name}/models/model.t7")

        # Add values to TensorBoard
        boardio.add_scalar("Train Loss", train_loss, epoch + 1)  # training loss
        boardio.add_scalar("Validation Loss", test_loss, epoch + 1)  # validation loss
        boardio.add_scalar(
            "Best Validation Loss", best_test_loss, epoch + 1
        )  # best validation loss
        boardio.add_scalar(
            "Train Accuracy", train_accuracy, epoch + 1
        )  # training accuracy
        boardio.add_scalar(
            "Validation Accuracy", test_accuracy, epoch + 1
        )  # validation accuracy
        boardio.add_scalars(
            "Loss",
            {"Training Loss": train_loss, "Validation Loss": test_loss},
            epoch + 1,
        )  # training and validation loss

        # Print and save losses and accuracies
        textio.cprint(
            "EPOCH:: %d, Training Loss: %f, Validation Loss: %f, Best Loss: %f"
            % (epoch + 1, train_loss, test_loss, best_test_loss)
        )
        textio.cprint(
            "EPOCH:: %d, Training Accuracy: %f Validation Accuracy: %f"
            % (epoch + 1, train_accuracy, test_accuracy)
        )
        
        # Added section below for early stopping
        if early_stopping is True:
            if test_loss > best_test_loss:
                triggertimes += 1
                textio.cprint(f"Trigger Times:{triggertimes}")

                if triggertimes >= patience:
                    textio.cprint("Early Stopping")
                    return test_loss

            else:
                textio.cprint("Trigger Times: 0")
                triggertimes = 0
        
        
def main(pretrained="", augment=True, num_augs=num_augs):
    # Set up TensorBoard summary writer
    boardio = SummaryWriter(log_dir="checkpoints/" + model_name)
    _init_(model_name)

    # Set up logger
    textio = IOStream("checkpoints/" + model_name + "/run.log")
    textio.cprint(model_name)

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
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    model = PointCNN(numfeatures=len(use_columns), numclasses=len(classes))

    # Checkpoint
    checkpoint = None

    # Load existing model
    if pretrained:
        assert os.path.isfile(pretrained)
        model.load_state_dict(torch.load(pretrained, map_location="cpu"))

    # Send model to defined device
    model.to(device)

    # Run testing
    if pretrained:
        finished = test(
            device=device, model=model, test_loader=test_loader, textio=textio
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
            boardio=boardio,
            textio=textio,
            checkpoint=checkpoint,
        )
        
        
# Runtime
if __name__ == "__main__":
    main()
    # value = main(pretrained=pretrained)