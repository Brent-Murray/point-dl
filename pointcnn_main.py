import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.tools import IOStream, PointCloudsInFiles, _init_, make_confusion_matrix, delete_files
from models.pointcnn import PointCNN


train_dataset_path = r"D:\MurrayBrent\data\RMF_ITD\PLOT_LAS\BUF_5M_SC\train"
test_dataset_path = r"D:\MurrayBrent\data\RMF_ITD\PLOT_LAS\BUF_5M_SC\test"
max_points = 2048
use_columns = ["intensity"]
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
model_name = f"PointCNN_{max_points}_{len(classes)}"

def test_one_epoch(device, model, test_loader):
    model.eval()  # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    test_loss = 0.0
    pred = 0.0
    count = 0
    y_pred = torch.tensor([], device=device)
    y_true = torch.tensor([], device=device)

    for i, data in enumerate(
        tqdm(test_loader, desc="Testing", leave=False, colour="green")
    ):
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

        y_true = torch.cat((y_true, data.y), 0)
        y_pred = torch.cat((y_pred, pred1), 0)

    # Calculate test_loss and accuracy
    test_loss = float(test_loss) / count
    accuracy = float(pred) / count

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    conf_mat = confusion_matrix(y_true, y_pred)
    cls_rpt = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        labels=np.arange(len(classes)),
        output_dict=True,
        zero_division=1,
    )

    return test_loss, accuracy, conf_mat, cls_rpt


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

    for epoch in tqdm(
        range(start_epoch, epochs), desc="Total", leave=False, colour="red"
    ):
        # Train Model
        train_loss, train_accuracy = train_one_epoch(
            device, model, train_loader, optimizer, epoch + 1
        )

        # Test Model
        test_loss, test_accuracy, conf_mat, cls_rpt = test_one_epoch(
            device, model, test_loader
        )

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

        # Save model
        torch.save(snap, f"checkpoints/{model_name}/models/model_snap.t7")
        torch.save(model.state_dict, f"checkpoints/{model_name}/models/model.t7")

        boardio.add_scalar("Train Loss", train_loss, epoch + 1)
        boardio.add_scalar("Test Loss", test_loss, epoch + 1)
        boardio.add_scalar("Best Test Loss", best_test_loss, epoch + 1)
        boardio.add_scalar("Train Accuracy", train_accuracy, epoch + 1)
        boardio.add_scalar("Test Accuracy", test_accuracy, epoch + 1)
        boardio.add_scalars(
            "Loss", {"Training Loss": train_loss, "Test Loss": test_loss}, epoch + 1
        )

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
    # max_points 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240,
    # 11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432, 19456, 20480
    trainset = PointCloudsInFiles(
        train_dataset_path, "*.laz", "Class", max_points=max_points, use_columns=use_columns
    )
    testset = PointCloudsInFiles(
        test_dataset_path, "*.laz", "Class", max_points=max_points, use_columns=use_columns
    )

    # Load training and test datasets
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PointCNN(numfeatures=len(use_columns), numclasses=len(classes))

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