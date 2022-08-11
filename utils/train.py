import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from utils.tools import make_confusion_matrix, delete_files

def test_one_epoch(device, model, test_loader, classes, testing=False):
    model.eval()  # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    with torch.no_grad():
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
            if not torch.cuda.device_count() > 1:
                data.to(device)

            # Call model
            output = model(data)

            # Define validation loss using negative log likelihood loss and softmax
            if not torch.cuda.device_count() > 1:
                loss_val = torch.nn.functional.nll_loss(
                    torch.nn.functional.log_softmax(output, dim=1), 
                    target=data.y,
                )

            else:
                loss_val = torch.nn.functional.nll_loss(
                    torch.nn.functional.log_softmax(output, dim=1),
                    target=torch.cat([d.y for d in data]).to(output.device)
                )

            # Update test_lost and count
            test_loss += loss_val.item()
            count += output.size(0)

            # Update pred and true
            _, pred1 = output.max(dim=1)
            if not torch.cuda.device_count() > 1:
                y = data.y
            else:
                y = torch.cat([d.y for d in data]).to(output.device)
            ag = pred1 == y
            am = ag.sum()
            pred += am.item()

            # y_true = torch.cat((y_true, data.y), 0)  # concatentate true values
            y_true = torch.cat((y_true, y), 0)  # concatentate true values
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
    
    
def test(device, model, test_loader, textio, classes):
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
            train_loader, desc="Epoch: " + str(epoch_number), leave=False, colour="cyan"
        )
    ):
        # Send data to device
        if not torch.cuda.device_count() > 1:
            data.to(device)

        # Call model
        output = model(data)

        # Define validation loss using negative log likelihood loss and softmax
        if not torch.cuda.device_count() > 1:
            loss_val = torch.nn.functional.nll_loss(
                torch.nn.functional.log_sofmax(output, dim=1), 
                target=data.y,
            )
            
        else:
            loss_val = torch.nn.functional.nll_loss(
                torch.nn.functional.log_softmax(output, dim=1),
                target=torch.cat([d.y for d in data]).to(output.device)
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
        if not torch.cuda.device_count() > 1:
            y = data.y
        else:
            y = torch.cat([d.y for d in data]).to(output.device)
        ag = pred1 == y
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
    classes,
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
        range(start_epoch, epochs), desc="Model Total: ", leave=False, colour="red"
    ):
        # Train Model
        train_loss, train_accuracy = train_one_epoch(
            device, model, train_loader, optimizer, epoch + 1
        )

        # Validate model: testing=False
        test_loss, test_accuracy, conf_mat, cls_rpt = test_one_epoch(
            device, model, test_loader, classes, testing=False
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