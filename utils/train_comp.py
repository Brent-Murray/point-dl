import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from utils.tools import create_comp_csv, delete_files


def test_one_epoch(device, model, test_loader, testing=False):
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        pred = 0.0
        count = 0

        # Empty Tensors
        y_true = torch.tensor([], device=device)
        y_pred = torch.tensor([], device=device)

        for i, data in enumerate(
            tqdm(test_loader, desc="Validation", leave=False, colour="green")
        ):
            # Send data to device
            if not torch.cuda.device_count() > 1:
                data.to(device)

            # Call model
            output = model(data)

            # Define validation loss
            if not torch.cuda.device_count() > 1:
                loss_val = F.mse_loss(F.softmax(output, dim=1), target=data.y)
            else:
                loss_val = F.mse_loss(
                    F.softmax(output, dim=1),
                    target=torch.stack([d.y for d in data]).to(output.device),
                )

            # Update test_loss and count
            test_loss += loss_val.item()
            count += output.size(0)

            # Get true y values
            if not torch.cuda.device_count() > 1:
                y = data.y
            else:
                y = torch.stack([d.y for d in data]).to(output.device)

            y_true = torch.cat((y_true, y), 0)  # concatenate true values
            y_pred = torch.cat(
                (y_pred, F.softmax(output, dim=1)), 0
            )  # concatenate true values

        y_true = y_true.cpu().numpy()  # convert to array and send to cpu
        y_pred = y_pred.cpu().numpy()  # convert to array and send to cpu
            
        r2 = r2_score(y_true.flatten(), y_pred.flatten().round(2))
            
        # Calculate average epoch test loss
        test_loss = float(test_loss) / count
        
        return test_loss, y_pred, y_true, r2


def test(device, mode, test_laoder, texio):
    test_loss, out = test_one_epoch(device, mode, test_loader, testing=True)

    return out


def train_one_epoch(device, model, train_loader, optimizer, epoch_number):
    model.train()
    train_loss = 0.0
    pred = 0.0
    count = 0

    # iterate through data in loader
    for i, data in enumerate(
        tqdm(
            train_loader, desc="Epoch: " + str(epoch_number), leave=False, colour="cyan"
        )
    ):
        # Send data to device if in single gpu
        if not torch.cuda.device_count() > 1:
            data.to(device)

        # Call Model
        output = model(data)

        # Define validation loss
        if not torch.cuda.device_count() > 1:
            loss_val = F.mse_loss(F.softmax(output, dim=1), target=data.y)
        else:
            loss_val = F.mse_loss(
                F.softmax(output, dim=1),
                target=torch.stack([d.y for d in data]).to(output.device),
            )

        # Forward + backward + optimize
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Update train_loss and count
        train_loss += loss_val.item()
        count += output.size(0)

    # Calculate train_loss
    train_loss = float(train_loss) / count

    return train_loss


def train(
    device,
    model,
    train_loader,
    test_loader,
    # boardio,
    textio,
    model_name,
    classes,
    adaptive_lr,
    lr,
    early_stopping,
    optimizer,
    start_epoch,
    epochs,
    patience,
    step_size,
    checkpoint=None,
):
    # Setup optimizer
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer == "Adam":  # Adam optimizer
        optimizer = torch.optim.Adam(learnable_params, lr=lr)
    if optimizer == "SGD":  # SGD optimizer
        optimizer = torch.optim.SGD(learnable_params, lr=lr)

    # Setup checkpoint
    if checkpoint is not None:
        min_loss = checkpoint["min_loss"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        
    # Use Adaptive Learning Rates
    if adaptive_lr is True:
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience) # reduce on plateu
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1) # reduce after 30 epochs
        # scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
        change = 0
        

    # Define best_test_loss
    best_test_loss = np.inf

    # Set triggertimes
    triggertimes = 0

    # Run for every epoch
    for epoch in tqdm(
        range(start_epoch, epochs), desc="Model Total: ", leave=False, colour="red"
    ):
        # Train Model
        train_loss = train_one_epoch(device, model, train_loader, optimizer, epoch + 1)

        # Validate Model
        test_loss, y_pred, y_true, r2 = test_one_epoch(
            device, model, test_loader, testing=False
        )
        
        # Save Best Model
        if test_loss < best_test_loss:
            best_test_loss = test_loss

            # Create Snap Dictionary
            snap = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "min_loss": best_test_loss,
                "optimizer": optimizer.state_dict,
            }

            # Save Best Snap Dictionary
            torch.save(snap, f"checkpoints/{model_name}/models/best_model_snap.t7")

            # Save Best Model
            torch.save(
                model.state_dict(), f"checkpoints/{model_name}/models/best_model.t7"
            )

            # Delete old file
            delete_files(f"checkpoints/{model_name}/output", "*.csv")

            # Create CSV of best model output
            create_comp_csv(
                y_true.flatten(),
                y_pred.flatten().round(2),
                classes,
                f"checkpoints/{model_name}/output/outputs_epoch{epoch+1}.csv",
            )

        # Save Most Recent Model
        torch.save(snap, f"checkpoints/{model_name}/models/model_snap.t7")
        torch.save(model.state_dict(), f"checkpoints/{model_name}/models/model.t7")

        # # Add Values to TensorBoard
        # boardio.add_scalar("Train Loss", train_loss, epoch + 1)  # training loss
        # boardio.add_scalar("Validation Loss", test_loss, epoch + 1)  # validation loss
        # boardio.add_scalar(
        #     "Best Validation Loss", best_test_loss, epoch + 1
        # )  # best validation loss
        # boardio.add_scalars(
        #     "Loss",
        #     {"Training Loss": train_loss, "Validation Loss": test_loss},
        #     epoch + 1,
        # )  # training and validation loss

        # Print and save losses and accuracies
        textio.cprint(
            "EPOCH:: %d, Training Loss: %f, Validation Loss: %f, Validation r2: %f, Best Loss: %f,"
            % (epoch + 1, train_loss, test_loss, r2, best_test_loss)
        )

        # Apply Addaptive Learning Rate
        if adaptive_lr is True:
            # textio.cprint(f"LR: {optimizer.param_groups[0]['lr']}") # print current learning rate
            if test_loss > best_test_loss:
                triggertimes += 1
                if triggertimes >= patience:
                    change = 1
            if change == 0:
                scheduler1.step(test_loss) # run first scheduler
                textio.cprint(f"LR: {scheduler1.optimizer.param_groups[0]['lr']}, Trigger Times: {triggertimes}, Scheduler: Plateau")
            else:
                scheduler2.step() # run second scheduler
                textio.cprint(f"LR: {scheduler2.optimizer.param_groups[0]['lr']}, Scheduler: Step")
                    
            
        # Early Stopping
        if early_stopping is True:
            if test_loss > best_test_loss:
                triggertimes += 1
                textio.cprint(f"Trigger Times: {triggertimes}")

                if triggertimes >= patience:
                    textio.cprint("Early Stopping")
                    return test_loss
            else:
                textio.cprint("Trigger Times: 0")
                triggertimes = 0