import tempfile

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from tqdm import tqdm

from src.es import EarlyStopper
from src.helpers import after_subplot, save_metrics, calculate_metrics


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one train_one_epoch epoch
    """
    if torch.cuda.is_available():
        model.cuda()
    model = model.train()

    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to GPU
        if torch.cuda.is_available():
            # workaround for RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

        # 1. clear the gradients of all optimized variables
        # :
        optimizer.zero_grad()
        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # 3. calculate the loss
        loss_value = loss(output, target)
        # 4. backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward()
        # 5. perform a single optimization step (parameter update)
        optimizer.step()
        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    y_true = []
    y_pred = []

    with torch.no_grad():

        # set the model to evaluation mode
        # 
        model.eval()
        if torch.cuda.is_available():
            model.cuda()

        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # move data to GPU
            if torch.cuda.is_available():
                target = target.type(torch.LongTensor)
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)  # 
            # 2. calculate the loss
            loss_value = loss(output, target)  # 

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )
            y_true.extend(target.cpu().numpy())
            y_pred.extend(output.cpu().data.max(1)[1].numpy())

    accuracy, precision, recall, f1, mcc = calculate_metrics(y_true, y_pred)

    print(
        f"Accuracy: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, "
        f"F1 score: {f1:.4f}, "
        f"MCC: {mcc:.4f}"
    )

    return valid_loss, accuracy, precision, recall, f1, mcc


def optimize(
    data_loaders, model, optimizer, loss, n_epochs,
    model_save_path, result_save_path,
    early_stopper: EarlyStopper | None = None
):
    valid_loss_min = None

    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a
    # plateau
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    from torch.optim.lr_scheduler import ExponentialLR
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss
        )

        valid_loss, accuracy, precision, recall, f1, mcc = valid_one_epoch(
            data_loaders["valid"], model, loss,
        )

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
            (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), model_save_path)
            valid_loss_min = valid_loss

        # Update learning rate, i.e., make a step in the learning rate scheduler
        scheduler.step()

        save_metrics(accuracy, precision, recall, f1, mcc, result_save_path)
        if early_stopper:
            if early_stopper.is_early_stop(
                f1 if early_stopper.early_stopping_metrics == "f1" else mcc
            ):
                break


def one_epoch_test(test_dataloader, model, loss):
    # function to test by batching
    y_true = []
    y_pred = []

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    with torch.no_grad():

        # set the model to evaluation mode
        # 
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(
            enumerate(test_dataloader),
            desc='Testing',
            total=len(test_dataloader),
            leave=True,
            ncols=80
        ):
            # move data to GPU
            if torch.cuda.is_available():
                target = target.type(torch.LongTensor)
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)  # 
            # 2. calculate the loss
            loss_value = loss(logits, target)  # 

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            # HINT: the predicted class is the index of the max of the logits
            pred = logits.data.max(1, keepdim=True)[1]  # 

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss


def test(test_dataloader, model):
    y_true = []
    y_pred = []

    # set the model to evaluation mode
    model.eval()

    # turn off gradient calculation
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()

        for data, target in tqdm(
            test_dataloader,
            desc='Testing',
            total=len(test_dataloader),
            leave=True,
            ncols=80
        ):
            # move data to GPU
            if torch.cuda.is_available():
                target = target.type(torch.LongTensor)
                data, target = data.cuda(), target.cuda()

            # forward pass
            logits = model(data)
            # convert logits to predicted class
            pred = logits.data.max(1)[1]

            # append true and predicted labels to lists
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    accuracy, precision, recall, f1, mcc = calculate_metrics(y_true, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 score: {f1:.4f}')
    print(f'MCC: {mcc:.4f}')

    return accuracy, precision, recall, f1, mcc
