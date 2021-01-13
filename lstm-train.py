import torch
import torch.nn as nn
from models.VideoRNN import VideoRNN
from utils.lsfb_dataset_loader import load_lsfb_dataset
from torchvision import datasets, transforms
from datasets.lsfb_dataset import LsfbDataset
import mlflow
import numpy as np
import json
import pickle
import argparse

from transforms.video_transforms import (
    ChangeVideoShape,
    ResizeVideo,
    RandomCropVideo,
    CenterCropVideo,
    I3DPixelsValue,
    TrimVideo,
    PadVideo,
)


# Loading gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Setup ressources
params = {
    "epoch": 30,
    "batch_size": 6,
    "learning_rate": 0.1,
    "hidden_size": 2048,
    "cumulation": 42,
    "lstm_layer": 1,
}

epoch = params["epoch"]
batch_size = params["batch_size"]
RUN_NAME = f"epoch:{epoch}-batch:{batch_size}"

# Parsing the args
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Path to the input video directory")
parser.add_argument("-o", "--output", help="Path to the output directory")
parser.add_argument("-n", "--name", help="Name of the MLflow experiment")
parser.add_argument("-l", "--load", help="Indicate to load model weight")
parser.add_argument("-w", "--workers", help="Number of workders", default=4, type=int)

args = parser.parse_args()

input_file = args.input
output_file = args.output
experiment_name = args.name
model_weights = args.load
nb_workers = args.workers


# Using

# setup dataset
# Transformations for train images
composed_train = transforms.Compose(
    [
        TrimVideo(50),
        PadVideo(50, loop=False),
        ResizeVideo(270, interpolation="linear"),
        RandomCropVideo((224, 224)),
        ChangeVideoShape("TCHW"),
    ]
)

# Transformation for test images
compose_test = transforms.Compose(
    [
        TrimVideo(50),
        PadVideo(50, loop=False),
        ResizeVideo(270, interpolation="linear"),
        CenterCropVideo((224, 224)),
        ChangeVideoShape("TCHW"),
    ]
)


## Loading data and setup the batch loader
data = load_lsfb_dataset(input_file)
train = data[data["subset"] == "train"]
test = data[data["subset"] == "test"]

print(len(data))

train_dataset = LsfbDataset(train, transforms=composed_train)

# Saving label mapping
labels = train_dataset.labels

with open(f"{output_file}/labels.json", "w") as f:
    json.dump(labels, f)

test_dataset = LsfbDataset(test, transforms=compose_test, labels=labels)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=nb_workers
)

val_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=nb_workers,
)


n_class = len(labels)
print(f"N class = {n_class}")
net = VideoRNN(params["hidden_size"], n_class, device, 2)

if model_weights != None:
    net.load_state_dict(torch.load(model_weights))
    print("Weights loaded")


# Chosing optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), lr=params["learning_rate"], momentum=0.9, weight_decay=0.0000001
)
lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


def pack_sequence(X):
    """ Transform a zero padded sequence batch to a pytorch
        PackedSequence object
    """
    numpy_X = X.numpy()
    lengths = []

    for i in range(len(numpy_X)):
        seq = numpy_X[i]
        l = 0
        for frame in seq:
            l += 1
            if np.sum(frame) == 0:
                break

        lengths.append(l)

    packed = torch.nn.utils.rnn.pack_padded_sequence(
        X, lengths, batch_first=True, enforce_sorted=False
    )

    return packed


def train_model(
    model, criterion, optimizer, lr_scheduler, loader, device, batch_size, cumulation=1
):

    epoch_loss = 0.0
    accuracy = 0

    model.train().to(device)
    batch_idx = 0

    for data in loader:
        print(f"\rBatch : {batch_idx+1} / {len(loader)}", end="\r")
        batch_idx += 1

        X, y = data
        # Correcting type of the tensors
        X = X.type(torch.FloatTensor)

        X = pack_sequence(X)
        X = X.to(device)
        y = y.to(device)

        output = model(X)

        loss = criterion(output, y)
        loss.backward()
        epoch_loss += loss.item()

        _, preds = torch.max(output, 1)

        # Cumulated gradient
        if batch_idx % cumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        accuracy += torch.sum(preds == y.data)

    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()

    epoch_loss = epoch_loss / len(loader)
    train_acc = accuracy.double() / (len(loader) * batch_size)

    return epoch_loss, train_acc


def eval_model(model, criterion, loader, device, batch_size):
    eval_loss = 0
    eval_acc = 0

    model.eval().to(device)
    batch_idx = 0
    raw_predictions = []

    for data in loader:
        print(f"\rBatch : {batch_idx+1} / {len(loader)}", end="\r")
        batch_idx += 1

        X, y = data

        # Correcting type of the tensors
        X = X.type(torch.FloatTensor)

        X = pack_sequence(X)
        X = X.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(False):
            output = model(X)

            loss = criterion(output, y)
            eval_loss += loss.item()

            _, preds = torch.max(output, 1)
            eval_acc += torch.sum(preds == y.data)

            numpy_pred = output.cpu().detach().numpy()
            for i in range(len(numpy_pred)):
                item = y[i].item()
                list_pred = numpy_pred[i].tolist()
                raw_predictions.append((item, list_pred))

    with open(f"{output_file}/predictions.pkl", "wb") as f:
        pickle.dump(raw_predictions, f)

    eval_loss = eval_loss / len(loader)
    eval_acc = eval_acc.double() / (len(loader) * batch_size)
    return eval_loss, eval_acc


# Training loop
mlflow.set_experiment(experiment_name)
current_min_loss = 3000
last_improvement = 0

with mlflow.start_run(run_name="Adding scheduler"):
    mlflow.log_params(params)

    for iter in range(1, epoch + 1):
        print(f"{iter}/{epoch}\n")

        train_loss, train_acc = train_model(
            net,
            criterion,
            optimizer,
            lr_sched,
            train_dataloader,
            device,
            batch_size,
            params["cumulation"],
        )
        print(f"train_loss : {train_loss}  train_acc : {train_acc}")
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("train_acc", train_acc.item())

        eval_loss, eval_acc = eval_model(
            net, criterion, val_dataloader, device, batch_size
        )

        print(f"eval_loss : {eval_loss}  eval_acc : {eval_acc}")
        mlflow.log_metric("eval_loss", eval_loss)
        mlflow.log_metric("eval_acc", eval_acc.item())

        if eval_loss < current_min_loss:
            current_min_loss = eval_loss
            torch.save(net.state_dict(), f"{output_file}/model.pt")
            last_improvement = 0
        else:
            last_improvement += 1

        if last_improvement > 3:
            print("No improvement since 3 epochs. Shutting down")
            break
