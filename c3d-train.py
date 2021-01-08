# https://towardsdatascience.com/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3

from utils.lsfb_dataset_loader import load_lsfb_dataset
from models.C3D import C3D
import mlflow
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from datasets.lsfb_dataset import LsfbDataset

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

path = "/home/jeromefink/Documents/unamur/signLanguage/Data/most_frequents_20"
# path = "./mock-data"
params = {
    "batch_size": 5,
    "max_frames": 48,
    "epochs": 20,
    "lr": 0.01,
    "dataset": path.split("/")[-1],
    "cumulation": 20,
}


## Loading data and setup the batch loader
#
data = load_lsfb_dataset(path)

train = data[data["subset"] == "train"]
test = data[data["subset"] == "test"]

# setup dataset
# Transformations for train images
composed_train = transforms.Compose(
    [
        TrimVideo(params["max_frames"]),
        PadVideo(params["max_frames"], loop=False),
        ResizeVideo(270, interpolation="linear"),
        RandomCropVideo((224, 224)),
        ChangeVideoShape("CTHW"),
    ]
)

# Transformation for test images
compose_test = transforms.Compose(
    [
        TrimVideo(params["max_frames"]),
        PadVideo(params["max_frames"], loop=False),
        ResizeVideo(270, interpolation="linear"),
        CenterCropVideo((224, 224)),
        ChangeVideoShape("CTHW"),
    ]
)

train_dataset = LsfbDataset(train, transforms=composed_train)

labels = train_dataset.labels

test_dataset = LsfbDataset(test, transforms=compose_test, labels=labels)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=2
)

val_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=2,
)


params["n_class"] = len(labels)
net = C3D(params["n_class"])
print(net)

# Chosing optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), lr=params["lr"], momentum=0.9, weight_decay=0.0000001
)
lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


# Training function


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

    epoch_loss = epoch_loss / len(loader)
    train_acc = accuracy.double() / (len(loader) * batch_size)

    return epoch_loss, train_acc


def eval_model(model, criterion, loader, device, batch_size):
    eval_loss = 0
    eval_acc = 0

    model.eval().to(device)
    batch_idx = 0

    for data in loader:
        print(f"\rBatch : {batch_idx+1} / {len(loader)}", end="\r")
        batch_idx += 1

        X, y = data
        X = X.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(False):
            output = model(X)

            loss = criterion(output, y)
            eval_loss += loss.item()

            _, preds = torch.max(output, 1)
            eval_acc += torch.sum(preds == y.data)

    eval_loss = eval_loss / len(loader)
    eval_acc = eval_acc.double() / (len(loader) * batch_size)
    return eval_loss, eval_acc


# Training loop
mlflow.set_experiment("C3D")

with mlflow.start_run(run_name=params["dataset"]):
    mlflow.log_params(params)

    for iter in range(1, params["epochs"] + 1):
        epochs = params["epochs"]
        print(f"{iter}/{epochs}\n")

        train_loss, train_acc = train_model(
            net,
            criterion,
            optimizer,
            lr_sched,
            train_dataloader,
            device,
            params["batch_size"],
            params["cumulation"],
        )
        print(f"train_loss : {train_loss}  train_acc : {train_acc}")
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("train_acc", train_acc.item())

        eval_loss, eval_acc = eval_model(
            net, criterion, val_dataloader, device, params["batch_size"]
        )
        print(f"eval_loss : {eval_loss}  eval_acc : {eval_acc}")
        mlflow.log_metric("eval_loss", eval_loss)
        mlflow.log_metric("eval_acc", eval_acc.item())

