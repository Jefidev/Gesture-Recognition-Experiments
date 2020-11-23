import os
import json
import pickle
import json


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys


from transforms.video_transforms import (
    ChangeVideoShape,
    ResizeVideo,
    RandomCropVideo,
    CenterCropVideo,
    I3DPixelsValue,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms


import numpy as np
from models.pytorch_i3d import InceptionI3d
from utils.lsfb_dataset_loader import load_lsfb_dataset
from datasets.lsfb_dataset import LsfbDataset
import mlflow


init_lr = 0.01
max_steps = 50
mode = "lsfb410"
batch_size = 6
test_batch_size = 2
model_path = "./checkpoints/lsfb_410.pt"
cumulation = 64  # accum gradient
path = "/home/jeromefink/Documents/unamur/signLanguage/Data/most_frequents_410"
nbr_frames = 48
experiment_name = "I3D RGB LSFB-410"
predictions_file = "./checkpoints/predictions_410.pkl"
labels_file = "./checkpoints/labels_410.json"

params_ml_flow = {
    "init_lr": init_lr,
    "initial_weights": mode,
    "batch_size": batch_size,
    "dataset": path,
    "nbr_frames": nbr_frames,
}

# setup dataset
# Transformations for train images
composed_train = transforms.Compose(
    [
        ResizeVideo(256, interpolation="linear"),
        RandomCropVideo((224, 224)),
        I3DPixelsValue(),
        ChangeVideoShape("CTHW"),
    ]
)

# Transformation for test images
compose_test = transforms.Compose(
    [
        ResizeVideo(256, interpolation="linear"),
        CenterCropVideo((224, 224)),
        I3DPixelsValue(),
        ChangeVideoShape("CTHW"),
    ]
)

data = load_lsfb_dataset(path)

train = data[data["subset"] == "train"]
test = data[data["subset"] == "test"]

train_dataset = LsfbDataset(
    train,
    nbr_frames,
    sequence_label=True,
    transforms=composed_train,
    one_hot=True,
    padding="loop",
)

labels = train_dataset.labels

test_dataset = LsfbDataset(
    test,
    nbr_frames,
    sequence_label=True,
    transforms=compose_test,
    one_hot=True,
    padding="loop",
    labels=labels,
)

dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=6
)

val_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=3,
)

dataloaders = {"train": dataloader, "val": val_dataloader}
datasets = {"train": train_dataset, "val": test_dataset}

with open(labels_file, "w") as f:
    json.dump(labels, f)

nbr_class = len(labels)

# setup the model
if mode == "flow":
    i3d = InceptionI3d(400, in_channels=2)
    i3d.load_state_dict(torch.load("checkpoints/flow_imagenet.pt"))
    i3d.replace_logits(nbr_class)
    print("Flow kinetic loaded")
elif mode == "rgb_msasl":
    i3d = InceptionI3d(100, in_channels=3)
    i3d.load_state_dict(torch.load("checkpoints/MSASL.pt"))
    # i3d.replace_logits(nbr_class)
    print("MSASL loaded")
elif mode == "charades":
    i3d = InceptionI3d(157, in_channels=3)
    i3d.load_state_dict(torch.load("checkpoints/rgb_charades.pt"))
    i3d.replace_logits(nbr_class)
    print("RGB Charade loaded")
elif mode == "lsfb410":
    i3d = InceptionI3d(401, in_channels=3)
    i3d.load_state_dict(torch.load("checkpoints/lsfb_410.pt"))
    # i3d.replace_logits(nbr_class)
    print("RGB LSFB 410 loaded")
else:
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load("checkpoints/rgb_imagenet.pt"))
    i3d.replace_logits(nbr_class)
    print("RGB kinetic loaded")

i3d.cuda()
i3d = nn.DataParallel(i3d)

optimizer = optim.SGD(
    i3d.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0000001
)
lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
criterion = nn.CrossEntropyLoss()


def train_i3d(
    dataloader,
    model,
    optimizer,
    scheduler,
    criterion,
    num_cumulation,
    batch_size,
    save_path,
):

    batch_loss = 0.0
    batch_accuracy = 0.0

    cumulation = 0
    counter = 0

    epoch_loss = 0
    epoch_acc = 0

    # Iterate over data.
    size = len(dataloader)
    for data in dataloader:
        cumulation += 1
        counter += 1
        print(f"{counter}/{size}")

        # get the inputs
        inputs, labels = data

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        t = inputs.size(2)
        labels = Variable(labels.cuda())

        per_frame_logits = model(inputs)
        # upsample to input size
        per_frame_logits = F.upsample(per_frame_logits, t, mode="linear")

        tmp = torch.max(labels, dim=2)[0]
        loss = (
            criterion(torch.max(per_frame_logits, dim=2)[0], torch.max(tmp, dim=1)[1])
            / num_cumulation
        )
        minibatch_loss = loss.item()
        batch_loss += minibatch_loss
        epoch_loss += minibatch_loss

        # Compute accuracy
        logits_tmp = torch.max(per_frame_logits, dim=2)[0]
        majority_pred = torch.max(logits_tmp, dim=1)[1]

        tmp = torch.max(labels, dim=2)[0]
        majority_truth = torch.max(tmp, dim=1)[1]

        minibatch_acc = torch.sum(majority_pred == majority_truth).item()

        batch_accuracy += minibatch_acc
        epoch_acc += minibatch_acc

        loss.backward()

        if cumulation == num_cumulation:

            cumulation = 0
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            print(
                "{} Loss: {:.4f}  Accuracy: {:.4f}".format(
                    phase, batch_loss, batch_accuracy / (num_cumulation * batch_size),
                )
            )
            # save model

            torch.save(
                model.module.state_dict(), save_path,
            )

            batch_accuracy = batch_loss = 0.0

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    return epoch_loss, epoch_acc


def eval_i3d(dataloader, model, criterion, raw_pred_path):
    counter = 0

    eval_loss = 0
    eval_acc = 0

    raw_predictions = []

    # Iterate over data.
    size = len(dataloader)
    for data in dataloader:
        counter += 1
        print(f"{counter}/{size}")

        # get the inputs
        inputs, labels = data

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        t = inputs.size(2)
        labels = Variable(labels.cuda())

        per_frame_logits = i3d(inputs)
        # upsample to input size
        per_frame_logits = F.upsample(per_frame_logits, t, mode="linear")

        tmp = torch.max(labels, dim=2)[0]
        loss = criterion(
            torch.max(per_frame_logits, dim=2)[0], torch.max(tmp, dim=1)[1]
        )

        eval_loss += loss.item()

        # Compute accuracy
        logits_tmp = torch.max(per_frame_logits, dim=2)[0]
        majority_pred = torch.max(logits_tmp, dim=1)[1]

        tmp = torch.max(labels, dim=2)[0]
        majority_truth = torch.max(tmp, dim=1)[1]

        eval_acc += torch.sum(majority_pred == majority_truth).item()

        # Saving raw pred in a list for further investigation later
        numpy_pred = logits_tmp.cpu().detach().numpy()
        for i in range(len(numpy_pred)):
            item = majority_truth[i].item()
            list_pred = numpy_pred[i].tolist()
            raw_predictions.append((item, list_pred))

    with open(raw_pred_path, "wb") as f:
        pickle.dump(raw_predictions, f)

    return eval_loss, eval_acc


mlflow.set_experiment("I3D")

with mlflow.start_run(run_id="ae47157afd9b45ceb283fa1526785ddf") as run:
    # with mlflow.start_run(run_name=experiment_name) as run:
    params_ml_flow["run_id"] = run.info.run_id
    # mlflow.log_params(params_ml_flow)

    steps = 0
    while steps < max_steps:  # for epoch in range(num_epochs):
        print("Step {}/{}".format(steps, max_steps))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                i3d.train(True)
                dataloader = dataloaders[phase]
                size = len(dataloader)

                loss, acc = train_i3d(
                    dataloader,
                    i3d,
                    optimizer,
                    lr_sched,
                    criterion,
                    cumulation,
                    batch_size,
                    model_path,
                )

                epoch_loss = (loss * cumulation) / size
                epoch_acc = acc / (size * batch_size)

                print(
                    "{}  Loss: {:.4f}  Accuracy: {:.4f}".format(
                        phase, epoch_loss, epoch_acc
                    )
                )

                mlflow.log_metric("train_loss", epoch_loss)
                mlflow.log_metric("train_acc", epoch_acc)

            else:
                i3d.train(False)  # Set model to evaluate mode
                dataloader = dataloaders[phase]
                size = len(dataloader)

                tot_loss, accuracy = eval_i3d(
                    dataloader, i3d, criterion, predictions_file
                )

                tot_loss = tot_loss / size
                accuracy = accuracy / (size * test_batch_size)

                print(
                    "{}  Loss: {:.4f}  Accuracy: {:.4f}".format(
                        phase, tot_loss, accuracy
                    )
                )

                mlflow.log_metric("val_loss", tot_loss)
                mlflow.log_metric("val_acc", accuracy)

        steps += 1

