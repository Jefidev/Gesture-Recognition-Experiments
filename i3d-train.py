import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np

from transforms.video_transforms import (
    ChangeVideoShape,
    ResizeVideo,
    RandomCropVideo,
    CenterCropVideo,
)
from models.pytorch_i3d import InceptionI3d
from utils.lsfb_dataset_loader import load_lsfb_dataset
import mlflow
from utils.train_eval import train_model, eval_model

import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import Compose
from datasets.lsfb_dataset import LsfbDataset
from torch.utils.data import DataLoader
import os
import time
import pickle
import json


# Loading gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path = "/home/jeromefink/Documents/unamur/signLanguage/Data/most_frequents_25"

params = {
    "batch_size": 3,
    "resize": 256,
    "frames": 48,
    "epochs": 20,
    "lr": 0.001,
    "dataset": path.split("/")[-1],
    "cumulation": 25,
    "classification_loss": 0.9,
    "pretrained": "charades",
}

composed_train = Compose(
    [
        ResizeVideo(params["resize"], interpolation="bilinear"),
        RandomCropVideo((224, 224)),
        ChangeVideoShape("CTHW"),
    ]
)

compose_test = Compose(
    [
        ResizeVideo(params["resize"], interpolation="bilinear"),
        CenterCropVideo((224, 224)),
        ChangeVideoShape("CTHW"),
    ]
)

epoch = params["epochs"]
batch_size = params["batch_size"]
frames = params["frames"]
RUN_NAME = f"epoch:{epoch}-batch:{batch_size}-frames:{frames}"
RUN_FOLDER = "./results/" + time.strftime("%Y%m%d-%H%M%S")

os.makedirs(RUN_FOLDER)

## Loading data and setup the batch loader
#
data = load_lsfb_dataset(path)

train = data[data["subset"] == "train"]
test = data[data["subset"] == "test"]

train_dataset = LsfbDataset(
    train, frames, sequence_label=True, transforms=composed_train, one_hot=True
)

test_dataset = LsfbDataset(
    test, frames, sequence_label=True, transforms=compose_test, one_hot=True
)


labels = train_dataset.labels
params["n_class"] = len(labels)

with open(f"{RUN_FOLDER}/labels.json", "w") as fp:
    json.dump(labels, fp)


if params["pretrained"] == "charades":
    net = InceptionI3d(157, in_channels=3)
    net.load_state_dict(torch.load("models/rgb_charades.pt"))
else:
    net = InceptionI3d(400, in_channels=3)
    net.load_state_dict(torch.load("models/rgb_imagenet.pt"))

net.replace_logits(params["n_class"])
net = net.to(device)


lr = params["lr"]
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


mlflow.set_experiment("I3D")

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_params(params)

    epoch = 0
    best_accuracy = 0
    # train it
    while epoch < params["epochs"]:
        print("Epoch {}/{}".format(epoch, params["epochs"] - 1))
        print("-" * 10)
        epoch += 1

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            print("\n\n")

            if phase == "train":
                net.train(True)
                current_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=2,
                )
            else:
                net.train(False)  # Set model to evaluate mode
                current_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=2,
                )

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            accuracy = 0
            raw_predictions = []

            # Iterate over data.
            for idx, batch in enumerate(current_loader):
                print(f"\rBatch : {idx+1} / {len(current_loader)}", end="\r")

                # get the inputs
                inputs, targets = batch
                inputs = inputs.type(torch.FloatTensor)
                targets = targets.type(torch.FloatTensor)

                # wrap them in Variable
                inputs = inputs.to(device)
                t = inputs.size(2)
                targets = targets.to(device)

                per_frame_logits = net(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode="linear")

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, targets)
                tot_loc_loss += loc_loss.data

                # Removing the padding
                ground_truth = torch.narrow(
                    torch.max(targets, dim=2)[0], 1, 1, params["n_class"] - 1
                )

                raw_pred = torch.narrow(
                    torch.max(per_frame_logits, dim=2)[0], 1, 1, params["n_class"] - 1
                )

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(raw_pred, ground_truth)

                tot_cls_loss += cls_loss.data

                pred = torch.max(raw_pred, dim=1)[1]
                ground_truth = torch.max(ground_truth, dim=1)[1]

                accuracy += torch.sum(pred == ground_truth)

                reste = 1 - params["classification_loss"]
                loss = reste * loc_loss + params["classification_loss"] * cls_loss
                tot_loss += loss.data

                if idx % params["cumulation"] == 0 and phase == "train":
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()

                if phase == "train":
                    if idx % 50 == 0:
                        print(
                            "{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accuracy: {:.4f}\n".format(
                                phase,
                                tot_loc_loss / idx,
                                tot_cls_loss / idx,
                                tot_loss / idx,
                                accuracy.double() / (idx * params["batch_size"]),
                            )
                        )
                elif phase == "val":
                    numpy_pred = raw_pred.cpu().detach().numpy()
                    for i in range(len(numpy_pred)):
                        raw_predictions.append(
                            (ground_truth[i].item(), numpy_pred[i].tolist())
                        )

            if phase == "val":
                val_m = {}
                val_m["val_loc_loss"] = (tot_loc_loss / len(current_loader)).item()
                val_m["val_cls_loss"] = (tot_cls_loss / len(current_loader)).item()
                val_m["val_tot_loss"] = (tot_loss / len(current_loader)).item()
                val_m["val_accuracy"] = float(
                    accuracy.double() / (len(current_loader) * params["batch_size"])
                )
                print(
                    "{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accuracy: {:.4f}".format(
                        phase,
                        val_m["val_loc_loss"],
                        val_m["val_cls_loss"],
                        val_m["val_tot_loss"],
                        val_m["val_accuracy"],
                    )
                )

                if best_accuracy < val_m["val_accuracy"]:
                    best_accuracy = val_m["val_accuracy"]
                    model_path = RUN_FOLDER + "/best_model.pt"
                    pred_path = RUN_FOLDER + "/model_eval_preds.pkl"
                    torch.save(net.state_dict(), model_path)

                    with open(pred_path, "wb") as f:
                        pickle.dump(raw_predictions, f)

                    mlflow.log_artifacts(RUN_FOLDER)

                mlflow.log_metrics(val_m)
            else:
                val_m = {}
                val_m["train_loc_loss"] = (tot_loc_loss / len(current_loader)).item()
                val_m["train_cls_loss"] = (tot_cls_loss / len(current_loader)).item()
                val_m["train_tot_loss"] = (tot_loss / len(current_loader)).item()
                val_m["train_accuracy"] = float(
                    accuracy.double() / (len(current_loader) * params["batch_size"])
                )

                mlflow.log_metrics(val_m)

