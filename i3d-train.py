import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np

from models.pytorch_i3d import InceptionI3d
from batch_loaders.I3DLoader import I3DLoader
from utils.lsfb_dataset_loader import load_lsfb_dataset
import mlflow
from utils.train_eval import train_model, eval_model

import torch.nn.functional as F
from torch.autograd import Variable


# Loading gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path = "/home/jeromefink/Documents/unamur/signLanguage/Data/most_frequents_25"
# path = "./mock-data"

params = {
    "batch_size": 5,
    "height": 224,
    "width": 224,
    "max_frames": 20,
    "epochs": 20,
    "lr": 0.001,
    "dataset": path.split("/")[-1],
    "cumulation": 15,
}

## Loading data and setup the batch loader
#
data = load_lsfb_dataset(path)

train = data[data["subset"] == "train"]
test = data[data["subset"] == "test"]

loader = I3DLoader(
    train,
    params["batch_size"],
    params["max_frames"],
    (params["height"], params["width"]),
)

val_loader = I3DLoader(
    test,
    params["batch_size"],
    params["max_frames"],
    (params["height"], params["width"]),
)


params["n_class"] = len(loader.get_label_mapping())

net = InceptionI3d(400, in_channels=3)
net.load_state_dict(torch.load("models/rgb_imagenet.pt"))
net.replace_logits(params["n_class"] + 1)
net = net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=params["lr"], momentum=0.9, weight_decay=0.0000001
)
lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


epoch = 0
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
            current_loader = loader
        else:
            net.train(False)  # Set model to evaluate mode
            current_loader = val_loader

        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        num_iter = 0
        accuracy = 0

        # Iterate over data.
        for idx in range(len(current_loader)):
            print(f"\rBatch : {idx+1} / {len(current_loader)}", end="\r")

            # get the inputs
            inputs, labels = current_loader.get_batch(idx)

            # wrap them in Variable
            inputs = inputs.to(device)
            t = inputs.size(2)
            labels = labels.to(device)

            per_frame_logits = net(inputs)
            # upsample to input size
            per_frame_logits = F.upsample(per_frame_logits, t, mode="linear")

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.data

            # Removing the padding
            ground_truth = torch.narrow(
                torch.max(labels, dim=2)[0], 1, 1, params["n_class"]
            )

            pred = torch.narrow(
                torch.max(per_frame_logits, dim=2)[0], 1, 1, params["n_class"]
            )

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

            tot_cls_loss += cls_loss.data

            pred = torch.max(pred, dim=1)[1]
            ground_truth = torch.max(ground_truth, dim=1)[1]

            accuracy += torch.sum(pred == ground_truth)

            loss = 0.2 * loc_loss + 0.8 * cls_loss
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

        if phase == "val":
            print(
                "{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accuracy: {:.4f}".format(
                    phase,
                    tot_loc_loss / len(current_loader),
                    tot_cls_loss / len(current_loader),
                    tot_loss / len(current_loader),
                    accuracy.double() / (len(current_loader) * params["batch_size"]),
                )
            )

