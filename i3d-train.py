import os
import json


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
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


init_lr = 0.01
max_steps = 64e3
mode = "rgb_msasl"
batch_size = 4
save_model = ""


path = "/home/jeromefink/Documents/unamur/signLanguage/Data/MS-ASL/MSASL"
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
    64,
    sequence_label=True,
    transforms=composed_train,
    one_hot=True,
    padding="loop",
)

test_dataset = LsfbDataset(
    test,
    64,
    sequence_label=True,
    transforms=compose_test,
    one_hot=True,
    padding="loop",
)

dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=3
)

val_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=2, shuffle=True, num_workers=2,
)

dataloaders = {"train": dataloader, "val": val_dataloader}
datasets = {"train": train_dataset, "val": test_dataset}

# setup the model
if mode == "flow":
    i3d = InceptionI3d(400, in_channels=2)
    i3d.load_state_dict(torch.load("checkpoints/flow_imagenet.pt"))
    i3d.replace_logits(100)
    print("Flow kinetic loaded")
elif mode == "rgb_msasl":
    i3d = InceptionI3d(100, in_channels=3)
    i3d.load_state_dict(torch.load("checkpoints/MSASL.pt"))
    print("MSASL loaded")
elif mode == "charades":
    i3d = InceptionI3d(157, in_channels=3)
    i3d.load_state_dict(torch.load("checkpoints/rgb_charades.pt"))
    i3d.replace_logits(100)
    print("RGB Charade loaded")
else:
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load("checkpoints/rgb_imagenet.pt"))
    i3d.replace_logits(100)
    print("RGB kinetic loaded")

i3d.cuda()
i3d = nn.DataParallel(i3d)

lr = 0.01
optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
criterion = nn.CrossEntropyLoss()

num_steps_per_update = 64  # accum gradient
steps = 0
# train it
while steps < max_steps:  # for epoch in range(num_epochs):
    print("Step {}/{}".format(steps, max_steps))
    print("-" * 10)

    # Each epoch has a training and validation phase
    for phase in ["train", "val"]:
        if phase == "train":
            i3d.train(True)
        else:
            i3d.train(False)  # Set model to evaluate mode

        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        accuracy = 0.0
        num_iter = 0
        counter = 0

        # Iterate over data.
        size = len(dataloaders[phase])
        for data in dataloaders[phase]:
            num_iter += 1
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

            # compute localization loss
            """
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.item()
            """

            # compute classification loss (with max-pooling along time B x C x T)
            """
            cls_loss = F.binary_cross_entropy_with_logits(
                torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0]
            )
            tot_cls_loss += cls_loss.item()
            """
            # loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update

            tmp = torch.max(labels, dim=2)[0]
            loss = (
                criterion(
                    torch.max(per_frame_logits, dim=2)[0], torch.max(tmp, dim=1)[1]
                )
                / num_steps_per_update
            )
            tot_loss += loss.item()

            if phase == "train":
                loss.backward()

            tmp = torch.max(per_frame_logits, dim=2)[0]
            majority_pred = torch.max(tmp, dim=1)[1]

            tmp = torch.max(labels, dim=2)[0]
            majority_truth = torch.max(tmp, dim=1)[1]

            accuracy += torch.sum(majority_pred == majority_truth).item()

            if num_iter == num_steps_per_update and phase == "train":
                steps += 1
                num_iter = 0
                optimizer.step()
                optimizer.zero_grad()
                lr_sched.step()
                if steps % 2 == 0:
                    print(
                        "{} Loss: {:.4f}  Accuracy: {:.4f}".format(
                            phase,
                            tot_loss / 2,
                            accuracy / (2 * num_steps_per_update * batch_size),
                        )
                    )
                    # save model

                    torch.save(
                        i3d.module.state_dict(), "MSASL.pt",
                    )

                    accuracy = tot_loss = 0.0

        if phase == "val":
            print(
                "{}  Loss: {:.4f}  Accuracy: {:.4f}".format(
                    phase,
                    (tot_loss * num_steps_per_update) / num_iter,
                    accuracy / (num_iter * batch_size),
                )
            )
        elif phase == "train":
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            lr_sched.step()

