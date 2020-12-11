from datasets.lsfb_dataset import LsfbDataset
from utils.lsfb_dataset_loader import load_lsfb_dataset
from torchvision.transforms import Compose
import numpy as np
import cv2
import random
from transforms.video_transforms import (
    ChangeVideoShape,
    ResizeVideo,
    RandomCropVideo,
    CenterCropVideo,
    I3DPixelsValue,
    RandomTrimVideo,
    PadVideo,
)
from torch.utils.data import DataLoader
import pandas as pd


path = "./test-video"

d = {
    "label": ["U"],
    "label_nbr": [0],
    "path": ["./test-video/t1.mp4"],
    "subset": ["test"],
}
data = pd.DataFrame(data=d)

composed = Compose(
    [
        RandomTrimVideo(48),
        PadVideo(48),
        ResizeVideo(280, interpolation="bilinear"),
        RandomCropVideo((224, 224)),
    ]
)

lsfb_dataset = LsfbDataset(
    data, sequence_label=True, transforms=composed, one_hot=True, padding="loop",
)


def write_video(video, idx):
    name = f"{path}/randomcrop2_{idx}.avi"
    fourc = cv2.VideoWriter_fourcc(*"DIVX")

    s = (video.shape[2], video.shape[1])
    out = cv2.VideoWriter(name, fourc, 25.0, s)

    for frame in video:
        image = frame * 255
        r, g, b = cv2.split(image)
        image = cv2.merge([b, g, r])
        out.write(image.astype(np.uint8))
    out.release()


for j in range(0, 5):
    for i in range(len(lsfb_dataset)):
        sequence = lsfb_dataset[i]
        write_video(sequence[0], j)

