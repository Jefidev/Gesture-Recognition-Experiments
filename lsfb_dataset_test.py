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
)
from torch.utils.data import DataLoader

path = "/home/jeromefink/Documents/unamur/signLanguage/Data/MS-ASL/MSASL"
data = load_lsfb_dataset(path)

composed = Compose(
    [
        ResizeVideo(256, interpolation="bilinear"),
        CenterCropVideo((224, 224)),
        I3DPixelsValue(),
    ]
)

lsfb_dataset = LsfbDataset(
    data, 60, sequence_label=True, transforms=composed, one_hot=True, padding="loop",
)


def write_video(video):
    i = random.randint(1, 10000)
    name = f"{path}/{i}.avi"
    fourc = cv2.VideoWriter_fourcc(*"DIVX")

    s = (video.shape[2], video.shape[1])
    out = cv2.VideoWriter(name, fourc, 25.0, s)

    for frame in video:
        image = frame * 255
        r, g, b = cv2.split(image)
        image = cv2.merge([b, g, r])
        out.write(image.astype(np.uint8))
    out.release()


print(lsfb_dataset.labels)
for i in range(len(lsfb_dataset)):
    sequence = lsfb_dataset[i]
    print(sequence[0])

