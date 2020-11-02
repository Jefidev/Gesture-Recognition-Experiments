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
)
from torch.utils.data import DataLoader

path = "./mock-data"

composed = Compose(
    [ResizeVideo(256, interpolation="bilinear"), CenterCropVideo((224, 224))]
)
data = load_lsfb_dataset(path)
lsfb_dataset = LsfbDataset(
    data, 30, sequence_label=True, transforms=composed, one_hot=True
)


def write_video(video):
    i = random.randint(1, 10000)
    name = f"{path}/{i}.avi"
    fourc = cv2.VideoWriter_fourcc(*"DIVX")

    s = (video.shape[2], video.shape[1])
    out = cv2.VideoWriter(name, fourc, 25.0, s)

    for frame in video:
        image = frame * 255
        out.write(image.astype(np.uint8))
    out.release()


print(lsfb_dataset._get_label_mapping())

dataloader = DataLoader(lsfb_dataset, 4, shuffle=True)

for i in range(len(lsfb_dataset)):
    sequence = lsfb_dataset[i]
    # print(sequence[1])

    # write_video(sequence[0])

    if i > 5:
        break


for i, val in enumerate(dataloader):
    input, label = val
    print(input.size())
    print(label.size())
    print(i)

