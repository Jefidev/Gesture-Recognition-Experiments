from torch.utils.data import Dataset
from typing import Tuple, Dict
import torch
import cv2
import numpy as np
import random


class LsfbDataset(Dataset):
    """ Load the LSFB video based on a dataframe containing their path"""

    def __init__(
        self,
        data,
        padding="loop",
        sequence_label=False,
        one_hot=False,
        transforms=None,
        labels=None,
    ):
        """
        data : A pandas dataframe containing ...
        nbr_frames : Number of frames to sample per video
        padding : Ensure all video have same lenght. To value possible. 
                  zero for zero padding or loop to loop the video.
        sequence_label : Return one label per video frame
        transforms : transformations to apply to the frames.
        """
        self.data = data
        self.padding = padding
        self.sequence_label = sequence_label
        self.transforms = transforms
        self.one_hot = one_hot

        if labels == None:
            self.labels = self._get_label_mapping()
        else:
            self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data.iloc[idx]

        capture = cv2.VideoCapture(item["path"])
        video = self.extract_frames(capture)
        video_len = len(video)

        # Apply the transformations to the video :
        if self.transforms:
            video = self.transforms(video)

        # If the video was trimmed and not padded
        if len(video) < video_len:
            video_len = len(video)

        y = item["label_nbr"]

        if self.sequence_label:

            # Retrieve the class number associated to the padding
            if self.padding == "zero":
                pad_nbr = list(self.labels.keys())[
                    list(self.labels.values()).index("SEQUENCE-PADDING")
                ]
                pad_len = len(video) - video_len
                y = np.array([y] * video_len + [pad_nbr] * pad_len)

            elif self.padding == "loop":
                y = np.array([y] * len(video))

        if self.one_hot:
            nbr_labels = len(self.labels)
            if isinstance(y, int):
                tmp = np.zeros(nbr_labels)
                tmp[y] = 1
            else:
                tmp = np.zeros((nbr_labels, len(video)))
                for idx, label in enumerate(y):
                    tmp[label][idx] = 1
            y = tmp

        return video, y

    def _get_label_mapping(self) -> Dict[int, str]:
        labels = self.data.label.unique()

        mapping = {}

        for label in labels:
            subset = self.data[self.data["label"] == label]
            class_number = subset["label_nbr"].iloc[0]

            mapping[class_number] = label

        if self.padding == "zero" and self.sequence_label:
            nbr_class = len(mapping)
            mapping[nbr_class] = "SEQUENCE-PADDING"

        return mapping

    def extract_frames(self, capture: cv2.VideoCapture):

        frame_array = []
        success, frame = capture.read()

        # Select
        frame_count = 0
        while success:
            frame_count += 1

            b, g, r = cv2.split(frame)
            frame = cv2.merge([r, g, b])
            frame_array.append(frame / 255)
            success, frame = capture.read()

            # Avoid memory saturation by stopping reading of
            # video if it is > 150 frames (5sec)
            if frame_count > 150:
                break

        return np.array(frame_array)
