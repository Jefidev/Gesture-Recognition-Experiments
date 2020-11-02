from torch.utils.data import Dataset
from typing import Tuple, Dict
import torch
import cv2
import numpy as np


class LsfbDataset(Dataset):
    """ Load the LSFB video based on a dataframe containing their path"""

    def __init__(
        self,
        data,
        nbr_frames,
        padding=True,
        sequence_label=False,
        one_hot=False,
        transforms=None,
    ):
        """
        data : A pandas dataframe containing ...
        nbr_frames : Number of frames to sample per video
        padding : Ensure all video have same lenght by padding the shorter one.
        sequence_label : Return one label per video frame
        transforms : transformations to apply to the frames.
        """
        self.data = data
        self.nbr_frames = nbr_frames
        self.padding = padding
        self.sequence_label = sequence_label
        self.transforms = transforms
        self.one_hot = one_hot

        self.labels = self._get_label_mapping()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data.iloc[idx]

        capture = cv2.VideoCapture(item["path"])
        video = self.extract_frames(capture)
        video_len = len(video)
        y = item["label_nbr"]

        if self.padding and len(video) < self.nbr_frames:
            video = self._pad_sequence(video, self.nbr_frames)

        if self.sequence_label:
            # Retrieve the class number associated to the padding
            pad_nbr = list(self.labels.keys())[
                list(self.labels.values()).index("SEQUENCE-PADDING")
            ]

            pad_len = len(video) - video_len
            y = np.array([y] * video_len + [pad_nbr] * pad_len)

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

        if self.transforms:
            video = self.transforms(video)

        return video, y

    def _get_label_mapping(self) -> Dict[int, str]:
        labels = self.data.label.unique()

        mapping = {}

        for label in labels:
            subset = self.data[self.data["label"] == label]
            class_number = subset["label_nbr"].iloc[0]

            mapping[class_number] = label

        if self.padding and self.sequence_label:
            nbr_class = len(mapping)
            mapping[nbr_class] = "SEQUENCE-PADDING"

        return mapping

    def extract_frames(self, capture: cv2.VideoCapture):

        frame_array = []
        success, frame = capture.read()

        # Select
        while success:
            frame_array.append(frame / 255)
            success, frame = capture.read()

        if len(frame_array) > self.nbr_frames:
            # List envenly spaced indice
            resized_frame_array = []
            idx = np.round(
                np.linspace(0, len(frame_array) - 1, self.nbr_frames)
            ).astype(int)

            for i in idx:
                resized_frame_array.append(frame_array[i])

            frame_array = resized_frame_array

        return np.array(frame_array)

    def _pad_sequence(self, sequence, length):
        shape = sequence.shape
        new_shape = (length, shape[1], shape[2], shape[3])

        zero_arr = np.zeros(new_shape)
        zero_arr[: shape[0]] = sequence

        return zero_arr
