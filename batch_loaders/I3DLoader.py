from batch_loaders.BaseLoader import BaseLoader
from typing import Tuple
import cv2
import torch
import numpy as np
import pandas as pd


class I3DLoader(BaseLoader):
    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int,
        nbr_frames: int,
        frame_size: Tuple[int, int],
        shuffle_data: bool = True,
    ):
        """
        frame_size (height, width)
        """
        super(I3DLoader, self).__init__(data, batch_size, frame_size, shuffle_data)
        self.nbr_frames = nbr_frames
        self.nbr_class = len(self.get_label_mapping())

    # I3D expected input : (B x C x T x H x W)
    def get_batch(self, offset: int):
        start = offset * self.batch_size
        end = start + self.batch_size

        batch = self.data.iloc[start:end, :]

        size = []
        X = torch.zeros(
            self.batch_size, 3, self.nbr_frames, self.resolution[0], self.resolution[1]
        )

        # Adding one class for the frame without labels (the padding)
        y = torch.zeros(
            (self.batch_size, self.nbr_class + 1, self.nbr_frames), dtype=torch.float
        )

        # Fill
        batch_elem_idx = 0
        for index, row in batch.iterrows():
            # Reading video, should get and resize frame

            capture = cv2.VideoCapture(row["path"])
            frames = self.extract_frames(capture)

            nb_frame = 0
            for f_idx, frame in enumerate(frames):
                X[batch_elem_idx][0][f_idx] = torch.from_numpy(frame[0])
                X[batch_elem_idx][1][f_idx] = torch.from_numpy(frame[1])
                X[batch_elem_idx][2][f_idx] = torch.from_numpy(frame[2])

                y[batch_elem_idx, row["label_nbr"] + 1, f_idx] = 1

            ## Adding the label for the padded frames
            for idx in range(len(frames), self.nbr_frames):
                y[batch_elem_idx, 0, idx] = 1

            batch_elem_idx += 1

        return X, y

    def extract_frames(self, capture: cv2.VideoCapture):

        frame_array = []
        success, frame = capture.read()

        # Select
        while success:
            new_size = (self.resolution[1], self.resolution[0])
            resized = cv2.resize(frame, new_size)
            rearanged = np.transpose(resized, (2, 0, 1)) / 255
            frame_array.append(rearanged)

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

        return frame_array

