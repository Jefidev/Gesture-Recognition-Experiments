import pandas as pd
import numpy as np
from typing import Tuple
from batch_loaders.BaseLoader import BaseLoader
import torch
import cv2

# https://www.marktechpost.com/2020/04/12/implementing-batching-for-seq2seq-models-in-pytorch/


class RNNBatchLoader(BaseLoader):
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
        super(RNNBatchLoader, self).__init__(data, batch_size, frame_size, shuffle_data)
        self.nbr_frames = nbr_frames

    def get_batch(self, offset: int):
        start = offset * self.batch_size
        end = start + self.batch_size

        batch = self.data.iloc[start:end, :]

        max_frame = 0
        videos_array = []
        lengths = []
        nbr_label = len(self.get_label_mapping())
        y = torch.zeros((self.batch_size), dtype=torch.long)

        i = 0
        for index, row in batch.iterrows():
            # Reading video, should get and resize frame
            capture = cv2.VideoCapture(row["path"])
            frames = self.extract_frames(capture)
            videos_array.append(frames)
            lengths.append(len(frames))

            y[i] = row["label_nbr"]
            i += 1

        # Creating the torch tensor for the batch
        max_length = max(lengths)
        height, width = self.resolution
        X = torch.zeros(max_length, self.batch_size, 3, height, width)

        for vid_index, video in enumerate(videos_array):
            for frame_index, frame in enumerate(video):
                X[frame_index][vid_index] = torch.from_numpy(frame)

        # Using torch utility to handle the variable length of each data in the parser
        X_final = torch.nn.utils.rnn.pack_padded_sequence(
            X, lengths, enforce_sorted=False
        )

        return X_final, y

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

