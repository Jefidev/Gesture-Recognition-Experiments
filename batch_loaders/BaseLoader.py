from typing import Tuple, Dict
from math import floor
import pandas as pd
from sklearn.utils import shuffle


class BaseLoader:
    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int,
        frame_size: Tuple[int, int],
        shuffle_data: bool = True,
    ):
        """
      frame_size (height, width)
    """

        if shuffle_data:
            self.data = shuffle(data)
            self.data.reset_index()
        else:
            self.data = data

        self.batch_size = batch_size
        self.resolution = frame_size

    def get_label_mapping(self) -> Dict[int, str]:
        labels = self.data.label.unique()
        mapping = {}

        for label in labels:
            subset = self.data[self.data["label"] == label]
            class_number = subset["label_nbr"].iloc[0]

            mapping[class_number] = label

        return mapping

    def __len__(self):
        nbr_data = len(self.data)
        return floor(nbr_data / self.batch_size)

    def get_batch(self, offset: int):
        raise NotImplementedError
