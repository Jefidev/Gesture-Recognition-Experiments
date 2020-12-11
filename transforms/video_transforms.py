import numbers
import random
import numpy as np
import PIL
import skimage.transform
import torchvision
import math
import torch

import utils.video_utils as F


class I3DPixelsValue(object):
    """
    Scale the pixel value between -1 and 1 insted of 0 and 1 (required for I3D)
    """

    def __call__(self, sample):
        return sample * 2 - 1


class ChangeVideoShape(object):
    """
    Expect to receive a ndarray of chape (Time, Height, Width, Channel) which is the default format
    of cv2 or PIL. Change the shape of the ndarray to TCHW or CTHW.
    """

    def __init__(self, shape: str):
        """
        shape : a string with the value "CTHW" or "TCHW".
        """

        self.shape = shape

    def __call__(self, sample):

        if self.shape == "CTHW":
            sample = np.transpose(sample, (3, 0, 1, 2))
        elif self.shape == "TCHW":
            sample = np.transpose(sample, (0, 3, 1, 2))
        else:
            raise ValueError(f"Received {self.shape}. Expecting TCHW or CTHW.")

        return sample


class ResizeVideo(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation="nearest"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = F.resize_clip(clip, self.size, interpolation=self.interpolation)
        return np.array(resized)


class RandomCropVideo(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )
        if w > im_w or h > im_h:
            error_msg = (
                "Initial image size should be larger then "
                "cropped size but got cropped sizes : ({w}, {h}) while "
                "initial image is ({im_w}, {im_h})".format(
                    im_w=im_w, im_h=im_h, w=w, h=h
                )
            )
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return np.array(cropped)


class CenterCropVideo(object):
    """Extract center crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )
        if w > im_w or h > im_h:
            error_msg = (
                "Initial image size should be larger then "
                "cropped size but got cropped sizes : ({w}, {h}) while "
                "initial image is ({im_w}, {im_h})".format(
                    im_w=im_w, im_h=im_h, w=w, h=h
                )
            )
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.0))
        y1 = int(round((im_h - h) / 2.0))
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return np.array(cropped)


class TrimVideo(object):
    """Trim each video the same way. Waiting shape TCHW
    """

    def __init__(self, size, offset=None):
        self.end = size
        self.begin = 0

        if self.offset != None:
            self.begin = offset
            self.end += offset

    def __call__(self, clip):
        resized = clip[self.beging : self.end]
        return np.array(resized)


class RandomTrimVideo(object):
    """Trim randomly the video. Waiting shape TCHW
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        resized = clip

        if len(clip) > self.size:
            diff = len(resized) - self.size

            start = random.randint(0, diff)
            end = start + self.size

            resized = resized[start:end]

        return np.array(resized)


class PadVideo(object):
    def __init__(self, size, loop=True):
        self.size = size
        self.loop = loop

    def __call__(self, clip):
        if self.loop:
            resized = self._loop_sequence(clip, self.size)
        else:
            resized = self._pad_sequence(clip, self.size)

        return np.array(resized)

    def _pad_sequence(self, sequence, length):
        shape = sequence.shape
        new_shape = (length, shape[1], shape[2], shape[3])

        zero_arr = np.zeros(new_shape)
        zero_arr[: shape[0]] = sequence

        return zero_arr

    def _loop_sequence(self, sequence, length):
        shape = sequence.shape
        new_shape = (length, shape[1], shape[2], shape[3])
        zero_arr = np.zeros(new_shape)

        video_len = len(sequence)

        for i in range(length):
            vid_idx = i % video_len
            zero_arr[i] = sequence[vid_idx]

        return zero_arr

