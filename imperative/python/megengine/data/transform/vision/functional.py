# -*- coding: utf-8 -*-
import collections.abc
import functools
import random

import cv2
import numpy as np


def wrap_keepdims(func):
    r"""Wraper to keep the dimension of input images unchanged."""

    @functools.wraps(func)
    def wrapper(image, *args, **kwargs):
        if len(image.shape) != 3:
            raise ValueError(
                "image must have 3 dims, but got {} dims".format(len(image.shape))
            )
        ret = func(image, *args, **kwargs)
        if len(ret.shape) == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    return wrapper


@wrap_keepdims
def to_gray(image):
    r"""Change BGR format image's color space to gray.

    Args:
        image: input BGR format image, with `(H, W, C)` shape.

    Returns:
        gray format image, with `(H, W, C)` shape.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


@wrap_keepdims
def to_bgr(image):
    r"""Change gray format image's color space to BGR.

    Args:
        image: input Gray format image, with `(H, W, C)` shape.

    Returns:
        BGR format image, with `(H, W, C)` shape.
    """
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


@wrap_keepdims
def pad(input, size, value):
    r"""Pad input data with *value* and given *size*.

    Args:
        input: input data, with `(H, W, C)` shape.
        size: padding size of input data, it could be integer or sequence.
            If it is an integer, the input data will be padded in four directions.
            If it is a sequence contains two integer, the bottom and right side
            of input data will be padded.
            If it is a sequence contains four integer, the top, bottom, left, right
            side of input data will be padded with given size.
        value: padding value of data, could be a sequence of int or float.
            If it is float value, the dtype of image will be casted to float32 also.

    Returns:
        padded image.
    """
    if isinstance(size, int):
        size = (size, size, size, size)
    elif isinstance(size, collections.abc.Sequence) and len(size) == 2:
        size = (0, size[0], 0, size[1])
    if np.array(value).dtype == float:
        input = input.astype(np.float32)
    return cv2.copyMakeBorder(input, *size, cv2.BORDER_CONSTANT, value=value)


@wrap_keepdims
def flip(image, flipCode):
    r"""Accordding to the flipCode (the type of flip), flip the input image.

    Args:
        image: input image, with `(H, W, C)` shape.
        flipCode: code that indicates the type of flip.

            * 1 : Flip horizontally
            * 0 : Flip vertically
            * -1: Flip horizontally and vertically

    Returns:
        BGR format image, with `(H, W, C)` shape.
    """
    return cv2.flip(image, flipCode=flipCode)


@wrap_keepdims
def resize(input, size, interpolation=cv2.INTER_LINEAR):
    r"""Resize the input data to given size.

    Args:
        input: input data, could be image or masks, with `(H, W, C)` shape.
        size: target size of input data, with (height, width) shape.
        interpolation: interpolation method.

    Returns:
        resized data, with `(H, W, C)` shape.
    """
    if len(size) != 2:
        raise ValueError("resize needs (h, w), but got {}".format(size))

    if isinstance(interpolation, collections.abc.Sequence):
        interpolation = random.choice(interpolation)
    return cv2.resize(input, size[::-1], interpolation=interpolation)
