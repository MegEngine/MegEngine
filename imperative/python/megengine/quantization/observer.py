# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math
from abc import abstractmethod

import numpy as np

from .. import functional as F
from ..core.tensor.dtype import _metadata_dict, get_quantized_dtype
from ..distributed import WORLD, get_rank, is_distributed
from ..functional.distributed import all_reduce_max, all_reduce_min
from ..module import Module
from ..tensor import Tensor
from .utils import QuantMode, Round, get_qparam_dict


class Observer(Module):
    r"""
    A base class for Observer Module.

    :param dtype: a string indicating to collect scale and zero_point of which dtype.
    :param narrow_range: whether the absolute value of ``qmin`` is the same as ``qmax``,
        instead of 1 greater. Usually True for weight and False for activation.
    """

    def __init__(self, dtype: str, narrow_range: bool = False):
        super().__init__()
        if dtype not in _metadata_dict.keys():
            raise ValueError(
                "unknown dtype: {}, only support {}".format(
                    dtype, _metadata_dict.keys()
                )
            )
        self.dtype = dtype
        self.narrow_range = narrow_range
        self.qmin = (
            -_metadata_dict[dtype].qmax if narrow_range else _metadata_dict[dtype].qmin
        )
        self.qmax = _metadata_dict[dtype].qmax
        self.enabled = True

    def get_dtype(self):
        q_dict = self.get_qparams()
        numpy_scale = None if "scale" not in q_dict else q_dict["scale"].numpy()
        numpy_zero_point = (
            None if "zero_point" not in q_dict else q_dict["zero_point"].numpy()
        )
        return get_quantized_dtype(self.dtype, numpy_scale, numpy_zero_point)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def train(self, mode: bool = True, recursive: bool = True) -> None:
        super().train(mode, recursive)
        if mode:
            self.enable()
        else:
            self.disable()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_qparams(self, **kwargs):
        pass


class MinMaxObserver(Observer):
    def __init__(
        self,
        mode=QuantMode.SYMMERTIC,
        eps=0.00001,
        dtype="qint8",
        narrow_range: bool = False,
    ):
        super().__init__(dtype, narrow_range)
        self.mode = mode
        self.min_val = Tensor(np.finfo(np.float32).max, dtype=np.float32)
        self.max_val = Tensor(np.finfo(np.float32).min, dtype=np.float32)
        self.scale_limit = eps

    def _calculate_qparams(self, inp_min_val, inp_max_val):
        min_val = F.minimum(0.0, inp_min_val)
        max_val = F.maximum(0.0, inp_max_val)
        q_dict = get_qparam_dict(self.mode)
        q_dict["min_val"] = inp_min_val
        q_dict["max_val"] = inp_max_val
        q_dict["enable_observer"] = self.enable
        if self.mode == QuantMode.SYMMERTIC:
            symmetric_max_vals = F.maximum(-min_val, max_val)
            # use maximun to avoid scale too small at the begin
            q_dict["scale"] = F.maximum(
                symmetric_max_vals / ((self.qmax - self.qmin) / 2), self.scale_limit
            )
            # zero_point = self.zero_point
        else:
            # use maximun to avoid scale too small at the begin
            q_dict["scale"] = F.maximum(
                (max_val - min_val) / (self.qmax - self.qmin), self.scale_limit,
            )
            # caculate zero_point
            q_dict["zero_point"] = self.qmin - Round()((min_val / q_dict["scale"]))

        return q_dict

    def get_qparams(self):
        return self._calculate_qparams(self.min_val, self.max_val)

    def forward(self, x_orig):
        if self.enabled:
            # stop gradient
            x = x_orig.detach()
            # find max and min
            self.min_val._reset(F.minimum(self.min_val, x.min()))
            self.max_val._reset(F.maximum(self.max_val, x.max()))
        return x_orig


class SyncMinMaxObserver(MinMaxObserver):
    def forward(self, x_orig):
        if self.enable:
            x = x_orig.detach()
            if is_distributed():
                min_x = all_reduce_min(x.min(), WORLD)
                max_x = all_reduce_max(x.max(), WORLD)
            else:
                min_x = x.min()
                max_x = x.max()
            self.min_val._reset(F.minimum(self.min_val, min_x))
            self.max_val._reset(F.maximum(self.max_val, max_x))
        return x_orig


class ExponentialMovingAverageObserver(MinMaxObserver):
    def __init__(
        self,
        momentum=0.9,
        mode=QuantMode.SYMMERTIC,
        eps=0.00001,
        dtype="qint8",
        narrow_range: bool = False,
    ):
        super().__init__(mode, eps, dtype, narrow_range)
        self.momentum = Tensor(momentum)
        self.runtime_momentum = Tensor(0.0)

    def set_momentum(self, momentum):
        self.momentum._reset(momentum)

    def forward(self, x_orig):
        if self.enabled:
            # stop gradient
            x = x_orig.detach()
            # Exponential Moving Average
            self.min_val._reset(
                self.min_val * self.runtime_momentum
                + (1 - self.runtime_momentum) * x.min()
            )
            self.max_val._reset(
                self.max_val * self.runtime_momentum
                + (1 - self.runtime_momentum) * x.max()
            )
            self.runtime_momentum = self.momentum

        return x_orig


class SyncExponentialMovingAverageObserver(ExponentialMovingAverageObserver):
    def forward(self, x_orig):
        if self.enabled:
            x = x_orig.detach()
            if is_distributed:
                min_x = all_reduce_min(x.min(), WORLD)
                max_x = all_reduce_max(x.max(), WORLD)
            else:
                min_x = x.min()
                max_x = x.max()
            self.min_val._reset(
                self.min_val * self.runtime_momentum
                + (1 - self.runtime_momentum) * min_x
            )
            self.max_val._reset(
                self.max_val * self.runtime_momentum
                + (1 - self.runtime_momentum) * max_x
            )
            self.runtime_momentum = self.momentum
        return x_orig


class HistogramObserver(MinMaxObserver):
    def __init__(
        self,
        bins=2048,
        upsample_rate=128,
        mode=QuantMode.SYMMERTIC,
        eps=0.00001,
        dtype="qint8",
        narrow_range: bool = False,
    ):
        super().__init__(mode, eps, dtype, narrow_range)
        self.bins = bins
        self.upsample_rate = upsample_rate
        self.dst_nbins = _metadata_dict[dtype].qmax - _metadata_dict[dtype].qmin + 1
        self.histogram = Tensor([-1] + [0.0] * (bins - 1), dtype="float32")

    def _non_linear_param_search(self):
        r"""
        Non-linear parameter search.
        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        """

        np_min_val = self.min_val.numpy()
        np_max_val = self.max_val.numpy()
        np_histogram = self.histogram.numpy()
        assert len(np_histogram) == self.bins, "bins mistmatch"
        bin_width = (np_max_val - np_min_val) / self.bins

        def _get_norm(delta_begin, delta_end, density, norm_type):
            r"""
            Compute the norm of the values uniformaly distributed between
            delta_begin and delta_end.
            norm = density * (integral_{begin, end} x^2)
                 = density * (end^3 - begin^3) / 3
            """
            assert norm_type == "L2", "Only L2 norms are currently supported"
            norm = 0.0
            if norm_type == "L2":
                norm = (
                    delta_end * delta_end * delta_end
                    - delta_begin * delta_begin * delta_begin
                ) / 3
            return density * norm

        def _compute_quantization_error(next_start_bin, next_end_bin, norm_type):
            r"""
            Compute the quantization error if we use start_bin to end_bin as the
            min and max to do the quantization.
            """

            norm = 0.0
            dst_bin_width = (
                bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
            )
            if dst_bin_width == 0.0:
                return 0.0
            for src_bin in range(self.bins):
                # distances from the beginning of first dst_bin to the beginning and
                # end of src_bin
                src_bin_begin = (src_bin - next_start_bin) * bin_width
                src_bin_end = src_bin_begin + bin_width

                # which dst_bins the beginning and end of src_bin belong to?
                dst_bin_of_begin = min(
                    self.dst_nbins - 1,
                    max(0.0, math.floor(src_bin_begin / dst_bin_width)),
                )
                dst_bin_of_end = min(
                    self.dst_nbins - 1,
                    max(0.0, math.floor(src_bin_end / dst_bin_width)),
                )
                dst_bin_of_begin_center = (
                    dst_bin_of_begin * dst_bin_width + dst_bin_width / 2
                )

                density = np_histogram[src_bin] / bin_width
                if dst_bin_of_begin == dst_bin_of_end:
                    # if src_bin is entirely within 1 dst_bin
                    delta_begin = src_bin_begin - dst_bin_of_begin_center
                    delta_end = src_bin_end - dst_bin_of_begin_center
                    norm = norm + _get_norm(delta_begin, delta_end, density, norm_type)
                else:
                    delta_begin = src_bin_begin - dst_bin_of_begin_center
                    delta_end = dst_bin_width / 2
                    norm = norm + _get_norm(delta_begin, delta_end, density, norm_type)

                    norm = norm + (dst_bin_of_end - dst_bin_of_begin - 1) * _get_norm(
                        -dst_bin_width / 2, dst_bin_width / 2, density, norm_type
                    )

                    dst_bin_of_end_center = (
                        dst_bin_of_end * dst_bin_width + dst_bin_width / 2
                    )

                    delta_begin = -dst_bin_width / 2
                    delta_end = src_bin_end - dst_bin_of_end_center
                    norm = norm + _get_norm(delta_begin, delta_end, density, norm_type)
            return norm

        # cumulative sum
        total = sum(np_histogram)
        cSum = np.cumsum(np_histogram, axis=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = _compute_quantization_error(next_start_bin, next_end_bin, "L2")

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + Tensor(bin_width * start_bin, dtype=np.float32)
        new_max = self.min_val + Tensor(bin_width * (end_bin + 1), dtype=np.float32)
        return new_min, new_max

    def get_qparams(self):
        new_min, new_max = self._non_linear_param_search()
        return self._calculate_qparams(new_min, new_max)

    def _combine_histograms(
        self, orig_hist, new_hist, upsample_rate, downsample_rate, start_idx, Nbins
    ):
        # First up-sample the histogram with new data by a factor of L
        # This creates an approximate probability density thats piecwise constant
        upsampled_histogram = new_hist.repeat(upsample_rate)
        # Now insert the upsampled histogram into the output
        # histogram, which is initialized with zeros.
        # The offset at which the histogram is introduced is determined
        # by the start index as the output histogram can cover a wider range
        histogram_with_output_range = np.zeros((Nbins * downsample_rate))
        histogram_with_output_range[
            start_idx : Nbins * upsample_rate + start_idx
        ] = upsampled_histogram
        # Compute integral histogram, double precision is needed to ensure
        # that there are no overflows
        integral_histogram = np.cumsum(histogram_with_output_range, 0)[
            downsample_rate - 1 :: downsample_rate
        ]
        # Finally perform interpolation
        shifted_integral_histogram = np.zeros((Nbins))
        shifted_integral_histogram[1:Nbins] = integral_histogram[0:-1]
        interpolated_histogram = (
            integral_histogram - shifted_integral_histogram
        ) / upsample_rate
        orig_hist = orig_hist + interpolated_histogram
        return orig_hist

    def _adjust_min_max(self, combined_min, combined_max, upsample_rate):
        # We ensure that:
        # (combined_max - combined_min)/(downsample_rate*Nbins) = (max - min)/(upsample_rate*Nbins)
        # This allows us to have a common grid of resolution s, where we can align
        # the input histogram
        # start_idx maps min_val to the histogram bin index.
        np_min_val = self.min_val.numpy()
        np_max_val = self.max_val.numpy()

        hist_bin_width = (np_max_val - np_min_val) / (self.bins * upsample_rate)
        downsample_rate = int(
            np.ceil((combined_max - combined_min) / (self.bins * hist_bin_width))
        )
        e = downsample_rate * (self.bins * hist_bin_width) - (
            combined_max - combined_min
        )
        combined_max = combined_max + e / 2
        combined_min = combined_min - e / 2
        start_idx = int(np.round((np_min_val - combined_min) / hist_bin_width))

        return combined_min, combined_max, downsample_rate, start_idx

    def sideeffect_forward(self, x_orig):
        x = x_orig.numpy()
        min_val = self.min_val.numpy()
        max_val = self.max_val.numpy()
        histogram = self.histogram.numpy()
        new_min = x.min()
        new_max = x.max()
        if histogram[0] == -1:
            new_histogram, _ = np.histogram(x, self.bins, (new_min, new_max))
        else:
            new_min = min(new_min, min_val)
            new_max = max(new_max, max_val)
            # combine the existing histogram and new histogram into 1 histogram
            # We do this by first upsampling the histogram to a dense grid
            # and then downsampling the histogram efficiently
            (new_min, new_max, downsample_rate, start_idx,) = self._adjust_min_max(
                new_min, new_max, self.upsample_rate
            )

            new_histogram, _ = np.histogram(x, self.bins, (new_min, new_max))
            new_histogram = new_histogram.astype(np.float64)
            if new_min == min_val and new_max == max_val:
                new_histogram += histogram
            else:
                new_histogram = self._combine_histograms(
                    new_histogram,
                    histogram,
                    self.upsample_rate,
                    downsample_rate,
                    start_idx,
                    self.bins,
                )

        self.histogram._reset(new_histogram)
        self.min_val._reset(new_min)
        self.max_val._reset(new_max)

    def forward(self, x_orig):
        self.sideeffect_forward(x_orig)
        return x_orig
