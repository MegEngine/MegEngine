/**
 * \file dnn/src/naive/lowbit_utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/naive/lowbit_utils.h"

// =================================quint4======================================
void megdnn::naive::uint4_to_uint8(const TensorND& in, const TensorND& out) {
    auto in_ptr = static_cast<uint8_t*>(in.raw_ptr()) + in.layout.span().low_byte;
    auto out_ptr = out.compatible_ptr<uint8_t>() + out.layout.span().low_byte;
    const auto& ly = in.layout;
    auto dim_in = ly.shape[ly.ndim - 1];
    auto elems = ly.total_nr_elems();
    auto dim_out = elems / dim_in;
    auto stride_out = div_ceil(dim_in, 2_z);
    for (size_t i = 0; i < dim_out; ++i) {
        for (size_t j = 0; j < dim_in; j += 2) {
            uint8_t val = in_ptr[j / 2];
            out_ptr[j] = val & 0xF;
            if (j + 1 < dim_in)
                out_ptr[j + 1] = (val >> 4) & 0xF;
        }
        in_ptr += stride_out;
        out_ptr += dim_in;
    }
}

void megdnn::naive::uint8_to_uint4(const TensorND& in, const TensorND& out) {
    auto in_ptr = static_cast<uint8_t*>(in.raw_ptr()) + in.layout.span().low_byte;
    auto out_ptr = static_cast<uint8_t*>(out.raw_ptr()) + out.layout.span().low_byte;
    const auto& ly = in.layout;
    auto dim_in = ly.shape[ly.ndim - 1];
    auto elems = ly.total_nr_elems();
    auto dim_out = elems / dim_in;
    auto stride_out = div_ceil(dim_in, 2_z);
    for (size_t i = 0; i < dim_out; ++i) {
        for (size_t j = 0; j < dim_in; j += 2) {
            uint8_t a = in_ptr[j];
            uint8_t b = 0;
            if (j + 1 < dim_in)
                b = in_ptr[j + 1];
            a = std::min(a, DTypeTrait<dtype::Quantized4Asymm>::max());
            b = std::min(b, DTypeTrait<dtype::Quantized4Asymm>::max());
            out_ptr[j / 2] = a + (b << 4);
        }
        in_ptr += dim_in;
        out_ptr += stride_out;
    }
}

void megdnn::naive::uint4_to_int8(const TensorND& in, const TensorND& out) {
    auto in_ptr = static_cast<uint8_t*>(in.raw_ptr()) + in.layout.span().low_byte;
    auto out_ptr = out.compatible_ptr<int8_t>() + out.layout.span().low_byte;
    const auto& ly = in.layout;
    int8_t zero_point = (int8_t)ly.dtype.param<dtype::Quantized4Asymm>().zero_point;
    auto dim_in = ly.shape[ly.ndim - 1];
    auto elems = ly.total_nr_elems();
    auto dim_out = elems / dim_in;
    auto stride_out = div_ceil(dim_in, 2_z);
    for (size_t i = 0; i < dim_out; ++i) {
        for (size_t j = 0; j < dim_in; j += 2) {
            uint8_t val = in_ptr[j / 2];
            out_ptr[j] = (int8_t)(val & 0xF) - zero_point;
            if (j + 1 < dim_in)
                out_ptr[j + 1] = (int8_t)((val >> 4) & 0xF) - zero_point;
        }
        in_ptr += stride_out;
        out_ptr += dim_in;
    }
}

void megdnn::naive::int8_to_uint4(const TensorND& in, const TensorND& out) {
    auto in_ptr = static_cast<int8_t*>(in.raw_ptr()) + in.layout.span().low_byte;
    auto out_ptr = static_cast<uint8_t*>(out.raw_ptr()) + out.layout.span().low_byte;
    auto zero_point = out.layout.dtype.param<dtype::Quantized4Asymm>().zero_point;
    const auto& ly = in.layout;
    auto dim_in = ly.shape[ly.ndim - 1];
    auto elems = ly.total_nr_elems();
    auto dim_out = elems / dim_in;
    auto stride_out = div_ceil(dim_in, 2_z);
    for (size_t i = 0; i < dim_out; ++i) {
        for (size_t j = 0; j < dim_in; j += 2) {
            uint8_t a = (uint8_t)std::max((int32_t)in_ptr[j] + zero_point, 0);
            uint8_t b = 0;
            if (j + 1 < dim_in)
                b = (uint8_t)std::max((int32_t)in_ptr[j + 1] + zero_point, 0);
            a = std::min(a, DTypeTrait<dtype::Quantized4Asymm>::max());
            b = std::min(b, DTypeTrait<dtype::Quantized4Asymm>::max());
            out_ptr[j / 2] = a + (b << 4);
        }
        in_ptr += dim_in;
        out_ptr += stride_out;
    }
}

// ==================================qint4======================================
void megdnn::naive::int4_to_int8(const TensorND& in, const TensorND& out) {
    auto in_ptr = static_cast<int8_t*>(in.raw_ptr()) + in.layout.span().low_byte;
    auto out_ptr = static_cast<int8_t*>(out.raw_ptr()) + out.layout.span().low_byte;
    const auto& ly = in.layout;
    auto dim_in = ly.shape[ly.ndim - 1];
    auto elems = ly.total_nr_elems();
    auto dim_out = elems / dim_in;
    auto stride_out = div_ceil(dim_in, 2_z);
    for (size_t i = 0; i < dim_out; ++i) {
        for (size_t j = 0; j < dim_in; j += 2) {
            int8_t cur = in_ptr[j / 2];
            out_ptr[j] = cur << 4;
            out_ptr[j] = out_ptr[j] >> 4;
            if (j + 1 < dim_in)
                out_ptr[j + 1] = cur >> 4;
        }
        in_ptr += stride_out;
        out_ptr += dim_in;
    }
}

void megdnn::naive::int8_to_int4(const TensorND& in, const TensorND& out) {
    auto in_ptr = static_cast<int8_t*>(in.raw_ptr()) + in.layout.span().low_byte;
    auto out_ptr = static_cast<int8_t*>(out.raw_ptr()) + out.layout.span().low_byte;
    const auto& ly = in.layout;
    auto dim_in = ly.shape[ly.ndim - 1];
    auto elems = ly.total_nr_elems();
    auto dim_out = elems / dim_in;
    auto stride_out = div_ceil(dim_in, 2_z);
    for (size_t i = 0; i < dim_out; ++i) {
        for (size_t j = 0; j < dim_in; j += 2) {
            int8_t a = in_ptr[j];
            int8_t b = 0;
            if (j + 1 < dim_in)
                b = in_ptr[j + 1];
            a = std::min(a, DTypeTrait<dtype::QuantizedS4>::max());
            a = std::max(a, DTypeTrait<dtype::QuantizedS4>::min());
            b = std::min(b, DTypeTrait<dtype::QuantizedS4>::max());
            b = std::max(b, DTypeTrait<dtype::QuantizedS4>::min());
            out_ptr[j / 2] = (a & 0xF) | (b << 4);
        }
        in_ptr += dim_in;
        out_ptr += stride_out;
    }
}
