/**
 * \file dnn/src/naive/lowbit_utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/lowbit_utils.h"

// =================================quint4======================================
void megdnn::naive::uint4_to_uint8(const TensorND& in, const TensorND& out) {
    auto in_ptr = static_cast<uint8_t*>(in.raw_ptr) + in.layout.span().low_byte;
    auto out_ptr = out.compatible_ptr<uint8_t>() + out.layout.span().low_byte;
    for (size_t i = 0; i < in.layout.span().dist_elem(); i += 2) {
        uint8_t val = in_ptr[i / 2];
        out_ptr[i] = val & 0xF;
        out_ptr[i + 1] = (val >> 4) & 0xF;
    }
}

void megdnn::naive::uint8_to_uint4(const TensorND& in, const TensorND& out) {
    auto in_ptr = static_cast<uint8_t*>(in.raw_ptr) + in.layout.span().low_byte;
    auto out_ptr =
            static_cast<uint8_t*>(out.raw_ptr) + out.layout.span().low_byte;
    for (size_t i = 0; i < out.layout.span().dist_elem(); i += 2) {
        uint8_t a = in_ptr[i], b = in_ptr[i + 1];
        a = std::min(a, DTypeTrait<dtype::Quantized4Asymm>::max());
        b = std::min(b, DTypeTrait<dtype::Quantized4Asymm>::max());
        out_ptr[i / 2] = a + (b << 4);
    }
}

// ==================================qint4======================================
void megdnn::naive::int4_to_int8(const TensorND& in, const TensorND& out) {
    auto in_ptr = static_cast<int8_t*>(in.raw_ptr) + in.layout.span().low_byte;
    auto out_ptr =
            static_cast<int8_t*>(out.raw_ptr) + out.layout.span().low_byte;

    for (size_t i = 0; i < in.layout.span().dist_elem(); i += 2) {
        int8_t cur = in_ptr[i / 2];
        out_ptr[i] = cur << 4;
        out_ptr[i] = out_ptr[i] >> 4;
        out_ptr[i + 1] = cur >> 4;
    }
}

void megdnn::naive::int8_to_int4(const TensorND& in, const TensorND& out) {
    auto in_ptr = static_cast<int8_t*>(in.raw_ptr) + in.layout.span().low_byte;
    auto out_ptr =
            static_cast<int8_t*>(out.raw_ptr) + out.layout.span().low_byte;
    for (size_t i = 0; i < out.layout.span().dist_elem(); i += 2) {
        int8_t a = in_ptr[i], b = in_ptr[i + 1];
        a = std::min(a, DTypeTrait<dtype::QuantizedS4>::max());
        a = std::max(a, DTypeTrait<dtype::QuantizedS4>::min());
        b = std::min(b, DTypeTrait<dtype::QuantizedS4>::max());
        b = std::max(b, DTypeTrait<dtype::QuantizedS4>::min());
        out_ptr[i / 2] = (a & 0xF) | (b << 4);
    }
}
