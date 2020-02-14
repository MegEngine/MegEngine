/**
 * \file dnn/test/naive/matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/naive/fixture.h"

#include "megdnn/oprs/linalg.h"
#include "test/common/checker.h"
#include "test/common/random_state.h"
#include "test/common/matrix_mul.h"

using namespace megdnn;
using namespace test;

namespace {

void run_matmul_mk_format(Handle* handle, param::MatrixMul::Format format,
                          DType Atype, DType Btype, DType Ctype) {
    using namespace matrix_mul;
    std::vector<TestArg> args = get_matmul_args();
    Checker<MatrixMul> checker(handle);

    auto extra_impl = [](const TensorNDArray& tensors, param::MatrixMul param,
                         Handle* handle, size_t pack_size) {
        megdnn_assert((param.format == param::MatrixMul::Format::MK4 ||
                      param.format == param::MatrixMul::Format::MK8) &&
                              tensors.size() == 3);
        param::MatrixMul new_param = param;
        new_param.format = param::MatrixMul::Format::DEFAULT;
        size_t M = tensors[2].layout[0] * pack_size;
        size_t N = tensors[2].layout[1];
        size_t K = tensors[0].layout[1 - param.transposeA] * pack_size;

        TensorLayoutArray default_layouts, mk4_layouts;
        if (param.transposeA) {
            default_layouts.emplace_back(tensors[0].layout.reshape({K, M}));
            mk4_layouts.emplace_back(
                    default_layouts.back()
                            .reshape({K / pack_size, M / pack_size, pack_size,
                                      pack_size})
                            .dimshuffle({0, 2, 1, 3}));
        } else {
            default_layouts.emplace_back(tensors[0].layout.reshape({M, K}));
            mk4_layouts.emplace_back(
                    default_layouts.back()
                            .reshape({M / pack_size, K / pack_size, pack_size,
                                      pack_size})
                            .dimshuffle({0, 3, 1, 2}));
        }
        if (param.transposeB) {
            default_layouts.emplace_back(tensors[1].layout.reshape({N, K}));
            mk4_layouts.emplace_back(
                    default_layouts.back()
                            .reshape({N, K / pack_size, pack_size})
                            .dimshuffle({0, 1, 2}));
        } else {
            default_layouts.emplace_back(tensors[1].layout.reshape({K, N}));
            mk4_layouts.emplace_back(
                    default_layouts.back()
                            .reshape({K / pack_size, N, pack_size})
                            .dimshuffle({0, 2, 1}));
        }

        default_layouts.emplace_back(tensors[2].layout.reshape({M, N}));
        mk4_layouts.emplace_back(default_layouts.back()
                                         .reshape({M / pack_size, N, pack_size})
                                         .dimshuffle({0, 2, 1}));

        auto matmul_opr = handle->create_operator<MatrixMul>();
        matmul_opr->param() = new_param;
        size_t matmul_workspace = matmul_opr->get_workspace_in_bytes(
                default_layouts[0], default_layouts[1], default_layouts[2]);
        auto relayout_opr = handle->create_operator<Relayout>();

        WorkspaceBundle wb(nullptr, {default_layouts[0].span().dist_byte(),
                                     default_layouts[1].span().dist_byte(),
                                     default_layouts[2].span().dist_byte(),
                                     matmul_workspace});
        wb.set(malloc(wb.total_size_in_bytes()));

        TensorNDArray default_tensors, mk4_tensors;
        for (size_t i = 0; i < 3; i++) {
            default_tensors.emplace_back(wb.get(i), default_layouts[i]);
            mk4_tensors.emplace_back(tensors[i].raw_ptr, mk4_layouts[i]);
        }
        relayout_opr->exec(mk4_tensors[0], default_tensors[0]);
        relayout_opr->exec(mk4_tensors[1], default_tensors[1]);
        matmul_opr->exec(default_tensors[0], default_tensors[1],
                         default_tensors[2], wb.get_workspace(3));
        relayout_opr->exec(default_tensors[2], mk4_tensors[2]);

        free(wb.ptr());
    };

    size_t pack_size = MatrixMulForward::pack_size(format);
    for (auto&& arg : args) {
        if (arg.m % pack_size != 0 || arg.k % pack_size != 0)
            continue;
        param::MatrixMul param;
        param.transposeA = arg.mask & 0x1;
        param.transposeB = arg.mask & 0x2;
        param.format = format;
        size_t m = arg.m, n = arg.n, k = arg.k;
        TensorShape A, B;
        if (param.transposeA) {
            A = TensorShape{k / pack_size, m / pack_size, pack_size, pack_size};
        } else {
            A = TensorShape{m / pack_size, k / pack_size, pack_size, pack_size};
        }
        if (param.transposeB) {
            B = TensorShape{n, k / pack_size, pack_size};
        } else {
            B = TensorShape{k / pack_size, n, pack_size};
        }

        checker.set_extra_opr_impl(std::bind(extra_impl, std::placeholders::_1,
                                             param, handle, pack_size));
        checker.set_dtype(0, Atype)
                .set_dtype(1, Btype)
                .set_dtype(2, Ctype)
                .set_epsilon(1e-3)
                .set_param(param)
                .execs({A, B, {}});
    }
}

}  // namespace

TEST_F(NAIVE, MATRIX_MUL_QUANTIZED4x4x32) {
    Checker<MatrixMul> checker(handle(), /* check_dispatch */false);
    auto GenTensorValueQuint4 = [](const TensorShape& shape,
                                   dtype::Quantized4Asymm dtype,
                                   const std::vector<int>& values) {
        TensorND tensor;
        tensor.layout = {shape, dtype};
        tensor.raw_ptr =
                static_cast<dt_byte*>(malloc(tensor.layout.span().dist_byte()));
        uint8_t* ptr = static_cast<uint8_t*>(tensor.raw_ptr);
        megdnn_assert(values.size() == tensor.layout.span().dist_elem());
        for (size_t i = 0; i < tensor.layout.span().dist_elem(); i += 2) {
            int val0 = values[i], val1 = values[i + 1];
            ptr[i / 2] = val0 | (val1 << 4);
        }
        return tensor;
    };
    using Param = MatrixMul::Param;
    Param param;
    checker.set_param(param);
    checker.set_dtype(2, dtype::QuantizedS32(0.3f * 0.3f));
    checker.exect(
            Testcase{
                    GenTensorValueQuint4(
                            {8, 8}, dtype::Quantized4Asymm(0.3f, (uint8_t)8),
                            {13, 2,  4, 13, 9,  3,  14, 14, 14, 5,  3,  3,  15,
                             11, 8,  8, 5,  7,  14, 15, 8,  2,  11, 1,  15, 9,
                             13, 14, 2, 3,  11, 11, 15, 10, 11, 0,  13, 12, 3,
                             11, 9,  9, 10, 5,  2,  5,  8,  4,  6,  9,  0,  0,
                             3,  9,  9, 8,  8,  15, 7,  5,  0,  3,  9,  10}),
                    GenTensorValueQuint4(
                            {8, 8}, dtype::Quantized4Asymm(0.3f, (uint8_t)8),
                            {5,  14, 13, 11, 4,  7,  12, 12, 11, 7,  13, 10, 5,
                             6,  4,  2,  3,  12, 2,  2,  13, 3,  14, 0,  15, 15,
                             0,  2,  2,  13, 3,  14, 10, 8,  9,  11, 0,  14, 15,
                             4,  14, 7,  1,  6,  13, 2,  12, 5,  2,  15, 7,  11,
                             13, 9,  8,  10, 0,  11, 6,  10, 12, 2,  2,  12}),
                    {}},
            Testcase{
                    {},
                    {},
                    TensorValue(
                            {8, 8}, dtype::QuantizedS32(0.3f * 0.3f),
                            {-90, 120, -3,   40,  -31, 58,  -54, 165, -5,  -19,
                             71,  87,  -51,  24,  92,  15,  27,  62,  -59, -82,
                             -40, 91,  11,   -16, -85, 138, -18, -36, 8,   -25,
                             -56, 75,  -46,  -34, 67,  53,  -4,  -83, 111, -86,
                             -29, -17, 45,   -9,  38,  -22, -3,  -19, -17, -95,
                             94,  78,  63,   -35, -51, 21,  -63, -14, 87,  31,
                             44,  -53, -107, 5}),
            });
}

TEST_F(NAIVE, MATRIX_MUL_QUANTIZED8x8x32) {
    Checker<MatrixMul> checker(handle(), /* check_dispatch */false);
    MatrixMul::Param param;
    param.transposeA = false;
    param.transposeB = false;

    checker.set_param(param).exect(
        Testcase{
          TensorValue({4, 7}, dtype::Quantized8Asymm(0.1f, (uint8_t)128),
                      {6,  97, 210,  47, 213, 246,  92,
                       121, 132, 133,  37,  31,  87,  71,
                       0,   5, 198,  11,  97, 141, 222,
                       166,  76, 212, 190, 108, 245, 143}),
          TensorValue({7, 5}, dtype::Quantized8Asymm(0.2f, (uint8_t)233),
                      { 89, 207,  79, 135,  43,
                        29, 235, 171,  40,  78,
                       119, 145, 254, 162, 184,
                       139, 248, 214, 201, 183,
                       127,  75,  48, 200,  96,
                       109,  63,  60, 100, 120,
                       111, 182, 150, 227,  92}),
          {}},
        Testcase{
          {},
          {},
          TensorValue({4, 5}, dtype::QuantizedS32(0.1f * 0.2f),
                      {  2908, -36975,  -9180,  -3574,   8114,
                        30496,  23588,  32433,  11467,  30974,
                        36748,  -6939,  26715,  33787,  35329,
                       -24486, -25049, -19828, -16627, -18972})});

    param.transposeA = true;
    checker.set_param(param).exect(
        Testcase{
            TensorValue({2, 1}, dtype::Quantized8Asymm(0.7f, (uint8_t)128),
                        {129, 129}),
            TensorValue({2, 1}, dtype::Quantized8Asymm(0.4f, (uint8_t)128),
                        {129, 129}),
            {}
        },
        Testcase{
          {},
          {},
          TensorValue({1, 1}, dtype::QuantizedS32(0.7f * 0.4f), {2})});
}

TEST_F(NAIVE, MATRIX_MUL_MK4) {
    run_matmul_mk_format(handle(), param::MatrixMul::Format::MK4,
                         dtype::Float32(), dtype::Float32(), dtype::Float32());
}

TEST_F(NAIVE, MATRIX_MUL_MK8) {
    run_matmul_mk_format(handle(), param::MatrixMul::Format::MK8,
                         dtype::Int16(), dtype::Int16(), dtype::Int32());
}

// vim: syntax=cpp.doxygen
