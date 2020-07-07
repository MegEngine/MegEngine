/**
 * \file dnn/test/common/warp_perspective.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "test/common/warp_perspective.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;
using namespace warp_perspective;

void WarpPerspectiveMatIdxProxy::deduce_layout(WarpPerspective*,
                                               TensorLayoutArray&) {}
void WarpPerspectiveMatIdxProxy::deduce_layout(WarpPerspectiveBackwardData*,
                                               TensorLayoutArray&) {}
void WarpPerspectiveMatIdxProxy::deduce_layout(WarpPerspectiveBackwardMat*,
                                               TensorLayoutArray&) {}

void WarpPerspectiveMatIdxProxy::exec(WarpPerspective* opr,
                                      const TensorNDArray& tensors) {
    if (!W.valid()) {
        W = WorkspaceWrapper(opr->handle(), 0);
    }
    megdnn_assert(tensors.size() == 4);
    W.update(opr->get_workspace_in_bytes(tensors[0].layout, tensors[1].layout,
                                         tensors[2].layout, tensors[3].layout));
    opr->exec(tensors[0], tensors[1], tensors[2], tensors[3], W.workspace());
}

void WarpPerspectiveMatIdxProxy::exec(WarpPerspectiveBackwardData* opr,
                                      const TensorNDArray& tensors) {
    if (!W.valid()) {
        W = WorkspaceWrapper(opr->handle(), 0);
    }
    megdnn_assert(tensors.size() == 4);
    W.update(opr->get_workspace_in_bytes(tensors[0].layout, tensors[1].layout,
                                         tensors[2].layout, tensors[3].layout));
    opr->exec(tensors[0], tensors[1], tensors[2], tensors[3], W.workspace());
}

void WarpPerspectiveMatIdxProxy::exec(WarpPerspectiveBackwardMat* opr,
                                      const TensorNDArray& tensors) {
    if (!W.valid()) {
        W = WorkspaceWrapper(opr->handle(), 0);
    }
    megdnn_assert(tensors.size() == 5);
    W.update(opr->get_workspace_in_bytes(tensors[0].layout, tensors[1].layout,
                                         tensors[2].layout, tensors[3].layout,
                                         tensors[4].layout));
    opr->exec(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
              W.workspace());
}

std::vector<TestArg> warp_perspective::get_cv_args() {
    std::vector<TestArg> args;

    // in warp_perspective_cv INTER_AREA == INTER_LINEAR
    using BorderMode = param::WarpPerspective::BorderMode;
    using InterpolationMode = param::WarpPerspective::InterpolationMode;
    param::WarpPerspective cur_param;

    for (size_t i = 4; i < 129; i *= 4) {
        for (size_t ic : {1, 2, 3}) {
            for (BorderMode bmode : {
                         BorderMode::REPLICATE,
                         BorderMode::REFLECT,
                         BorderMode::REFLECT_101,
                         BorderMode::WRAP,
                         BorderMode::CONSTANT,
                 }) {
                for (InterpolationMode imode :
                     {InterpolationMode::NEAREST, InterpolationMode::LINEAR,
                      InterpolationMode::CUBIC, InterpolationMode::LANCZOS4}) {
                    cur_param.bmode = bmode;
                    cur_param.format = param::WarpPerspective::Format::NHWC;

                    cur_param.imode = imode;
                    args.emplace_back(cur_param, TensorShape{1, i, i, ic},
                                      TensorShape{1, 3, 3}, TensorShape{1},
                                      TensorShape{1, i, i, ic});
                    args.emplace_back(cur_param, TensorShape{1, i, i * 2, ic},
                                      TensorShape{1, 3, 3}, TensorShape{1},
                                      TensorShape{1, i, i * 2, ic});
                    args.emplace_back(cur_param, TensorShape{1, i * 3, i, ic},
                                      TensorShape{1, 3, 3}, TensorShape{1},
                                      TensorShape{1, i * 3, i, ic});

                    cur_param.border_val = 0.78f;
                    args.emplace_back(cur_param, TensorShape{1, i, i, ic},
                                      TensorShape{1, 3, 3}, TensorShape{1},
                                      TensorShape{1, 8, 8, ic});
                    args.emplace_back(cur_param, TensorShape{1, i, i * 2, ic},
                                      TensorShape{1, 3, 3}, TensorShape{1},
                                      TensorShape{1, 8, 8, ic});
                    args.emplace_back(cur_param, TensorShape{1, i * 3, i, ic},
                                      TensorShape{1, 3, 3}, TensorShape{1},
                                      TensorShape{1, 8, 8, ic});
                }
            }
        }
    }
    return args;
}

void warp_perspective::run_mat_idx_test(Handle* handle) {
    constexpr int N_SRC = 5;
    Checker<WarpPerspectiveForward, WarpPerspectiveMatIdxProxy> checker(handle);
    WarpPerspectiveMatRNG mat_rng;
    checker.set_rng(1, &mat_rng);

    UniformIntRNG mat_idx_rng{0, N_SRC - 1};
    checker.set_dtype(2, dtype::Int32());
    checker.set_rng(2, &mat_idx_rng);

    WarpPerspective::Param param;
    param.bmode = WarpPerspective::Param::BorderMode::REFLECT;
    param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
    checker.set_param(param);
    checker.execs({{N_SRC, 3, 10, 11}, {2, 3, 3}, {2}, {2, 3, 11, 12}});
    checker.execs({{N_SRC, 14, 17, 13}, {123, 3, 3}, {123}, {123, 14, 16, 15}});

    // test NHWC
    param.format = WarpPerspective::Param::Format::NHWC;
        checker.set_param(param)
               .set_rng(2, &mat_idx_rng)
                   .set_epsilon(1e-1)
                   .set_dtype(2, dtype::Int32());
    checker.execs({{N_SRC, 10, 11, 3}, {2, 3, 3}, {2}, {2, 11, 12, 3}});
}

void warp_perspective::run_int8_test(Handle* handle) {
    using Param = WarpPerspective::Param;
    Checker<WarpPerspectiveForward> checker(handle);
    UniformIntRNG input_rng{-128, 127};
    WarpPerspectiveMatRNG mat_rng;
    class ResizeBy2xMatRNG : public RNG {
        void gen(const TensorND& tensor_) override {
            float* ptr = tensor_.ptr<float>();
            auto N = tensor_.layout.shape[0];
            megdnn_assert(tensor_.layout.is_contiguous() &&
                          tensor_.layout.ndim == 3 && tensor_.layout[1] == 3 &&
                          tensor_.layout[2] == 3);
            for (size_t n = 0; n < N; ++n) {
                //       | 1 0 0 |
                // mat = | 0 1 0 |
                //       | 0 0 2 |
                // resize_2x
                ptr[0] = ptr[4] = 1;
                ptr[8] = 2;
                ptr[1] = ptr[2] = ptr[3] = ptr[5] = ptr[6] = ptr[7] = 0;
                ptr += 9;
            }
        }
    } resize_2x_mat_rng;
    if (handle->type() == Handle::HandleType::CUDA) {
        // As currently the computation is performed in floating points instead
        // of full int, it could be slightly different on GPU.
        checker.set_epsilon(1.1).set_max_avg_error(7e-5);
    }
    checker.set_rng(0, &input_rng)
            .set_rng(1, &mat_rng)
            .set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int8())
            .set_param({Param::InterpolationMode::LINEAR,
                        Param::BorderMode::CONSTANT, Param::Format::NCHW, 0.f});
    checker.execs({{99, 48, 17, 17}, {99, 3, 3}, {99, 48, 22, 22}})
            .execs({{12, 3, 224, 224}, {12, 3, 3}, {12, 3, 256, 256}});

    checker.set_rng(1, &resize_2x_mat_rng);
    checker.execs({{98, 48, 17, 17}, {98, 3, 3}, {98, 48, 34, 34}})
            .execs({{13, 3, 224, 224}, {13, 3, 3}, {13, 3, 448, 448}});
}

void warp_perspective::run_quint8_test(Handle* handle) {
    using Param = WarpPerspective::Param;
    Checker<WarpPerspectiveForward> checker(handle);
    UniformIntRNG input_rng{0, 255};
    WarpPerspectiveMatRNG mat_rng;
    class ResizeBy2xMatRNG : public RNG {
        void gen(const TensorND& tensor_) override {
            float* ptr = tensor_.ptr<float>();
            auto N = tensor_.layout.shape[0];
            megdnn_assert(tensor_.layout.is_contiguous() &&
                          tensor_.layout.ndim == 3 && tensor_.layout[1] == 3 &&
                          tensor_.layout[2] == 3);
            for (size_t n = 0; n < N; ++n) {
                //       | 1 0 0 |
                // mat = | 0 1 0 |
                //       | 0 0 2 |
                // resize_2x
                ptr[0] = ptr[4] = 1;
                ptr[8] = 2;
                ptr[1] = ptr[2] = ptr[3] = ptr[5] = ptr[6] = ptr[7] = 0;
                ptr += 9;
            }
        }
    } resize_2x_mat_rng;
    if (handle->type() == Handle::HandleType::CUDA) {
        // As currently the computation is performed in floating points instead
        // of full int, it could be slightly different on GPU.
        checker.set_epsilon(1.1).set_max_avg_error(7e-5);
    }
    checker.set_rng(0, &input_rng)
            .set_rng(1, &mat_rng)
            .set_dtype(0,
                       dtype::Quantized8Asymm(0.6f, static_cast<uint8_t>(127)))
            .set_dtype(1, dtype::Float32())
            .set_dtype(2,
                       dtype::Quantized8Asymm(0.6f, static_cast<uint8_t>(127)))
            .set_param({Param::InterpolationMode::LINEAR,
                        Param::BorderMode::CONSTANT, Param::Format::NCHW, 0.f});
    checker.execs({{99, 48, 17, 17}, {99, 3, 3}, {99, 48, 22, 22}})
            .execs({{12, 3, 224, 224}, {12, 3, 3}, {12, 3, 256, 256}});

    checker.set_rng(1, &resize_2x_mat_rng);
    checker.execs({{98, 48, 17, 17}, {98, 3, 3}, {98, 48, 34, 34}})
            .execs({{13, 3, 224, 224}, {13, 3, 3}, {13, 3, 448, 448}});
}

// vim: syntax=cpp.doxygen
