/**
 * \file dnn/test/common/warp_perspective.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "test/common/rng.h"
#include "test/common/random_state.h"
#include "test/common/workspace_wrapper.h"

#include "megdnn/oprs/imgproc.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {

struct WarpPerspectiveMatIdxProxy {
    WorkspaceWrapper W;
    static void deduce_layout(WarpPerspective*, TensorLayoutArray&);
    static void deduce_layout(WarpPerspectiveBackwardData*, TensorLayoutArray&);
    static void deduce_layout(WarpPerspectiveBackwardMat*, TensorLayoutArray&);
    void exec(WarpPerspective* opr, const TensorNDArray& tensors);
    void exec(WarpPerspectiveBackwardData* opr, const TensorNDArray& tensors);
    void exec(WarpPerspectiveBackwardMat* opr, const TensorNDArray& tensors);
};

class WarpPerspectiveMatRNG final : public IIDRNG {
public:
    WarpPerspectiveMatRNG() : idx(0) {}
    dt_float32 gen_single_val() override {
        std::normal_distribution<float_t> dist;
        switch (idx) {
            case 6:
            case 7:
                dist = std::normal_distribution<float_t>(0.0f, 0.01f);
                break;
            case 8:
                dist = std::normal_distribution<float_t>(1.0f, 0.1f);
                break;
            default:
                dist = std::normal_distribution<float_t>(0.0f, 1.0f);
                break;
        }
        auto res = dist(RandomState::generator());
        idx = (idx + 1) % 9;
        return res;
    }

private:
    size_t idx;
};

class WarpPerspectiveMatRNG_V2 final : public IIDRNG {
public:
    WarpPerspectiveMatRNG_V2() : idx(0) {}
    void set_hw(size_t h, size_t w) {
        ih = h;
        iw = w;
        idx = 0;
        rng.seed(time(NULL));
    }
    dt_float32 gen_single_val() override {
        auto rand_real = [&](double lo, double hi) {
            return rng() / (std::mt19937::max() + 1.0) * (hi - lo) + lo;
        };
        auto rand_real2 = [&](double range) {
            return rand_real(-range, range);
        };
        dt_float32 res;
        switch (idx) {
            case 0:
                rot = rand_real(0, M_PI * 2);
                scale = rand_real(0.8, 1.2);
                sheer = rand_real(0.9, 1.1);
                res = cos(rot) * scale;
                break;
            case 1:
                res = -sin(rot) * scale;
                break;
            case 2:
                res = rand_real2(iw * 0.5);
                break;
            case 3:
                res = sin(rot) * scale * sheer;
                break;
            case 4:
                res = cos(rot) * scale * sheer;
                break;
            case 5:
                res = rand_real2(ih * 0.5);
                break;
            case 6:
                res = rand_real2(0.1 / iw);
                break;
            case 7:
                res = rand_real2(0.1 / ih);
                break;
            case 8:
                res = rand_real2(0.1) + 1;
                break;
        }
        idx = (idx + 1) % 9;
        return res;
    }

private:
    size_t idx, ih, iw;
    float rot, scale, sheer;
    std::mt19937 rng;
};

namespace warp_perspective {

struct TestArg {
    param::WarpPerspective param;
    TensorShape src;
    TensorShape trans;
	TensorShape mat_idx;
    TensorShape dst;
    TestArg(param::WarpPerspective param_, TensorShape src_, TensorShape trans_, TensorShape mat_idx_,
            TensorShape dst_)
            : param(param_), src(src_), trans(trans_), mat_idx(mat_idx_), dst(dst_) {}
};

//! Test args for the WarpPerspective with format NHWC
std::vector<TestArg> get_cv_args();

void run_mat_idx_test(Handle* handle);
void run_int8_test(Handle* handle);
void run_quint8_test(Handle* handle);

}  // namespace warp_perspective
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
