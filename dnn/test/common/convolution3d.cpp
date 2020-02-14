/**
 * \file dnn/test/common/convolution3d.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/checker.h"
#include "test/common/convolution3d.h"
#include <chrono>
#include <unordered_set>
#include <sstream>

using namespace megdnn;
using namespace test;
using namespace convolution3d;

std::vector<TestArg> convolution3d::get_1x1x1_args() {
    std::vector<TestArg> args;
    param::Convolution3D param;
    param.mode = param::Convolution3D::Mode::CROSS_CORRELATION; 
    // clang-format off
    for (size_t batch_size: {4, 8})
    for (size_t ic: {1, 4, 8})
    for (size_t oc: {ic})
    for (size_t id: {4, 16, 64})
    for (size_t ih : {id})
    for (size_t iw : {id}) {
        args.emplace_back(param, TensorShape{batch_size, ic, id, ih, iw},
                          TensorShape{oc, ic, 1, 1, 1});
    }
    // clang-format on
    return args;
}
#if MEGDNN_WITH_BENCHMARK 
std::vector<TestArg> convolution3d::get_speed_test_args() {
    std::vector<TestArg> args;
    std::vector<std::pair<size_t, size_t>> range;
    range.push_back(std::pair<size_t, size_t> (10, 16));
    // clang-format off
    for (size_t n:  {64})
    for (size_t id: {18, 32, 64})
    for (size_t ih: {id})
    for (size_t iw: {18, 64, 128})
    for (size_t oc: {16, 64})
    for (size_t ic: {oc})
    for (size_t fd: {1, 2, 3})
    for (size_t fh: {fd})
    for (size_t fw: {fh})
    for (size_t pd: {0, 1})
    for (size_t sd: {1, 2, 3})
    for (size_t dd: {1, 3}) 
    for (size_t cw: {false})
    for (bool xcorr: {false, true}) {
        param::Convolution3D param;
        param.mode = xcorr ? param::Convolution3D::Mode::CROSS_CORRELATION
                           : param::Convolution3D::Mode::CONVOLUTION;
        param.stride_d = param.stride_h = param.stride_w = sd;
        param.pad_d = param.pad_h = param.pad_w = pd;
        param.dilate_d = param.dilate_h = param.dilate_w = dd;
        if (cw)
            param.sparse = param::Convolution3D::Sparse::GROUP;
        args.emplace_back(param, TensorShape{n, ic, id, ih, iw},
                          !cw ? TensorShape{oc, ic, fd, fh, fw}
                              : TensorShape{ic, oc, 1, fd, fh, fw});
    }
    // clang-format on
    return args;
}
#endif
std::vector<TestArg> convolution3d::get_args() {
    std::vector<TestArg> args;
    std::vector<std::pair<size_t, size_t>> range;
    range.push_back(std::pair<size_t, size_t> (11, 13));
    // clang-format off
#if 1
    for (size_t n:  {4})
    for (size_t id: {12, 16})
    for (size_t ih: {id})
    for (size_t iw: {16})
    for (size_t ic: {5, 10})
    for (size_t oc: {ic})
    for (size_t fd: {1,2,3})
    for (size_t fh: {fd})
    for (size_t fw: {fh})
    for (size_t pd: {0, 4})
    for (size_t sd: {2})
#if CUDNN_MAJOR >= 6
    for (size_t dd: {1, 3, 4}) 
#else
    for (size_t dd: {1}) 
#endif
    for (size_t cw: {false})
    for (bool xcorr: {false, true}) {
        param::Convolution3D param;
        param.mode = xcorr ? param::Convolution3D::Mode::CROSS_CORRELATION
                           : param::Convolution3D::Mode::CONVOLUTION;
        param.stride_d = param.stride_h = param.stride_w = sd;
        param.pad_d = param.pad_h = param.pad_w = pd;
        param.dilate_d = param.dilate_h = param.dilate_w = dd;
        if (cw)
            param.sparse = param::Convolution3D::Sparse::GROUP;
        args.emplace_back(param, TensorShape{n, ic, id, ih, iw},
                          !cw ? TensorShape{oc, ic, fd, fh, fw}
                              : TensorShape{ic, oc, 1, fd, fh, fw});
    }
    return args;
#endif
    // clang-format on
    // clang-format off
    for (size_t n:  {8})
    for (size_t id: {20})
    for (size_t ih: {id})
    for (size_t iw: {id})
    for (size_t ic: {1})
    for (size_t oc: {ic})
    for (size_t fd: {3})
    for (size_t fh: {fd})
    for (size_t fw: {fh})
    for (size_t pd: {1, 2, 3})
    for (size_t sd: {2})
    for (size_t dd: {1, 2}) 
    for (size_t cw: {false})
    for (bool xcorr: {false, true}) {
        param::Convolution3D param;
        param.mode = xcorr ? param::Convolution3D::Mode::CROSS_CORRELATION
                           : param::Convolution3D::Mode::CONVOLUTION;
        param.stride_d = param.stride_h = param.stride_w = sd;
        param.pad_d = param.pad_h = param.pad_w = pd;
        param.dilate_d = param.dilate_h = param.dilate_w = dd;
        if (cw)
            param.sparse = param::Convolution3D::Sparse::GROUP;
        args.emplace_back(param, TensorShape{n, ic, id, ih, iw},
                          !cw ? TensorShape{oc, ic, fd, fh, fw}
                              : TensorShape{ic, oc, 1, fd, fh, fw});
    }
    // clang-format on
    return args;
    for (size_t i = range[0].first; i < range[0].second; ++i) {
        param::Convolution3D param;
        param.mode = param::Convolution3D::Mode::CONVOLUTION;
        args.emplace_back(param,
                TensorShape{4, 10, i, i+1, i+2},
                TensorShape{10, 10, 1, 1, 1});

        param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
        args.emplace_back(param,
                TensorShape{4, 10, i, i+1, i+2},
                TensorShape{4, 10, 1, 1, 1});
    }

    for (size_t i = 2; i < 6; ++i) {
        param::Convolution3D param;
        param.mode = param::Convolution3D::Mode::CONVOLUTION;
        args.emplace_back(param,
                TensorShape{1, 1, i, i+1, i+2},
                TensorShape{1, 1, 1, 2, 3});
    }
    for (size_t i = 2; i < 6; ++i) {
        param::Convolution3D param;
        param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
        args.emplace_back(param,
                TensorShape{1, 1, i, i+1, i+2},
                TensorShape{1, 1, 1, 2, 3});
    }
    for (size_t i = 2; i < 5; ++i) {
        param::Convolution3D param;
        param.mode = param::Convolution3D::Mode::CONVOLUTION;
        args.emplace_back(param,
                TensorShape{1, 1, i, i+1, i+2},
                TensorShape{1, 1, 2, 2, 2});
    }

    for (size_t i = range[0].first; i < range[0].second; ++i) {
        param::Convolution3D param;
        param.mode = param::Convolution3D::Mode::CONVOLUTION;
        args.emplace_back(param,
                TensorShape{5, 2, i, i+1, i+2},
                TensorShape{3, 2, 3, 4, 5});

        param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
        args.emplace_back(param,
                TensorShape{5, 2, i, i+1, i+2},
                TensorShape{3, 2, 3, 4, 5});
    }

    //padding case
    for (size_t i = range[0].first; i < range[0].second; ++i) {
        param::Convolution3D param;
        param.pad_d = 1;
        param.pad_h = 2;
        param.pad_w = 3;

        param.mode = param::Convolution3D::Mode::CONVOLUTION;
        args.emplace_back(param,
                TensorShape{5, 2, i, i+1, i+2},
                TensorShape{3, 2, 3, 4, 5});
        param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
        args.emplace_back(param,
                TensorShape{5, 2, i, i+1, i+2},
                TensorShape{3, 2, 3, 4, 5});
    }
    // large channel
    for (size_t i = range[0].first; i < range[0].second; ++i) {
        param::Convolution3D param;

        param.mode = param::Convolution3D::Mode::CONVOLUTION;
        args.emplace_back(param,
                TensorShape{2, 20, i, i+1, i+2},
                TensorShape{30, 20, 3, 4, 5});
        param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
        args.emplace_back(param,
                TensorShape{2, 20, i, i+1, i+2},
                TensorShape{30, 20, 3, 4, 5});
    }

    for (size_t i = range[0].first; i < range[0].second; ++i) {
        param::Convolution3D param;
        param.pad_d = 1;
        param.pad_h = 2;
        param.pad_w = 3;

        param.mode = param::Convolution3D::Mode::CONVOLUTION;
        args.emplace_back(param,
                TensorShape{2, 20, i, i+1, i+2},
                TensorShape{30, 20, 3, 4, 5});
        param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
        args.emplace_back(param,
                TensorShape{2, 20, i, i+1, i+2},
                TensorShape{30, 20, 3, 4, 5});
    }

    // 1x1x1
    for (size_t i = range[0].first; i < range[0].second; ++i) {
        param::Convolution3D param;

        param.mode = param::Convolution3D::Mode::CONVOLUTION;
        args.emplace_back(param,
                TensorShape{2, 20, i, i+1, i+2},
                TensorShape{30, 20, 1, 1, 1});
        param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
        args.emplace_back(param,
                TensorShape{2, 20, i, i+1, i+2},
                TensorShape{30, 20, 1, 1, 1});
    }

    // large filter
    for (size_t i = range[0].first; i < range[0].second; ++i) {
        param::Convolution3D param;

        param.mode = param::Convolution3D::Mode::CONVOLUTION;
        args.emplace_back(param,
                TensorShape{2, 2, i, i+1, i+2},
                TensorShape{3, 2, 7, 8, 9});
        param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
        args.emplace_back(param,
                TensorShape{2, 2, i, i+1, i+2},
                TensorShape{3, 2, 7, 8, 9});
    }

    // exhaustive search
    // clang-format off
    for (size_t n: {1, 2})
    for (size_t id: {7, 8})
    for (size_t ih: {id+1})
    for (size_t iw: {ih+1})
    for (size_t ic: {3})
    for (size_t oc: {4})
    for (size_t fd: {2, 4})
    for (size_t fh: {fd+1})
    for (size_t fw: {fh+1})
    for (size_t ph: {0, 1})
    for (size_t sh: {1, 2})
    for (bool xcorr: {false, true})
    {
        param::Convolution3D param;
        param.mode = xcorr ? param::Convolution3D::Mode::CROSS_CORRELATION
                           : param::Convolution3D::Mode::CONVOLUTION;
        param.stride_d = param.stride_h = param.stride_w = sh;
        param.pad_d = param.pad_h = param.pad_w = ph;
        args.emplace_back(param, TensorShape{n, ic, id, ih, iw},
                          TensorShape{oc, ic, fd, fh, fw});
    }
    // clang-format on

    // 4x4x4
    for (size_t oh = 1; oh < 10; ++oh) {
        param::Convolution3D param;
        param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
        args.emplace_back(param,
                TensorShape{4, 3, oh+3, oh+4, oh+5},
                TensorShape{2, 3, 4, 4, 4});
    }
    // large channels
    // clang-format off
    for (size_t n: {2})
    for (size_t id: {8})
    for (size_t ih: {id+1})
    for (size_t iw: {ih+1})
    for (size_t ic: {16})
    for (size_t oc: {16})
    for (size_t fd: {3, 6})
    for (size_t fh: {fd+1})
    for (size_t fw: {fh+1})
    for (size_t ph: {0, 1})
    for (size_t sh: {1, 2})
    for (bool xcorr: {false, true})
    {
        param::Convolution3D param;
        param.mode = xcorr ? param::Convolution3D::Mode::CROSS_CORRELATION
                           : param::Convolution3D::Mode::CONVOLUTION;
        param.stride_d = param.stride_h = param.stride_w = sh;
        param.pad_d = param.pad_h = param.pad_w = ph;
        args.emplace_back(param, TensorShape{n, ic, id, ih, iw},
                          TensorShape{oc, ic, fd, fh, fw});
    }
    // clang-format on
#if 0
    // x86 direct case 2
    for (size_t stride: {1, 2})
    for (size_t ker_size: {3, 5, 7})
    {
        param::Convolution3D param;
        param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
        param.stride_d = param.stride_h = param.stride_w = stride;
        param.pad_d = param.pad_h = param.pad_w = ker_size/2;
        args.emplace_back(param,
                TensorShape{2, 2, 20, 19, 18},
                TensorShape{3, 2, ker_size, ker_size, ker_size});
        args.emplace_back(param,
                TensorShape{2, 2, 20, 19, 18},
                TensorShape{1, 2, ker_size, ker_size, ker_size});
    }

    for (size_t sd: {1, 2})
    for (size_t sh: {1, 2})
    for (size_t sw: {1, 2})
    for (size_t pd: {0, 1, 2})
    for (size_t ph: {0, 1, 2})
    for (size_t pw: {0, 1, 2})
    for (size_t ker_size: {3, 4, 5, 7})
    for (size_t xcorr : {false, true})
    {
        param::Convolution3D param;
        param.mode = xcorr ?
            param::Convolution3D::Mode::CROSS_CORRELATION :
            param::Convolution3D::Mode::CONVOLUTION;
        param.stride_d = sd;
        param.stride_h = sh;
        param.stride_w = sw;
        param.pad_d = pd;
        param.pad_h = ph;
        param.pad_w = pw;
        args.emplace_back(param,
                TensorShape{2, 2, 10, 15, 20},
                TensorShape{3, 2, ker_size, ker_size, ker_size});
        args.emplace_back(param,
                TensorShape{2, 2, 10, 15, 20},
                TensorShape{1, 2, ker_size, ker_size, ker_size});
    }
    // fallback non-templated impl
    for (size_t sd: {1, 2})
    for (size_t sh: {1, 2})
    for (size_t sw: {1, 2})
    for (size_t pd: {0, 1, 2})
    for (size_t ph: {0, 1, 2})
    for (size_t pw: {0, 1, 2})
    for (size_t ker_size: {3, 4, 5})
    for (size_t xcorr : {false, true})
    {
        param::Convolution3D param;
        param.mode = xcorr ?
            param::Convolution3D::Mode::CROSS_CORRELATION :
            param::Convolution3D::Mode::CONVOLUTION;
        param.stride_d = sd;
        param.stride_h = sh;
        param.stride_w = sw;
        param.pad_d = pd;
        param.pad_h = ph;
        param.pad_w = pw;
        args.emplace_back(param,
                TensorShape{2, 2, 5, 15, 20}, TensorShape{3, 2, ker_size, ker_size+1, ker_size+2});
        args.emplace_back(param,
                TensorShape{2, 2, 5, 15, 20},
                TensorShape{1, 2, ker_size, ker_size+1, ker_size+2});
    }

    // x86 winograd algorithm
    for (size_t ic_size: {8, 16})
    {
        param::Convolution3D param;
        param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
        param.stride_d = param.stride_h = param.stride_w = 1;
        param.pad_d = param.pad_h = param.pad_w = 0;
        args.emplace_back(param,
                TensorShape{2, ic_size, 20, 18, 19},
                TensorShape{8, ic_size, 3, 3, 3});
    }
#endif
    return args;
}

std::vector<TestArg> convolution3d::get_chanwise_args() {
    std::vector<TestArg> args;
    // clang-format off
    for (size_t n : {4})
    for (size_t id : {35})
    for (size_t ih : {id + 1})
    for (size_t iw : {ih + 1})
    for (size_t c : {4, 8, 16})
    for (size_t fd : {3, 4, 7})
    for (size_t fh : {fd + 1})
    for (size_t fw : {fh + 1})
    for (size_t ph : {0, 1})
    for (size_t sh : {1, 2})
    for (size_t dh : {1}) {
        param::Convolution3D param;
        param.sparse = param::Convolution3D::Sparse::GROUP;
        param.stride_d = param.stride_h = param.stride_w = sh;
        param.pad_d = param.pad_h = param.pad_w = ph;
        param.dilate_d = param.dilate_h = param.dilate_w = dh;
        args.emplace_back(param, TensorShape{n, c, id, ih, iw},
                          TensorShape{c, 1, 1, fd, fh, fw});
    }
    // clang-format on
    return args;
}

std::vector<TestArg> convolution3d::get_dilated_args() {
    std::vector<TestArg> args;
    param::Convolution3D param;
    {
        param.pad_d = param.pad_h = param.pad_w = 2;
        param.dilate_d = param.dilate_h = param.dilate_w = 3;
        size_t n = 1, ic = 5, id = 24, ih = 24, iw = 24,
               fd = 3, fh = 3, fw = 3,
               oc = 6;
        args.emplace_back(param,
                TensorShape{n, ic, id, ih, iw},
                TensorShape{oc, ic, fd, fh, fw});
    }
    // exhaustive search
    // clang-format off
    for (size_t n : {2})
    for (size_t id : {32})
    for (size_t ih : {id + 1})
    for (size_t iw : {ih + 1})
    for (size_t ic : {3})
    for (size_t oc : {4})
    for (size_t fd : {2, 3, 4})
    for (size_t fh : {fd + 1})
    for (size_t fw : {fh + 1})
    for (size_t ph : {0, 1})
    for (size_t sh : {2, 3})
    for (size_t dh : {2, 3, 4}) {
        param::Convolution3D param;
        param.stride_d = param.stride_h = param.stride_w = sh;
        param.pad_d = param.pad_h = param.pad_w = ph;
        param.dilate_d = param.dilate_h = param.dilate_w = dh;
        args.emplace_back(param, TensorShape{n, ic, id, ih, iw},
                          TensorShape{oc, ic, fd, fh, fw});
    }
    // clang-format on
    return args;
}
// vim: syntax=cpp.doxygen

