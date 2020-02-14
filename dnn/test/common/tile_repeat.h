/**
 * \file dnn/test/common/tile_repeat.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace test {
namespace tile_repeat {

struct Arg {
    TensorShape times, src, dst;
    Arg(TensorShape times, TensorShape src) : times(times), src(src) {
        dst = src;
        for (size_t i = 0; i < src.ndim; ++i) {
            dst[i] *= times[i];
        }
    }
    TileRepeatBase::Param param() {
        TileRepeatBase::Param param;
        param.times = times;
        return param;
    }
};

inline std::vector<Arg> get_args() {
    std::vector<Arg> args;
    args.emplace_back(TensorShape{3}, TensorShape{10000});
    args.emplace_back(TensorShape{1, 1}, TensorShape{200, 300});
    args.emplace_back(TensorShape{1, 3}, TensorShape{200, 300});
    args.emplace_back(TensorShape{2, 1}, TensorShape{200, 300});
    args.emplace_back(TensorShape{2, 3}, TensorShape{200, 300});
    for (unsigned mask = 0; mask < 32; ++mask) {
        auto b = [mask](unsigned bit) { return (mask >> bit) & 1; };
        args.emplace_back(
                TensorShape{b(0) + 1, b(1) + 1, b(2) + 1, b(3) + 1, b(4) + 1},
                TensorShape{3, 4, 5, 6, 7});
    }
    for (size_t i = 1; i < 10; ++i)
        for (size_t j = 1; j < 10; ++j) {
            args.emplace_back(TensorShape{i, j}, TensorShape{3, 4});
        }
    return args;
}

}  // namespace tile_repeat
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
