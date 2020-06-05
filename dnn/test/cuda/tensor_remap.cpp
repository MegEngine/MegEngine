/**
 * \file dnn/test/cuda/tensor_remap.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/common/tensor_remap.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, TENSOR_REMAP_FORWARD) {
    Checker<IndexingRemapForward> checker(handle_cuda());
    TensorShape src{11, 13, 17}, map{3, 5, 7, 3}, dst{3, 5, 7};
    checker.set_dtype(1, dtype::Int32());
    for (auto dt : std::vector<DType>{dtype::Float32(), dtype::Int32()}) {
        checker.set_dtype(0, dt);
        checker.set_dtype(2, dt);
        using namespace tensor_remap;
        {
            MapRNG rng(src);
            checker.set_rng(1, &rng).execs({src, map, {}});
        }
        {
            NonoverlappingMapRNG rng(src);
            checker.set_rng(1, &rng).execs({src, map, {}});
        }
    }
}

TEST_F(CUDA, TENSOR_REMAP_BACKWARD) {
    Checker<IndexingRemapBackward> checker(handle_cuda());
    checker.set_dtype(1, dtype::Int32());
    TensorShape src{11, 13, 17}, map{3, 5, 7, 3}, dst{3, 5, 7};
    checker.set_dtype(1, dtype::Int32());
    for (auto dt : std::vector<DType>{dtype::Float32(), dtype::Int32()}) {
        checker.set_dtype(0, dt);
        checker.set_dtype(2, dt);
        using namespace tensor_remap;
        {
            MapRNG rng(src);
            checker.set_rng(1, &rng).execs({dst, map, src});
        }
        {
            NonoverlappingMapRNG rng(src);
            checker.set_rng(1, &rng).execs({dst, map, src});
        }
    }
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
