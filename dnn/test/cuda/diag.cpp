/**
 * \file dnn/test/cuda/diag.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, DIAG) {
    Checker<Diag> checker(handle_cuda());
    for (DType dtype :
         std::vector<DType>{dtype::Float16(), dtype::Int32(), dtype::Float32()})
        for (int k = -5; k < 5; ++k) {
            checker.set_param({k});
            checker.set_dtype(0, dtype);
            checker.set_dtype(1, dtype);
            size_t absk = static_cast<size_t>(std::abs(k));
            checker.exec(TensorShapeArray{{8}, {8 + absk, 8 + absk}});
            //! NOTE: diag for vector or matrix is a vector
            auto oshape = [&](int n, int m) -> TensorShape {
                size_t o = (k >= 0 ? std::min(m - k, n) : std::min(n + k, m));
                return {o};
            };
            checker.exec(TensorShapeArray{{8, 6}, oshape(8, 6)});
            checker.exec(TensorShapeArray{{6, 8}, oshape(6, 8)});
            checker.exec(TensorShapeArray{{8, 8}, oshape(8, 8)});
        }
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
