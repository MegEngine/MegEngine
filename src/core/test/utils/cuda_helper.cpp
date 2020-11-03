/**
 * \file src/core/test/utils/cuda_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain_build_config.h"
#include "megbrain/test/helper.h"
#include "megbrain/utils/cuda_helper.h"


#if MGB_CUDA
TEST(TestUtils, TestCudaIncludePath) {
    auto paths = mgb::get_cuda_include_path();
    int available = 0;
    for (auto path : paths) {
        FILE* file =
                fopen((path + "/cuda.h").c_str(), "r");
        if(file) {
            available ++;
            fclose(file);
        }
    }
    mgb_assert(available, "no available cuda include path found!");
}
#endif