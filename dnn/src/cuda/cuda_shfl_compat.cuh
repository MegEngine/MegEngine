/**
 * \file dnn/src/cuda/cuda_shfl_compat.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#if __CUDACC_VER_MAJOR__ >= 9
#define __shfl(x, y, z) __shfl_sync(0xffffffffu, x, y, z)
#define __shfl_up(x, y, z) __shfl_up_sync(0xffffffffu, x, y, z)
#define __shfl_down(x, y, z) __shfl_down_sync(0xffffffffu, x, y, z)
#define __shfl_xor(x, y, z) __shfl_xor_sync(0xffffffffu, x, y, z)
#endif

// vim: syntax=cpp.doxygen
