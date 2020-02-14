/**
 * \file dnn/src/cuda/local/cuda-convnet2/helper_cuda.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
/**
 * \file src/cuda/local/cuda-convnet2/helper_cuda.h
 *
 * This file is part of MegDNN, a deep neural network run-time library * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 */

#pragma once
#include "src/cuda/utils.cuh"
#include <cstdio>
#define checkCudaErrors(x) cuda_check(x)
#define getLastCudaError(x) cuda_check(cudaGetLastError())

// vim: syntax=cpp.doxygen
