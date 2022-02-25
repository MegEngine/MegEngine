/**
 * \file src/cuda/local/cuda-convnet2/helper_cuda.h
 *
 * This file is part of MegDNN, a deep neural network run-time library * developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 */

#pragma once
#include <cstdio>
#include "src/cuda/utils.cuh"
#define checkCudaErrors(x)  cuda_check(x)
#define getLastCudaError(x) cuda_check(cudaGetLastError())

// vim: syntax=cpp.doxygen
