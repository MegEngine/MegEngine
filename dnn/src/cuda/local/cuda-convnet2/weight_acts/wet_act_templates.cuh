/**
 * \file dnn/src/cuda/local/cuda-convnet2/weight_acts/wet_act_templates.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
/**
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * --------------------------------------------------------------------------
 * * This file has been modified by Megvii ("Megvii Modifications").
 * * All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
 * --------------------------------------------------------------------------
 */
#include "../nvmatrix.cuh"
#include "../cudaconv2.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

#define LO16(x)     ((x) & 0x0000FFFF)
#define HI16(x)     ((x) >> 16)

#define WA_LOOP(r) _Pragma("unroll") \
for (int c = 0; c < colorsPerThread; c++) { \
    _Pragma("unroll") \
    for (int f = 0; f < filtersPerThread; f++) { \
        prod[f][c] += shImages[threadIdx.y + c * B_Y][(r)] * shHidActs[threadIdx.x + f * B_X][(r)]; \
    } \
}

#define WA_LOOP2(r) _Pragma("unroll") \
for (int f = 0; f < filtersPerThread; f++) { \
    _Pragma("unroll") \
    for (int c = 0; c < colorsPerThread; c++) { \
        prod[f][c] += shImages[threadIdx.y + c * B_Y][(r)] * shHidActs[threadIdx.x + f * B_X][(r)]; \
    } \
}

#define WA_IMLOAD(r) imPreload[r] = im[(r) * B_X * B_Y / preloadCases * imgPixels * imgStride];
#define WA_IMLOAD_TX(r) imPreload[r] = tex1Dfetch<float>(images, imgOffset2 + (r) * B_X * B_Y / preloadCases * imgPixels * imgStride);
#define WA_HALOAD(r) haPreload[r] = ha[(r) * B_X * B_Y / preloadCases * numImages * numModules];
#define WA_HALOAD_TX(r) haPreload[r] = tex1Dfetch<float>(hidActs, hidActsOffset2 + (r) * B_X * B_Y / preloadCases * numImages * numModules);

#define C_KEP_PARAM float* images, float* hidActs, float* targets,      \
                    const int numImages, const int numFilters,          \
                    const int numModulesY, const int numModulesX,       \
                    const int imgSizeY, const int imgSizeX,             \
                    const int filterSize, const int paddingStart,       \
                    const int moduleStride, const int imgStride,        \
                    const int partialSum,                               \
                    const float scaleTargets, const float scaleOutputs
/*
 * Each block computes weight gradients for B_Y * pixelsPerThread pixels and B_X filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines pixel batch of B_Y * pixelsPerThread
 *
 * Number of filters must be divisible by B_X * filtersPerThread
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * numModules must be divisible by partialSum
 * pixelsPerThread must be divisible by pixelCache
 *
 * After adding pixelsPerThread, register usage went from 20 to 23 (when pixelsPerThread = 1)...
 * so the compiler is messing up here somehow. It's unable to optimize that case away.
 */
template <int B_Y, int B_X, int pixelCache, int pixelsPerThread, int filtersPerThread, int preloadCases, int numColors, bool scale, bool checkCaseBounds>
__global__ void conv_weight_acts_c_kepler(C_KEP_PARAM);



#define MC_MF_KEP_PARAM float* images,                      \
        float* hidActs, float* targets,                     \
        const int numImages, const int numFilters,          \
        const int numModulesY, const int numModulesX,       \
        const int imgSizeY, const int imgSizeX,             \
        const int filterSize, const int paddingStart,       \
        const int moduleStride, const int imgStride,        \
        const int numImgColors, const int numGroups,        \
        const int partialSum,                               \
        const float scaleTargets, const float scaleOutputs
/*
 * Each block computes weight gradients for 1 pixel, B_Y * colorsPerThread colors and B_X * filtersPerThread filters
 * threadIdx.x determines filter
 * threadIdx.y determines color
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines color batch of B_Y * colorsPerThread
 * blockIdx.z determines pixel in filter
 *            NOTE: blockIdx.z is limited to values < 2^16. This means that this routine will
 *                  fail for filters >= 256*256. I'm assuming I won't ever use such large filters.

 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters)

 * B_X * B_Y must be divisible by preloadCases
 */
template <int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases, bool scale>
__global__ void conv_weight_acts_mc_mf_kepler(MC_MF_KEP_PARAM);

#define MC_MF_KEP_SW_PARAM float* images,               \
    float* hidActs, float* targets,                     \
    const int numImages, const int numFilters,          \
    const int numModulesY, const int numModulesX,       \
    const int imgSizeY, const int imgSizeX, const       \
    int filterSize, const int paddingStart,             \
    const int moduleStride, const int imgStride,        \
    const int numImgColors, const int numGroups,        \
    const int sumWidth,                                 \
    const float scaleTargets, const float scaleOutputs
/*
 * Each block computes weight gradients for 1 pixel, B_Y * colorsPerThread colors and B_X * filtersPerThread filters
 * threadIdx.x determines filter
 * threadIdx.y determines color
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines color batch of B_Y * colorsPerThread
 * blockIdx.z determines pixel in filter
 *            NOTE: blockIdx.z is limited to values < 2^16. This means that this routine will
 *                  fail for filters >= 256*256. I'm assuming I won't ever use such large filters.

 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters)

 * B_X * B_Y must be divisible by preloadCases
 */
template <int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases, bool scale>
__global__ void conv_weight_acts_mc_mf_kepler_sw(MC_MF_KEP_SW_PARAM);




#define C_KEP_SW_PARAM float* images,                           \
            float* hidActs, float* targets,                     \
            const int numImages, const int numFilters,          \
            const int numModulesY, const int numModulesX,       \
            const int imgSizeY, const int imgSizeX,             \
            const int filterSize, const int paddingStart,       \
            const int moduleStride, const int imgStride,        \
            const int sumWidth,                                 \
            const float scaleTargets, const float scaleOutputs
/*
 * Each block computes weight gradients for B_Y * pixelsPerThread pixels and B_X filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines pixel batch of B_Y * pixelsPerThread
 *
 * Number of filters must be divisible by B_X * filtersPerThread
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * numModules must be divisible by partialSum
 * pixelsPerThread must be divisible by pixelCache
 *
 * After adding pixelsPerThread, register usage went from 20 to 23 (when pixelsPerThread = 1)...
 * so the compiler is messing up here somehow. It's unable to optimize that case away.
 */
template <int B_Y, int B_X, int pixelCache, int pixelsPerThread, int filtersPerThread, int preloadCases, int numColors, bool scale, bool checkCaseBounds>
__global__ void conv_weight_acts_c_kepler_sw(float* images, float* hidActs, float* targets,
                                   const int numImages, const int numFilters,
                                   const int numModulesY, const int numModulesX,
                                   const int imgSizeY, const int imgSizeX, const int filterSize,
                                   const int paddingStart, const int moduleStride, const int imgStride,
                                   const int sumWidth,
                                   const float scaleTargets, const float scaleOutputs);

} // namespace cuda
} // namespace megdnn
