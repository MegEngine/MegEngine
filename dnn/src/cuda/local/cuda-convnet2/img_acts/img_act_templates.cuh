/**
 * \file dnn/src/cuda/local/cuda-convnet2/img_acts/img_act_templates.cuh
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

#define MANYCOLOR_KEP_PARAM const float* hidActs,       \
        const float* filters, float* targets,           \
        const int numModulesY, const int numModulesX,   \
        const int numImages, const int numFilters,      \
        const int filterSize, const int imgSizeY,       \
        const int imgSizeX, const int paddingStart,     \
        const int moduleStride,                         \
        const int numImgColors, const int numGroups,    \
        const float scaleTargets, const float scaleOutputs

/*
 * Block size: B_YxB_X.
 * blockIdx.x determines case in batches of B_X*imgsPerThread, also color in batches of B_Y*colorsPerThread.
 *  In essence, blockIdx.x.x = 1..numImages/(B_X*imgsPerThread)
 *              blockIdx.x.y = 1..numImgColors/(B_Y*colorsPerThread)
 * blockIdx.y determines image pixel in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines color.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numFilterColors, filterPixels, numFilters)                             if conv
 *              (numModulesY, numModulesX, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:     (numImageColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one B_Y*colorsPerThread colors from 1 pixel from B_X*imgsPerThread cases.
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false.
 * numFiltersPerGroup must be divisible by filterCacheF.
 *
 * numFilterColors must be divisible by B_Y*colorsPerThread.
 * B_X*B_Y must be divisible by filterCacheF
 * filterCacheF must be divisible by filterCacheH
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads filterCacheF weights at a time, so those aren't fully coalesced (depending on size of filterCacheF).
 *
 * To be used when there are >= 16 color channels.
 */
template <int B_Y, int B_X,
    int imgsPerThread, int colorsPerThread,
    int filterCacheF, int filterCacheH,
    bool scale, bool checkCaseBounds, bool conv>
__global__ void conv_img_acts_manycolor_kepler(MANYCOLOR_KEP_PARAM);



#define MED_COLOR_KEP_PARAM const float* hidActs,           \
        const float* filters, float* targets,               \
        const int numModulesY, const int numModulesX,       \
        const int numImages, const int numFilters,          \
        const int filterSize,                               \
        const int imgSizeY, const int imgSizeX,             \
        const int paddingStart, const int moduleStride,     \
        const int numImgColors, const int numGroups,        \
        const float scaleTargets, const float scaleOutputs
/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread, also color in batches of colorsPerThread.
 *  In essence, blockIdx.x.x = 1..numImages/(16*imgsPerThread)
 *              blockIdx.x.y = 1..numImgColors/colorsPerThread
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numFilterColors, filterPixels, numFilters)                             if conv
 *              (numModulesY, numModulesX, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:     (numImageColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * numImages must be divisible by 16*imgsPerThread if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 * numImageColors/numGroups must be divisible by colorsPerThread.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 *
 * To be used when there are 4-16 color channels.
 */
template <int imgsPerThread, int colorsPerThread,  bool scale, bool checkCaseBounds, bool conv>
__global__ void img_acts_mediumcolor(MED_COLOR_KEP_PARAM);


#define COLOR_KEP_PARAM const float* hidActs,               \
        const float* filters, float* targets,               \
        const int numModulesY, const int numModulesX,       \
        const int numImages, const int numFilters,          \
        const int filterSize,                               \
        const int imgSizeY, const int imgSizeX,             \
        const int paddingStart, const int moduleStride,     \
        const float scaleTargets, const float scaleOutputs

/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread.
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numColors, filterPixels, numFilters)                               if conv
 *              (numModulesY, numModulesX, numColors, filterPixels, numFilters)     otherwise
 * targets:     (numColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * Number of filters must be divisible by 16.
 * Number of images must be divisible by 16*imgsPerThread  if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 */
template <int imgsPerThread, int numColors, bool scale, bool checkCaseBounds, bool conv>
__global__ void img_acts_color(COLOR_KEP_PARAM);

} // namespace megdnn
} // namespace cuda
