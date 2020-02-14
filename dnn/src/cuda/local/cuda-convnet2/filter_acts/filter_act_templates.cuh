/**
 * \file dnn/src/cuda/local/cuda-convnet2/filter_acts/filter_act_templates.cuh
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

__device__ inline void
    filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords
        (int filterSize, int imgSizeX,
         int imgLoadModPosY, int imgLoadModPosX,
         int imgY, int imgX, int& fPidx, int& iPidx) {
    int filterPxY = imgY - imgLoadModPosY;
    int filterPxX = imgX - imgLoadModPosX;
    fPidx = filterPxY * filterSize + filterPxX;
    iPidx = imgY * imgSizeX + imgX; // Pixel index in img
}

#define FILTER_ACTS_PARAMS  cudaTextureObject_t images,                         \
                            cudaTextureObject_t filters, float* targets,        \
                            const int numImages, const int numFilters,          \
                            const int imgSizeY, const int imgSizeX,             \
                            const int filterSize, const int paddingStart,       \
                            const int moduleStride,                             \
                            const int numModulesY, const int numModulesX,       \
                            const int imgStride, const int numImgColors,        \
                            const int numGroups,                                \
                            const float scaleTargets, const float scaleOutputs, \
                            const bool conv/*, const bool noloads*/
/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex (FILTER_ACTS_PARAMS);



#define FILTER_COLOR_PARAMS float* images, float* filters, float* targets,      \
                            const int numImages, const int numFilters,          \
                            const int imgSizeY, const int imgSizeX,             \
                            const int filterSize, const int paddingStart,       \
                            const int moduleStride,                             \
                            const int numModulesY, const int numModulesX,       \
                            const int imgStride,                                \
                            const float scaleTargets, const float scaleOutputs, \
                            const bool conv
/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of module and B_Y * filtersPerThread
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numColors, filterPixels, numFilters) if conv
 *              (numModules, numColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 *
 * Number of filters per module should be divisible by B_Y * filtersPerThread
 * checkImgBounds indicates whether number of images is divisible by B_X * imgsPerThread
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
 template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, int pixelCache, bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_color(FILTER_COLOR_PARAMS);




#define FILTER_SPARSE2_PARAMS float* images, float* filters, float* targets,        \
                              const int numImages, const int numFilters,            \
                              const int imgSizeY, const int imgSizeX,               \
                              const int filterSize, const int paddingStart,         \
                              const int moduleStride,                               \
                              const int numModulesY, const int numModulesX,         \
                              const int imgStride, const int numImgColors,          \
                              const int numGroups,                                  \
                              const float scaleTargets, const float scaleOutputs,   \
                              const bool conv
/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of B_Y * filtersPerThread
 *
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 * colorCache: how many colors to put into shmem
 *
 * numFilters should be divisible by B_Y * filtersPerThread
 * numImages be divisible by B_X * imgsPerThread
 * numFilterColors should be divisible by colorCache.
 * numImgColors must be even.
 * numFilters must be divisible by numGroups.
 * no restrictions on pixelCache
 * The imgSize here is the size of the actual image without the padding.
 * As always, try to make B_X * imgsPerThread == B_Y * filtersPerThread for maximum efficiency.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse2(FILTER_SPARSE2_PARAMS);

} // namespace megdnn
} // namespace cuda
