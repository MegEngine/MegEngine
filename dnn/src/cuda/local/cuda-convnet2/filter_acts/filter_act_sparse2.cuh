/**
 * \file dnn/src/cuda/local/cuda-convnet2/filter_acts/filter_act_sparse2.cuh
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
#include "filter_act_templates.cuh"

namespace megdnn {
namespace cuda {

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
__global__ void filterActs_YxX_sparse2(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX,
                                       const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX,
                                       const int imgStride, const int numImgColors,
                                       const int numGroups,
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv) {
    __shared__ float shFilters[colorCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[colorCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
    fill_shared_mem<float>((float *)shFilters, sizeof(shFilters)/sizeof(float), 0);
    fill_shared_mem<float>((float *)shImages, sizeof(shImages)/sizeof(float), 0);
    __syncthreads();
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = DIVUP(numFilters, (B_Y*filtersPerThread));
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx + shFilterLoadX
            + shFilterLoadY * numFilters * filterPixels;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }
    bool active_thread_y = (blockFilterIdx + shFilterLoadX) < numFilters;

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y) * numImages * numModules
            + myImgIdx;

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }
    const int imgStartX = MAX(0, imgLoadModPosX);
    const int imgStartY = MAX(0, imgLoadModPosY);
    const int imgEndX = MIN(imgLoadModPosX + filterSize, imgSizeX);
    const int imgEndY = MIN(imgLoadModPosY + filterSize, imgSizeY);
//    __shared__ int imgPos[]

    for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
        const int filterPxY = imgY - imgLoadModPosY;
        for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
            const int filterPxX = imgX - imgLoadModPosX;
            const int p = filterPxY * filterSize + filterPxX;
            for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)

                /*
                 * Load a pixel from B_Y*filtersPerThread filters
                 * This condition covers the case when B_X is not divisible by filtersPerThread.
                 * In this case, not all of the threads will participate in the loading operation.
                 * This ensures that in each loop iteration, an integer number of rows of shFilters
                 * are filled, which makes indexing simple.

                 * nvcc is behaving in a completely insane way: removing this condition under
                 * template parameters that guarantee it to be true actually slows down
                 * the computation.
                 *
                 */
                if (/*B_X % filtersPerThread == 0 ||*/ shFilterLoadY < B_X/filtersPerThread) {
                    #pragma unroll
                    for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
                        if (colorCache % (B_X/filtersPerThread) == 0 || c + shFilterLoadY < colorCache) {
                            if (active_thread_y) {
                                shFilters[c + shFilterLoadY][shFilterLoadX] = filters[((oc+c) * filterPixels + p) * numFilters];
                            } else {
                                shFilters[c + shFilterLoadY][shFilterLoadX] = 0;
                            }
                        }
                    }
                }

                /*
                 * Load a pixel from B_X*imgsPerThread images.
                 */
                const int pixIdx = imgY * imgSizeX + imgX;// Pixel index in img

                float* m = &images[imgStride * (oc * imgPixels + pixIdx)];
                #pragma unroll
                for (int c = 0; c < colorCache; c += B_Y) {
                    if (colorCache % B_Y == 0 || threadIdx.y + c < colorCache) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                                shImages[c + threadIdx.y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
                            } else {
                                shImages[c + threadIdx.y][threadIdx.x + i * B_X] = 0;
                            }
                        }
                    }
                }

                __syncthreads();

                for (int c = 0; c < colorCache; c++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        #pragma unroll
                        for(int f = 0; f < filtersPerThread; f++) {
                            prod[f][g] += shImages[c][g * B_X + threadIdx.x] * shFilters[c][threadIdx.y + f * B_Y];
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    int filtersThisThread = filtersPerThread;
    //if(checkFilterBounds) {
        int filtersThisBlock = numFilters - (blockIdx.y % blocksPerModule)
                               * (B_Y*filtersPerThread);
        if (filtersThisBlock < (B_Y * filtersPerThread)) {
            filtersThisThread = (filtersThisBlock - threadIdx.y + filtersPerThread - 1) / filtersPerThread;
        }
    //}

    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersThisThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
        #pragma unroll
        for (int f = 0; f < filtersThisThread; f++) {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
                }
            }
        }
    }
}

#define FILTER_SPARSE2_HEAD template __global__ void filterActs_YxX_sparse2

// <B_Y, B_X, imgsPerThread, filtersPerThread, colorCache, scale, checkImgBounds>
#define FILTER_SPARSE2(scale, ckImg) \
FILTER_SPARSE2_HEAD < 4, 32, 4, 8, 8, scale, ckImg > (FILTER_SPARSE2_PARAMS);   \
FILTER_SPARSE2_HEAD < 4, 32, 4, 4, 8, scale, ckImg > (FILTER_SPARSE2_PARAMS);   \
\
FILTER_SPARSE2_HEAD < 8, 32, 2, 16, 8, scale, ckImg > (FILTER_SPARSE2_PARAMS);  \
FILTER_SPARSE2_HEAD < 4, 32, 2, 16, 8, scale, ckImg > (FILTER_SPARSE2_PARAMS);  \
FILTER_SPARSE2_HEAD < 4, 32, 2, 8, 8, scale, ckImg > (FILTER_SPARSE2_PARAMS);   \
FILTER_SPARSE2_HEAD < 4, 32, 2, 4, 8, scale, ckImg > (FILTER_SPARSE2_PARAMS);   \
\
FILTER_SPARSE2_HEAD < 8, 32, 1, 16, 8, scale, ckImg > (FILTER_SPARSE2_PARAMS);  \
FILTER_SPARSE2_HEAD < 4, 32, 1, 16, 8, scale, ckImg > (FILTER_SPARSE2_PARAMS);  \
FILTER_SPARSE2_HEAD < 4, 32, 1, 8, 8, scale, ckImg > (FILTER_SPARSE2_PARAMS);   \
FILTER_SPARSE2_HEAD < 4, 32, 1, 4, 8, scale, ckImg > (FILTER_SPARSE2_PARAMS);   \
\
FILTER_SPARSE2_HEAD < 4, 32, 4, 16, 4, scale, ckImg > (FILTER_SPARSE2_PARAMS);  \
FILTER_SPARSE2_HEAD < 4, 32, 4, 8, 4, scale, ckImg > (FILTER_SPARSE2_PARAMS);   \
FILTER_SPARSE2_HEAD < 4, 32, 4, 4, 4, scale, ckImg > (FILTER_SPARSE2_PARAMS);   \
\
FILTER_SPARSE2_HEAD < 4, 32, 2, 16, 4, scale, ckImg > (FILTER_SPARSE2_PARAMS);  \
FILTER_SPARSE2_HEAD < 4, 32, 2, 8, 4, scale, ckImg > (FILTER_SPARSE2_PARAMS);   \
FILTER_SPARSE2_HEAD < 4, 32, 2, 4, 4, scale, ckImg > (FILTER_SPARSE2_PARAMS);   \
\
FILTER_SPARSE2_HEAD < 4, 32, 1, 16, 4, scale, ckImg > (FILTER_SPARSE2_PARAMS);  \
FILTER_SPARSE2_HEAD < 4, 32, 1, 8, 4, scale, ckImg > (FILTER_SPARSE2_PARAMS);   \
FILTER_SPARSE2_HEAD < 4, 32, 1, 4, 4, scale, ckImg > (FILTER_SPARSE2_PARAMS);

} // namespace cuda
} // namespace megdnn
