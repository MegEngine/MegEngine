/**
 * \file dnn/src/cuda/local/cuda-convnet2/filter_acts/filter_act_color.cuh
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
__global__ void filterActs_YxX_color(FILTER_COLOR_PARAMS) {
    __shared__ float shFilters[pixelCache*numColors][B_Y * filtersPerThread]; // pre-load pixelCache pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[pixelCache*numColors][B_X * imgsPerThread]; // pre-load pixelCache pixels from B_X*imgsPerThread images
    fill_shared_mem<float>((float *)shFilters, sizeof(shFilters)/sizeof(float), 0);
    fill_shared_mem<float>((float *)shImages, sizeof(shImages)/sizeof(float), 0);
    __syncthreads();
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;

    const int blocksPerModule = DIVUP(numFilters, (B_Y*filtersPerThread));
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;
    const int numModules = numModulesY * numModulesX;
    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
    images += myImgIdx;
    filters += blockFilterIdx
             + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numColors * filterPixels * numFilters;
    }
    bool active_thread_y = (blockFilterIdx + shFilterLoadX) < numFilters;

    targets += moduleIdx * numImages
            + myImgIdx
            + (blockFilterIdx + threadIdx.y*filtersPerThread) * numImages * numModulesY * numModulesX;


    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }
    //float* shImgLoad = &shImages[0][threadIdx.x];
    for (int p = 0; p < filterPixels; p += pixelCache) {
        /*
         * Load pixelCache pixels from B_Y*filtersPerThread filters
         * This condition covers the case when B_X is not divisible by filtersPerThread.
         * In this case, not all of the threads will participate in the loading operation.
         * This ensures that in each loop iteration, an integer number of rows of shFilters
         * are filled, which makes indexing simple.
         */
        if (B_X % filtersPerThread == 0 || shFilterLoadY < B_X/filtersPerThread) {
            #pragma unroll
            for (int p2 = 0; p2 < pixelCache; p2 += B_X/filtersPerThread) {
                const bool omit = pixelCache % (B_X / filtersPerThread) == 0;
                const int preloadPx = shFilterLoadY + p2;
                if (omit || preloadPx < pixelCache) {
                    if (p + preloadPx < filterPixels && active_thread_y) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shFilters[shFilterLoadY + p2 + c * pixelCache][shFilterLoadX] = filters[(c * filterPixels + p + p2) * numFilters];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shFilters[shFilterLoadY + p2 + c * pixelCache][shFilterLoadX] = 0;
                        }
                    }
                }
            }
        }

        /*
         * Load pixelCache pixels from B_X*imgsPerThread images.
         */
        #pragma unroll
        for (int ly = 0; ly < pixelCache; ly += B_Y) {
            const int preloadPx = ly + threadIdx.y;
            const int pixIdx = p + preloadPx;
            const bool omit = pixelCache % B_Y == 0; // Compile-time condition
            /*
             * Don't load any image pixels corresponding to filter pixels that don't exist.
             */
            if (pixIdx < filterPixels && (omit || preloadPx < pixelCache)) {
                const int x = imgLoadModPosX + pixIdx % filterSize;
                const int y = imgLoadModPosY + pixIdx / filterSize;

                if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                    float* m = &images[imgStride * (y * imgSizeX + x)];

                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                                shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = m[c * imgStride * imgPixels + i * B_X];
                            } else {
                                shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = 0;
                        }
                    }
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < pixelCache*numColors; i++) {
            #pragma unroll
            for(int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for(int g = 0; g < imgsPerThread; g++) {
                    prod[f][g] += shImages[i][g + threadIdx.x * imgsPerThread]
                                  * shFilters[i][threadIdx.y * filtersPerThread + f];
                }
            }
        }
        __syncthreads();
    }

    int filtersThisThread = numFilters - blockFilterIdx - threadIdx.y * filtersPerThread;
    if (filtersThisThread > filtersPerThread) {
        filtersThisThread = filtersPerThread;
    }

    //active_thread_y = (blockFilterIdx + threadIdx.y * filtersPerThread) < numFilters;
    if (scale) {
        #pragma unroll
        for (int f = 0; f < filtersThisThread; f++) {
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                    targets[g * B_X + f * numImages * numModules] =
                        scaleTargets * targets[g * B_X + f * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersThisThread; f++) {
                    //if (active_thread_y) {
                        targets[g * B_X + f * numImages * numModules] = scaleOutputs * prod[f][g];
                    //}
                }
            }
        }
    }
}


#define FILTER_COLOR_HEAD template __global__ void filterActs_YxX_color

#define FILTER_COLOR(scale, ckImg) \
FILTER_COLOR_HEAD < 4, 32, 4, 8, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
FILTER_COLOR_HEAD < 4, 32, 4, 4, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
                                                            \
FILTER_COLOR_HEAD < 4, 32, 2, 16, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 2, 12, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 2, 8, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
FILTER_COLOR_HEAD < 4, 32, 2, 4, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
                                                            \
FILTER_COLOR_HEAD < 4, 32, 1, 16, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 1, 12, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 1, 8, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
FILTER_COLOR_HEAD < 4, 32, 1, 4, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
                                                            \
FILTER_COLOR_HEAD < 4, 32, 4, 16, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 4, 12, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 4, 8, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
FILTER_COLOR_HEAD < 4, 32, 4, 4, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
                                                            \
FILTER_COLOR_HEAD < 4, 32, 2, 16, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 2, 12, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 2, 8, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
FILTER_COLOR_HEAD < 4, 32, 2, 4, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
                                                            \
FILTER_COLOR_HEAD < 4, 32, 1, 16, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 1, 12, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 1, 8, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
FILTER_COLOR_HEAD < 4, 32, 1, 4, 2, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
                                                            \
FILTER_COLOR_HEAD < 4, 32, 4, 16, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 4, 12, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 4, 8, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
FILTER_COLOR_HEAD < 4, 32, 4, 4, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
                                                            \
FILTER_COLOR_HEAD < 4, 32, 2, 16, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 2, 12, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 2, 8, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
FILTER_COLOR_HEAD < 4, 32, 2, 4, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
                                                            \
FILTER_COLOR_HEAD < 4, 32, 1, 16, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 1, 12, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 1, 8, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
FILTER_COLOR_HEAD < 4, 32, 1, 4, 1, 4, scale, ckImg > (FILTER_COLOR_PARAMS);        \
\
FILTER_COLOR_HEAD < 4, 32, 4, 16, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \
FILTER_COLOR_HEAD < 4, 32, 4, 12, 3, 4, scale, ckImg > (FILTER_COLOR_PARAMS);       \

} // namespace cuda
} // namespace megdnn
