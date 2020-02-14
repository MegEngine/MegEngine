/**
 * \file dnn/src/cuda/local/cuda-convnet2/img_acts/img_act_color.cuh
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
#include "img_act_templates.cuh"

namespace megdnn {
namespace cuda {

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
__global__ void img_acts_color(const float* hidActs, const float* filters, float* targets,
                                   const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                                   const int filterSize, const int imgSizeY, const int imgSizeX,
                                   const int paddingStart, const int moduleStride,
                                   const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[numColors*16][16 + 1];
    __shared__ float shHidActs[16][16*imgsPerThread];
    fill_shared_mem<float>((float *)shFilters, sizeof(shFilters)/sizeof(float), 0);
    fill_shared_mem<float>((float *)shHidActs, sizeof(shHidActs)/sizeof(float), 0);
    __syncthreads();

    const int blockCaseIdx = blockIdx.x * 16*imgsPerThread;
    const int numRegionsX = DIVUP(imgSizeX, 4);
    const int blockRegionIdx = blockIdx.y;
    const int blockRegionIdxX = blockRegionIdx % numRegionsX;
    const int blockRegionIdxY = blockRegionIdx / numRegionsX;
    const int blockRegionLeft = blockRegionIdxX * 4;
    const int blockRegionTop = blockRegionIdxY * 4;
    const int pxYInRegion = threadIdx.y / 4, pxXInRegion = threadIdx.y % 4;
    const int pxY = blockRegionTop + pxYInRegion;
    const int pxX = blockRegionLeft + pxXInRegion;
    const int pxIdx = pxY * imgSizeX + pxX;
    const bool isPxInImg = pxY < imgSizeY && pxX < imgSizeX;
    const int numModules = numModulesY * numModulesX;
    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeX * imgSizeY;
    const int tidx = threadIdx.y * 16 + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    hidActs += blockCaseIdx + loadY * numImages * numModules + loadX;
    filters += threadIdx.x;
    targets += pxIdx * numImages + blockCaseIdx + threadIdx.x;


    float prod[numColors][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < numColors; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }
    const int startY = blockRegionTop - paddingStart < filterSize ? 0
                        : 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
    const int startX = blockRegionLeft - paddingStart < filterSize ? 0
                        : 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);

    float* shilterLoad = &shFilters[threadIdx.y][threadIdx.x];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInModuleY = pxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInModuleX = pxX - moduleLeft;

            const bool isPxInModule = pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
            const int pxIdxInModule = pxInModuleY * filterSize + pxInModuleX;

            for (int f = 0; f < numFilters; f += 16) { // multiply with 16 filters at a time
                // Now the threads split up into half-warps, and each half-warp decides if it's interested.
                const float* hLoad = &hidActs[(moduleIdx + f * numModules) * numImages];
                #pragma unroll
                for (int i = 0; i < imgsPerThread * 16; i += 32) {
                    if (!checkCaseBounds || blockCaseIdx + i + loadX < numImages) {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            if (f + loadY + j < numFilters) {
                                shHidActLoad[j * 16 * imgsPerThread + i] = hLoad[j * numModules * numImages + i];
                            } else {
                                shHidActLoad[j * 16 * imgsPerThread + i] = 0;
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * 16 * imgsPerThread + i] = 0;
                        }
                    }
                }

                if (isPxInImg && isPxInModule) {
                    // This half-warp is interested, so it's going to load the weights from this module to its pixel.
                    // Not fully coalesced read :(
                    // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
                    const float* fLoad = conv ? &filters[pxIdxInModule * numFilters + f]
                                              : &filters[(moduleIdx * numColors * filterPixels + pxIdxInModule) * numFilters + f];
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        if (f + threadIdx.x < numFilters) {
                            shilterLoad[c * 16 * (16 + 1)] = fLoad[c * filterPixels * numFilters];
                        } else {
                            shilterLoad[c * 16 * (16 + 1)] = 0;
                        }
                    }


                }

                __syncthreads();
                // Do some actual computation
                if (isPxInImg && isPxInModule) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        #pragma unroll
                        for (int w = 0; w < 16; w++) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread; i++) {
                                prod[c][i] += shFilters[threadIdx.y + c * 16][w] * shHidActs[w][threadIdx.x + i * 16];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
    if (isPxInImg) {
        if (scale) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleTargets * targets[c * imgPixels * numImages + i * 16] + scaleOutputs * prod[c][i];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleOutputs * prod[c][i];
                    }
                }
            }
        }
    }
}

#define IMG_COLOR_K_HEAD template __global__ void img_acts_color
#define IMG_COLOR_K(scale, ckCase, conv) \
    IMG_COLOR_K_HEAD < 8, 2, scale, ckCase, conv >(COLOR_KEP_PARAM); \
    IMG_COLOR_K_HEAD < 4, 2, scale, ckCase, conv >(COLOR_KEP_PARAM); \
    IMG_COLOR_K_HEAD < 2, 2, scale, ckCase, conv >(COLOR_KEP_PARAM); \
    IMG_COLOR_K_HEAD < 8, 3, scale, ckCase, conv >(COLOR_KEP_PARAM); \
    IMG_COLOR_K_HEAD < 4, 3, scale, ckCase, conv >(COLOR_KEP_PARAM); \
    IMG_COLOR_K_HEAD < 2, 3, scale, ckCase, conv >(COLOR_KEP_PARAM); \
    IMG_COLOR_K_HEAD < 8, 1, scale, ckCase, conv >(COLOR_KEP_PARAM); \
    IMG_COLOR_K_HEAD < 4, 1, scale, ckCase, conv >(COLOR_KEP_PARAM); \
    IMG_COLOR_K_HEAD < 2, 1, scale, ckCase, conv >(COLOR_KEP_PARAM);

} // namespace cuda
} // namespace megdnn
