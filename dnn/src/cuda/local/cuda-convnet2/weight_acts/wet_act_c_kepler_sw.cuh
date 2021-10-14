/**
 * \file dnn/src/cuda/local/cuda-convnet2/weight_acts/wet_act_c_kepler_sw.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
 * * All Megvii Modifications are Copyright (C) 2014-2021 Megvii Inc. All rights
 * reserved.
 * --------------------------------------------------------------------------
 */
#include "wet_act_templates.cuh"

namespace megdnn {
namespace cuda {

/*
 * Each block computes weight gradients for B_Y * pixelsPerThread pixels and B_X filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of
 * partialSum blockIdx.y determines pixel batch of B_Y * pixelsPerThread
 *
 * Number of filters must be divisible by B_X * filtersPerThread
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is
 * false.
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
 * After adding pixelsPerThread, register usage went from 20 to 23 (when pixelsPerThread
 * = 1)... so the compiler is messing up here somehow. It's unable to optimize that case
 * away. To be used when numFilterColors <= 3
 */
template <
        int B_Y, int B_X, int pixelCache, int pixelsPerThread, int filtersPerThread,
        int preloadCases, int numColors, bool scale, bool checkCaseBounds>
__global__ void conv_weight_acts_c_kepler_sw(
        float* images, float* hidActs, float* targets, const int numImages,
        const int numFilters, const int numModulesY, const int numModulesX,
        const int imgSizeY, const int imgSizeX, const int filterSize,
        const int paddingStart, const int moduleStride, const int imgStride,
        const int sumWidth, const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImages[pixelCache * B_Y * numColors]
                             [preloadCases];  // preload preloadCases cases of B_Y *
                                              // pixelsPerThread pixels
    __shared__ float
            shHidActs[B_X * filtersPerThread]
                     [preloadCases + 1];  // preload preloadCases cases of B_X hidActs
    fill_shared_mem<float>((float*)shImages, sizeof(shImages) / sizeof(float), 0);
    fill_shared_mem<float>((float*)shHidActs, sizeof(shHidActs) / sizeof(float), 0);
    __syncthreads();
    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = DIVUP(numFilters, B_X * filtersPerThread);

    const int blockModuleChunkIdx = blockIdx.x / numFilterBlocks;

    const int numModuleChunksX = DIVUP(numModulesX, sumWidth);
    //    const int numModuleChunksY = DIVUP(numModulesY, sumWidth);

    const int blockModuleChunkX = blockModuleChunkIdx % numModuleChunksX;
    const int blockModuleChunkY = blockModuleChunkIdx / numModuleChunksX;

    const int blockModuleStartX = blockModuleChunkX * sumWidth;
    const int blockModuleStartY = blockModuleChunkY * sumWidth;

    const int blockFilterIdx = B_X * filtersPerThread * (blockIdx.x % numFilterBlocks);

    //    const int moduleStride = (imgSize - filterSize + 1) / numModulesX;
    const int numModules = numModulesY * numModulesX;

    const int blockPixelOffset = blockIdx.y * B_Y * pixelsPerThread;

    images += loadX;
    hidActs += blockFilterIdx * numImages * numModules
             //            + loadY * numImages * numModules
             + loadX;

    targets += (blockModuleChunkIdx * numFilters) * filterPixels * numColors +
               blockPixelOffset * numFilters + blockFilterIdx +
               threadIdx.y * numFilters + threadIdx.x;

    // float* shImgLoad = &shImages[loadY][loadX];
    // float* shHidActLoad = &shHidActs[loadY][loadX];

    float prod[numColors][pixelsPerThread][filtersPerThread];
#pragma unroll
    for (int c = 0; c < numColors; c++) {
#pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
#pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                prod[c][p][f] = 0;
            }
        }
    }
    const int mStartX = blockModuleStartX;
    const int mStartY = blockModuleStartY;
    const int mEndX = min(numModulesX, blockModuleStartX + sumWidth);
    const int mEndY = min(numModulesY, blockModuleStartY + sumWidth);

    //    if (mStartY == mEndY || mStartX == mEndX) {
    //        return;
    //    }

    const int fYOff = (blockPixelOffset + tidx) / filterSize;
    const int fXOff = (blockPixelOffset + tidx) % filterSize;
    __shared__ int pxIdxes[B_Y * pixelsPerThread];
    fill_shared_mem<int>((int*)pxIdxes, sizeof(pxIdxes) / sizeof(int), 0);
    __syncthreads();
    for (int my = mStartY; my < mEndY; my++) {
        const int imgLoadModPosY = paddingStart + my * moduleStride;
        for (int mx = mStartX; mx < mEndX; mx++) {
            const int m = my * numModulesX + mx;

            __syncthreads();
            const int imgLoadModPosX = paddingStart + mx * moduleStride;
            if (tidx < B_Y * pixelsPerThread) {
                //                const int imgLoadModPosY = paddingStart + my *
                //                moduleStride; const int imgLoadModPosX = paddingStart
                //                + mx * moduleStride;
                int pxY = (imgLoadModPosY + fYOff);
                int pxX = (imgLoadModPosX + fXOff);
                int pixIdx = (pxY * imgSizeX + pxX) * imgStride;
                pxIdxes[tidx] = pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX
                                      ? pixIdx
                                      : -1;
            }
            __syncthreads();
            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                if (  // loadY < B_X*filtersPerThread &&
                        (!checkCaseBounds || caseIdx + loadX < numImages)) {
#pragma unroll
                    for (int y = 0; y < B_X * filtersPerThread;
                         y += (B_X * B_Y) / preloadCases) {
                        const int fIdx = ((loadY + y) % filtersPerThread) * B_X +
                                         (loadY + y) / filtersPerThread;
                        // Make sure number of rows in the array is divisible by number
                        // of rows filled per iteration
                        if ((B_X * filtersPerThread) % (B_X * B_Y / preloadCases) ==
                                    0 ||
                            loadY + y < B_X * filtersPerThread) {
                            if (blockFilterIdx + fIdx < numFilters) {
                                shHidActs[loadY + y][loadX] =
                                        hidActs[caseIdx +
                                                (fIdx * numModules + m) * numImages];
                            } else {
                                shHidActs[loadY + y][loadX] = 0;
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int y = 0; y < B_X * filtersPerThread;
                         y += (B_X * B_Y) / preloadCases) {
                        //                        const int fIdx = ((loadY + y) %
                        //                        filtersPerThread) * B_X + (loadY + y)
                        //                        / filtersPerThread;
                        // Make sure number of rows in the array is divisible by number
                        // of rows filled per iteration
                        if ((B_X * filtersPerThread) % (B_X * B_Y / preloadCases) ==
                                    0 ||
                            loadY + y < B_X * filtersPerThread) {
                            shHidActs[loadY + y][loadX] = 0;
                        }
                    }
                }
#pragma unroll
                for (int pp = 0; pp < pixelsPerThread; pp += pixelCache) {
// if (loadY < B_Y * pixelCache) { // This condition is not necessary for correctness,
// but it speeds things a bit
//
// As long as B_Y * B_X is divisible by preloadCases this will loop the right
// number of times.
//
// This will load some imgGrads from filter pixels that don't exit (it'll set those to
// 0), but the code does not produce any output for those pixels (see last lines).
//
#pragma unroll
                    for (int y = 0; y < B_Y * pixelCache;
                         y += (B_X * B_Y) / preloadCases) {
                        // Make sure number of rows in the array is divisible by number
                        // of rows filled per iteration
                        if ((B_Y * pixelCache) % (B_X * B_Y / preloadCases) == 0 ||
                            y + loadY < B_Y * pixelCache) {
                            const int pxIdx =
                                    pp * B_Y + loadY + y;  // pixel idx in filter

                            if (pxIdx + blockPixelOffset < filterPixels &&
                                (!checkCaseBounds || caseIdx + loadX < numImages)) {
                                const int pixIdx = pxIdxes
                                        [pxIdx];  //(pxY * imgSizeX + pxX) * imgStride;

                                if (pixIdx >= 0) {
#pragma unroll
                                    for (int c = 0; c < numColors; c++) {
                                        shImages[loadY + y + c * pixelCache * B_Y]
                                                [loadX] =
                                                        images[caseIdx +
                                                               c * imgPixels *
                                                                       imgStride +
                                                               pixIdx];
                                    }
                                } else {
#pragma unroll
                                    for (int c = 0; c < numColors; c++) {
                                        shImages[loadY + y + c * pixelCache * B_Y]
                                                [loadX] = 0;
                                    }
                                }
                            } else {
#pragma unroll
                                for (int c = 0; c < numColors; c++) {
                                    shImages[loadY + y + c * pixelCache * B_Y][loadX] =
                                            0;
                                }
                            }
                        }
                    }
                    //}

                    __syncthreads();

#pragma unroll
                    for (int c = 0; c < numColors; c++) {
#pragma unroll
                        for (int i = 0; i < preloadCases; i++) {
#pragma unroll
                            for (int p = 0; p < pixelCache; p++) {
#pragma unroll
                                for (int f = 0; f < filtersPerThread; f++) {
                                    if (blockFilterIdx + threadIdx.x + f * B_X <
                                        numFilters) {
                                        prod[c][pp + p][f] +=
                                                shImages
                                                        [threadIdx.y +
                                                         (p + c * pixelCache) * B_Y]
                                                        [i] *
                                                shHidActs
                                                        [threadIdx.x *
                                                                 filtersPerThread +
                                                         f][i];
                                    }
                                }
                            }
                        }
                    }

                    __syncthreads();
                }
            }
        }
    }
    if (scale) {
#pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
#pragma unroll
                for (int c = 0; c < numColors; c++) {
#pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        targets[(p * B_Y + c * filterPixels) * numFilters + f * B_X] =
                                scaleTargets * targets[(p * B_Y + c * filterPixels) *
                                                               numFilters +
                                                       f * B_X] +
                                scaleOutputs * prod[c][p][f];
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
#pragma unroll
                for (int c = 0; c < numColors; c++) {
#pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        if (blockFilterIdx + threadIdx.x + f * B_X < numFilters) {
                            targets[(p * B_Y + c * filterPixels) * numFilters +
                                    f * B_X] = scaleOutputs * prod[c][p][f];
                        }
                    }
                }
            }
        }
    }
}

#define WET_ACT_C_KEPLER_SW_HEAD template __global__ void conv_weight_acts_c_kepler_sw

}  // namespace cuda
}  // namespace megdnn
