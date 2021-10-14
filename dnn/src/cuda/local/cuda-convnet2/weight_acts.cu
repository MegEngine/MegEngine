/**
 * \file dnn/src/cuda/local/cuda-convnet2/weight_acts.cu
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

#include "cudaconv2.cuh"

#include <utility>
#include "nvmatrix.cuh"
#include "weight_acts/wet_act_templates.cuh"

#ifdef _WIN32
#define _Pragma(x)
#endif

namespace megdnn {
namespace cuda {

__device__ __forceinline__ void
conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
        const int my, const int mx, const int paddingStart, const int numModulesX,
        const int moduleStride, const int blockPixelY, const int blockPixelX,
        const int imgSizeX, const int imgStride, int& pixIdx, int& m) {
    const int imgLoadModPosY = paddingStart + my * moduleStride;
    const int imgLoadModPosX = paddingStart + mx * moduleStride;
    const int pxY = imgLoadModPosY + blockPixelY;  // pixel x,y coords in image
    const int pxX = imgLoadModPosX + blockPixelX;
    pixIdx = (pxY * imgSizeX + pxX) * imgStride;  // pixel idx in image
    m = my * numModulesX + mx;
}

#define WA_C3_LOOP(pp, c)                                                           \
    _Pragma("unroll") for (int i = 0; i < preloadCases; i++) {                      \
        _Pragma("unroll") for (int p = 0; p < pixelCache; p++) {                    \
            _Pragma("unroll") for (int f = 0; f < filtersPerThread; f++) {          \
                prod[c][(pp) + p][f] +=                                             \
                        shImages[threadIdx.y + p * B_Y + (c)*pixelCache * B_Y][i] * \
                        shHidActs[threadIdx.x * filtersPerThread + f][i];           \
            }                                                                       \
        }                                                                           \
    }

#define WA_C3_LOOP2(pp)                                                            \
    _Pragma("unroll") for (int p = 0; p < pixelCache; p++) {                       \
        _Pragma("unroll") for (int i = 0; i < preloadCases; i++) {                 \
            _Pragma("unroll") for (int f = 0; f < filtersPerThread; f++) {         \
                _Pragma("unroll") for (int c = 0; c < 3; ++c) {                    \
                    prod[c][(pp) + p][f] +=                                        \
                            shImages[threadIdx.y + p * B_Y + (c)*pixelCache * B_Y] \
                                    [i] *                                          \
                            shHidActs[threadIdx.x * filtersPerThread + f][i];      \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    }

#define WA_3_FIDX(y)                                                     \
    (((loadY + (y)*B_X * B_Y / preloadCases) % filtersPerThread) * B_X + \
     (loadY + (y)*B_X * B_Y / preloadCases) / filtersPerThread)

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
 * away.
 */
template <
        int B_Y, int B_X, int pixelCache, int pixelsPerThread, int filtersPerThread,
        int preloadCases, int numColors, bool scale, bool checkCaseBounds>
//__launch_bounds__(256,2)
__global__ void conv_weight_acts_c_preload_pc_2_pt_2_f_4_r_32_c_3(
        cudaTextureObject_t images, cudaTextureObject_t hidActs, float* targets,
        const int numImages, const int numFilters, const int numModulesY,
        const int numModulesX, const int imgSizeY, const int imgSizeX,
        const int filterSize, const int paddingStart, const int moduleStride,
        const int imgStride, const int sumWidth, const float scaleTargets,
        const float scaleOutputs) {
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

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);

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
    const int imgOffset = loadX;
    const int hidActsOffset = blockFilterIdx * numImages * numModules + loadX;
    //    images += loadX;
    //    hidActs += blockFilterIdx * numImages * numModules
    //            + loadX;

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

    const bool doWork = mStartY < mEndY && mStartX < mEndX;
    //    if (!doWork) {
    //        hidActs -=
    //    }
    //    if (mStartY == mEndY || mStartX == mEndX) {
    //        return;
    //    }

    //    float imPreload[pixelCache * numColors * preloadCases / B_X]; // [12]
    float haPreload[filtersPerThread * preloadCases / B_Y];  // [8]
    //    if (blockIdx.x != 0 || blockIdx.y !=0) {
    //        return;
    //    }
    //    printf("mStartX: %d, mStartX: %d, mStartX: %d, mStartX: %d\n", mStartX,
    //    mStartY, mEndX, mEndY);
    const int fYOff = (blockPixelOffset + tidx) / filterSize;
    const int fXOff = (blockPixelOffset + tidx) % filterSize;
    __shared__ int pxIdxes[B_Y * pixelsPerThread];
    fill_shared_mem<int>((int*)pxIdxes, sizeof(pxIdxes) / sizeof(int), 0);
    __syncthreads();
    //    __shared__ int fidx[filtersPerThread * preloadCases / B_Y]; // [8]

    int m = mStartY * numModulesX + mStartX;

    int fidx[filtersPerThread * preloadCases / B_Y];
    if (doWork) {
#pragma unroll
        for (int y = 0; y < filtersPerThread * preloadCases / B_Y; ++y) {
            const int fIdx = WA_3_FIDX(y);
            //            if (doWork) {
            haPreload[y] = tex1Dfetch<float>(
                    hidActs,
                    hidActsOffset + fIdx * numImages * numModules + m * numImages);
            //            }
            fidx[y] = fIdx * numImages * numModules;
        }
    }

    for (int my = mStartY; my < mEndY; my++) {
        const int imgLoadModPosY = paddingStart + my * moduleStride;
        for (int mx = mStartX; mx < mEndX; mx++) {
            m = my * numModulesX + mx;

            //            __syncthreads();
            const int imgLoadModPosX = paddingStart + mx * moduleStride;
            if (tidx < B_Y * pixelsPerThread) {
                //                const int imgLoadModPosY = paddingStart + my *
                //                moduleStride; const int imgLoadModPosX = paddingStart
                //                + mx * moduleStride;
                const int pxY = (imgLoadModPosY + fYOff);
                const int pxX = (imgLoadModPosX + fXOff);
                const int pixIdx = (pxY * imgSizeX + pxX) * imgStride;
                pxIdxes[tidx] = pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX
                                      ? pixIdx
                                      : -1;
            }
            __syncthreads();

            int myNext = my, mxNext = mx, mNext = m;
            const bool lastModule = my == mEndY - 1 && mx == mEndX - 1;

            if (!lastModule) {
                mxNext = mx + 1 == mEndX ? mStartX : mx + 1;
                myNext = my + (mx + 1 == mEndX);
                mNext = myNext * numModulesX + mxNext;
            }

            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                const bool lastBatch = caseIdx + preloadCases == numImages;
                //                const float* im = &images[caseIdx + preloadCases +
                //                pixIdx]; const float* ha = &hidActs[caseIdx +
                //                preloadCases + m * numImages];
                int hidActsOffset2 =
                        hidActsOffset + caseIdx + preloadCases + m * numImages;

                if (lastBatch) {
                    //                    ha = &hidActs[mNext * numImages];
                    hidActsOffset2 = hidActsOffset + mNext * numImages;
                }

#pragma unroll
                for (int y = 0; y < B_X * filtersPerThread;
                     y += (B_X * B_Y) / preloadCases) {
                    shHidActs[loadY + y][loadX] =
                            haPreload[y * preloadCases / (B_X * B_Y)];
                }

/* ==================================================================================
 * Iteration 0
 * ==================================================================================
 */
#pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
#pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shImages[loadY + y + c * pixelCache * B_Y][loadX] = 0;
                    }
                }
#pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    const int pxIdx = 0 * B_Y + loadY + y;  // pixel idx in filter
                    if (pxIdx + blockPixelOffset < filterPixels) {
                        const int pixIdx =
                                pxIdxes[pxIdx];  //(pxY * imgSizeX + pxX) * imgStride;
                        if (pixIdx >= 0) {
#pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shImages[loadY + y + c * pixelCache * B_Y][loadX] =
                                        tex1Dfetch<float>(
                                                images,
                                                imgOffset + caseIdx +
                                                        c * imgPixels * imgStride +
                                                        pixIdx);
                            }
                        }
                    }
                }

                __syncthreads();

                haPreload[0] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[0]);
                haPreload[1] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[1]);
                WA_C3_LOOP(0, 0);
                haPreload[2] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[2]);
                haPreload[3] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[3]);
                WA_C3_LOOP(0, 1);
                haPreload[4] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[4]);
                haPreload[5] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[5]);
                WA_C3_LOOP(0, 2);
                haPreload[6] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[6]);
                haPreload[7] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[7]);

                __syncthreads();
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
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters +
                                f * B_X] =
                                scaleTargets * targets[p * B_Y * numFilters +
                                                       c * filterPixels * numFilters +
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
                        //                        if (threadIdx.x == 3)
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters +
                                f * B_X] = scaleOutputs * prod[c][p][f];
                    }
                }
            }
        }
    }
}

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
 * away.
 */
template <
        int B_Y, int B_X, int pixelCache, int pixelsPerThread, int filtersPerThread,
        int preloadCases, int numColors, bool scale, bool checkCaseBounds>
__launch_bounds__(256, 2) __global__
        void conv_weight_acts_c_preload_pc_2_pt_4_f_3_r_32_c_3(
                cudaTextureObject_t images, cudaTextureObject_t hidActs, float* targets,
                const int numImages, const int numFilters, const int numModulesY,
                const int numModulesX, const int imgSizeY, const int imgSizeX,
                const int filterSize, const int paddingStart, const int moduleStride,
                const int imgStride, const int sumWidth, const float scaleTargets,
                const float scaleOutputs) {
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

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);

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
    const int imgOffset = loadX;
    const int hidActsOffset = blockFilterIdx * numImages * numModules + loadX;
    //    images += loadX;
    //    hidActs += blockFilterIdx * numImages * numModules
    //            + loadX;

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

    const bool doWork = mStartY < mEndY && mStartX < mEndX;
    //    if (mStartY == mEndY || mStartX == mEndX) {
    //        return;
    //    }

    //    float imPreload[pixelCache * numColors * preloadCases / B_X]; // [12]
    float haPreload[filtersPerThread * preloadCases / B_Y];  // [6]
    //    if (blockIdx.x != 0 || blockIdx.y !=0) {
    //        return;
    //    }
    //    printf("mStartX: %d, mStartX: %d, mStartX: %d, mStartX: %d\n", mStartX,
    //    mStartY, mEndX, mEndY);
    const int fYOff = (blockPixelOffset + tidx) / filterSize;
    const int fXOff = (blockPixelOffset + tidx) % filterSize;
    __shared__ int pxIdxes[B_Y * pixelsPerThread];
    fill_shared_mem<int>((int*)pxIdxes, sizeof(pxIdxes) / sizeof(int), 0);
    __syncthreads();
    //    __shared__ int fidx[filtersPerThread * preloadCases / B_Y]; // [6]

    int m = mStartY * numModulesX + mStartX;
    int fidx[filtersPerThread * preloadCases / B_Y];
    //    if (doWork) {
#pragma unroll
    for (int y = 0; y < filtersPerThread * preloadCases / B_Y; ++y) {
        fidx[y] = WA_3_FIDX(y) * numImages * numModules;
        if (doWork) {  // Not actually necessary, I think
            haPreload[y] =
                    tex1Dfetch<float>(hidActs, hidActsOffset + fidx[y] + m * numImages);
        }
    }
    //    }
    int mNext = mStartY * numModulesX + mStartX;
    for (int my = mStartY; my < mEndY; my++) {
        //        const int imgLoadModPosY = paddingStart + my * moduleStride;
        for (int mx = mStartX; mx < mEndX; mx++) {
            m = mNext;  // my * numModulesX + mx;

            //            __syncthreads();
            //            const int imgLoadModPosX = paddingStart + mx * moduleStride;
            if (tidx < B_Y * pixelsPerThread) {
                const int imgLoadModPosY = paddingStart + my * moduleStride;
                const int imgLoadModPosX = paddingStart + mx * moduleStride;
                const int pxY = (imgLoadModPosY + fYOff);
                const int pxX = (imgLoadModPosX + fXOff);
                const int pixIdx = (pxY * imgSizeX + pxX) * imgStride;
                pxIdxes[tidx] = pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX
                                      ? pixIdx
                                      : -1;
            }
            __syncthreads();

            const bool lastModule = my == mEndY - 1 && mx == mEndX - 1;
            mNext = lastModule * m +
                    !lastModule * ((my + (mx + 1 == mEndX)) * numModulesX +
                                   (mx + 1 == mEndX ? mStartX : mx + 1));
            //            if (!lastModule) {
            //                const int mxNext = mx + 1 == mEndX ? mStartX : mx + 1;
            //                const int myNext = my + (mx + 1 == mEndX);
            //                mNext = myNext * numModulesX + mxNext;
            //            }

            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                const bool lastBatch = caseIdx + preloadCases == numImages;
                //                const float* im = &images[caseIdx + preloadCases +
                //                pixIdx]; const float* ha = hidActs + !lastBatch *
                //                (caseIdx + preloadCases + m * numImages) + lastBatch *
                //                mNext * numImages;
                const int hidActsOffset2 =
                        hidActsOffset +
                        !lastBatch * (caseIdx + preloadCases + m * numImages) +
                        lastBatch * mNext * numImages;
                //                if (lastBatch) {
                //                    ha = &hidActs[mNext * numImages];
                //                }

#pragma unroll
                for (int y = 0; y < B_X * filtersPerThread;
                     y += (B_X * B_Y) / preloadCases) {
                    shHidActs[loadY + y][loadX] =
                            haPreload[y * preloadCases / (B_X * B_Y)];
                }

/* ==================================================================================
 * Iteration 0
 * ==================================================================================
 */
#pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of
                    // rows filled per iteration
                    if ((B_Y * pixelCache) % (B_X * B_Y / preloadCases) == 0 ||
                        y + loadY < B_Y * pixelCache) {
#pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[loadY + y + c * pixelCache * B_Y][loadX] = 0;
                        }
                    }
                }
#pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of
                    // rows filled per iteration
                    if ((B_Y * pixelCache) % (B_X * B_Y / preloadCases) == 0 ||
                        y + loadY < B_Y * pixelCache) {
                        const int pxIdx = 0 * B_Y + loadY + y;  // pixel idx in filter
                        const int pixIdx =
                                pxIdxes[pxIdx];  //(pxY * imgSizeX + pxX) * imgStride;
                        if (pixIdx >= 0 && pxIdx + blockPixelOffset < filterPixels &&
                            (!checkCaseBounds || caseIdx + loadX < numImages)) {
#pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shImages[loadY + y + c * pixelCache * B_Y][loadX] =
                                        tex1Dfetch<float>(
                                                images,
                                                imgOffset + caseIdx +
                                                        c * imgPixels * imgStride +
                                                        pixIdx);
                            }
                        }
                    }
                }

                __syncthreads();

                haPreload[0] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[0]);
                haPreload[1] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[1]);
                haPreload[2] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[2]);
                haPreload[3] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[3]);
                haPreload[4] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[4]);
                haPreload[5] = tex1Dfetch<float>(hidActs, hidActsOffset2 + fidx[5]);

                WA_C3_LOOP2(0);

                __syncthreads();

/* ==================================================================================
 * Iteration 1
 * ==================================================================================
 */
#pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of
                    // rows filled per iteration
                    if ((B_Y * pixelCache) % (B_X * B_Y / preloadCases) == 0 ||
                        y + loadY < B_Y * pixelCache) {
                        //                        const int pxIdx = 2 * B_Y + loadY + y;
                        //                        // pixel idx in filter
#pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[loadY + y + c * pixelCache * B_Y][loadX] = 0;
                        }
                    }
                }

#pragma unroll
                for (int y = 0; y < B_Y * pixelCache; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of
                    // rows filled per iteration
                    if ((B_Y * pixelCache) % (B_X * B_Y / preloadCases) == 0 ||
                        y + loadY < B_Y * pixelCache) {
                        const int pxIdx = 2 * B_Y + loadY + y;  // pixel idx in filter
                        const int pixIdx =
                                pxIdxes[pxIdx];  //(pxY * imgSizeX + pxX) * imgStride;
                        if (pixIdx >= 0 && pxIdx + blockPixelOffset < filterPixels &&
                            (!checkCaseBounds || caseIdx + loadX < numImages)) {
#pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shImages[loadY + y + c * pixelCache * B_Y][loadX] =
                                        tex1Dfetch<float>(
                                                images,
                                                imgOffset + caseIdx +
                                                        c * imgPixels * imgStride +
                                                        pixIdx);
                            }
                        }
                    }
                }

                __syncthreads();

                WA_C3_LOOP2(2);

                __syncthreads();
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
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters +
                                f * B_X] =
                                scaleTargets * targets[p * B_Y * numFilters +
                                                       c * filterPixels * numFilters +
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
                        targets[p * B_Y * numFilters + c * filterPixels * numFilters +
                                f * B_X] = scaleOutputs * prod[c][p][f];
                    }
                }
            }
        }
    }
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numFilterColors, filterPixels,
 * numFilters)
 */
template <
        int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases,
        bool scale>
__launch_bounds__(128, 4) __global__
        void conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_16_f_4_c_8_r_16(
                cudaTextureObject_t images, cudaTextureObject_t hidActs, float* targets,
                const int numImages, const int numFilters, const int numModulesY,
                const int numModulesX, const int imgSizeY, const int imgSizeX,
                const int filterSize, const int paddingStart, const int moduleStride,
                const int imgStride, const int numImgColors, const int numGroups,
                const int sumWidth, const float scaleTargets,
                const float scaleOutputs) {
    __shared__ float shImages[colorsPerThread * B_Y]
                             [preloadCases];  // preload preloadCases cases
    __shared__ float
            shHidActs[filtersPerThread * B_X]
                     [preloadCases + 1];  // preload preloadCases cases of B_X hidacts
    fill_shared_mem<float>((float*)shImages, sizeof(shImages) / sizeof(float), 0);
    fill_shared_mem<float>((float*)shHidActs, sizeof(shHidActs) / sizeof(float), 0);
    __syncthreads();

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);
    const int blockModuleChunkIdx = blockIdx.x / numFilterBlocks;

    const int numModuleChunksX = DIVUP(numModulesX, sumWidth);
    //    const int numModuleChunksY = DIVUP(numModulesY, sumWidth);

    const int blockModuleChunkX = blockModuleChunkIdx % numModuleChunksX;
    const int blockModuleChunkY = blockModuleChunkIdx / numModuleChunksX;

    const int blockModuleStartX = blockModuleChunkX * sumWidth;
    const int blockModuleStartY = blockModuleChunkY * sumWidth;

    //    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = filtersPerThread * B_X * (blockIdx.x % numFilterBlocks);
    const int numModules = numModulesY * numModulesX;

    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
    const int numFilterColors = numImgColors / numGroups;

    const int blockPixelOffset = blockIdx.z;  // pixel idx in filter
    const int blockPixelY = blockPixelOffset / filterSize,
              blockPixelX = blockPixelOffset % filterSize;
    const int blockFilterColorIdx = blockIdx.y * B_Y * colorsPerThread;
    const int imgColorIdx = blockFilterColorIdx + blockGroupIdx * numFilterColors;
    const int imgOffset = (imgColorIdx + loadY) * imgPixels * imgStride + loadX;
    //    images += (imgColorIdx + loadY) * imgPixels * imgStride + loadX;
    const int hidActsOffset = blockFilterIdx * numImages * numModules +
                              loadY * numImages * numModules + loadX;
    //
    //    hidActs +=
    //             blockFilterIdx * numImages * numModules
    //            + loadY * numImages * numModules
    //            + loadX;

    targets += blockModuleChunkIdx * numFilters * filterPixels * numFilterColors +
               (blockFilterColorIdx + threadIdx.y) * filterPixels * numFilters +
               blockPixelOffset * numFilters + blockFilterIdx + threadIdx.x;
    //    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;

    const int mStartX =
            max(blockModuleStartX, DIVUP(-blockPixelX - paddingStart, moduleStride));
    const int mStartY =
            max(blockModuleStartY, DIVUP(-blockPixelY - paddingStart, moduleStride));
    const int mEndX =
            min(numModulesX,
                min(blockModuleStartX + sumWidth,
                    DIVUP(imgSizeX - blockPixelX - paddingStart, moduleStride)));
    const int mEndY =
            min(numModulesY,
                min(blockModuleStartY + sumWidth,
                    DIVUP(imgSizeY - blockPixelY - paddingStart, moduleStride)));

    //    if (mStartY == mEndY || mStartX == mEndX) {
    //        return;
    //    }
    //    const bool doWork = mStartY < mEndY && mStartX < mEndX;

    float* shHidActLoad = &shHidActs[loadY][loadX];
    float* shImgLoad = &shImages[loadY][loadX];

    float imPreload[preloadCases * colorsPerThread / B_X];   // [8]
    float haPreload[preloadCases * filtersPerThread / B_Y];  // [8]

    float prod[filtersPerThread][colorsPerThread];

#pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            prod[f][c] = 0;
        }
    }
    int pixIdx, pixIdxNext, m, mNext;

    conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
            mStartY, mStartX, paddingStart, numModulesX, moduleStride, blockPixelY,
            blockPixelX, imgSizeX, imgStride, pixIdx, m);

#pragma unroll
    for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
        // It's bizarre, but this is the fastest way I've found to get it not to load
        // nonexistent pixels. All other ways cause crazy excessive register usage.
        const int idx = (mStartY < mEndY && mStartX < mEndX) *
                        (0 + y * imgPixels * imgStride + pixIdx);
        imPreload[y * preloadCases / (B_X * B_Y)] =
                tex1Dfetch<float>(images, imgOffset + idx);
    }
#pragma unroll
    for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
        // Almost certainly not necessary here.
        const int idx = (mStartY < mEndY && mStartX < mEndX) *
                        (0 + y * numImages * numModules + m * numImages);
        haPreload[y * preloadCases / (B_X * B_Y)] =
                tex1Dfetch<float>(hidActs, hidActsOffset + idx);
    }

    for (int my = mStartY; my < mEndY; my++) {
        for (int mx = mStartX; mx < mEndX; mx++) {
            int myNext = my, mxNext = mx;
            const bool lastModule = my == mEndY - 1 && mx == mEndX - 1;

            if (!lastModule) {
                mxNext = mx + 1 == mEndX ? mStartX : mx + 1;
                myNext = my + (mx + 1 == mEndX);
            }

            conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
                    myNext, mxNext, paddingStart, numModulesX, moduleStride,
                    blockPixelY, blockPixelX, imgSizeX, imgStride, pixIdxNext, mNext);

            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
#pragma unroll
                for (int y = 0; y < B_Y * colorsPerThread;
                     y += (B_X * B_Y) / preloadCases) {
                    shImgLoad[(y)*preloadCases] =
                            imPreload[y * preloadCases / (B_X * B_Y)];
                }
                //                const float* im = &images[caseIdx + preloadCases +
                //                pixIdx]; const float* ha = &hidActs[caseIdx +
                //                preloadCases + m * numImages];
                int imgOffset2 = imgOffset + caseIdx + preloadCases + pixIdx;
                int hidActsOffset2 =
                        hidActsOffset + caseIdx + preloadCases + m * numImages;
                if (caseIdx + preloadCases == numImages) {
                    pixIdx = pixIdxNext;
                    m = mNext;
                    imgOffset2 = imgOffset + pixIdxNext;
                    hidActsOffset2 = hidActsOffset + mNext * numImages;
                }
#pragma unroll
                for (int y = 0; y < B_X * filtersPerThread;
                     y += (B_X * B_Y) / preloadCases) {
                    shHidActLoad[y * (preloadCases + 1)] =
                            haPreload[y * preloadCases / (B_X * B_Y)];
                }

                __syncthreads();

#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    WA_IMLOAD_TX(z);
                    WA_LOOP2(z);
                }

#pragma unroll
                for (int z = 0; z < 8; ++z) {
                    WA_HALOAD_TX(z);
                    WA_LOOP2(z + 8);
                }
                __syncthreads();
            }
        }
    }

    if (scale) {
#pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
#pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] =
                        scaleTargets *
                                targets[c * B_Y * filterPixels * numFilters + f * B_X] +
                        scaleOutputs * prod[f][c];
            }
        }
    } else {
#pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
#pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] =
                        scaleOutputs * prod[f][c];
            }
        }
    }
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numFilterColors, filterPixels,
 * numFilters)
 */
template <
        int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases,
        bool scale>
__launch_bounds__(256, 2) __global__
        void conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_6_r_32(
                cudaTextureObject_t images, cudaTextureObject_t hidActs, float* targets,
                const int numImages, const int numFilters, const int numModulesY,
                const int numModulesX, const int imgSizeY, const int imgSizeX,
                const int filterSize, const int paddingStart, const int moduleStride,
                const int imgStride, const int numImgColors, const int numGroups,
                const int sumWidth, const float scaleTargets,
                const float scaleOutputs) {
    __shared__ float shImages[colorsPerThread * B_Y]
                             [preloadCases];  // preload preloadCases cases
    __shared__ float
            shHidActs[filtersPerThread * B_X]
                     [preloadCases + 1];  // preload preloadCases cases of B_X hidacts
    fill_shared_mem<float>((float*)shImages, sizeof(shImages) / sizeof(float), 0);
    fill_shared_mem<float>((float*)shHidActs, sizeof(shHidActs) / sizeof(float), 0);
    __syncthreads();

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);
    const int blockModuleChunkIdx = blockIdx.x / numFilterBlocks;

    const int numModuleChunksX = DIVUP(numModulesX, sumWidth);
    //    const int numModuleChunksY = DIVUP(numModulesY, sumWidth);

    const int blockModuleChunkX = blockModuleChunkIdx % numModuleChunksX;
    const int blockModuleChunkY = blockModuleChunkIdx / numModuleChunksX;

    const int blockModuleStartX = blockModuleChunkX * sumWidth;
    const int blockModuleStartY = blockModuleChunkY * sumWidth;

    //    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = filtersPerThread * B_X * (blockIdx.x % numFilterBlocks);
    const int numModules = numModulesY * numModulesX;

    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
    const int numFilterColors = numImgColors / numGroups;

    const int blockPixelOffset = blockIdx.z;  // pixel idx in filter
    const int blockPixelY = blockPixelOffset / filterSize,
              blockPixelX = blockPixelOffset % filterSize;
    const int blockFilterColorIdx = blockIdx.y * B_Y * colorsPerThread;
    const int imgColorIdx = blockFilterColorIdx + blockGroupIdx * numFilterColors;

    const int imgOffset = (imgColorIdx + loadY) * imgPixels * imgStride + loadX;
    const int hidActsOffset = blockFilterIdx * numImages * numModules +
                              loadY * numImages * numModules + loadX;
    //    images += (imgColorIdx + loadY) * imgPixels * imgStride + loadX;
    //
    //    hidActs +=
    //             blockFilterIdx * numImages * numModules
    //            + loadY * numImages * numModules
    //            + loadX;

    targets += blockModuleChunkIdx * numFilters * filterPixels * numFilterColors +
               (blockFilterColorIdx + threadIdx.y) * filterPixels * numFilters +
               blockPixelOffset * numFilters + blockFilterIdx + threadIdx.x;
    //    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;

    const int mStartX =
            max(blockModuleStartX, DIVUP(-blockPixelX - paddingStart, moduleStride));
    const int mStartY =
            max(blockModuleStartY, DIVUP(-blockPixelY - paddingStart, moduleStride));
    const int mEndX =
            min(numModulesX,
                min(blockModuleStartX + sumWidth,
                    DIVUP(imgSizeX - blockPixelX - paddingStart, moduleStride)));
    const int mEndY =
            min(numModulesY,
                min(blockModuleStartY + sumWidth,
                    DIVUP(imgSizeY - blockPixelY - paddingStart, moduleStride)));

    //    if (mStartY == mEndY || mStartX == mEndX) {
    //        return;
    //    }
    const bool doWork = mStartY < mEndY && mStartX < mEndX;

    float* shHidActLoad = &shHidActs[loadY][loadX];
    float* shImgLoad = &shImages[loadY][loadX];

    float imPreload[preloadCases * colorsPerThread / B_X];   // [6]
    float haPreload[preloadCases * filtersPerThread / B_Y];  // [16]

    float prod[filtersPerThread][colorsPerThread];

#pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            prod[f][c] = 0;
        }
    }
    int pixIdx, pixIdxNext, m, mNext;

    conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
            mStartY, mStartX, paddingStart, numModulesX, moduleStride, blockPixelY,
            blockPixelX, imgSizeX, imgStride, pixIdx, m);

    if (doWork) {
#pragma unroll
        for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
            imPreload[y * preloadCases / (B_X * B_Y)] = tex1Dfetch<float>(
                    images, imgOffset + y * imgPixels * imgStride + pixIdx);
        }

#pragma unroll
        for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
            haPreload[y * preloadCases / (B_X * B_Y)] = tex1Dfetch<float>(
                    hidActs,
                    hidActsOffset + y * numImages * numModules + m * numImages);
        }
    }
    //    if (mStartY > mEndY || mStartX > mEndX) {
    //        printf("crzy!!\n");
    //    }

    for (int my = mStartY; my < mEndY; my++) {
        for (int mx = mStartX; mx < mEndX; mx++) {
            int myNext = my, mxNext = mx;
            const bool lastModule = my == mEndY - 1 && mx == mEndX - 1;

            if (!lastModule) {
                mxNext = mx + 1 == mEndX ? mStartX : mx + 1;
                myNext = my + (mx + 1 == mEndX);
            }

            conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
                    myNext, mxNext, paddingStart, numModulesX, moduleStride,
                    blockPixelY, blockPixelX, imgSizeX, imgStride, pixIdxNext, mNext);

            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
#pragma unroll
                for (int y = 0; y < B_Y * colorsPerThread;
                     y += (B_X * B_Y) / preloadCases) {
                    shImgLoad[(y)*preloadCases] =
                            imPreload[y * preloadCases / (B_X * B_Y)];
                }

#pragma unroll
                for (int y = 0; y < B_X * filtersPerThread;
                     y += (B_X * B_Y) / preloadCases) {
                    shHidActLoad[y * (preloadCases + 1)] =
                            haPreload[y * preloadCases / (B_X * B_Y)];
                }

                __syncthreads();

                //                const float* im = &images[caseIdx + preloadCases +
                //                pixIdx]; const float* ha = &hidActs[caseIdx +
                //                preloadCases + m * numImages];
                int imgOffset2 = imgOffset + caseIdx + preloadCases + pixIdx;
                int hidActsOffset2 =
                        hidActsOffset + caseIdx + preloadCases + m * numImages;
                if (caseIdx + preloadCases == numImages) {
                    pixIdx = pixIdxNext;
                    m = mNext;
                    imgOffset2 = imgOffset + pixIdxNext;
                    hidActsOffset2 = hidActsOffset + mNext * numImages;
                }

                WA_LOOP(0);
                WA_LOOP(1);
                WA_LOOP(2);
                WA_LOOP(3);
                WA_LOOP(4);

                WA_LOOP(5);
                WA_IMLOAD_TX(0);
                WA_LOOP(6);
                WA_IMLOAD_TX(1);
                WA_LOOP(7);
                WA_IMLOAD_TX(2);
                WA_LOOP(8);
                WA_IMLOAD_TX(3);
                WA_LOOP(9);
                WA_IMLOAD_TX(4);
                WA_LOOP(10);
                WA_IMLOAD_TX(5);

                WA_LOOP(11);
                WA_HALOAD_TX(0);
                WA_LOOP(12);
                WA_HALOAD_TX(1);
                WA_LOOP(13);
                WA_HALOAD_TX(2);
                WA_LOOP(14);
                WA_HALOAD_TX(3);
                WA_LOOP(15);
                WA_HALOAD_TX(4);
                WA_LOOP(16);
                WA_HALOAD_TX(5);
                WA_LOOP(17);
                WA_HALOAD_TX(6);
                WA_LOOP(18);
                WA_HALOAD_TX(7);
                WA_LOOP(19);
                WA_HALOAD_TX(8);
                WA_LOOP(20);
                WA_HALOAD_TX(9);
                WA_LOOP(21);
                WA_HALOAD_TX(10);
                WA_LOOP(22);
                WA_HALOAD_TX(11);
                WA_LOOP(23);
                WA_HALOAD_TX(12);
                WA_LOOP(24);
                WA_HALOAD_TX(13);
                WA_LOOP(25);
                WA_HALOAD_TX(14);
                WA_LOOP(26);
                WA_HALOAD_TX(15);

                WA_LOOP(27);
                WA_LOOP(28);
                WA_LOOP(29);
                WA_LOOP(30);
                WA_LOOP(31);

                __syncthreads();
            }
        }
    }

    if (scale) {
#pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
#pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] =
                        scaleTargets *
                                targets[c * B_Y * filterPixels * numFilters + f * B_X] +
                        scaleOutputs * prod[f][c];
            }
        }
    } else {
#pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
#pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] =
                        scaleOutputs * prod[f][c];
            }
        }
    }
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numFilterColors, filterPixels,
 * numFilters)
 */
template <
        int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases,
        bool scale>
__launch_bounds__(256, 2) __global__
        void conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16(
                cudaTextureObject_t images, cudaTextureObject_t hidActs, float* targets,
                const int numImages, const int numFilters, const int numModulesY,
                const int numModulesX, const int imgSizeY, const int imgSizeX,
                const int filterSize, const int paddingStart, const int moduleStride,
                const int imgStride, const int numImgColors, const int numGroups,
                const int sumWidth, const float scaleTargets,
                const float scaleOutputs) {
    __shared__ float shImages[colorsPerThread * B_Y]
                             [preloadCases];  // preload preloadCases cases
    __shared__ float
            shHidActs[filtersPerThread * B_X]
                     [preloadCases + 1];  // preload preloadCases cases of B_X hidacts
    fill_shared_mem<float>((float*)shImages, sizeof(shImages) / sizeof(float), 0);
    fill_shared_mem<float>((float*)shHidActs, sizeof(shHidActs) / sizeof(float), 0);
    __syncthreads();

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);
    const int blockModuleChunkIdx = blockIdx.x / numFilterBlocks;

    const int numModuleChunksX = DIVUP(numModulesX, sumWidth);
    //    const int numModuleChunksY = DIVUP(numModulesY, sumWidth);

    const int blockModuleChunkX = blockModuleChunkIdx % numModuleChunksX;
    const int blockModuleChunkY = blockModuleChunkIdx / numModuleChunksX;

    const int blockModuleStartX = blockModuleChunkX * sumWidth;
    const int blockModuleStartY = blockModuleChunkY * sumWidth;

    //    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = filtersPerThread * B_X * (blockIdx.x % numFilterBlocks);
    const int numModules = numModulesY * numModulesX;

    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
    const int numFilterColors = numImgColors / numGroups;

    const int blockPixelOffset = blockIdx.z;  // pixel idx in filter
    const int blockPixelY = blockPixelOffset / filterSize,
              blockPixelX = blockPixelOffset % filterSize;
    const int blockFilterColorIdx = blockIdx.y * B_Y * colorsPerThread;
    const int imgColorIdx = blockFilterColorIdx + blockGroupIdx * numFilterColors;
    const int imgOffset = (imgColorIdx + loadY) * imgPixels * imgStride + loadX;
    //    images += (imgColorIdx + loadY) * imgPixels * imgStride + loadX;
    const int hidActsOffset = blockFilterIdx * numImages * numModules +
                              loadY * numImages * numModules + loadX;
    //
    //    hidActs +=
    //             blockFilterIdx * numImages * numModules
    //            + loadY * numImages * numModules
    //            + loadX;

    targets += blockModuleChunkIdx * numFilters * filterPixels * numFilterColors +
               (blockFilterColorIdx + threadIdx.y) * filterPixels * numFilters +
               blockPixelOffset * numFilters + blockFilterIdx + threadIdx.x;
    //    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;

    const int mStartX =
            max(blockModuleStartX, DIVUP(-blockPixelX - paddingStart, moduleStride));
    const int mStartY =
            max(blockModuleStartY, DIVUP(-blockPixelY - paddingStart, moduleStride));
    const int mEndX =
            min(numModulesX,
                min(blockModuleStartX + sumWidth,
                    DIVUP(imgSizeX - blockPixelX - paddingStart, moduleStride)));
    const int mEndY =
            min(numModulesY,
                min(blockModuleStartY + sumWidth,
                    DIVUP(imgSizeY - blockPixelY - paddingStart, moduleStride)));

    const bool doWork = mStartY < mEndY && mStartX < mEndX;

    float* shHidActLoad = &shHidActs[loadY][loadX];
    float* shImgLoad = &shImages[loadY][loadX];

    float imPreload[preloadCases * colorsPerThread / B_X];   // [4]
    float haPreload[preloadCases * filtersPerThread / B_Y];  // [8]

    float prod[filtersPerThread][colorsPerThread];

#pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            prod[f][c] = 0;
        }
    }
    int pixIdx, pixIdxNext, m, mNext;

    conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
            mStartY, mStartX, paddingStart, numModulesX, moduleStride, blockPixelY,
            blockPixelX, imgSizeX, imgStride, pixIdx, m);

    if (doWork && loadY < B_Y * colorsPerThread) {
#pragma unroll
        for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
            imPreload[y * preloadCases / (B_X * B_Y)] = tex1Dfetch<float>(
                    images, imgOffset + y * imgPixels * imgStride + pixIdx);
        }
    }

    if (doWork && loadY < B_X * filtersPerThread) {
#pragma unroll
        for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
            haPreload[y * preloadCases / (B_X * B_Y)] = tex1Dfetch<float>(
                    hidActs,
                    hidActsOffset + y * numImages * numModules + m * numImages);
        }
    }

    for (int my = mStartY; my < mEndY; my++) {
        for (int mx = mStartX; mx < mEndX; mx++) {
            int myNext = my, mxNext = mx;
            const bool lastModule = my == mEndY - 1 && mx == mEndX - 1;

            if (!lastModule) {
                mxNext = mx + 1 == mEndX ? mStartX : mx + 1;
                myNext = my + (mx + 1 == mEndX);
            }

            conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16_setCoords(
                    myNext, mxNext, paddingStart, numModulesX, moduleStride,
                    blockPixelY, blockPixelX, imgSizeX, imgStride, pixIdxNext, mNext);

            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                //                const float* im = &images[caseIdx + preloadCases +
                //                pixIdx];
                int imgOffset2 = imgOffset + caseIdx + preloadCases + pixIdx;
                int hidActsOffset2 =
                        hidActsOffset + caseIdx + preloadCases + m * numImages;
                //                const float* ha = &hidActs[caseIdx + preloadCases + m
                //                * numImages];

                if (caseIdx + preloadCases == numImages) {
                    pixIdx = pixIdxNext;
                    m = mNext;
                    //                    im = &images[pixIdxNext];
                    imgOffset2 = imgOffset + pixIdxNext;
                    hidActsOffset2 = hidActsOffset + mNext * numImages;

                    //                    ha = &hidActs[mNext * numImages];
                }

                if (loadY < B_Y * colorsPerThread) {
#pragma unroll
                    for (int y = 0; y < B_Y * colorsPerThread;
                         y += (B_X * B_Y) / preloadCases) {
                        shImgLoad[(y)*preloadCases] =
                                imPreload[y * preloadCases / (B_X * B_Y)];
                    }
                }

                if (loadY < B_X * filtersPerThread) {
#pragma unroll
                    for (int y = 0; y < B_X * filtersPerThread;
                         y += (B_X * B_Y) / preloadCases) {
                        shHidActLoad[y * (preloadCases + 1)] =
                                haPreload[y * preloadCases / (B_X * B_Y)];
                    }
                }

                __syncthreads();

                WA_LOOP(0);
                WA_IMLOAD_TX(0);
                WA_LOOP(1);
                WA_IMLOAD_TX(1);
                WA_LOOP(2);
                WA_IMLOAD_TX(2);
                WA_LOOP(3);
                WA_IMLOAD_TX(3);
                WA_LOOP(4);
                WA_HALOAD_TX(0);
                WA_LOOP(5);
                WA_HALOAD_TX(1);
                WA_LOOP(6);
                WA_HALOAD_TX(2);
                WA_LOOP(7);
                WA_HALOAD_TX(3);
                WA_LOOP(8);
                WA_HALOAD_TX(4);
                WA_LOOP(9);
                WA_HALOAD_TX(5);
                WA_LOOP(10);
                WA_HALOAD_TX(6);
                WA_LOOP(11);
                WA_HALOAD_TX(7);
                WA_LOOP(12);
                WA_LOOP(13);
                WA_LOOP(14);
                WA_LOOP(15);

                __syncthreads();
            }
        }
    }

    if (scale) {
#pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
#pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] =
                        scaleTargets *
                                targets[c * B_Y * filterPixels * numFilters + f * B_X] +
                        scaleOutputs * prod[f][c];
            }
        }
    } else {
#pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
#pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] =
                        scaleOutputs * prod[f][c];
            }
        }
    }
}

std::pair<int, int> getWeightActsOutputSize(
        int numModulesY, int numModulesX, int numFilterColors, int filterSize,
        int numFilters, int sumWidth) {
    const int outputModuleChunksX = DIVUP(numModulesX, sumWidth);
    const int outputModuleChunksY = DIVUP(numModulesY, sumWidth);
    const int outputModuleChunks = outputModuleChunksX * outputModuleChunksY;
    return std::pair<int, int>(
            outputModuleChunks * numFilterColors * filterSize * filterSize, numFilters);
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModules, numImages)
 *
 * targets:     (numModuleY*numModulesX/partialSum, numFilterColors, filterPixels,
 * numFilters)
 *
 * TODO: you can get a slight speed boost for local non-convolutional units by writing
 * special routines for partialSum = 1. But I dunno if the code duplication is worth
 * it...
 *
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128.
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast.
 */
void _weightActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int filterSize,
        int paddingStart, int moduleStride, int numImgColors, int numGroups,
        int sumWidth, float scaleTargets, float scaleOutput) {
    int numFilterColors = numImgColors / numGroups;
    int imgStride = images.getStride();
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    int numModules = numModulesY * numModulesX;
    int numFilters = hidActs.getNumRows() / numModules;
    int numFiltersPerGroup = numFilters / numGroups;

    megdnn_assert_internal(numImgColors % numGroups == 0);
    // megdnn_assert_internal(numFilters % (16*numGroups) == 0);
    bool previous_limit = numFilters % (16 * numGroups) == 0;

    megdnn_assert_internal(
            numGroups > 1 ||
            (numImgColors > 0 /*&& (numImgColors <= 3 || numImgColors % 16 == 0)*/));
    previous_limit &= numImgColors % 16 == 0;
    megdnn_assert_internal(numGroups == 1 || numFilterColors % 16 == 0);

    megdnn_assert_internal(imgSizeY * imgSizeX == imgPixels);
    megdnn_assert_internal(images.getNumRows() == imgPixels * numImgColors);

    int filterPixels = filterSize * filterSize;
    int outputModuleChunksX = DIVUP(numModulesX, sumWidth);
    int outputModuleChunksY = DIVUP(numModulesY, sumWidth);
    int outputModuleChunks = outputModuleChunksX * outputModuleChunksY;
    //    partialSum = partialSum == 0 ? numModules : partialSum;

    //    megdnn_assert_internal(numModules % partialSum == 0);
    megdnn_assert_internal(hidActs.getNumCols() == numImages);

    // These routines don't handle the case when only part of the image is visited in
    // the convolution
    megdnn_assert_internal(paddingStart <= 0);
    megdnn_assert_internal(
            paddingStart + (numModulesX - 1) * moduleStride + filterSize >= imgSizeX);
    megdnn_assert_internal(
            paddingStart + (numModulesY - 1) * moduleStride + filterSize >= imgSizeY);
    megdnn_assert_internal(moduleStride <= filterSize);

    megdnn_assert_internal(numModules * numFilters == hidActs.getNumRows());

    megdnn_assert_internal(!images.isTrans());
    megdnn_assert_internal(!hidActs.isTrans());
    megdnn_assert_internal(hidActs.isContiguous());

    megdnn_assert_internal(!targets.isTrans());
    megdnn_assert_internal(targets.isContiguous());

    int preloadCases = 32;

    dim3 blocks, threads;
    int bx, by;
    int pixelsPerThread = 0, filtersPerThread = 0, colorsPerThread = 0;
    // Worth playing with these parameters to find best values for your problem.
    // These values work relatively well, but not optimal for all problems.
    if (numFilterColors > 3) {
        filtersPerThread = numFiltersPerGroup % 64 == 0 ? 4
                         : numFiltersPerGroup % 32 == 0 ? 2
                                                        : 1;
        colorsPerThread = numFilterColors % 64 == 0 ? 8
                        : numFilterColors % 48 == 0 ? 6
                        : numFilterColors % 32 == 0 ? 8
                                                    : 4;
        by = (numFilterColors / colorsPerThread) % 8 == 0 ? 8 : 4;
        bx = numFiltersPerGroup % 128 == 0 ? 32 : 16;
        preloadCases = filtersPerThread * colorsPerThread < 32 ? 32 : 16;
        blocks =
                dim3(outputModuleChunks * DIVUP(numFilters, bx * filtersPerThread),
                     DIVUP(numFilterColors, (by * colorsPerThread)), filterPixels);

        // megdnn_assert_internal(numFilterColors % (by*colorsPerThread) == 0);
        previous_limit &= numFilterColors % (by * colorsPerThread) == 0;

    } else {  // This is ugly but it's nice to spell it out clearly
        megdnn_assert_internal(numGroups == 1);  // Just for sanity
        // NOTE: these things are only optimized for colors = 3. I didn't really test
        // other cases.
        if (numFilters % 64 ==
            0) {  // TODO: having a separate case for 128 would make things faster, but
                  // I probably don't care about 128
            filtersPerThread = 4;
            pixelsPerThread = 2;
            by = 16;
            bx = 16;
            preloadCases = 32;
        } else if (numFilters % 48 == 0) {
            filtersPerThread = 3;
            pixelsPerThread = 4;
            by = 16;
            bx = 16;
            preloadCases = 32;
        } else if (numFilters % 32 == 0) {
            filtersPerThread = 2;
            pixelsPerThread = 2;
            by = 8;
            bx = 16;
            preloadCases = 16;
        } else {  // This case is completely untested. It might be really slow. But no
                  // time now.
            filtersPerThread = 1;
            pixelsPerThread = 16;
            by = 16;
            bx = 16;
            preloadCases = 32;
        }
        blocks =
                dim3(outputModuleChunks * DIVUP(numFilters, bx * filtersPerThread),
                     DIVUP(filterPixels, by * pixelsPerThread));
    }
    megdnn_assert_internal((by * bx) % preloadCases == 0);
    // megdnn_assert_internal(numFilters % (bx * filtersPerThread) == 0);
    previous_limit &= numFilters % (bx * filtersPerThread) == 0;

    threads = dim3(bx, by);
    bool checkCaseBounds = numImages % preloadCases != 0;
    bool scale = scaleTargets != 0;
    std::pair<int, int> targetSize = getWeightActsOutputSize(
            numModulesY, numModulesX, numFilterColors, filterSize, numFilters,
            sumWidth);
    if (!scale) {
        targets.resize(targetSize.first, targetSize.second);
    } else {
        megdnn_assert_internal(targets.getNumRows() == targetSize.first);
        megdnn_assert_internal(targets.getNumCols() == targetSize.second);
    }

    if (scale == false) {
        if (checkCaseBounds == false) {
            if (numFilterColors > 3) {
                if (numFilterColors % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        if (previous_limit) {
                            cudaFuncSetCacheConfig(
                                    conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16<
                                            8, 32, 4, 8, 16, false>,
                                    cudaFuncCachePreferShared);
                            conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_8_r_16<
                                    8, 32, 4, 8, 16, false>
                                    <<<blocks, threads, 0, stream>>>(
                                            images.getTextureObject(),
                                            hidActs.getTextureObject(),
                                            targets.getDevData(), numImages, numFilters,
                                            numModulesY, numModulesX, imgSizeY,
                                            imgSizeX, filterSize, paddingStart,
                                            moduleStride, imgStride, numImgColors,
                                            numGroups, sumWidth, scaleTargets,
                                            scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(
                                    conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_16_f_4_c_8_r_16<
                                            8, 32, 4, 8, 16, false>,
                                    cudaFuncCachePreferShared);
                            conv_weight_acts_mc_mf_kepler_sw<8, 16, 2, 8, 32, false>
                                    <<<blocks, threads, 0, stream>>>(
                                            images.getDevData(), hidActs.getDevData(),
                                            targets.getDevData(), numImages, numFilters,
                                            numModulesY, numModulesX, imgSizeY,
                                            imgSizeX, filterSize, paddingStart,
                                            moduleStride, imgStride, numImgColors,
                                            numGroups, sumWidth, scaleTargets,
                                            scaleOutput);
                        }
                    } else if (numFiltersPerGroup % 64 == 0) {
                        if (previous_limit) {
                            cudaFuncSetCacheConfig(
                                    conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_16_f_4_c_8_r_16<
                                            8, 16, 4, 8, 16, false>,
                                    cudaFuncCachePreferShared);
                            conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_16_f_4_c_8_r_16<
                                    8, 16, 4, 8, 16, false>
                                    <<<blocks, threads, 0, stream>>>(
                                            images.getTextureObject(),
                                            hidActs.getTextureObject(),
                                            targets.getDevData(), numImages, numFilters,
                                            numModulesY, numModulesX, imgSizeY,
                                            imgSizeX, filterSize, paddingStart,
                                            moduleStride, imgStride, numImgColors,
                                            numGroups, sumWidth, scaleTargets,
                                            scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(
                                    conv_weight_acts_mc_mf_kepler_sw<
                                            8, 16, 4, 8, 16, false>,
                                    cudaFuncCachePreferShared);
                            conv_weight_acts_mc_mf_kepler_sw<8, 16, 4, 8, 16, false>
                                    <<<blocks, threads, 0, stream>>>(
                                            images.getDevData(), hidActs.getDevData(),
                                            targets.getDevData(), numImages, numFilters,
                                            numModulesY, numModulesX, imgSizeY,
                                            imgSizeX, filterSize, paddingStart,
                                            moduleStride, imgStride, numImgColors,
                                            numGroups, sumWidth, scaleTargets,
                                            scaleOutput);
                        }
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 16, 2, 8, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 16, 2, 8, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 16, 1, 8, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 16, 1, 8, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    }
                } else if (numFilterColors % 48 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        if (previous_limit) {
                            cudaFuncSetCacheConfig(
                                    conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_6_r_32<
                                            8, 32, 4, 6, 32, false>,
                                    cudaFuncCachePreferShared);
                            conv_weight_acts_mc_mf_kepler_preload_ty_8_tx_32_f_4_c_6_r_32<
                                    8, 32, 4, 6, 32, false>
                                    <<<blocks, threads, 0, stream>>>(
                                            images.getTextureObject(),
                                            hidActs.getTextureObject(),
                                            targets.getDevData(), numImages, numFilters,
                                            numModulesY, numModulesX, imgSizeY,
                                            imgSizeX, filterSize, paddingStart,
                                            moduleStride, imgStride, numImgColors,
                                            numGroups, sumWidth, scaleTargets,
                                            scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(
                                    conv_weight_acts_mc_mf_kepler_sw<
                                            8, 32, 4, 6, 32, false>,
                                    cudaFuncCachePreferShared);
                            conv_weight_acts_mc_mf_kepler_sw<8, 32, 4, 6, 32, false>
                                    <<<blocks, threads, 0, stream>>>(
                                            images.getDevData(), hidActs.getDevData(),
                                            targets.getDevData(), numImages, numFilters,
                                            numModulesY, numModulesX, imgSizeY,
                                            imgSizeX, filterSize, paddingStart,
                                            moduleStride, imgStride, numImgColors,
                                            numGroups, sumWidth, scaleTargets,
                                            scaleOutput);
                        }
                    } else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 16, 4, 6, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 16, 4, 6, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 16, 2, 6, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 16, 2, 6, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 16, 1, 6, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 16, 1, 6, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    }
                } else if (numFilterColors % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 32, 4, 8, 16, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 32, 4, 8, 16, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 4, 8, 16, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 4, 8, 16, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 2, 8, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 2, 8, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 1, 8, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 1, 8, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    }
                } else if (numFilterColors % 1 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 32, 4, 4, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 32, 4, 4, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 4, 4, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 4, 4, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 2, 4, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 2, 4, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 1, 4, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 1, 4, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    }
                }
            } else if (numFilterColors <= 3) {
                if (numFilterColors == 3) {
                    if (numFiltersPerGroup % 64 == 0) {
                        if (previous_limit) {
                            cudaFuncSetCacheConfig(
                                    conv_weight_acts_c_preload_pc_2_pt_2_f_4_r_32_c_3<
                                            16, 16, 2, 2, 4, 32, 3, false, false>,
                                    cudaFuncCachePreferShared);
                            conv_weight_acts_c_preload_pc_2_pt_2_f_4_r_32_c_3<
                                    16, 16, 2, 2, 4, 32, 3, false, false>
                                    <<<blocks, threads, 0, stream>>>(
                                            images.getTextureObject(),
                                            hidActs.getTextureObject(),
                                            targets.getDevData(), numImages, numFilters,
                                            numModulesY, numModulesX, imgSizeY,
                                            imgSizeX, filterSize, paddingStart,
                                            moduleStride, imgStride, sumWidth,
                                            scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(
                                    conv_weight_acts_c_kepler_sw<
                                            16, 16, 2, 2, 4, 32, 3, false, false>,
                                    cudaFuncCachePreferShared);
                            conv_weight_acts_c_kepler_sw<
                                    16, 16, 2, 2, 4, 32, 3, false, false>
                                    <<<blocks, threads, 0, stream>>>(
                                            images.getDevData(), hidActs.getDevData(),
                                            targets.getDevData(), numImages, numFilters,
                                            numModulesY, numModulesX, imgSizeY,
                                            imgSizeX, filterSize, paddingStart,
                                            moduleStride, imgStride, sumWidth,
                                            scaleTargets, scaleOutput);
                        }
                    } else if (numFiltersPerGroup % 48 == 0) {
                        if (previous_limit) {
                            cudaFuncSetCacheConfig(
                                    conv_weight_acts_c_preload_pc_2_pt_4_f_3_r_32_c_3<
                                            16, 16, 2, 4, 3, 32, 3, false, false>,
                                    cudaFuncCachePreferShared);
                            conv_weight_acts_c_preload_pc_2_pt_4_f_3_r_32_c_3<
                                    16, 16, 2, 4, 3, 32, 3, false, false>
                                    <<<blocks, threads, 0, stream>>>(
                                            images.getTextureObject(),
                                            hidActs.getTextureObject(),
                                            targets.getDevData(), numImages, numFilters,
                                            numModulesY, numModulesX, imgSizeY,
                                            imgSizeX, filterSize, paddingStart,
                                            moduleStride, imgStride, sumWidth,
                                            scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(
                                    conv_weight_acts_c_kepler_sw<
                                            16, 16, 2, 4, 3, 32, 3, false, false>,
                                    cudaFuncCachePreferShared);
                            conv_weight_acts_c_kepler_sw<
                                    16, 16, 2, 4, 3, 32, 3, false, false>
                                    <<<blocks, threads, 0, stream>>>(
                                            images.getDevData(), hidActs.getDevData(),
                                            targets.getDevData(), numImages, numFilters,
                                            numModulesY, numModulesX, imgSizeY,
                                            imgSizeX, filterSize, paddingStart,
                                            moduleStride, imgStride, sumWidth,
                                            scaleTargets, scaleOutput);
                        }
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        8, 16, 2, 2, 2, 16, 3, false, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                8, 16, 2, 2, 2, 16, 3, false, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 16, 1, 32, 3, false, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 16, 1, 32, 3, false, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                } else if (numFilterColors == 2) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 2, 4, 32, 2, false, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 2, 4, 32, 2, false, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 4, 3, 32, 2, false, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 4, 3, 32, 2, false, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        8, 16, 2, 2, 2, 16, 2, false, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                8, 16, 2, 2, 2, 16, 2, false, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 16, 1, 32, 2, false, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 16, 1, 32, 2, false, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                } else if (numFilterColors == 1) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 2, 4, 32, 1, false, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 2, 4, 32, 1, false, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 4, 3, 32, 1, false, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 4, 3, 32, 1, false, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        8, 16, 2, 2, 2, 16, 1, false, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                8, 16, 2, 2, 2, 16, 1, false, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 16, 1, 32, 1, false, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 16, 1, 32, 1, false, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
            }
        } else if (checkCaseBounds == true) {
            if (numFilterColors > 3) {
                if (numFilterColors % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 32, 4, 8, 16, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 32, 4, 8, 16, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 16, 4, 8, 16, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 16, 4, 8, 16, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 16, 2, 8, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 16, 2, 8, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 16, 1, 8, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 16, 1, 8, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    }
                } else if (numFilterColors % 48 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 32, 4, 6, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 32, 4, 6, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 16, 4, 6, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 16, 4, 6, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 16, 2, 6, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 16, 2, 6, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        8, 16, 1, 6, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<8, 16, 1, 6, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    }
                } else if (numFilterColors % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 32, 4, 8, 16, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 32, 4, 8, 16, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 4, 8, 16, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 4, 8, 16, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 2, 8, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 2, 8, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 1, 8, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 1, 8, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    }
                } else if (numFilterColors % 1 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 32, 4, 4, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 32, 4, 4, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 4, 4, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 4, 4, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 2, 4, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 2, 4, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_mc_mf_kepler_sw<
                                        4, 16, 1, 4, 32, false>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_mc_mf_kepler_sw<4, 16, 1, 4, 32, false>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, numImgColors, numGroups, sumWidth,
                                        scaleTargets, scaleOutput);
                    }
                }
            } else if (numFilterColors <= 3) {
                if (numFilterColors == 3) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 2, 4, 32, 3, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 2, 4, 32, 3, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 4, 3, 32, 3, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 4, 3, 32, 3, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        8, 16, 2, 2, 2, 16, 3, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<8, 16, 2, 2, 2, 16, 3, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 16, 1, 32, 3, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 16, 1, 32, 3, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                } else if (numFilterColors == 2) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 2, 4, 32, 2, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 2, 4, 32, 2, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 4, 3, 32, 2, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 4, 3, 32, 2, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        8, 16, 2, 2, 2, 16, 2, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<8, 16, 2, 2, 2, 16, 2, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 16, 1, 32, 2, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 16, 1, 32, 2, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                } else if (numFilterColors == 1) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 2, 4, 32, 1, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 2, 4, 32, 1, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 4, 3, 32, 1, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 4, 3, 32, 1, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        8, 16, 2, 2, 2, 16, 1, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<8, 16, 2, 2, 2, 16, 1, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    } else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(
                                conv_weight_acts_c_kepler_sw<
                                        16, 16, 2, 16, 1, 32, 1, false, true>,
                                cudaFuncCachePreferShared);
                        conv_weight_acts_c_kepler_sw<
                                16, 16, 2, 16, 1, 32, 1, false, true>
                                <<<blocks, threads, 0, stream>>>(
                                        images.getDevData(), hidActs.getDevData(),
                                        targets.getDevData(), numImages, numFilters,
                                        numModulesY, numModulesX, imgSizeY, imgSizeX,
                                        filterSize, paddingStart, moduleStride,
                                        imgStride, sumWidth, scaleTargets, scaleOutput);
                    }
                }
            }
        }
    }

    getLastCudaError("weightActs: kernel execution failed");
}

void convWeightActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int filterSize,
        int paddingStart, int moduleStride, int numImgColors, int numGroups,
        int partialSum) {
    _weightActs(
            stream, images, hidActs, targets, imgSizeY, numModulesY, numModulesX,
            filterSize, paddingStart, moduleStride, numImgColors, numGroups, partialSum,
            0, 1);
}

void convWeightActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int filterSize,
        int paddingStart, int moduleStride, int numImgColors, int numGroups,
        int partialSum, float scaleTargets, float scaleOutput) {
    _weightActs(
            stream, images, hidActs, targets, imgSizeY, numModulesY, numModulesX,
            filterSize, paddingStart, moduleStride, numImgColors, numGroups, partialSum,
            scaleTargets, scaleOutput);
}

void localWeightActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int filterSize,
        int paddingStart, int moduleStride, int numImgColors, int numGroups) {
    _weightActs(
            stream, images, hidActs, targets, imgSizeY, numModulesY, numModulesX,
            filterSize, paddingStart, moduleStride, numImgColors, numGroups, 1, 0, 1);
}

void localWeightActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int filterSize,
        int paddingStart, int moduleStride, int numImgColors, int numGroups,
        float scaleTargets, float scaleOutput) {
    _weightActs(
            stream, images, hidActs, targets, imgSizeY, numModulesY, numModulesX,
            filterSize, paddingStart, moduleStride, numImgColors, numGroups, 1,
            scaleTargets, scaleOutput);
}

}  // namespace cuda
}  // namespace megdnn
