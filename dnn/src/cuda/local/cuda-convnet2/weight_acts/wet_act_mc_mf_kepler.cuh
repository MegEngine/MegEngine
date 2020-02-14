/**
 * \file dnn/src/cuda/local/cuda-convnet2/weight_acts/wet_act_mc_mf_kepler.cuh
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
#include "wet_act_templates.cuh"

namespace megdnn {
namespace cuda {
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
__global__ void conv_weight_acts_mc_mf_kepler(float* images, float* hidActs, float* targets,
                                       const int numImages, const int numFilters,
                                       const int numModulesY, const int numModulesX,
                                       const int imgSizeY, const int imgSizeX, const int filterSize,
                                       const int paddingStart, const int moduleStride, const int imgStride,
                                       const int numImgColors, const int numGroups, const int partialSum,
                                       const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImages[colorsPerThread * B_Y][preloadCases]; // preload preloadCases cases
    __shared__ float shHidActs[filtersPerThread * B_X][preloadCases + 1]; // preload preloadCases cases of B_X hidacts
    fill_shared_mem<float>((float *)shImages, sizeof(shImages)/sizeof(float), 0);
    fill_shared_mem<float>((float *)shHidActs, sizeof(shHidActs)/sizeof(float), 0);
    __syncthreads();

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);
    const int outputModuleIdx = blockIdx.x / numFilterBlocks;
    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = filtersPerThread * B_X * (blockIdx.x % numFilterBlocks);
    const int numModules = numModulesY * numModulesX;

    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
    const int numFilterColors = numImgColors / numGroups;

    const int blockPixelOffset = blockIdx.z; // pixel idx in filter
    const int blockPixelY = blockPixelOffset / filterSize, blockPixelX = blockPixelOffset % filterSize;
    const int blockFilterColorIdx = blockIdx.y  * B_Y * colorsPerThread;
    const int imgColorIdx = blockFilterColorIdx + blockGroupIdx * numFilterColors;

    images += (imgColorIdx + loadY) * imgPixels * imgStride + loadX;

    hidActs +=
             blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;

    targets += outputModuleIdx * numFilters * filterPixels * numFilterColors
            + (blockFilterColorIdx + threadIdx.y) * filterPixels * numFilters
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.x;
    //if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;
    float* shHidActLoad = &shHidActs[loadY][loadX];
    float* shImgLoad = &shImages[loadY][loadX];
    float prod[colorsPerThread][filtersPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            prod[c][f] = 0;
        }
    }

    for (int m = moduleIdx; m < moduleIdx + partialSum; m++) {
        const int imgLoadModPosY = paddingStart + (m / numModulesX) * moduleStride;
        const int imgLoadModPosX = paddingStart + (m % numModulesX) * moduleStride;
        const int pxY = imgLoadModPosY + blockPixelY; // pixel x,y coords in image
        const int pxX = imgLoadModPosX + blockPixelX;
        const int pixIdx = (pxY * imgSizeX + pxX) * imgStride; // pixel idx in image
        if (pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX) {
            for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
                // Checking this condition actually makes things faster ... :/
                // So I've removed the !checkCaseBounds flag and just check it all the time.
                if (caseIdx + loadX < numImages) {
                    /*
                     * As long as B_Y * B_X is divisible by preloadCases this will loop the right
                     * number of times.
                     *
                     * This will load some images from filter pixels that don't exist (it'll set those to 0),
                     * but the code does not produce any output for those pixels (see last lines).
                     */
                    if (loadY < B_Y * colorsPerThread) {
                        #pragma unroll
                        for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
                            // Make sure number of rows in the array is divisible by number of rows filled per iteration
                            if ((B_Y*colorsPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y*colorsPerThread) {
                                shImgLoad[(y) * preloadCases] = images[caseIdx + y * imgPixels * imgStride + pixIdx];
                            }
                        }
                    }

                    if (loadY < B_X * filtersPerThread) {
                        #pragma unroll
                        for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                            // Make sure number of rows in the array is divisible by number of rows filled per iteration
                            if ((B_X * filtersPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X * filtersPerThread) {
                                shHidActLoad[y * (preloadCases + 1)] = hidActs[caseIdx + (y * numModules + m) * numImages];
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int y = 0; y < B_Y * colorsPerThread; y += (B_X * B_Y) / preloadCases) {
                        // Make sure number of rows in the array is divisible by number of rows filled per iteration
                        if ((B_Y*colorsPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y*colorsPerThread) {
                            shImgLoad[(y) * preloadCases] = 0;
                        }
                    }
                    #pragma unroll
                    for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                        // Make sure number of rows in the array is divisible by number of rows filled per iteration
                        if ((B_X * filtersPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X * filtersPerThread) {
                            shHidActLoad[y * (preloadCases + 1)] = 0;
                        }
                    }
                }

                __syncthreads();
                #pragma unroll
                for (int i = 0; i < preloadCases; i++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        #pragma unroll
                        for (int c = 0; c < colorsPerThread; c++) {
                            prod[c][f] += shImages[threadIdx.y + c * B_Y][i] * shHidActs[threadIdx.x + f * B_X][i];
                        }
                    }
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
                targets[c * B_Y * filterPixels * numFilters + f * B_X] = scaleTargets * targets[c * B_Y * filterPixels * numFilters + f * B_X] + scaleOutputs * prod[c][f];
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[c * B_Y * filterPixels * numFilters + f * B_X] = scaleOutputs * prod[c][f];
            }
        }
    }
}

} // namespace cuda
} // namespace megdnn
