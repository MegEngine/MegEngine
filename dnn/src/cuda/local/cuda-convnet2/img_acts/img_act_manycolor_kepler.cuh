/**
 * \file dnn/src/cuda/local/cuda-convnet2/img_acts/img_act_manycolor_kepler.cuh
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
template <int B_Y, int B_X, int imgsPerThread, int colorsPerThread, int filterCacheF, int filterCacheH, bool scale, bool checkCaseBounds, bool conv>
__global__ void conv_img_acts_manycolor_kepler(const float* hidActs, const float* filters, float* targets,
                                          const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                                          const int filterSize, const int imgSizeY, const int imgSizeX, const int paddingStart, const int moduleStride,
                                          const int numImgColors, const int numGroups,
                                          const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[colorsPerThread*B_Y][filterCacheF];
    __shared__ float shHidActs[filterCacheH][B_X*imgsPerThread];
    fill_shared_mem<float>((float *)shFilters, sizeof(shFilters)/sizeof(float), 0);
    fill_shared_mem<float>((float *)shHidActs, sizeof(shHidActs)/sizeof(float), 0);
    __syncthreads();

    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockCaseIdx = (blockIdx.x % numImgBlocks) * B_X*imgsPerThread;

    const int imgColorIdx = (blockIdx.x / numImgBlocks) * B_Y*colorsPerThread; // color idx globally
    const int numFilterColors = numImgColors / numGroups;
    const int blockGroupIdx = imgColorIdx / numFilterColors;
    const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;

    const int blockPixelIdx = blockIdx.y;
    const int blockPixelIdxX = blockPixelIdx % imgSizeX;
    const int blockPixelIdxY = blockPixelIdx / imgSizeX;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int hidActLoadY = threadIdx.y, hidActLoadX = threadIdx.x;
    //const int hidActLoadY = tidx / (B_X*imgsPerThread), hidActLoadX = tidx % (B_X*imgsPerThread);
    const int filtersLoadY = tidx / filterCacheF, filtersLoadX = tidx % filterCacheF;
    // nvcc is behaving idiotically again, these useless declarations save registers
    //const int outputY = threadIdx.y, outputX = threadIdx.x;
    //const int ty = threadIdx.y, tx = threadIdx.x;
    const int numModules = numModulesY * numModulesX;

    hidActs += blockCaseIdx + (blockFilterIdx + hidActLoadY) * numImages * numModules + hidActLoadX;
    filters += blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
    targets += (imgColorIdx + threadIdx.y) * imgPixels * numImages + blockPixelIdx * numImages + blockCaseIdx + threadIdx.x;
    //bool active_t = filtersLoadX < numFilters;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }

    const int startY = blockPixelIdxY - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxY - paddingStart - filterSize) / moduleStride;
    const int endY = min(numModulesY, 1 + (blockPixelIdxY - paddingStart) / moduleStride);
    const int startX = blockPixelIdxX - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxX - paddingStart - filterSize) / moduleStride;
    const int endX = min(numModulesX, 1 + (blockPixelIdxX - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[filtersLoadY][filtersLoadX];
    float* shHidActLoad = &shHidActs[hidActLoadY][hidActLoadX];
    //const bool noFLoop = filterCacheF == filterCacheH;
    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInFilterY = blockPixelIdxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInFilterX = blockPixelIdxX - moduleLeft;

            const int pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX;

            for (int f = 0; f < numFiltersPerGroup; f += filterCacheF) { // multiply with filterCacheF filters at a time
                const float* fLoad = conv ? &filters[pxIdxInFilter * numFilters + f]
                                          : &filters[(moduleIdx * numFilterColors * filterPixels + pxIdxInFilter) * numFilters + f];
                #pragma unroll
                for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/filterCacheF) {
                    if (((colorsPerThread*B_Y) % (B_X*B_Y/filterCacheF) == 0 ||
                            i + filtersLoadY < colorsPerThread*B_Y) &&
                            f + filtersLoadX < numFiltersPerGroup)  {
                            shFilterLoad[i * filterCacheF] = fLoad[i * filterPixels * numFilters];
                    } else {
                        shFilterLoad[i * filterCacheF] = 0;

                    }
                }
                //#pragma unroll
                for (int fh = f; fh < f + filterCacheF; fh += filterCacheH) {
                    //conv_img_acts_manycolor_dummy_fhLoop<B_Y, B_X, imgsPerThread, colorsPerThread, filterCacheF, filterCacheH, checkCaseBounds>(hidActs, shHidActLoad, shHidActs, shFilters, moduleIdx, numImages, hidActLoadY, hidActLoadX, blockCaseIdx, numModules, f, fh, prod);

                    const float* hLoad = &hidActs[(moduleIdx + fh * numModules) * numImages];
                    int hload_offset = blockFilterIdx + hidActLoadY + fh;
                    #pragma unroll
                    for (int j = 0; j < filterCacheH; j += B_Y) {
                        if (filterCacheH % B_Y == 0 || hidActLoadY + j < filterCacheH) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread*B_X; i += B_X) {
                                if ((!checkCaseBounds || blockCaseIdx + hidActLoadX + i < numImages)
                                    && hload_offset + j < numFilters) {
                                    shHidActLoad[j * B_X * imgsPerThread + i] = hLoad[j * numModules * numImages + i];
                                } else {
                                    shHidActLoad[j * B_X * imgsPerThread + i] = 0;
                                }
                            }
                        }
                    }
                    __syncthreads();

                    // Do some actual computation
                    // Using these variables causes register usage to go from 161 --> 123.
                    // But nonetheless, the high-register version is faster.
                    //const float* shF = &shFilters[threadIdx.y][fh-f];
                    //const float* const shF2 = &shFilters[threadIdx.y][fh];
                    //const float*  shH = &shHidActs[0][threadIdx.x];
                    #pragma unroll
                    for (int w = 0; w < filterCacheH; w++) {
                        #pragma unroll
                        for (int c = 0; c < colorsPerThread; c++) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread; i++) {
                                // for test (checking result)
                                //float hid_val = shHidActs[w][threadIdx.x + i * B_X];
                                //if (isnan(hid_val)) {
                                //    hid_val = 0;
                                //}
                                prod[c][i] += shFilters[c * B_Y + threadIdx.y][fh-f + w] * shHidActs[w][threadIdx.x + i * B_X];

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
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[c * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[c][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleOutputs * prod[c][i];
                }
            }
        }
    }
}

#define IMG_MANY_COLOR_K_HEAD template __global__ void conv_img_acts_manycolor_kepler
#define IMG_MANY_COLOR_K(scale, ckCase, conv) \
    IMG_MANY_COLOR_K_HEAD< 8, 32, 4, 8, 32, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 8, 32, 2, 8, 32, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 8, 32, 1, 8, 32, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
 \
    IMG_MANY_COLOR_K_HEAD< 8, 32, 4, 8, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 8, 32, 2, 8, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 8, 32, 1, 8, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
 \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 4, 12, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 2, 12, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 1, 12, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
 \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 4, 8, 32, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 2, 8, 32, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 1, 8, 32, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
 \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 4, 8, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 2, 8, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 1, 8, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
 \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 4, 4, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 2, 4, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 1, 4, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
 \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 4, 2, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 2, 2, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \
    IMG_MANY_COLOR_K_HEAD< 4, 32, 1, 2, 16, 16, scale, ckCase, conv > (MANYCOLOR_KEP_PARAM); \

// ftt
//< 8, 32, 1, 8, 32, 16, scale, conv, conv >
//< 8, 32, 1, 8, 16, 16, scale, conv, conv >
//< 4, 32, 1, 12, 16, 16, scale, conv, conv >
//< 4, 32, 1, 8, 32, 16, scale, conv, conv >
//< 4, 32, 1, 8, 16, 16, scale, conv, conv >
//< 4, 32, 1, 4, 16, 16, scale, conv, conv >
//< 4, 32, 1, 2, 16, 16, scale, conv, conv >

} // namespace cuda
} // namespace megdnn
