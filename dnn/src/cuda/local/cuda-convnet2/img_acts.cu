/**
 * \file dnn/src/cuda/local/cuda-convnet2/img_acts.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
/*
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

#include "cudaconv2.cuh"

#include "nvmatrix.cuh"
#include "img_acts/img_act_templates.cuh"

#ifdef _WIN32
#define _Pragma(x)
#endif

namespace megdnn {
namespace cuda {
/*
 * New Titan-optimized stuff.
 */

__device__ __forceinline__ void conv_img_acts_manycolor_preload_ty_8_tx_32_c_8_ff_32_fh_16_setCoords(const int my, const int mx, const int numModulesX,
        const int paddingStart, const int moduleStride, const int blockPixelIdxY, const int blockPixelIdxX, const int filterSize, int &moduleIdx, int &pxIdxInFilter) {
    const int moduleTop = paddingStart + my * moduleStride;
    const int pxInFilterY = blockPixelIdxY - moduleTop;

    moduleIdx = my * numModulesX + mx; // out
    const int moduleLeft = paddingStart + mx * moduleStride;
    const int pxInFilterX = blockPixelIdxX - moduleLeft;

    pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX; // out
}

#define IA_PRELOAD_LOOP(w,offset) _Pragma("unroll") \
for (int i = 0; i < imgsPerThread; i++) { \
    _Pragma("unroll") \
    for (int c = 0; c < colorsPerThread; c++) { \
        prod[c][i] += shFilters[c * B_Y + threadIdx.y][(w)+(offset)] * shHidActs[w][threadIdx.x * imgsPerThread + i]; \
    } \
} \

/*
 * Same loop as above but inverted.
 */
#define IA_PRELOAD_LOOP2(w,offset) _Pragma("unroll") \
for (int c = 0; c < colorsPerThread; c++) { \
    _Pragma("unroll") \
    for (int i = 0; i < imgsPerThread; i++) { \
        prod[c][i] += shFilters[c * B_Y + threadIdx.y][(w)+(offset)] * shHidActs[w][threadIdx.x * imgsPerThread + i]; \
    } \
} \

#define IA_PRELOAD_LOOP3(i,offset) _Pragma("unroll") \
for (int w = 0; w < filterCacheH; w++) { \
    _Pragma("unroll") \
    for (int c = 0; c < colorsPerThread; c++) { \
        prod[c][i] += shFilters[c * B_Y + threadIdx.y][(w)+(offset)] * shHidActs[w][threadIdx.x * imgsPerThread + i]; \
    } \
} \

#define IA_PRELOAD_W(z) wPreload[z] = fLoad[(z) * B_X*B_Y/filterCacheF * filterPixels * numFilters];
#define IA_PRELOAD_W_TX(z) wPreload[z] = tex1Dfetch<float>(filters, filtersLoadOffset + (z) * B_X*B_Y/filterCacheF * filterPixels * numFilters);
#define IA_PRELOAD_H(y,x) if (!checkCaseBounds || myCaseIdx + (x) * B_X < numImages) { \
    hPreload[y][x] =  hLoad[(y) * B_Y * numModules * numImages + (x) * B_X]; \
}
#define IA_PRELOAD_H_TX(y,x) if (!checkCaseBounds || myCaseIdx + (x) * B_X < numImages) { \
    hPreload[y][x] =  tex1Dfetch<float>(hidActs, hidActsLoadOffset + (y) * B_Y * numModules * numImages + (x) * B_X); \
}

template <int B_Y, int B_X, int imgsPerThread, int colorsPerThread, int filterCacheF, int filterCacheH, bool scale, bool checkCaseBounds, bool conv>
__global__ void
__launch_bounds__(256, 2)   // 256 threads per block, 2 blocks per multiprocessor
                            // These launch bounds ensure 25% occupancy (128 registers used)
                            // as oppposed to 13% (130 registers) achieved by defaults.
conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex(cudaTextureObject_t hidActs, cudaTextureObject_t filters, float* targets,
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
    const int myCaseIdx = blockCaseIdx + threadIdx.x;

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
//    const int hidActLoadY = threadIdx.y % B_Y, hidActLoadX = threadIdx.x % B_X;
    //const int hidActLoadY = tidx / (B_X*imgsPerThread), hidActLoadX = tidx % (B_X*imgsPerThread);
    const int filtersLoadY = tidx / filterCacheF, filtersLoadX = tidx % filterCacheF;
    // nvcc is behaving idiotically again, these useless declarations save registers
    //const int outputY = threadIdx.y, outputX = threadIdx.x;
    //const int ty = threadIdx.y, tx = threadIdx.x;
    const int numModules = numModulesY * numModulesX;
    const int hidActsOffset = (blockFilterIdx + threadIdx.y) * numImages * numModules + myCaseIdx;
    const int filtersOffset = blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
//    hidActs += (blockFilterIdx + threadIdx.y) * numImages * numModules + myCaseIdx;
//    filters += blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
    targets += (imgColorIdx + threadIdx.y) * imgPixels * numImages + blockPixelIdx * numImages + myCaseIdx;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
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
    float* shHidActLoad = &shHidActs[threadIdx.y][threadIdx.x * imgsPerThread];
    //const bool noFLoop = filterCacheF == filterCacheH;

    /*
     * Initial preload
     */
    float hPreload[filterCacheH/B_Y][imgsPerThread]; // [2][4]
    float wPreload[filterCacheF*colorsPerThread/B_X]; // [8]

    int moduleIdx, pxIdxInFilter;
    conv_img_acts_manycolor_preload_ty_8_tx_32_c_8_ff_32_fh_16_setCoords(startY, startX, numModulesX, paddingStart, moduleStride, blockPixelIdxY,
                                                                         blockPixelIdxX, filterSize, moduleIdx, pxIdxInFilter);
//    const float* fLoad = conv ? &filters[pxIdxInFilter * numFilters + 0]
//                              : &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + 0];
    int filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilter * numFilters + 0
                                                  : moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters);
    #pragma unroll
    for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/filterCacheF) {
        if ((colorsPerThread*B_Y) % (B_X*B_Y/filterCacheF) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
            wPreload[i * filterCacheF/(B_X*B_Y)] = tex1Dfetch<float>(filters, filtersLoadOffset + i * filterPixels * numFilters);
        }
    }

//    const float* hLoad = &hidActs[(moduleIdx + 0 * numModules) * numImages];
    int hidActsLoadOffset = hidActsOffset + (moduleIdx + 0 * numModules) * numImages;
    #pragma unroll
    for (int j = 0; j < filterCacheH; j += B_Y) {
        if (filterCacheH % B_Y == 0 || threadIdx.y + j < filterCacheH) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    hPreload[j/B_Y][i] = tex1Dfetch<float>(hidActs, hidActsLoadOffset + j * numModules * numImages + i * B_X);
                }
            }
        }
    }

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInFilterY = blockPixelIdxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInFilterX = blockPixelIdxX - moduleLeft;

            pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX;
            int myNext = my, mxNext = mx, moduleIdxNext, pxIdxInFilterNext;
            const bool lastModule = my == endY - 1 && mx == endX - 1;
            if (!lastModule) {
                mxNext = mx + 1 == endX ? startX : mx + 1;
                myNext = my + (mx + 1 == endX);
            }
            conv_img_acts_manycolor_preload_ty_8_tx_32_c_8_ff_32_fh_16_setCoords(myNext, mxNext, numModulesX, paddingStart, moduleStride, blockPixelIdxY,
                                                                                 blockPixelIdxX, filterSize, moduleIdxNext, pxIdxInFilterNext);
            for (int f = 0; f < numFiltersPerGroup; f += filterCacheF) { // multiply with filterCacheF filters at a time
                #pragma unroll
                for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/filterCacheF) {
                    if ((colorsPerThread*B_Y) % (B_X*B_Y/filterCacheF) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
                        shFilterLoad[i * filterCacheF] = wPreload[i * filterCacheF/(B_X*B_Y)];
                    }
                }

                filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilter * numFilters + f + filterCacheF
                                                          : moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + f + filterCacheF);
                if (f == numFiltersPerGroup - filterCacheF) {
                    filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilterNext * numFilters
                                                              : moduleIdxNext * numFilterColors * filterPixels * numFilters + pxIdxInFilterNext * numFilters);
                }

                #pragma unroll
                for (int j = 0; j < filterCacheH; j += B_Y) {
                    if (filterCacheH % B_Y == 0 || threadIdx.y + j < filterCacheH) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            // NOTE: bank conflicts here!
                            if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                                shHidActLoad[j * B_X * imgsPerThread + i] = hPreload[j/B_Y][i];
                            }
                        }
                    }
                }

                __syncthreads();

                hidActsLoadOffset = hidActsOffset + (moduleIdx + (f + filterCacheH) * numModules) * numImages;

                #pragma unroll
                for (int z = 0; z < 4; ++z) {
                    IA_PRELOAD_LOOP(z,0);
                    IA_PRELOAD_W_TX(z);
                }

                #pragma unroll
                for (int z = 4; z < 12; ++z) {
                    IA_PRELOAD_LOOP(z,0);
                    IA_PRELOAD_H_TX((z-4)/4,z%4);
                }

                #pragma unroll
                for (int z = 12; z < 16; ++z) {
                    IA_PRELOAD_LOOP(z,0);
                }

                __syncthreads();

                #pragma unroll
                for (int j = 0; j < filterCacheH; j += B_Y) {
                    if (filterCacheH % B_Y == 0 || threadIdx.y + j < filterCacheH) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                                shHidActLoad[j * B_X * imgsPerThread + i] = hPreload[j/B_Y][i];
                            }
                        }
                    }
                }

                __syncthreads();

                hidActsLoadOffset = hidActsOffset + (moduleIdx + (f + filterCacheF) * numModules) * numImages;
                if (f == numFiltersPerGroup - filterCacheF) {
                    hidActsLoadOffset = hidActsOffset + moduleIdxNext * numImages;
                }

                #pragma unroll
                for (int z = 0; z < 4; ++z) {
                    IA_PRELOAD_LOOP(z,filterCacheH);
                    IA_PRELOAD_W_TX(z+4);
                }

                #pragma unroll
                for (int z = 4; z < 12; ++z) {
                    IA_PRELOAD_LOOP(z,filterCacheH);
                    IA_PRELOAD_H_TX((z-4)/4, z%4);
                }

                #pragma unroll
                for (int z = 12; z < 16; ++z) {
                    IA_PRELOAD_LOOP(z,filterCacheH);
                }

                __syncthreads();
            }
        }
    }
    if (scale) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[c * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[c][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleOutputs * prod[c][i];
                }
            }
        }
    }
}


template <int B_Y, int B_X, int imgsPerThread, int colorsPerThread, int filterCacheF, int filterCacheH, bool scale, bool checkCaseBounds, bool conv>
__global__ void
//__launch_bounds__(128, 3)   // 128 threads per block, 3 blocks per multiprocessor
conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16(cudaTextureObject_t hidActs, cudaTextureObject_t filters, float* targets,
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
    const int myCaseIdx = blockCaseIdx + threadIdx.x;

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
//    const int hidActLoadY = threadIdx.y % B_Y, hidActLoadX = threadIdx.x % B_X;
    //const int hidActLoadY = tidx / (B_X*imgsPerThread), hidActLoadX = tidx % (B_X*imgsPerThread);
    const int filtersLoadY = tidx / filterCacheF, filtersLoadX = tidx % filterCacheF;
    // nvcc is behaving idiotically again, these useless declarations save registers
    //const int outputY = threadIdx.y, outputX = threadIdx.x;
    //const int ty = threadIdx.y, tx = threadIdx.x;
    const int numModules = numModulesY * numModulesX;

    const int hidActsOffset = (blockFilterIdx + threadIdx.y) * numImages * numModules + myCaseIdx;
    const int filtersOffset = blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;

//    hidActs += (blockFilterIdx + threadIdx.y) * numImages * numModules + myCaseIdx;
//    filters += blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
    targets += (imgColorIdx + threadIdx.y) * imgPixels * numImages + blockPixelIdx * numImages + myCaseIdx;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
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
    float* shHidActLoad = &shHidActs[threadIdx.y][threadIdx.x * imgsPerThread];
    //const bool noFLoop = filterCacheF == filterCacheH;

    /*
     * Initial preload
     */
    float hPreload[filterCacheH/B_Y][imgsPerThread]; // [4][4]
    float wPreload[filterCacheF*colorsPerThread/B_X]; // [6]

    int moduleIdx, pxIdxInFilter;
    conv_img_acts_manycolor_preload_ty_8_tx_32_c_8_ff_32_fh_16_setCoords(startY, startX, numModulesX, paddingStart, moduleStride, blockPixelIdxY,
                                                                         blockPixelIdxX, filterSize, moduleIdx, pxIdxInFilter);
//    const float* fLoad = conv ? &filters[pxIdxInFilter * numFilters + 0]
//                              : &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + 0];
    int filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilter * numFilters
                                                : moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters);
    #pragma unroll
    for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/filterCacheF) {
        if ((colorsPerThread*B_Y) % (B_X*B_Y/filterCacheF) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
            wPreload[i * filterCacheF/(B_X*B_Y)] = tex1Dfetch<float>(filters, filtersLoadOffset + i * filterPixels * numFilters);
        }
    }

//    const float* hLoad = &hidActs[moduleIdx * numImages];
    int hidActsLoadOffset = hidActsOffset + moduleIdx * numImages;
    #pragma unroll
    for (int j = 0; j < filterCacheH; j += B_Y) {
        if (filterCacheH % B_Y == 0 || threadIdx.y + j < filterCacheH) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    hPreload[j/B_Y][i] = tex1Dfetch<float>(hidActs, hidActsLoadOffset + j * numModules * numImages + i * B_X);
                }
            }
        }
    }

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInFilterY = blockPixelIdxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInFilterX = blockPixelIdxX - moduleLeft;

            pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX;
            int myNext = my, mxNext = mx, moduleIdxNext, pxIdxInFilterNext;
            const bool lastModule = my == endY - 1 && mx == endX - 1;
            if (!lastModule) {
                mxNext = mx + 1 == endX ? startX : mx + 1;
                myNext = my + (mx + 1 == endX);
            }
            conv_img_acts_manycolor_preload_ty_8_tx_32_c_8_ff_32_fh_16_setCoords(myNext, mxNext, numModulesX, paddingStart, moduleStride, blockPixelIdxY,
                                                                                 blockPixelIdxX, filterSize, moduleIdxNext, pxIdxInFilterNext);
            for (int f = 0; f < numFiltersPerGroup; f += filterCacheF) { // multiply with filterCacheF filters at a time
                #pragma unroll
                for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/filterCacheF) {
                    if ((colorsPerThread*B_Y) % (B_X*B_Y/filterCacheF) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
                        shFilterLoad[i * filterCacheF] = wPreload[i * filterCacheF/(B_X*B_Y)];
                    }
                }

                filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilter * numFilters + f + filterCacheF
                                                          : moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + f + filterCacheF);
                if (f == numFiltersPerGroup - filterCacheF) {
                    filtersLoadOffset = filtersOffset + (conv ? pxIdxInFilterNext * numFilters
                                                              : moduleIdxNext * numFilterColors * filterPixels * numFilters + pxIdxInFilterNext * numFilters);
                }

                #pragma unroll
                for (int j = 0; j < filterCacheH; j += B_Y) {
                    if (filterCacheH % B_Y == 0 || threadIdx.y + j < filterCacheH) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            // NOTE: bank conflicts here!
                            if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                                shHidActLoad[j * B_X * imgsPerThread + i] = hPreload[j/B_Y][i];
                            }
                        }
                    }
                }
                hidActsLoadOffset = hidActsOffset + (moduleIdx + (f + filterCacheF) * numModules) * numImages;
                if (f == numFiltersPerGroup - filterCacheF) {
                    hidActsLoadOffset = hidActsOffset + moduleIdxNext * numImages;
                }

                __syncthreads();

                // It seems that there is no point explicitly interleaving loads
                // and computations because the scheduler does that anyway.

                IA_PRELOAD_LOOP2(0,0);
                IA_PRELOAD_LOOP2(1,0);
                IA_PRELOAD_LOOP2(2,0);
                IA_PRELOAD_LOOP2(3,0);
                IA_PRELOAD_LOOP2(4,0);
                IA_PRELOAD_LOOP2(5,0);
                IA_PRELOAD_LOOP2(6,0);
                IA_PRELOAD_LOOP2(7,0);
                IA_PRELOAD_LOOP2(8,0);
                IA_PRELOAD_LOOP2(9,0);
                IA_PRELOAD_LOOP2(10,0);
                IA_PRELOAD_LOOP2(11,0);
                IA_PRELOAD_LOOP2(12,0);
                IA_PRELOAD_LOOP2(13,0);
                IA_PRELOAD_LOOP2(14,0);
                IA_PRELOAD_LOOP2(15,0);

                IA_PRELOAD_W_TX(0);
                IA_PRELOAD_W_TX(1);
                IA_PRELOAD_W_TX(2);
                IA_PRELOAD_W_TX(3);
                IA_PRELOAD_W_TX(4);
                IA_PRELOAD_W_TX(5);

                IA_PRELOAD_H_TX(0,0);
                IA_PRELOAD_H_TX(0,1);
                IA_PRELOAD_H_TX(0,2);
                IA_PRELOAD_H_TX(0,3);
                IA_PRELOAD_H_TX(1,0);
                IA_PRELOAD_H_TX(1,1);
                IA_PRELOAD_H_TX(1,2);
                IA_PRELOAD_H_TX(1,3);
                IA_PRELOAD_H_TX(2,0);
                IA_PRELOAD_H_TX(2,1);
                IA_PRELOAD_H_TX(2,2);
                IA_PRELOAD_H_TX(2,3);
                IA_PRELOAD_H_TX(3,0);
                IA_PRELOAD_H_TX(3,1);
                IA_PRELOAD_H_TX(3,2);
                IA_PRELOAD_H_TX(3,3);

                __syncthreads();
            }
        }
    }
    if (scale) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[c * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[c][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || myCaseIdx + i * B_X < numImages) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleOutputs * prod[c][i];
                }
            }
        }
    }
}

/*
 * hidActs:         (numFilters, numModules, numImages)
 * filters:         (numFilterColors, filterPixels, numFilters)               if conv
 *                  (numModules, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:         (overSample, numImgColors, imgPixels, numImages)
 *
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128.
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast.
 */
void _imgActs(cudaStream_t stream, NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
              int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
              float scaleTargets, float scaleOutput, bool conv) {
    int numFilterColors = numImgColors / numGroups;
    int numImages = hidActs.getNumCols();
    int numFilters = filters.getNumCols();
    int numModules = hidActs.getNumRows() / numFilters;
    int filterModuleMult = conv ? 1 : numModules;
    int filterPixels = filters.getNumRows() / (filterModuleMult * numFilterColors);
    int filterSize = sqrt(filterPixels);
    int imgPixels = imgSizeY * imgSizeX;
    int numModulesX = numModules / numModulesY;

    megdnn_assert_internal(numImgColors % numGroups == 0);
    //megdnn_assert_internal(numFilters % (16*numGroups) == 0); // TODO: insisting on 32 filters due to bug in calling code below. fix that.
    bool previous_limit = (numFilters % (16 * numGroups)) == 0;

    megdnn_assert_internal(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
    megdnn_assert_internal(numGroups == 1 || numFilterColors % 4 == 0);

    megdnn_assert_internal(filterPixels == filterSize * filterSize);
    megdnn_assert_internal(hidActs.getNumRows() == numModules * numFilters);
    megdnn_assert_internal(filters.getNumRows() == filterModuleMult * numFilterColors * filterPixels);
    megdnn_assert_internal(numModules == numModulesY * numModulesX);

    megdnn_assert_internal(hidActs.isContiguous());
    megdnn_assert_internal(filters.isContiguous());

    megdnn_assert_internal(!hidActs.isTrans());
    megdnn_assert_internal(!filters.isTrans());
    megdnn_assert_internal(!targets.isTrans());
    // These routines don't handle the case when only part of the image is visited in the convolution
    megdnn_assert_internal(paddingStart <= 0);
    megdnn_assert_internal(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    megdnn_assert_internal(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
    megdnn_assert_internal(moduleStride <= filterSize);

    megdnn_assert_internal(targets.isContiguous()); // no stride support here!

    dim3 blocks;
    dim3 threads;
    int colorsPerThread = 0, imgsPerThread = 0;
    if (numFilterColors % 8 == 0) {
        threads = dim3(32, numFilterColors % 64 == 0 ? 8 : 4);
        colorsPerThread = numFilterColors % 64 == 0 ? 8
                        : numFilterColors % 48 == 0 ? 12
                        : numFilterColors % 32 == 0 ? 8
                        : numFilterColors % 16 == 0 ? 4
                        : 2;
        imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
        megdnn_assert_internal(numFilterColors % (threads.y * colorsPerThread) == 0);
        //previous_limit = numFilterColors % (threads.y * colorsPerThread) == 0;

        blocks = dim3(DIVUP(numImages, threads.x*imgsPerThread) * (numImgColors/(threads.y*colorsPerThread)), imgPixels);
        // NOTE: the case when channels % 32 == 0 but channels % 48 != 0 and channels % 64 != 0 has not been optimized!!
    } else if (numFilterColors > 3) {
        // NOTE: THIS CASE HAS NOT BEEN OPTIMIZED FOR KEPLER!!
        imgsPerThread = numImages % 128 == 0 ? 8 : numImages % 64 == 0 ? 4 : 2;
        threads = dim3(16, 16);
        colorsPerThread = numFilterColors % 4 == 0 ? 4 : 2;
        blocks = dim3(DIVUP(numImages,threads.x*imgsPerThread) * (numImgColors / colorsPerThread), DIVUP(imgSizeY,4) * DIVUP(imgSizeX,4));
    } else {
        // NOTE: THIS CASE HAS NOT BEEN OPTIMIZED FOR KEPLER!!
        imgsPerThread = numImages % 128 == 0 ? 8 : numImages % 64 == 0 ? 4 : 2;
        threads = dim3(16, 16);
        blocks = dim3(DIVUP(numImages,threads.x*imgsPerThread), DIVUP(imgSizeY,4) * DIVUP(imgSizeX,4));
    }
    bool checkCaseBounds = numImages % (threads.x * imgsPerThread) != 0;

    if (scaleTargets == 0) { // do not scale or use targets matrix
        targets.resize(numImgColors*imgPixels, numImages);
    } else {
        megdnn_assert_internal(targets.getNumRows() == numImgColors * imgPixels);
        megdnn_assert_internal(targets.getNumCols() == numImages);
    }
    const bool scale = scaleTargets != 0;
//    cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, false, false, true >, cudaFuncCachePreferShared);
//    conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, false, false, true ><<<blocks, threads, 0, stream>>>(
//            hidActs.getTextureObject(), filters.getTextureObject(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize,
//            imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);

    //return;
//    printf("conv: %d\n", conv);
//    printf("scale: %d\n", scale);
//    printf("checkCaseBounds: %d\n", checkCaseBounds);
//    printf("numFilterColors: %d\n", numFilterColors);
//    printf("numImages: %d\n", numImages);
//    cudaStream_t stream = NVMatrix::getDefaultStream();

    if (conv == false) {
        if (scale == false) {
            if (checkCaseBounds == false) {
                if (numFilterColors % 8 == 0) {
                    if (numFilterColors % 64 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 128 == 0) {
                                if (previous_limit) {
                                    cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex< 8, 32, 4, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                    conv_img_acts_manycolor_preloadfh_ty_8_tx_32_c_8_ff_32_fh_16_tex< 8, 32, 4, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getTextureObject(), filters.getTextureObject(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                                } else {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                                }
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if ((numFilters % 1 == 0)) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 4, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 2, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 48 == 0) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 128 == 0) {
                                if (previous_limit) {
                                    cudaFuncSetCacheConfig(conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                    conv_img_acts_manycolor_preloadfh_ty_4_tx_32_c_12_ff_16_fh_16< 4, 32, 4, 12, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getTextureObject(), filters.getTextureObject(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                                } else {
                                    cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 12, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                    conv_img_acts_manycolor_kepler < 4, 32, 4, 12, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                                }
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 12, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 12, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 32 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if ((numFilters % 1 == 0)) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 16 == 0) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 4, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 4, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 4, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 4, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 8 == 0) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 4, 2, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 4, 2, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 2, 2, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 2, 2, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors > 3) {
                    if (numFilterColors == 4) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 8, 4, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 8, 4, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 4, 4, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 4, 4, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors <= 3) {
                    if (numFilterColors == 3) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 3, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 3, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 3, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 3, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 1) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 128 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 8, 1, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 8, 1, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 64 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 4, 1, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 4, 1, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 32 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                            else if (numImages % 16 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, false, false, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, false, false, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
            }
            else if (checkCaseBounds == true) {
                if (numFilterColors % 8 == 0) {
                    if (numFilterColors % 64 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 32, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if ((numFilters % 1 == 0)) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 8, 32, 1, 8, 16, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 48 == 0) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 12, 16, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 32 == 0) {
                        if (numFilters % 32 == 0) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 32, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                        else if ((numFilters % 1 == 0)) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 8, 16, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 16 == 0) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 4, 16, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors % 8 == 0) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, true, false >, cudaFuncCachePreferShared);
                                conv_img_acts_manycolor_kepler < 4, 32, 1, 2, 16, 16, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
                else if (numFilterColors > 3) {
                    if (numFilterColors == 4) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_mediumcolor < 2, 4, false, true, false >, cudaFuncCachePreferShared);
                                img_acts_mediumcolor < 2, 4, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    /*
                    else if (numFilterColors == 2) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    */
                }
                else if (numFilterColors <= 3) {
                    if (numFilterColors == 3) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 3, false, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 3, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 2) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 2, false, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 2, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                    else if (numFilterColors == 1) {
                        if ((numFilters % 1 == 0)) {
                            if (numImages % 1 == 0) {
                                cudaFuncSetCacheConfig(img_acts_color < 2, 1, false, true, false >, cudaFuncCachePreferShared);
                                img_acts_color < 2, 1, false, true, false ><<<blocks, threads, 0, stream>>>(hidActs.getDevData(), filters.getDevData(), targets.getDevData(), numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput);
                            }
                        }
                    }
                }
            }
        }
    }

    getLastCudaError("imgActs: kernel execution failed");
}


void convImgActs(cudaStream_t stream, NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                    int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups) {
    _imgActs(stream, hidActs, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, numGroups, 0, 1, true);
}

void convImgActs(cudaStream_t stream, NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                    int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                    float scaleTargets, float scaleOutput) {
    _imgActs(stream, hidActs, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, true);
}

void localImgActs(cudaStream_t stream, NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                    int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups) {
    _imgActs(stream, hidActs, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, numGroups, 0, 1, false);
}

void localImgActs(cudaStream_t stream, NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                    int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                    float scaleTargets, float scaleOutput) {
    _imgActs(stream, hidActs, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, false);
}

} // namespace cuda
} // namespace megdnn

