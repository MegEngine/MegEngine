/**
 * \file dnn/src/cuda/local/cuda-convnet2/filter_acts.cu
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

#include "nvmatrix.cuh"
#include "cudaconv2.cuh"
#include "src/cuda/utils.cuh"
#include "filter_acts/filter_act_templates.cuh"

namespace megdnn {
namespace cuda {

__device__ __forceinline__ void filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(int fPidx, int imgLoadModPosY, int imgLoadModPosX,
    int imgSizeX, int filterSize, int& iPidx) {
    int x = imgLoadModPosX + (fPidx) % filterSize;
    int y = imgLoadModPosY + (fPidx) / filterSize;
    iPidx = y >= 0 && y < imgSizeX && x >= 0 && x < imgSizeX ? y * imgSizeX + x : -1;
}

#define FA_COLOR3_IMPRELOAD(c,i) imPreload[c][i] = iPidxNext < 0 || (checkImgBounds && myImgIdx + i * B_X >= numImages) ? 0 : mm[c * imgPixels * imgStride + i * B_X];
#define FA_COLOR3_IMPRELOAD_TX(c,i) imPreload[c][i] = iPidxNext < 0 || (checkImgBounds && myImgIdx + i * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imagesOffset2 + c * imgPixels * imgStride + i * B_X);


/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, int pixelCache,
          bool scale, bool checkImgBounds>
//__launch_bounds__(128,3)
__global__ void filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex(cudaTextureObject_t images, cudaTextureObject_t filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride,
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv/*, const bool noloads*/) {
    __shared__ float shFilters[numColors][pixelCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[numColors][pixelCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
    fill_shared_mem<float>((float *)shFilters, sizeof(shFilters)/sizeof(float), 0);
    fill_shared_mem<float>((float *)shImages, sizeof(shImages)/sizeof(float), 0);
    __syncthreads();
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);

    const int numModules = numModulesX * numModulesY;
    // Another fun insanity: the % B_X makes things faster, even thought threadIdx.x is
    // in the range 0..31. It appears that this allows the compiler to optimize?
    const int tx = threadIdx.x % B_X;
    const int ty = threadIdx.y % B_Y;
    const int tidx = ty * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

//    images += myImgIdx;
//    filters += blockFilterIdx
//            + shFilterLoadY * numFilters + shFilterLoadX;
//    if (!conv) { // NOTE: UNTESTED!
//        filters += moduleIdx * numColors * filterPixels * numFilters;
//    }

    const int imagesOffset = myImgIdx;
    const int filtersOffset = blockFilterIdx + shFilterLoadY * numFilters + shFilterLoadX
                            + (conv ? 0 : moduleIdx * numColors * filterPixels * numFilters);

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y * filtersPerThread) * numImages * numModules
            + myImgIdx;

    float prod[imgsPerThread][filtersPerThread];
    #pragma unroll
    for(int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for(int f = 0; f < filtersPerThread; f++) {
            prod[i][f] = 0;
        }
    }

    int iPidxNext;
    float imPreload[numColors][imgsPerThread];
    float fPreload[numColors][pixelCache*filtersPerThread/B_X];

    #pragma unroll
    for (int c = 0; c < numColors; ++c) {
        #pragma unroll
        for (int p = 0; p < pixelCache; p += B_X/filtersPerThread) {
            if (p + shFilterLoadY < filterPixels) {
                fPreload[c][p*filtersPerThread/B_X] = tex1Dfetch<float>(filters, filtersOffset + p * numFilters + c * numFilters * filterPixels);
            } else{
                fPreload[c][p*filtersPerThread/B_X] = 0;
            }
        }
    }

    filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(ty, imgLoadModPosY, imgLoadModPosX, imgSizeX, filterSize, iPidxNext);

    #pragma unroll
    for (int c = 0; c < numColors; ++c) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (iPidxNext >= 0 && (!checkImgBounds || myImgIdx + i * B_X < numImages)) {
                imPreload[c][i] = tex1Dfetch<float>(images, imagesOffset + (c * imgPixels + iPidxNext) * imgStride + i * B_X);
            } else {
                imPreload[c][i] =  0;
            }
        }
    }

    for (int p = 0; p < filterPixels; p += pixelCache) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int c = 0; c < numColors; ++c) {
                // NOTE: bank conflicts here!
                shImages[c][ty][tx * imgsPerThread + i] = imPreload[c][i];
            }
        }

        const int fPidxNext = p + pixelCache >= filterPixels ? 0 : p + pixelCache;
        filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(fPidxNext + ty, imgLoadModPosY, imgLoadModPosX, imgSizeX, filterSize, iPidxNext);

//        const float* ff = &filters[numFilters * fPidxNext];
//        const float* mm = &images[imgStride * iPidxNext];
        const int filtersOffset2 = filtersOffset + numFilters * fPidxNext;
        const int imagesOffset2 = imagesOffset + imgStride * iPidxNext;

        FA_COLOR3_IMPRELOAD_TX(0,0);
        FA_COLOR3_IMPRELOAD_TX(0,1);
        FA_COLOR3_IMPRELOAD_TX(0,2);
        FA_COLOR3_IMPRELOAD_TX(0,3);

        #pragma unroll
        for (int c = 0; c < numColors; ++c) {
            #pragma unroll
            for (int pp = 0; pp < pixelCache; pp += B_X/filtersPerThread) {
                shFilters[c][pp + shFilterLoadY][shFilterLoadX] = fPreload[c][pp*filtersPerThread/B_X];
            }
        }

        __syncthreads();
        FA_COLOR3_IMPRELOAD_TX(1,0);
        FA_COLOR3_IMPRELOAD_TX(1,1);
        FA_COLOR3_IMPRELOAD_TX(1,2);
        FA_COLOR3_IMPRELOAD_TX(1,3);
        FA_COLOR3_IMPRELOAD_TX(2,0);
        FA_COLOR3_IMPRELOAD_TX(2,1);
        FA_COLOR3_IMPRELOAD_TX(2,2);
        FA_COLOR3_IMPRELOAD_TX(2,3);
        #pragma unroll
        for (int c = 0; c < numColors; c++) {
            #pragma unroll
            for (int pp = 0; pp < pixelCache*filtersPerThread/B_X; pp++) {
                fPreload[c][pp] = fPidxNext + pp*(B_X/filtersPerThread) + shFilterLoadY >= filterPixels ? 0 : tex1Dfetch<float>(filters, filtersOffset2 + c * numFilters* filterPixels + pp*(B_X/filtersPerThread) * numFilters);
            }
        }
        #pragma unroll
        for (int pp = 0; pp < pixelCache; pp++) {
            #pragma unroll
            for (int c = 0; c < numColors; c++) {
                #pragma unroll
                for(int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for(int i = 0; i < imgsPerThread; i++) {
                        prod[i][f] += shImages[c][pp][tx * imgsPerThread + i] * shFilters[c][pp][ty * filtersPerThread + f];
                    }
                }
            }
        }

        __syncthreads();
    }

    if (scale) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleTargets * targets[i * B_X + f * numImages * numModules] + scaleOutputs * prod[i][f];
                }
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleOutputs * prod[i][f];
                }
            }
        }
    }
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * This won't be pretty.
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, int pixelCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex(cudaTextureObject_t images, cudaTextureObject_t filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride,
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv/*, const bool noloads*/) {
    __shared__ float shFilters[numColors][pixelCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[numColors][pixelCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
    fill_shared_mem<float>((float *)shFilters, sizeof(shFilters)/sizeof(float), 0);
    fill_shared_mem<float>((float *)shImages, sizeof(shImages)/sizeof(float), 0);
    __syncthreads();
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);

    const int numModules = numModulesX * numModulesY;
    // Another fun insanity: the % B_X makes things faster, even though threadIdx.x is
    // in the range 0..31. It appears that this allows the compiler to optimize?
    const int tx = threadIdx.x % B_X;
    const int ty = threadIdx.y % B_Y;
    const int tidx = ty * B_X + threadIdx.x;
    const int warp = tidx / 32;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

//    images += myImgIdx;
//    filters += blockFilterIdx
//            + shFilterLoadY * numFilters + shFilterLoadX;
//    if (!conv) { // NOTE: UNTESTED!
//        filters += moduleIdx * numColors * filterPixels * numFilters;
//    }

    const int imagesOffset = myImgIdx;
    const int filtersOffset = blockFilterIdx + shFilterLoadY * numFilters + shFilterLoadX
                            + (conv ? 0 : moduleIdx * numColors * filterPixels * numFilters);

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y * filtersPerThread) * numImages * numModules
            + myImgIdx;

    float prod[imgsPerThread][filtersPerThread];
    #pragma unroll
    for(int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for(int f = 0; f < filtersPerThread; f++) {
            prod[i][f] = 0;
        }
    }

    int iPidxNext;
    float imPreload[numColors][imgsPerThread];
    float fPreload[numColors][DIVUP(pixelCache*filtersPerThread,B_X)];

    if (warp < 3) {
        #pragma unroll
        for (int c = 0; c < numColors; ++c) {
            #pragma unroll
            for (int p = 0; p < pixelCache; p += 2) {
                if (p + shFilterLoadY < filterPixels) {
                    fPreload[c][p/2] = tex1Dfetch<float>(filters, filtersOffset + p * numFilters + c * numFilters * filterPixels);
                } else {
                    fPreload[c][p/2] = 0;
                }
            }
        }
    }

    filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(ty, imgLoadModPosY, imgLoadModPosX, imgSizeX, filterSize, iPidxNext);

    #pragma unroll
    for (int c = 0; c < numColors; ++c) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (iPidxNext >= 0 && (!checkImgBounds || myImgIdx + i * B_X < numImages)) {
                imPreload[c][i] = tex1Dfetch<float>(images, imagesOffset + (c * imgPixels + iPidxNext) * imgStride + i * B_X);
            } else {
                imPreload[c][i] =  0;
            }
        }
    }

    for (int p = 0; p < filterPixels; p += pixelCache) {
        const int fPidxNext = p + pixelCache >= filterPixels ? 0 : p + pixelCache;
        filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(fPidxNext + ty, imgLoadModPosY, imgLoadModPosX, imgSizeX, filterSize, iPidxNext);

        #pragma unroll
        for (int c = 0; c < numColors; ++c) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                // NOTE: bank conflicts here!
                shImages[c][ty][tx * imgsPerThread + i] = imPreload[c][i];
            }
        }

        if (warp < 3) {
            #pragma unroll
            for (int c = 0; c < numColors; ++c) {
                #pragma unroll
                for (int pp = 0; pp < pixelCache; pp += 2) {
                    shFilters[c][pp + shFilterLoadY][shFilterLoadX] = fPreload[c][pp/2];
                }
            }
        }

        __syncthreads();
//        const float* ff = &filters[numFilters * fPidxNext];
//        const float* mm = &images[imgStride * iPidxNext];
        const int filtersOffset2 = filtersOffset + numFilters * fPidxNext;
        const int imagesOffset2 = imagesOffset + imgStride * iPidxNext;

        #pragma unroll
        for (int i = 0; i < imgsPerThread; ++i) {
            #pragma unroll
            for (int c = 0; c < numColors; c++) {
                FA_COLOR3_IMPRELOAD_TX(c,i);
            }
        }

        #pragma unroll
        for (int c = 0; c < numColors; c++) {
            #pragma unroll
            for (int pp = 0; pp < 2; pp++) {
                fPreload[c][pp] = warp >= 3 || fPidxNext + pp*2 + shFilterLoadY >= filterPixels ? 0 : tex1Dfetch<float>(filters, filtersOffset2 +  c * numFilters* filterPixels + pp*2 * numFilters);
            }
            #pragma unroll
            for (int pp = 0; pp < pixelCache; pp++) {
                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[c][pp][tx * imgsPerThread + i] * shFilters[c][pp][ty * filtersPerThread + f];
                    }
                }
            }

        }
        __syncthreads();
    }

    if (scale) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleTargets * targets[i * B_X + f * numImages * numModules] + scaleOutputs * prod[i][f];
                }
            }
        }
    } else {
        // Note: reversing order of these loops costs 2 registers, but saves time
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleOutputs * prod[i][f];
                }
            }
        }
    }
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * Note: in git there's a 1.5% faster version of this which sues 167 registers instead of 154...
 * it's basically the same thing, but it doesn't do the next-pixel computation. It just avoids
 * pre-loading when it rolls over to the next pixel.
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups,
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv/*, const bool noloads*/) {
    __shared__ float shFilters[colorCache][B_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
    __shared__ float shImages[colorCache][B_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
    fill_shared_mem<float>((float *)shFilters, sizeof(shFilters)/sizeof(float), 0);
    fill_shared_mem<float>((float *)shImages, sizeof(shImages)/sizeof(float), 0);
    __syncthreads();
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;
    // Another fun insanity: the % B_X makes things faster, even thought threadIdx.x is
    // in the range 0..31. It appears that this allows the compiler to optimize?
    const int tx = threadIdx.x % B_X;
    const int ty = threadIdx.y % B_Y;
    const int tidx = ty * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y * filtersPerThread) * numImages * numModules
            + myImgIdx;

    float prod[imgsPerThread][filtersPerThread];
//    float fCache[filtersPerThread];
    #pragma unroll
    for(int i = 0; i < imgsPerThread; i++) {
        #pragma unroll
        for(int f = 0; f < filtersPerThread; f++) {
            prod[i][f] = 0;
        }
    }
    // NOTE: these max/min functions increase register usage as compared to my macros
    const int imgStartX = max(0, imgLoadModPosX);
    const int imgStartY = max(0, imgLoadModPosY);
    const int imgEndX = min(imgLoadModPosX + filterSize, imgSizeX);
    const int imgEndY = min(imgLoadModPosY + filterSize, imgSizeY);
//    __shared__ int imgPos[]

    int fPidx, iPidx;
    float imPreload[imgsPerThread];
    float fPreload[colorCache*filtersPerThread/B_X];
//    float fCache[filtersPerThread];

    filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgStartY, imgStartX, fPidx, iPidx);

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
            imPreload[i] = images[imgStride * iPidx + i * B_X];
        } else {
            imPreload[i] = 0;
        }
    }
    if (/*B_X % filtersPerThread == 0 ||*/ shFilterLoadY < B_X/filtersPerThread) { // This if statement reduces reg usage..
        #pragma unroll
        for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
            fPreload[c*filtersPerThread/B_X] = filters[(c * filterPixels + fPidx) * numFilters];
        }
    }
    for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
//        const int filterPxY = imgY - imgLoadModPosY;
        for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
//            const int filterPxX = imgX - imgLoadModPosX;
//            const int p = filterPxY * filterSize + filterPxX;
//            const int pixIdx = imgY * imgSizeX + imgX;// Pixel index in img
//            setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgY, imgX, &p, &pixIdx);
//            float* m = &images[imgStride * pixIdx];
            const bool lastPixel = imgY == imgEndY - 1 && imgX == imgEndX - 1;
            int imgYNext = imgY;
            int imgXNext = imgX;
            int fPidxNext, iPidxNext;
            if (!lastPixel) {
                imgYNext = imgY + (imgX + 1 == imgEndX);
                imgXNext = imgX + 1 == imgEndX ? imgStartX : imgX + 1;
            }
            filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgYNext, imgXNext, fPidxNext, iPidxNext);
            for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
                const float* ff = &filters[numFilters * ((oc + colorCache) * filterPixels + fPidx)];
                const float* mm = &images[imgStride * ((oc + colorCache) * imgPixels + iPidx)];
                if (oc == numFilterColors - colorCache) {
                    ff = &filters[fPidxNext * numFilters];
                    mm = &images[iPidxNext * imgStride];
                    fPidx = fPidxNext;
                    iPidx = iPidxNext;
                }

                #pragma unroll
                for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
                    shFilters[c + shFilterLoadY][shFilterLoadX] = fPreload[c*filtersPerThread/B_X];
                }

                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    // NOTE: bank conflicts here!
                    shImages[ty][tx * imgsPerThread + i] = imPreload[i];
                }
                imPreload[0] = (checkImgBounds && myImgIdx + 0 * B_X >= numImages) ? 0 : mm[0 * B_X];
                imPreload[1] = (checkImgBounds && myImgIdx + 1 * B_X >= numImages) ? 0 : mm[1 * B_X];
                imPreload[2] = (checkImgBounds && myImgIdx + 2 * B_X >= numImages) ? 0 : mm[2 * B_X];

                __syncthreads();

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[0][threadIdx.x * imgsPerThread + i] * shFilters[0][threadIdx.y * filtersPerThread + f];
                    }
                }

                fPreload[0] = ff[0];

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[1][threadIdx.x * imgsPerThread + i] * shFilters[1][threadIdx.y * filtersPerThread + f];
                    }
                }

                fPreload[1] = ff[(B_X/filtersPerThread * filterPixels) * numFilters];

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[2][threadIdx.x * imgsPerThread + i] * shFilters[2][threadIdx.y * filtersPerThread + f];
                    }
                }

                imPreload[3] = (checkImgBounds && myImgIdx + 3 * B_X >= numImages) ? 0 : mm[3 * B_X];

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[3][threadIdx.x * imgsPerThread + i] * shFilters[3][threadIdx.y * filtersPerThread + f];
                    }
                }
                __syncthreads();
            }
        }
    }

    if (scale) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleTargets * targets[i * B_X + f * numImages * numModules] + scaleOutputs * prod[i][f];
                }
            }
        }
    } else {
        // Note: reversing order of these loops saves 2 registers, but costs time
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                    targets[i * B_X + f * numImages * numModules] = scaleOutputs * prod[i][f];
                }
            }
        }
    }
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters)             if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModules, numImages)
 *
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128.
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast.
 */
 void _filterActs(cudaStream_t stream, NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput, bool conv) {
    int numFilterColors = numImgColors / numGroups;
    int numFilters = filters.getNumCols();
    int numModules = numModulesY * numModulesX;
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows()/numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    int filterModuleMult = conv ? 1 : numModules;

    megdnn_assert_internal(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 4 == 0)));
    megdnn_assert_internal(numGroups == 1 || numFilterColors % 4 == 0);
    //megdnn_assert_internal(numFilters % (16 * numGroups) == 0);
    megdnn_assert_internal(numImgColors % numGroups == 0);
    bool previous_limit = (numFilters % (16 * numGroups)) == 0;

    //images.printShape("images");
    //printf("rows: %d, pixels: %d, colors: %d\n", images.getNumRows(), imgPixels, numImgColors);
    //images.printShape("images");
    megdnn_assert_internal(images.getNumRows() == imgPixels * numImgColors);
    megdnn_assert_internal(imgSizeY * imgSizeX == imgPixels);
    int numFiltersPerGroup = numFilters / numGroups;

    int imgStride = images.getStride(); // images does not need to be a contiguous matrix

    int filterPixels = filters.getNumRows() / (filterModuleMult * numFilterColors);
    int filterSize = int(sqrt(filterPixels));
    megdnn_assert_internal(filterSize * filterSize == filterPixels);
    megdnn_assert_internal(filters.getNumRows() == filterModuleMult * numFilterColors * filterPixels);

    // These routines don't handle the case when only part of the image is visited in the convolution
    megdnn_assert_internal(paddingStart <= 0);
    megdnn_assert_internal(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    megdnn_assert_internal(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
    megdnn_assert_internal(moduleStride <= filterSize);

    megdnn_assert_internal(!images.isTrans());
    megdnn_assert_internal(!filters.isTrans());
    megdnn_assert_internal(!targets.isTrans());

    megdnn_assert_internal(filters.isContiguous());
    megdnn_assert_internal(targets.isContiguous());
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    int filtersPerThread, threadsY = 4;
    if (numImgColors <= 3) {
        // Special kernels written for colors = 3, filters = 64 and colors = 3, filters = 48 cases.
        // The remaining cases use the old routines.
        // TODO: Modernize the remaining cases if you care about them.
        filtersPerThread = numFiltersPerGroup % 64 == 0 ? 16 : numFiltersPerGroup % 48 == 0 ? 12 : numFiltersPerGroup % 32 == 0 ? 8 : 4;
    } else {
        filtersPerThread = numFiltersPerGroup % 64 == 0 ? 16 : numFiltersPerGroup % 32 == 0 ? 8 : 4;
        threadsY = numFiltersPerGroup % 128 == 0 && numFilterColors % 8 == 0  && imgsPerThread != 4 ?  8 : 4;
    }
    int threadsX = 32;
    dim3 threads(threadsX, threadsY);
    dim3 blocks = dim3(DIVUP(numImages, threads.x * imgsPerThread), numModules * DIVUP(numFilters, (threads.y * filtersPerThread)));

    bool checkImgBounds = numImages % (threads.x*imgsPerThread) != 0;
    bool scale = scaleTargets != 0;
    if (scaleTargets == 0) {
        targets.resize(numFilters * numModules, numImages);
    } else {
        megdnn_assert_internal(targets.getNumRows() == numFilters * numModules);
        megdnn_assert_internal(targets.getNumCols() == numImages);
    }

    // Auto-generated calling code...
    // NOTE: The calling code is set up such that if checkImgBounds is true, then imgsPerThread = 1.
    // In principle it doesn't have to be this way, and you may want to optimize for that case.

    if (scale == false) {
        if (checkImgBounds == false) {
            if (numFilterColors % 8 == 0) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        if (previous_limit) {
                            if (images.getNumDataBytes() < TEXTURE_SIZE_MAX) {
                                cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
                                filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                            } else {
                                cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
                                filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                            }
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        if (previous_limit) {
                            if (images.getNumDataBytes() < TEXTURE_SIZE_MAX) {
                                cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
                                filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                            } else {
                                cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
                                filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                            }
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 8, 8, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 2, 16, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 8, 32, 2, 16, 8, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 16, 8, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 8, 8, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors % 4 == 0) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 3) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        if (previous_limit) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getTextureObject(), filters.getTextureObject(), targets.getDevData(),numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 3, 4, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 16, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        if (previous_limit) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getTextureObject(), filters.getTextureObject(), targets.getDevData(),numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 3, 4, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 12, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 2) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 16, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 12, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 1) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 16, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 12, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
        else if (checkImgBounds == true) {
            if (numFilterColors % 8 == 0) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors % 4 == 0) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 3) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 2) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 1) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
    }
    else if (scale == true) {
        if (checkImgBounds == false) {
            if (numFilterColors % 8 == 0) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        if (previous_limit) {
                            if (images.getNumDataBytes() < TEXTURE_SIZE_MAX) {
                                cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
                                filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                            } else {
                                cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
                                filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                            }
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        if (previous_limit) {
                            if (images.getNumDataBytes() < TEXTURE_SIZE_MAX) {
                                cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
                                filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                            } else {
                                cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
                                filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                            }
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 8, 8, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 2, 16, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 8, 32, 2, 16, 8, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 16, 8, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 8, 8, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors % 4 == 0) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 3) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        if (previous_limit) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getTextureObject(), filters.getTextureObject(), targets.getDevData(),numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 3, 4, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 16, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        if (previous_limit) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getTextureObject(), filters.getTextureObject(), targets.getDevData(),numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 3, 4, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 12, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 2) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 16, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 12, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 1) {
                if (numImages % 128 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 16, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 12, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 8, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 4, 4, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 64 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 16, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 12, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 8, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 2, 4, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
                else if (numImages % 32 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, false > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
        else if (checkImgBounds == true) {
            if (numFilterColors % 8 == 0) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors % 4 == 0) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 128 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 3) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 2) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
            else if (numFilterColors == 1) {
                if (numImages % 1 == 0) {
                    if (numFiltersPerGroup % 64 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 48 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                    else if (numFiltersPerGroup % 1 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, true > <<<blocks, threads, 0, stream>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
    }

    getLastCudaError("filterActs: kernel execution failed");
}

void convFilterActs(cudaStream_t stream, NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numGroups) {
    convFilterActs(stream, images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, 0, 1);
}

void convFilterActs(cudaStream_t stream, NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput) {
     _filterActs(stream, images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, true);
}

void localFilterActs(cudaStream_t stream, NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numGroups) {
    localFilterActs(stream, images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, 0, 1);
}

void localFilterActs(cudaStream_t stream, NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput) {
     _filterActs(stream, images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, false);
}

} // namespace cuda
} // namespace megdnn
