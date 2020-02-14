/**
 * \file dnn/src/cuda/local/cuda-convnet2/filter_acts/filter_act_sparse2_y4x32i4f16c4_tex.cu
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

template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex (FILTER_ACTS_PARAMS) {
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
    const int imgOffset = (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;

//    images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
    const int filterOffset = blockFilterIdx
            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX + (conv ? 0 : moduleIdx * numFilterColors * filterPixels * numFilters);
//    filters +=blockFilterIdx
//            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX;
//    if (!conv) {
//        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
//    }

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
    float imPreload[imgsPerThread]; // [4]
    float fPreload[colorCache*filtersPerThread/B_X]; // [2]
//    float fCache[filtersPerThread];

    filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgStartY, imgStartX, fPidx, iPidx);

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
            imPreload[i] = tex1Dfetch<float>(images, imgOffset + imgStride * iPidx + i * B_X);
        } else {
            imPreload[i] = 0;
        }
    }
    if (/*B_X % filtersPerThread == 0 ||*/ shFilterLoadY < B_X/filtersPerThread) { // This if statement reduces reg usage..
        #pragma unroll
        for (int c = 0; c < colorCache; c += B_X/filtersPerThread) {
            fPreload[c*filtersPerThread/B_X] = tex1Dfetch<float>(filters, filterOffset + (c * filterPixels + fPidx) * numFilters);
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
//                const float* ff = &filters[numFilters * ((oc + colorCache) * filterPixels + fPidx)];
//                const float* mm = &images[imgStride * ((oc + colorCache) * imgPixels + iPidx)];
                int imgOffset2 = imgOffset + imgStride * ((oc + colorCache) * imgPixels + iPidx);
                int filterOffset2 = filterOffset + numFilters * ((oc + colorCache) * filterPixels + fPidx);
                if (oc == numFilterColors - colorCache) {
                    filterOffset2 = filterOffset + fPidxNext * numFilters;
                    imgOffset2 = imgOffset + iPidxNext * imgStride;
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
                imPreload[0] = (checkImgBounds && myImgIdx + 0 * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imgOffset2 + 0 * B_X);
                imPreload[1] = (checkImgBounds && myImgIdx + 1 * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imgOffset2 + 1 * B_X);
                imPreload[2] = (checkImgBounds && myImgIdx + 2 * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imgOffset2 + 2 * B_X);

                __syncthreads();

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[0][threadIdx.x * imgsPerThread + i] * shFilters[0][threadIdx.y * filtersPerThread + f];
                    }
                }

                fPreload[0] = tex1Dfetch<float>(filters, filterOffset2 + 0);

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[1][threadIdx.x * imgsPerThread + i] * shFilters[1][threadIdx.y * filtersPerThread + f];
                    }
                }

                fPreload[1] = tex1Dfetch<float>(filters, filterOffset2 + (B_X/filtersPerThread * filterPixels) * numFilters);

                #pragma unroll
                for(int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for(int f = 0; f < filtersPerThread; f++) {
                        prod[i][f] += shImages[2][threadIdx.x * imgsPerThread + i] * shFilters[2][threadIdx.y * filtersPerThread + f];
                    }
                }

                imPreload[3] = (checkImgBounds && myImgIdx + 3 * B_X >= numImages) ? 0 : tex1Dfetch<float>(images, imgOffset2 + 3 * B_X);

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

template __global__ void
filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex
< 4, 32, 4, 16, 4, false, false >(FILTER_ACTS_PARAMS);

template __global__ void
filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex
< 4, 32, 4, 16, 4, true, false >(FILTER_ACTS_PARAMS);

} // namespace cuda
} // namespace megdnn
