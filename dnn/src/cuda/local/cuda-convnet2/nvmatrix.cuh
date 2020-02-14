/**
 * \file dnn/src/cuda/local/cuda-convnet2/nvmatrix.cuh
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

#pragma once
#include "src/cuda/utils.cuh"
#include <cublas_v2.h>

namespace megdnn {
namespace cuda {

const int TEXTURE_SIZE_MAX = 1<<29;

struct MemorySegment {
    float *data;
    MemorySegment(float *data): data(data)
    {}
};

struct NVMatrix {
    NVMatrix(MemorySegment *seg, int row, int col):
        seg(seg), row(row), col(col), stride(col), _texObj(0)
    {
    }
    NVMatrix(MemorySegment *seg, int row, int col, int stride):
        seg(seg), row(row), col(col), stride(stride), _texObj(0)
    {
    }
    float *getDevData()
    {
        return seg->data;
    }
    MemorySegment *seg;
    int row, col, stride;
    cudaTextureObject_t _texObj;
    // target must be initialized before transpose.
    void transpose(const NVMatrix &target, cublasHandle_t handle,
            float *one, float *zero)
    {
        cublas_check(cublasSgeam(handle,
                CUBLAS_OP_T, CUBLAS_OP_T,
                row, col,
                one,
                seg->data, this->stride,
                zero,
                seg->data, this->stride,
                target.seg->data, target.stride));
    }
    cudaTextureObject_t getTextureObject() {
        if (_texObj == 0) {
            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeLinear;
            resDesc.res.linear.devPtr = getDevData();
            resDesc.res.linear.sizeInBytes = getNumDataBytes();
            resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0,
                    cudaChannelFormatKindFloat);
            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            cuda_check(cudaCreateTextureObject(&_texObj, &resDesc, &texDesc, NULL));
        }
        megdnn_assert_internal(_texObj != 0);
        return _texObj;
    }
    ~NVMatrix()
    {
        if (_texObj) {
            cuda_check(cudaDestroyTextureObject(_texObj));
        }
    }
    int getNumDataBytes()
    {
        return row * col * sizeof(float);
    }
    int getNumRows()
    {
        return row;
    }
    int getNumCols()
    {
        return col;
    }
    int getStride()
    {
        return stride;
    }
    bool isTrans()
    {
        return false;
    }
    bool isContiguous()
    {
        return true;
    }
    void resize(int row, int col)
    {
        megdnn_assert_internal(row * col == this->row * this->col);
        this->row = row;
        this->col = col;
    }
};

}
}
