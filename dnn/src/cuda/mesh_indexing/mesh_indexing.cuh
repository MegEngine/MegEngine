/**
 * \file dnn/src/cuda/mesh_indexing/mesh_indexing.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cstdio>
#include "megdnn/basic_types.h"
#include "src/cuda/error_info.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace mesh_indexing {

// template <int ndim>
struct KernIndexer {
    int ndim;
    int* ptrs[TensorShape::MAX_NDIM];
    int origin_stride[TensorShape::MAX_NDIM];
    int indexed_strde[TensorShape::MAX_NDIM];
    int desc_stride[TensorShape::MAX_NDIM];
    uint32_t indexed_shape[TensorShape::MAX_NDIM];
    uint32_t origin_shape[TensorShape::MAX_NDIM];

    void* error_tracker;
    megcore::AsyncErrorInfo* error_info;
    bool batch_mode;
    uint32_t batch_stride;
    uint32_t size;

    KernIndexer(const TensorLayout& origin_layout,
                const TensorLayout& indexed_layout, int** _ptrs,
                const TensorLayout* desc_layouts,
                void* _err_tracker = nullptr,
                megcore::AsyncErrorInfo* _err_info = nullptr,
                bool _batch_mode = false)
            : error_tracker(_err_tracker),
              error_info(_err_info),
              batch_mode(_batch_mode),
              size(indexed_layout.total_nr_elems()) {
        ndim = origin_layout.ndim;
        for (int i = 0; i < ndim; ++i) {
            origin_stride[i] = origin_layout.stride[i];
            indexed_strde[i] = indexed_layout.stride[i];
            origin_shape[i] = origin_layout[i];
            indexed_shape[i] = indexed_layout[i];
            ptrs[i] = _ptrs[i];
            desc_stride[i] = desc_layouts[i].stride[0];
        }
    }

    int __device__ __forceinline__ convert_indxer(uint32_t& index) const {
        int data_offset = 0;
        int value_offset = 0;
        uint32_t n = 0;
        if (batch_mode) {
            n = index;
            for (int i = ndim - 1; i >= 1; --i) {
                n /= indexed_shape[i];
            }
            n %= indexed_shape[0];
        }
        for (int i = ndim - 1; i >= 0; --i) {
            int pos = index % indexed_shape[i];
            value_offset += pos * indexed_strde[i];
            if (ptrs[i]) {
                pos += n * desc_stride[i];
                pos = ptrs[i][pos];
                pos += (pos < 0 ? origin_shape[i] : 0);
            }
            if (static_cast<uint32_t>(pos) >= origin_shape[i]) {
                set_async_error_info(error_info, error_tracker,
                                     "invalid mesh indexing: "
                                     "indexer=%d idx=%d shape=%d",
                                     i, pos, origin_shape[i]);
            }
            data_offset += pos * origin_stride[i];
            index /= indexed_shape[i];
        }

        index = value_offset;
        return data_offset;
    }
};

template <typename T, class Opr>
void mesh_indexing_proxy(T* origin, T* indexed, KernIndexer* indexer,
                         cudaStream_t stream);
}  // namespace mesh_indexing
}  // namespace cuda
}  // namespace megdnn
