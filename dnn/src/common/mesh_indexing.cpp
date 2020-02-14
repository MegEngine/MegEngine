/**
 * \file dnn/src/common/mesh_indexing.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/general.h"
#include "src/common/utils.h"

namespace megdnn {

/* ============================== MeshIndexing ============================= */

void MeshBase::check_exec(const TensorLayout& origin,
                          const TensorLayout& indexed, const IndexDesc& desc) {
    megdnn_assert(origin.dtype == indexed.dtype);
    megdnn_assert(origin.ndim == indexed.ndim);
    for (auto&& index : desc) {
        megdnn_assert(index.vec.layout.dtype == dtype::Int32());
    }
}

void NormalMeshBase::check_exec(const TensorLayout& src,
                                const TensorLayout& dst,
                                const IndexDesc& desc) {
    MeshBase::check_exec(src, dst, desc);
    for (auto&& index : desc) {
        size_t ndim = index.vec.layout.ndim;
        megdnn_assert(ndim == 1, "index must be 1-dim vector, while dim %zu",
                      ndim);
        megdnn_assert(dst.shape[index.axis] == index.vec.layout[0]);
    }
}

void BatchedMeshBase::check_exec(const TensorLayout& src,
                                 const TensorLayout& dst,
                                 const IndexDesc& desc) {
    MeshBase::check_exec(src, dst, desc);
    megdnn_assert(src[0] == dst[0], "batch mismatch, src %zu, dst %zu", src[0],
                  dst[0]);
    for (auto&& index : desc) {
        size_t ndim = index.vec.layout.ndim;
        megdnn_assert(ndim == 2, "index must be a 2-dim matrix, while ndim %zu",
                      ndim);
        megdnn_assert(dst[0] == index.vec.layout[0] &&
                              dst[index.axis] == index.vec.layout[1],
                      "require each index shape equals (%zu, %zu), but got "
                      "(%zu, %zu)",
                      dst[0], dst[index.axis], index.vec.layout[0],
                      index.vec.layout[1]);
        megdnn_assert(index.axis != 0,
                      "index axis should be 0-th dim when executing "
                      "BatchedMeshIndexing");
    }
}

void MeshIndexing::deduce_layout(const TensorLayout& inp,
                                 const IndexDescLayoutOnly& layouts,
                                 TensorLayout& out_layout) {
    out_layout = inp;
    for (auto&& index : layouts) {
        megdnn_assert(index.layout.ndim == 1, 
                "mesh indexing require index being 1-dim vector");
        out_layout[index.axis] = index.layout[0];
    }
    out_layout.init_contiguous_stride();
}

void BatchedMeshIndexing::deduce_layout(const TensorLayout& inp,
                                        const IndexDescLayoutOnly& layouts,
                                        TensorLayout& out_layout) {
    out_layout = inp;
    for (auto&& index : layouts) {
        megdnn_assert(index.layout.ndim == 2, 
                "batch mesh indexing require index being 2-dim matrix");
        out_layout[index.axis] = index.layout[1];
    }
    out_layout.init_contiguous_stride();
}

}  // namespace megdnn
