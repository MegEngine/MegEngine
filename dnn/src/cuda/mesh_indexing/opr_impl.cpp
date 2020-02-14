/**
 * \file dnn/src/cuda/mesh_indexing/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "opr_impl.h"
#include "mesh_indexing.cuh"
#include "src/common/indexing_multi_axis_vec_kdef.h"
#include "src/cuda/indexing_multi_axis_vec/kern.cuh"
#include "src/cuda/utils.h"

namespace {
using namespace megdnn;
using namespace cuda;
using namespace mesh_indexing;
KernIndexer get_indexer(const TensorND& origin, const TensorND& indexed,
                        const MeshBase::IndexDesc& desc, void* error_tracker,
                        megcore::AsyncErrorInfo* error_info, bool batched) {
    int* tmp_ptrs[TensorShape::MAX_NDIM] = {nullptr};
    TensorLayout desc_layouts[TensorShape::MAX_NDIM];
    for (size_t i = 0; i < desc.size(); ++i) {
        auto axis = desc[i].axis;
        megdnn_assert(axis < TensorShape::MAX_NDIM);
        tmp_ptrs[axis] = desc[i].vec.ptr<int>();
        desc_layouts[axis] = desc[i].vec.layout;
    }
    return {origin.layout, indexed.layout, tmp_ptrs, desc_layouts,
            error_tracker, error_info,     batched};
}

template <typename ctype, class Opr, bool batched>
void do_exec(const TensorND& data, const TensorND& value,
             const MeshBase::IndexDesc& desc, Handle* handle,
             void* error_tracker) {
    auto error_info = async_error_info(handle);
    auto indexer =
            get_indexer(data, value, desc, error_tracker, error_info, batched);

    auto stream = cuda_stream(handle);
    mesh_indexing::mesh_indexing_proxy<ctype, Opr>(
            data.ptr<ctype>(), value.ptr<ctype>(), &indexer, stream);
}

}  // namespace

namespace megdnn {
namespace cuda {

/* =========================== MeshIndexing ============================ */

void MeshIndexingImpl::exec(_megdnn_tensor_in src, const IndexDesc& desc,
                            _megdnn_tensor_out dst, _megdnn_workspace) {
    check_exec(src.layout, dst.layout, desc);
#define cb(DType)                                                    \
    if (dst.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {      \
        using ctype = typename DTypeTrait<DType>::ctype;             \
        do_exec<ctype, indexing_multi_axis_vec_kdef::OprFwd, false>( \
                src, dst, desc, handle(), m_error_tracker);          \
        return;                                                      \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

/* ========================= BatchedMeshIndexing ========================== */

void BatchedMeshIndexingImpl::exec(_megdnn_tensor_in src, const IndexDesc& desc,
                                   _megdnn_tensor_out dst, _megdnn_workspace) {
    check_exec(src.layout, dst.layout, desc);

#define cb(DType)                                                   \
    if (dst.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {     \
        using ctype = typename DTypeTrait<DType>::ctype;            \
        do_exec<ctype, indexing_multi_axis_vec_kdef::OprFwd, true>( \
                src, dst, desc, handle(), m_error_tracker);         \
        return;                                                     \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

/* ============================ Mesh ============================= */

void IncrMeshIndexingImpl::exec(_megdnn_tensor_inout data,
                                _megdnn_tensor_in value, const IndexDesc& desc,
                                _megdnn_workspace) {
    check_exec(data.layout, value.layout, desc);

#define cb(DType)                                                      \
    if (data.layout.dtype == DType()) {                                \
        using ctype = typename DTypeTrait<DType>::ctype;               \
        do_exec<ctype, indexing_multi_axis_vec::OprAtomicIncr, false>( \
                data, value, desc, handle(), m_error_tracker);         \
        return;                                                        \
    }

    cb(dtype::Float32);
    cb(dtype::Int32);
#undef cb
    megdnn_assert_internal(0);
}

void SetMeshIndexingImpl::exec(_megdnn_tensor_inout data,
                               _megdnn_tensor_in value, const IndexDesc& desc,
                               _megdnn_workspace) {
    check_exec(data.layout, value.layout, desc);

#define cb(DType)                                                    \
    if (data.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {     \
        using ctype = typename DTypeTrait<DType>::ctype;             \
        do_exec<ctype, indexing_multi_axis_vec_kdef::OprSet, false>( \
                data, value, desc, handle(), m_error_tracker);       \
        return;                                                      \
    }

    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

/* ========================== BatchedMesh ============================= */
void BatchedIncrMeshIndexingImpl::exec(_megdnn_tensor_inout data,
                                       _megdnn_tensor_in value,
                                       const IndexDesc& desc,
                                       _megdnn_workspace) {
    check_exec(data.layout, value.layout, desc);

#define cb(DType)                                                     \
    if (data.layout.dtype == DType()) {                               \
        using ctype = typename DTypeTrait<DType>::ctype;              \
        do_exec<ctype, indexing_multi_axis_vec::OprAtomicIncr, true>( \
                data, value, desc, handle(), m_error_tracker);        \
        return;                                                       \
    }
    cb(dtype::Float32);
    cb(dtype::Int32);
#undef cb
    megdnn_assert_internal(0);
}

void BatchedSetMeshIndexingImpl::exec(_megdnn_tensor_inout data,
                                      _megdnn_tensor_in value,
                                      const IndexDesc& desc,
                                      _megdnn_workspace) {
    check_exec(data.layout, value.layout, desc);

#define cb(DType)                                                   \
    if (data.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {    \
        using ctype = typename DTypeTrait<DType>::ctype;            \
        do_exec<ctype, indexing_multi_axis_vec_kdef::OprSet, true>( \
                data, value, desc, handle(), m_error_tracker);      \
        return;                                                     \
    }

    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

}  // namespace cuda
}  // namespace megdnn
