/**
 * \file dnn/src/naive/mesh_indexing/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <numeric>

#include "megdnn/tensor_iter.h"
#include "src/naive/handle.h"
#include "src/naive/mesh_indexing/opr_impl.h"

namespace {
using namespace megdnn;
size_t get_index(const TensorND& t0, const TensorND& t1,
                 const MeshIndexing::IndexDesc& desc, int* index) {
    int* index_ptr[TensorShape::MAX_NDIM] = {nullptr};
    for (auto&& i : desc) {
        size_t stride = (i.vec.layout.ndim == 1 ? 0 : i.vec.layout.stride[0]);
        index_ptr[i.axis] = i.vec.ptr<int>() + index[0] * stride;
    }
    size_t ret = 0;
    for (size_t i = 0; i < t1.layout.ndim; ++i) {
        int& pos = index[i];
        if (index_ptr[i]) {
            pos = index_ptr[i][pos];
        }
        if (pos < 0) {
            pos += t0.layout.shape[i];
        }
        ret += pos * t0.layout.stride[i];
    }
    return ret;
}
}  // namespace

namespace megdnn {
namespace naive {

/* =========================== MeshIndexing ============================ */

template <typename T>
void MeshIndexingImpl::exec_mesh_indexing(const TensorND& src_tensor,
                                          const IndexDesc& desc,
                                          const TensorND& dst_tensor) {
    // normal mesh indexing.
    auto iter = tensor_iter<T>(dst_tensor).begin();
    size_t ndim = dst_tensor.layout.ndim;
    auto ptr = src_tensor.ptr<T>();
    for (size_t dst_idx = 0; dst_idx < dst_tensor.layout.total_nr_elems();
         ++dst_idx) {
        int index[TensorShape::MAX_NDIM];
        std::copy(iter.idx(), iter.idx() + ndim, index);
        size_t src_idx = get_index(src_tensor, dst_tensor, desc, index);
        *iter = ptr[src_idx];
        ++iter;
    }
}

void MeshIndexingImpl::exec(_megdnn_tensor_in src, const IndexDesc& desc,
                            _megdnn_tensor_out dst,
                            _megdnn_workspace workspace) {
    MEGDNN_MARK_USED_VAR(workspace);
    check_exec(src.layout, dst.layout, desc);
#define cb(DType)                                               \
    if (dst.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype;        \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                           \
                exec_mesh_indexing<ctype>(src, desc, dst));     \
        return;                                                 \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

/* ========================= BatchedMeshIndexing =========================== */

template <typename T>
void BatchedMeshIndexingImpl::do_exec(const TensorND& src_tensor,
                                      const IndexDesc& desc,
                                      const TensorND& dst_tensor) {
    auto iter = tensor_iter<T>(dst_tensor).begin();
    size_t ndim = dst_tensor.layout.ndim;
    auto ptr = src_tensor.ptr<T>();
    for (size_t dst_idx = 0; dst_idx < dst_tensor.layout.total_nr_elems();
         ++dst_idx) {
        int index[TensorShape::MAX_NDIM];
        std::copy(iter.idx(), iter.idx() + ndim, index);
        size_t src_idx = get_index(src_tensor, dst_tensor, desc, index);
        *iter = ptr[src_idx];
        ++iter;
    }
}

void BatchedMeshIndexingImpl::exec(_megdnn_tensor_in src, const IndexDesc& desc,
                                   _megdnn_tensor_out dst, _megdnn_workspace) {
    check_exec(src.layout, dst.layout, desc);

#define cb(DType)                                                     \
    if (dst.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                                \
        using ctype = typename DTypeTrait<DType>::ctype;              \
        MEGDNN_DISPATCH_CPU_KERN_OPR(do_exec<ctype>(src, desc, dst)); \
        return;                                                       \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

/* ============================ Mesh ============================= */

template <typename T>
void IncrMeshIndexingImpl::do_exec(const TensorND& data, const TensorND& value,
                                   const IndexDesc& desc) {
    auto iter = tensor_iter<T>(value).begin();
    size_t ndim = value.layout.ndim;
    auto ptr = data.ptr<T>();
    for (size_t idx = 0; idx < value.layout.total_nr_elems(); ++idx) {
        int index[TensorShape::MAX_NDIM];
        std::copy(iter.idx(), iter.idx() + ndim, index);
        size_t data_idx = get_index(data, value, desc, index);
        ptr[data_idx] += *iter;
        ++iter;
    }
}

void IncrMeshIndexingImpl::exec(_megdnn_tensor_inout data,
                                _megdnn_tensor_in value, const IndexDesc& desc,
                                _megdnn_workspace workspace) {
    MEGDNN_MARK_USED_VAR(workspace);
    check_exec(data.layout, value.layout, desc);
#define cb(DType)                                                        \
    if (data.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                                  \
        using ctype = typename DTypeTrait<DType>::ctype;                 \
        MEGDNN_DISPATCH_CPU_KERN_OPR(do_exec<ctype>(data, value, desc)); \
        return;                                                          \
    }

    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

template <typename T>
void SetMeshIndexingImpl::do_exec(const TensorND& data, const TensorND& value,
                                  const IndexDesc& desc) {
    auto iter = tensor_iter<T>(value).begin();
    size_t ndim = value.layout.ndim;
    auto ptr = data.ptr<T>();
    for (size_t idx = 0; idx < value.layout.total_nr_elems(); ++idx) {
        int index[TensorShape::MAX_NDIM];
        std::copy(iter.idx(), iter.idx() + ndim, index);
        size_t data_idx = get_index(data, value, desc, index);
        ptr[data_idx] = *iter;
        ++iter;
    }
}

void SetMeshIndexingImpl::exec(_megdnn_tensor_inout data,
                               _megdnn_tensor_in value, const IndexDesc& desc,
                               _megdnn_workspace workspace) {
    MEGDNN_MARK_USED_VAR(workspace);
    check_exec(data.layout, value.layout, desc);
#define cb(DType)                                                        \
    if (data.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {         \
        using ctype = typename DTypeTrait<DType>::ctype;                 \
        MEGDNN_DISPATCH_CPU_KERN_OPR(do_exec<ctype>(data, value, desc)); \
        return;                                                          \
    }

    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

/* =========================== BatchedMesh =========================== */

template <typename T>
void BatchedIncrMeshIndexingImpl::do_exec(const TensorND& data,
                                          const TensorND& value,
                                          const IndexDesc& desc) {
    auto iter = tensor_iter<T>(value).begin();
    size_t ndim = value.layout.ndim;
    auto ptr = data.ptr<T>();
    for (size_t idx = 0; idx < value.layout.total_nr_elems(); ++idx) {
        int index[TensorShape::MAX_NDIM];
        std::copy(iter.idx(), iter.idx() + ndim, index);
        size_t data_idx = get_index(data, value, desc, index);
        ptr[data_idx] += *iter;
        ++iter;
    }
}

void BatchedIncrMeshIndexingImpl::exec(_megdnn_tensor_inout data,
                                       _megdnn_tensor_in value,
                                       const IndexDesc& desc,
                                       _megdnn_workspace workspace) {
    MEGDNN_MARK_USED_VAR(workspace);
    check_exec(data.layout, value.layout, desc);
#define cb(DType)                                                        \
    if (data.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {         \
        using ctype = typename DTypeTrait<DType>::ctype;                 \
        MEGDNN_DISPATCH_CPU_KERN_OPR(do_exec<ctype>(data, value, desc)); \
        return;                                                          \
    }

    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

template <typename T>
void BatchedSetMeshIndexingImpl::do_exec(const TensorND& data,
                                         const TensorND& value,
                                         const IndexDesc& desc) {
    auto iter = tensor_iter<T>(value).begin();
    size_t ndim = value.layout.ndim;
    auto ptr = data.ptr<T>();
    for (size_t idx = 0; idx < value.layout.total_nr_elems(); ++idx) {
        int index[TensorShape::MAX_NDIM];
        std::copy(iter.idx(), iter.idx() + ndim, index);
        size_t data_idx = get_index(data, value, desc, index);
        ptr[data_idx] = *iter;
        ++iter;
    }
}

void BatchedSetMeshIndexingImpl::exec(_megdnn_tensor_inout data,
                                      _megdnn_tensor_in value,
                                      const IndexDesc& desc,
                                      _megdnn_workspace workspace) {
    MEGDNN_MARK_USED_VAR(workspace);
    check_exec(data.layout, value.layout, desc);
#define cb(DType)                                                        \
    if (data.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {         \
        using ctype = typename DTypeTrait<DType>::ctype;                 \
        MEGDNN_DISPATCH_CPU_KERN_OPR(do_exec<ctype>(data, value, desc)); \
        return;                                                          \
    }

    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

}  // namespace naive
}  // namespace megdnn
