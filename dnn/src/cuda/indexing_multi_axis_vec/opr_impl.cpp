/**
 * \file dnn/src/cuda/indexing_multi_axis_vec/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "./kern.cuh"

#include "src/cuda/utils.h"
#include "src/common/indexing_multi_axis_vec_kdef.h"

using namespace megdnn;
using namespace cuda;
using namespace indexing_multi_axis_vec;

namespace {
    class ExecImplHelper {
        template<int nidx>
        void dispatch_gen_offset_base_nidx();

        void dispatch_gen_offset_base();
    protected:
        using IndexDesc = IndexingMultiAxisVec::IndexDesc;
        using ExecInfo = IndexingMultiAxisVec::ExecInfo;

        cudaStream_t m_stream;
        const TensorND * const m_data;
        const TensorND * const m_value;
        const IndexDesc * const m_index;
        const ExecInfo* const m_exec_info;
        int * const m_offset_base;
        TensorLayout m_value_layout_on_data;
        size_t m_idx_axis;
        int m_value_stride;

    public:
        ExecImplHelper(const TensorND &data, const TensorND &value,
                const IndexDesc &index, const Workspace &workspace,
                const ExecInfo &exec_info, cudaStream_t stream);
    };

    template<class Opr>
    class ExecImpl : public ExecImplHelper {

        void dispatch_exec();

        template<typename ctype>
        void dispatch_exec_ctype();

        template<typename ctype, int ndim>
        void dispatch_exec_ctype_ndim();

    public:
        using ExecImplHelper::ExecImplHelper;

        void operator() () {
            dispatch_exec();
            after_kernel_launch();
        }
    };
} // anonymous namespace

ExecImplHelper::ExecImplHelper(const TensorND &data, const TensorND &value,
        const IndexDesc &index, const Workspace &workspace,
        const ExecInfo &exec_info, cudaStream_t stream):
    m_stream{stream}, m_data{&data}, m_value{&value}, m_index{&index},
    m_exec_info{&exec_info}, m_offset_base{workspace.ptr<int>()}
{
    safe_size_in_kern(data.layout.total_nr_elems());
    dispatch_gen_offset_base();

    std::tie(m_value_layout_on_data, m_idx_axis) =
        IndexingMultiAxisVec::get_value_iter_optimized_layout(
            data.layout, value.layout, index, exec_info.idx_axis);
    m_value_stride = exec_info.value_stride;
}

template<int nidx>
void ExecImplHelper::dispatch_gen_offset_base_nidx() {

    GenOffsetBaseParam<nidx> param;
    param.size = m_value->layout.shape[m_exec_info->idx_axis];
    param.output = m_offset_base;
    param.error_tracker = m_exec_info->error_tracker;
    param.error_info = m_exec_info->error_info;
    for (int i = 0; i < nidx; ++ i) {
        auto &&dst = param.indexer[i];
        auto &&src = m_index->operator[](i);
        megdnn_assert(src.vec.layout.ndim == 1);
        dst.stride = src.vec.layout.stride[0];
        if (src.vec.layout.shape[0] == 1) {
            dst.stride = 0;
        }
        dst.ptr = src.vec.ptr<int>();
        param.data_shape[i] = m_data->layout.shape[src.axis];
        param.data_stride[i] = m_data->layout.stride[src.axis];
    }
    gen_offset_base(param, m_stream);
}

void ExecImplHelper::dispatch_gen_offset_base() {
    switch(m_index->size()) {
#define cb(_n) case _n:  return dispatch_gen_offset_base_nidx<_n>();
        MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
    }
    megdnn_throw("bad index size");
}

template<class Opr>
void ExecImpl<Opr>::dispatch_exec() {
    switch (m_data->layout.dtype.enumv()) {
#define cb(_dtype) \
        case DTypeTrait<_dtype>::enumv: \
            return dispatch_exec_ctype<DTypeTrait<_dtype>::ctype>();
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

template<class Opr>
template<typename ctype>
void ExecImpl<Opr>::dispatch_exec_ctype() {
    switch (m_value_layout_on_data.ndim) {
#define cb(_n) \
        case _n: return dispatch_exec_ctype_ndim<ctype, _n>();
        MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
        default:
            megdnn_throw("bad data ndim");
    }
}

template<class Opr>
template<typename ctype, int ndim>
void ExecImpl<Opr>::dispatch_exec_ctype_ndim() {
    ApplyOprParam<ctype, ndim> param;
    param.tot_size = safe_size_in_kern(m_value->layout.total_nr_elems());
    param.offset_base = m_offset_base;
    param.data = m_data->ptr<ctype>();
    param.value = m_value->ptr<ctype>();
    param.idx_axis = m_idx_axis;
    param.value_stride = m_value_stride;
    for (int i = 0; i < ndim; ++ i) {
        param.value_ly_on_data.stride[i] = m_value_layout_on_data.stride[i];
        if (i) {
            param.value_ly_on_data.shape[i - 1] =
                m_value_layout_on_data.shape[i];
        }
    }
    apply_opr<ctype, ndim, Opr>(param, m_stream);
}


size_t IndexingMultiAxisVecImpl::get_workspace_in_bytes(size_t dst_idx_size) {
    return dst_idx_size * sizeof(int);
}

void IndexingMultiAxisVecImpl::exec(
        _megdnn_tensor_in src, const IndexDesc &index,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    auto info = check_exec(src.layout, index, dst.layout, workspace.size);
    info.error_tracker = m_error_tracker;
    info.error_info = async_error_info(handle());
    ExecImpl<indexing_multi_axis_vec_kdef::OprFwd>{
            src, dst, index, workspace, info, cuda_stream(handle())}();
}

size_t IndexingSetMultiAxisVecImpl::get_workspace_in_bytes(
        size_t value_idx_size) {
    return value_idx_size * sizeof(int);
}

void IndexingSetMultiAxisVecImpl::exec(
        _megdnn_tensor_inout data, _megdnn_tensor_in value,
        const IndexDesc &index, _megdnn_workspace workspace) {
    auto info = check_exec(data.layout, value.layout, index, workspace.size);
    info.error_tracker = m_error_tracker;
    info.error_info = async_error_info(handle());
    ExecImpl<indexing_multi_axis_vec_kdef::OprSet>{
            data, value, index, workspace, info, cuda_stream(handle())}();
}

size_t IndexingIncrMultiAxisVecImpl::get_workspace_in_bytes(
        size_t value_idx_size) {
    return value_idx_size * sizeof(int);
}

void IndexingIncrMultiAxisVecImpl::exec(
        _megdnn_tensor_inout data, _megdnn_tensor_in value,
        const IndexDesc &index, _megdnn_workspace workspace) {
    MEGDNN_INC_FLOAT16(
            megdnn_assert(data.layout.dtype != dtype::Float16(),
            "float16 incr on cuda currently not supported"));
    auto info = check_exec(data.layout, value.layout, index, workspace.size);
    info.error_tracker = m_error_tracker;
    info.error_info = async_error_info(handle());
    ExecImpl<OprAtomicIncr>{data, value, index, workspace, info,
            cuda_stream(handle())}();
}

// vim: syntax=cpp.doxygen

