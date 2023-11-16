#include "./opr_impl.h"

#include <vector>
#include "./kern.mlu.h"
#include "src/cambricon/cnnl_wrapper/cnnl_op_descriptor.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"

using namespace megdnn;
using namespace cambricon;
using namespace indexing_multi_axis_vec;
namespace {
class ExecImplHelper {
    template <int nidx, int idx_ndim>
    void dispatch_gen_offset_base_nidx_ndim();
    template <int nidx>
    void dispatch_gen_offset_base_nidx();
    void dispatch_gen_offset_base();

protected:
    using IndexDesc = IndexingMultiAxisVec::IndexDesc;
    using ExecInfo = IndexingMultiAxisVec::ExecInfo;

    cnrtQueue_t m_queue;
    const TensorND* const m_data;
    const TensorND* const m_value;
    const IndexDesc* const m_index;
    const ExecInfo* const m_exec_info;
    int* const m_offset_base;
    TensorLayout m_value_layout_on_data;
    size_t m_idx_axis;
    TensorShape m_idx_shape;
    int m_value_stride;
    int m_cluster_count;
    int m_core_per_cluster;

public:
    ExecImplHelper(
            const TensorND& data, const TensorND& value, const IndexDesc& index,
            const Workspace& workspace, const ExecInfo& exec_info, int cluster_count,
            int core_per_cluster, cnrtQueue_t queue);
};

template <class Opr>
class ExecImpl : public ExecImplHelper {
    void dispatch_exec();

    template <typename ctype>
    void dispatch_exec_ctype();

    template <typename ctype, int ndim>
    void dispatch_exec_ctype_ndim();

public:
    using ExecImplHelper::ExecImplHelper;

    void operator()() {
        dispatch_exec();
        after_kernel_launch();
    }
};
}  // anonymous namespace

ExecImplHelper::ExecImplHelper(
        const TensorND& data, const TensorND& value, const IndexDesc& index,
        const Workspace& workspace, const ExecInfo& exec_info, int cluster_count,
        int core_per_cluster, cnrtQueue_t queue)
        : m_queue{queue},
          m_data{&data},
          m_value{&value},
          m_index{&index},
          m_exec_info{&exec_info},
          m_offset_base{workspace.ptr<int>()},
          m_cluster_count{cluster_count},
          m_core_per_cluster{core_per_cluster} {
    if (!data.layout.total_nr_elems()) {
        megdnn_throw(ssprintf(
                "invalid size of input for advanced indexing: %zu",
                data.layout.total_nr_elems()));
    }
    std::tie(m_value_layout_on_data, m_idx_axis, m_idx_shape) =
            IndexingMultiAxisVec::get_value_iter_optimized_layout(
                    data.layout, value.layout, index, exec_info.idx_axis);
    dispatch_gen_offset_base();
    m_value_stride = exec_info.value_stride;
}

template <int nidx, int idx_ndim>
void ExecImplHelper::dispatch_gen_offset_base_nidx_ndim() {
    GenOffsetBaseParam<nidx, idx_ndim> param;
    param.size = m_idx_shape.total_nr_elems();
    param.output = m_offset_base;
    megdnn_assert(m_idx_shape.ndim == idx_ndim);
    for (int i = 0; i < nidx; ++i) {
        auto&& dst = param.indexer[i];
        auto&& src = m_index->at(i);
        auto src_layout = src.vec.layout.broadcast(m_idx_shape);
        for (size_t i = 0; i < idx_ndim; ++i) {
            if (i) {
                dst.shape[i - 1] = src_layout.shape[i];
            }
            dst.stride[i] = src_layout.stride[i];
        }
        dst.ptr = src.vec.ptr<int>();
        param.data_shape[i] = m_data->layout.shape[src.axis];
        param.data_stride[i] = m_data->layout.stride[src.axis];
    }
    param.cluster_count = m_cluster_count;
    param.core_per_cluster = m_core_per_cluster;
    gen_offset_base(param, m_queue);
}

template <int nidx>
void ExecImplHelper::dispatch_gen_offset_base_nidx() {
    switch (m_idx_shape.ndim) {
#define cb(_n) \
    case _n:   \
        return dispatch_gen_offset_base_nidx_ndim<nidx, _n>();
        MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
    }
    megdnn_throw("bad index ndim");
}

void ExecImplHelper::dispatch_gen_offset_base() {
    switch (m_index->size()) {
#define cb(_n) \
    case _n:   \
        return dispatch_gen_offset_base_nidx<_n>();
        MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
    }
    megdnn_throw("bad index size");
}

template <class Opr>
void ExecImpl<Opr>::dispatch_exec() {
    switch (m_data->layout.dtype.enumv()) {
#define cb(_dtype)                  \
    case DTypeTrait<_dtype>::enumv: \
        return dispatch_exec_ctype<DTypeTrait<_dtype>::ctype>();
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
#undef cb
                default : megdnn_throw("bad dtype");
    }
}

template <class Opr>
template <typename ctype>
void ExecImpl<Opr>::dispatch_exec_ctype() {
    switch (m_value_layout_on_data.ndim) {
#define cb(_n) \
    case _n:   \
        return dispatch_exec_ctype_ndim<ctype, _n>();
        MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
        default:
            megdnn_throw("bad data ndim");
    }
}

template <class Opr>
template <typename ctype, int ndim>
void ExecImpl<Opr>::dispatch_exec_ctype_ndim() {
    ApplyOprParam<ndim> param;
    if (!m_value->layout.total_nr_elems()) {
        megdnn_throw(ssprintf(
                "invalid size of output for advanced indexing: %zu",
                m_value->layout.total_nr_elems()));
    }
    param.tot_size = m_value->layout.total_nr_elems();
    param.offset_base = m_offset_base;
    param.data = m_data->raw_ptr();
    param.value = m_value->raw_ptr();
    param.idx_axis = m_idx_axis;
    param.idx_axis_end = m_idx_axis + m_idx_shape.ndim;
    param.idx_nelems = m_idx_shape.total_nr_elems();
    param.value_stride = m_value_stride;
    param.cluster_count = m_cluster_count;
    param.core_per_cluster = m_core_per_cluster;
    for (int i = 0; i < ndim; ++i) {
        param.value_ly_on_data.stride[i] = m_value_layout_on_data.stride[i];
        if (i) {
            param.value_ly_on_data.shape[i - 1] = m_value_layout_on_data.shape[i];
        }
    }
    if (std::is_same<ctype, dt_float32>::value) {
        apply_opr<ndim, Opr>(param, CNNL_DTYPE_FLOAT, m_queue);
    } else if (std::is_same<ctype, dt_float16>::value) {
        apply_opr<ndim, Opr>(param, CNNL_DTYPE_HALF, m_queue);
#if __bang_arch >= 592
    } else if (std::is_same<ctype, dt_bfloat16>::value) {
        apply_opr<ndim, Opr>(param, CNNL_DTYPE_BFLOAT16, m_queue);
#endif
    } else if (
            std::is_same<ctype, dt_int32>::value &&
            !std::is_same<Opr, OprIncrCommon>::value) {
        apply_opr<ndim, Opr>(param, CNNL_DTYPE_INT32, m_queue);
    } else if (
            std::is_same<ctype, dt_int16>::value &&
            !std::is_same<Opr, OprIncrCommon>::value) {
        apply_opr<ndim, Opr>(param, CNNL_DTYPE_INT16, m_queue);
    } else if (
            std::is_same<ctype, dt_int8>::value &&
            !std::is_same<Opr, OprIncrCommon>::value) {
        apply_opr<ndim, Opr>(param, CNNL_DTYPE_INT8, m_queue);
    } else if (
            std::is_same<ctype, dt_uint8>::value &&
            !std::is_same<Opr, OprIncrCommon>::value) {
        apply_opr<ndim, Opr>(param, CNNL_DTYPE_UINT8, m_queue);
    } else if (std::is_same<ctype, dt_bool>::value) {
        apply_opr<ndim, Opr>(param, CNNL_DTYPE_BOOL, m_queue);
    } else {
        megdnn_throw("unsupported data type");
    }
}

size_t IndexingMultiAxisVecImpl::get_workspace_in_bytes(size_t dst_idx_size) {
    return 0;
}

bool is_axis_contiguous(std::vector<bool> is_indices_defined) {
    auto start = std::find(is_indices_defined.begin(), is_indices_defined.end(), true);
    auto end = std::find(is_indices_defined.rbegin(), is_indices_defined.rend(), true);
    auto it = std::find(start, end.base(), false);
    return it == end.base();
}

bool is_axis_contiguous(const IndexingMultiAxisVec::IndexDesc& index) {
    size_t min_axis = 8;
    size_t max_axis = 0;
    for (size_t i = 0; i < index.size(); ++i) {
        min_axis = std::min(min_axis, index[i].axis);
        max_axis = std::max(max_axis, index[i].axis);
    }
    return max_axis - min_axis + 1 == index.size();
}

// [src_transpose src_copy output_dim output_dims dst_copy]
WorkspaceBundle IndexingMultiAxisVecImpl::get_workspace_bundle(
        const TensorLayout& src_layout, const IndexDesc& index,
        const TensorLayout& dst_layout) {
    auto cambricon_handle = concrete_handle(this->handle());
    size_t src_transpose_size = 0;
    size_t src_copy_size = 0;
    if (!is_axis_contiguous(index)) {
        src_transpose_size = src_layout.dtype.size(src_layout.total_nr_elems());
        if (!src_layout.is_contiguous()) {
            src_copy_size = src_transpose_size;
        }
    }

    size_t output_dim_size = sizeof(int32_t);
    size_t output_dims_size = sizeof(int64_t) * 8;

    size_t dst_contig_size = 0;
    if (!dst_layout.is_contiguous()) {
        dst_contig_size = dst_layout.dtype.size(dst_layout.total_nr_elems());
    }

    return {nullptr,
            {src_transpose_size, src_copy_size, output_dim_size, output_dims_size,
             dst_contig_size},
            cambricon_handle->alignment_requirement()};
}

void broadcast_indices(
        std::vector<TensorND>& indices, std::vector<bool>& is_indices_defined) {
    size_t target_ndim = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (is_indices_defined[i]) {
            target_ndim = std::max(target_ndim, indices[i].layout.ndim);
        }
    }
    TensorLayout target;
    target.ndim = target_ndim;
    for (size_t i = 0; i < target_ndim; ++i) {
        size_t target_idx = target_ndim - i - 1;
        target.shape[target_idx] = 0;
        for (size_t j = 0; j < indices.size(); ++j) {
            if (!is_indices_defined[j] || indices[j].layout.ndim < i + 1) {
                continue;
            }
            size_t cur_idx = indices[j].layout.ndim - i - 1;
            if (indices[j].layout.shape[cur_idx] != 1 &&
                target.shape[target_idx] != 1 &&
                target.shape[target_idx] != indices[j].layout.shape[cur_idx] &&
                target.shape[target_idx] != 0) {
                megdnn_throw(ssprintf(
                        "mismatch index[%zu] in broadcast, got %s", j,
                        indices[j].layout.to_string().c_str()));
            }
            target.shape[target_idx] = std::max(
                    target.shape[target_idx], indices[j].layout.shape[cur_idx]);
        }
    }
    target.init_contiguous_stride();

    for (size_t i = 0; i < indices.size(); ++i) {
        if (is_indices_defined[i]) {
            indices[i].layout = indices[i].layout.broadcast(target);
        }
    }
}

std::pair<TensorND, std::vector<TensorND>> IndexingMultiAxisVecImpl::
        transpose_src_and_indices(
                const TensorND& src, const std::vector<TensorND>& indices,
                const std::vector<bool>& is_indices_defined,
                const WorkspaceBundle& wk_bundle) {
    auto cambricon_handle = concrete_handle(handle());

    std::vector<size_t> src_transpose_permute;
    std::vector<TensorND> indices_transpose;

    for (size_t i = 0; i < is_indices_defined.size() && i < src.layout.ndim; ++i) {
        if (is_indices_defined[i]) {
            src_transpose_permute.push_back(i);
            indices_transpose.push_back(indices[i]);
        }
    }
    for (size_t i = 0; i < is_indices_defined.size() && i < src.layout.ndim; ++i) {
        if (!is_indices_defined[i]) {
            src_transpose_permute.push_back(i);
            indices_transpose.push_back(indices[i]);
        }
    }

    CnnlTransposeDescriptor cnnl_transpose_desc;
    cnnl_transpose_desc.set(
            static_cast<int>(src.layout.ndim), src_transpose_permute.data());
    CnnlTensorDescriptor src_desc, src_transpose_desc;
    src_desc.set(&src);
    TensorLayout src_transpose_layout = src.layout.dimshuffle(src_transpose_permute);
    src_transpose_layout.init_contiguous_stride();
    src_transpose_desc.set(src_transpose_layout);
    void* src_transpose_ptr = wk_bundle.get(0);
    TensorND src_transpose(src_transpose_ptr, src_transpose_layout);

    if (src.layout.is_contiguous()) {
        cnnl_check(cnnlTranspose(
                cambricon_handle->cnnl_handle(), cnnl_transpose_desc.desc(),
                src_desc.desc(), src.raw_ptr(), src_transpose_desc.desc(),
                src_transpose_ptr));
    } else {
        void* src_collapse_ptr = wk_bundle.get(1);
        TensorLayout src_collapse_layout(src.layout);
        src_collapse_layout.init_contiguous_stride();
        CnnlTensorDescriptor src_collapse_desc;
        src_collapse_desc.set(src_collapse_layout);

        cnnl_check(cnnlCopy(
                cambricon_handle->cnnl_handle(), src_desc.desc(), src.raw_ptr(),
                src_collapse_desc.desc(), src_collapse_ptr));
        cnnl_check(cnnlTranspose(
                cambricon_handle->cnnl_handle(), cnnl_transpose_desc.desc(),
                src_collapse_desc.desc(), src_collapse_ptr, src_transpose_desc.desc(),
                src_transpose_ptr));
    }

    return std::make_pair(src_transpose, indices_transpose);
}

std::tuple<TensorND, std::vector<TensorND>, std::vector<bool>> IndexingMultiAxisVecImpl::
        prepare_src_and_index(
                const TensorND& src, const IndexDesc& index,
                const WorkspaceBundle& wk_bundle) {
    std::vector<TensorND> indices(8, TensorND());
    std::vector<bool> is_indices_defined(8, false);
    for (size_t i = 0; i < index.size(); ++i) {
        indices[index[i].axis] = index[i].vec;
        is_indices_defined[index[i].axis] = true;
    }

    broadcast_indices(indices, is_indices_defined);
    if (!is_axis_contiguous(is_indices_defined)) {
        auto transpose_ret =
                transpose_src_and_indices(src, indices, is_indices_defined, wk_bundle);
        for (size_t i = 0; i < index.size(); ++i) {
            is_indices_defined[i] = true;
        }
        for (size_t i = index.size(); i < 8; ++i) {
            is_indices_defined[i] = false;
        }
        return std::make_tuple(
                transpose_ret.first, transpose_ret.second, is_indices_defined);
    }
    return std::make_tuple(src, indices, is_indices_defined);
}

// TODO: check length of indices and delete is_indices_defined
// The indexed axes are contiguous just like [..., indexed_axes, ...]
void IndexingMultiAxisVecImpl::exec(
        _megdnn_tensor_in src, const IndexDesc& index, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    auto cambricon_handle = concrete_handle(handle());

    WorkspaceBundle wk_bundle = get_workspace_bundle(src.layout, index, dst.layout);
    size_t wk_size = wk_bundle.total_size_in_bytes();
    void* wk_ptr = cambricon_handle->alloc(wk_size);
    wk_bundle.set(wk_ptr);

    auto src_indices_and_defined = prepare_src_and_index(src, index, wk_bundle);
    auto prepared_src = std::get<0>(src_indices_and_defined);
    auto prepared_indices = std::get<1>(src_indices_and_defined);
    auto is_indices_defined = std::get<2>(src_indices_and_defined);

    CnnlTensorDescriptor cnnl_src_desc, cnnl_dst_desc, cnnl_output_dim_desc,
            cnnl_expand_output_dims_desc;
    cnnl_src_desc.set(&prepared_src);
    if (dst.layout.is_contiguous()) {
        cnnl_dst_desc.set(&dst);
    } else {
        TensorLayout dst_contig_layout(dst.layout);
        dst_contig_layout.init_contiguous_stride();
        cnnl_dst_desc.set(dst_contig_layout);
    }

    size_t len = 8;
    CnnlTensorDescriptor cnnl_full_index_desc[len];
    for (size_t i = 0; i < len; ++i) {
        if (!is_indices_defined[i]) {
        } else {
            cnnl_full_index_desc[i].set(prepared_indices[i].layout);
        }
    }
    cnnlTensorDescriptor_t cnnl_full_index_desc_t[len];
    for (size_t i = 0; i < 8; ++i) {
        cnnl_full_index_desc_t[i] = cnnl_full_index_desc[i].desc();
    }
    void* indices_ptr[len];
    for (size_t i = 0; i < len; ++i) {
        if (!is_indices_defined[i]) {
            indices_ptr[i] = nullptr;
        } else {
            indices_ptr[i] = prepared_indices[i].raw_ptr();
        }
    }

    void* output_dim = wk_bundle.get(2);
    void* output_dims = wk_bundle.get(3);
    std::vector<size_t> output_dim_shape(1, 1);
    cnnl_output_dim_desc.set(1, output_dim_shape, CNNL_DTYPE_INT32);
    std::vector<size_t> expand_output_dims_shape(1, 8);
    cnnl_expand_output_dims_desc.set(1, expand_output_dims_shape, CNNL_DTYPE_INT64);

    size_t advanced_index_wk_size = 0;
    cnnl_check(cnnlGetAdvancedIndexWorkspaceSize(
            cambricon_handle->cnnl_handle(), cnnl_src_desc.desc(),
            cnnl_full_index_desc_t, &advanced_index_wk_size));
    void* advanced_index_ptr = cambricon_handle->alloc(advanced_index_wk_size);

    if (dst.layout.is_contiguous()) {
        cnnl_check(cnnlAdvancedIndex_v2(
                cambricon_handle->cnnl_handle(), cnnl_src_desc.desc(),
                prepared_src.raw_ptr(), cnnl_full_index_desc_t, indices_ptr,
                advanced_index_ptr, advanced_index_wk_size, cnnl_dst_desc.desc(),
                dst.raw_ptr(), cnnl_output_dim_desc.desc(), output_dim,
                cnnl_expand_output_dims_desc.desc(), output_dims));
    } else {
        void* dst_contig_ptr = wk_bundle.get(4);
        cnnl_check(cnnlAdvancedIndex_v2(
                cambricon_handle->cnnl_handle(), cnnl_src_desc.desc(),
                prepared_src.raw_ptr(), cnnl_full_index_desc_t, indices_ptr,
                advanced_index_ptr, advanced_index_wk_size, cnnl_dst_desc.desc(),
                dst_contig_ptr, cnnl_output_dim_desc.desc(), output_dim,
                cnnl_expand_output_dims_desc.desc(), output_dims));
        CnnlTensorDescriptor cnnl_ori_dst_desc;
        cnnl_ori_dst_desc.set(dst.layout);
        cnnl_check(cnnlCopy(
                cambricon_handle->cnnl_handle(), cnnl_dst_desc.desc(), dst_contig_ptr,
                cnnl_ori_dst_desc.desc(), dst.raw_ptr()));
    }

    cambricon_handle->free(advanced_index_ptr);
    cambricon_handle->free(wk_bundle.ptr());
}

// [src_contig src_transpose]
SmallVector<size_t> get_workspace_size_in_transpose_src_and_indices_of_index_put(
        const TensorLayout& src_layout) {
    SmallVector<size_t> ret;
    if (src_layout.is_contiguous()) {
        ret.push_back(0);
        ret.push_back(src_layout.dtype.size(src_layout.total_nr_elems()));
    } else {
        ret.push_back(src_layout.dtype.size(src_layout.total_nr_elems()));
        ret.push_back(src_layout.dtype.size(src_layout.total_nr_elems()));
    }
    return ret;
}

// [value_contig [index_contig] src_contig src_tranpose]
SmallVector<size_t> get_workspace_size_in_prepare_of_index_put(
        const TensorLayout& src_layout,
        const IndexingMultiAxisVecBase::IndexDesc& index,
        const TensorLayout& value_layout) {
    SmallVector<size_t> ret;
    // workspace for non-contiguous value
    if (!value_layout.is_contiguous()) {
        ret.push_back(value_layout.dtype.size(value_layout.total_nr_elems()));
    } else {
        ret.push_back(0);
    }
    // workspace for non-contiguous index
    for (size_t i = 0; i < src_layout.ndim; ++i) {
        ret.push_back(0);
    }
    for (size_t i = 0; i < index.size(); ++i) {
        if (!index[i].vec.layout.is_contiguous()) {
            ret[1 + index[i].axis] = index[i].vec.layout.dtype.size(
                    index[i].vec.layout.total_nr_elems());
        }
    }

    if (!is_axis_contiguous(index)) {
        auto wk_trans = get_workspace_size_in_transpose_src_and_indices_of_index_put(
                src_layout);
        ret.append(wk_trans.begin(), wk_trans.end());
    } else if (!src_layout.is_contiguous()) {
        ret.append({src_layout.dtype.size(src_layout.total_nr_elems()), 0});
    } else {
        ret.append({0, 0});
    }
    return ret;
}

std::tuple<TensorND, std::vector<TensorND>, std::vector<size_t>>
transpose_src_and_indices_for_index_put(
        Handle* handle, const TensorND& src, const std::vector<TensorND>& indices,
        const std::vector<bool>& is_indices_defined, const WorkspaceBundle& wk_bundle) {
    auto cambricon_handle = concrete_handle(handle);

    std::vector<size_t> src_transpose_permute;
    std::vector<TensorND> indices_transpose;
    std::vector<size_t> src_transpose_re_permute(src.layout.ndim, -1);

    for (size_t i = 0; i < is_indices_defined.size() && i < src.layout.ndim; ++i) {
        if (is_indices_defined[i]) {
            src_transpose_permute.push_back(i);
            indices_transpose.push_back(indices[i]);
            src_transpose_re_permute[i] = src_transpose_permute.size() - 1;
        }
    }
    for (size_t i = 0; i < is_indices_defined.size() && i < src.layout.ndim; ++i) {
        if (!is_indices_defined[i]) {
            src_transpose_permute.push_back(i);
            indices_transpose.push_back(indices[i]);
            src_transpose_re_permute[i] = src_transpose_permute.size() - 1;
        }
    }

    CnnlTransposeDescriptor cnnl_transpose_desc;
    cnnl_transpose_desc.set(
            static_cast<int>(src.layout.ndim), src_transpose_permute.data());
    CnnlTensorDescriptor src_desc, src_transpose_desc;
    src_desc.set(&src);
    TensorLayout src_transpose_layout = src.layout.dimshuffle(src_transpose_permute);
    src_transpose_layout.init_contiguous_stride();
    src_transpose_desc.set(src_transpose_layout);
    void* src_transpose_ptr = wk_bundle.get(1 + src.layout.ndim + 1);
    TensorND src_transpose(src_transpose_ptr, src_transpose_layout);

    if (src.layout.is_contiguous()) {
        cnnl_check(cnnlTranspose(
                cambricon_handle->cnnl_handle(), cnnl_transpose_desc.desc(),
                src_desc.desc(), src.raw_ptr(), src_transpose_desc.desc(),
                src_transpose_ptr));
    } else {
        void* src_collapse_ptr = wk_bundle.get(1 + src.layout.ndim);
        TensorLayout src_collapse_layout(src.layout);
        src_collapse_layout.init_contiguous_stride();
        CnnlTensorDescriptor src_collapse_desc;
        src_collapse_desc.set(src_collapse_layout);

        cnnl_check(cnnlCopy(
                cambricon_handle->cnnl_handle(), src_desc.desc(), src.raw_ptr(),
                src_collapse_desc.desc(), src_collapse_ptr));
        cnnl_check(cnnlTranspose(
                cambricon_handle->cnnl_handle(), cnnl_transpose_desc.desc(),
                src_collapse_desc.desc(), src_collapse_ptr, src_transpose_desc.desc(),
                src_transpose_ptr));
    }

    return std::make_tuple(src_transpose, indices_transpose, src_transpose_re_permute);
}

std::tuple<
        TensorND, std::vector<TensorND>, std::vector<bool>, std::vector<size_t>,
        TensorND>
prepare_src_index_value_for_index_put(
        Handle* handle, const TensorND& src,
        const IndexingMultiAxisVecBase::IndexDesc& index, const TensorND& value,
        const WorkspaceBundle& wk_bundle) {
    auto cambricon_handle = concrete_handle(handle);

    // handle non-contiguous value
    TensorND value_contig = value;
    if (!value.layout.is_contiguous()) {
        CnnlTensorDescriptor cnnl_value_desc, cnnl_value_contig_desc;
        cnnl_value_desc.set(value.layout);
        TensorLayout value_contig_layout(value.layout);
        value_contig_layout.init_contiguous_stride();
        cnnl_value_contig_desc.set(value_contig_layout);
        void* value_contig_ptr = wk_bundle.get(0);
        cnnl_check(cnnlCopy(
                cambricon_handle->cnnl_handle(), cnnl_value_desc.desc(),
                value.raw_ptr(), cnnl_value_contig_desc.desc(), value_contig_ptr));
        value_contig = TensorND(value_contig_layout, value_contig_ptr);
    }

    // handle non-contiguous index
    std::vector<TensorND> indices(src.layout.ndim, TensorND());
    std::vector<bool> is_indices_defined(src.layout.ndim, false);
    for (size_t i = 0; i < index.size(); ++i) {
        if (index[i].vec.layout.is_contiguous()) {
            indices[index[i].axis] = index[i].vec;
        } else {
            CnnlTensorDescriptor cnnl_index_desc, cnnl_index_contig_desc;
            cnnl_index_desc.set(index[i].vec.layout);
            TensorLayout index_contig_layout(index[i].vec.layout);
            index_contig_layout.init_contiguous_stride();
            cnnl_index_contig_desc.set(index_contig_layout);
            void* index_contig_ptr = wk_bundle.get(1 + index[i].axis);
            cnnl_check(cnnlCopy(
                    cambricon_handle->cnnl_handle(), cnnl_index_desc.desc(),
                    index[i].vec.raw_ptr(), cnnl_index_contig_desc.desc(),
                    index_contig_ptr));
            indices[index[i].axis] = TensorND(index_contig_layout, index_contig_ptr);
        }
        is_indices_defined[index[i].axis] = true;
    }

    // broadcast index
    broadcast_indices(indices, is_indices_defined);
    // handle non-contiguous indexed axis
    if (!is_axis_contiguous(is_indices_defined)) {
        auto transpose_ret = transpose_src_and_indices_for_index_put(
                handle, src, indices, is_indices_defined, wk_bundle);
        for (size_t i = 0; i < index.size(); ++i) {
            is_indices_defined[i] = true;
        }
        for (size_t i = index.size(); i < src.layout.ndim; ++i) {
            is_indices_defined[i] = false;
        }
        return std::make_tuple(
                std::get<0>(transpose_ret), std::get<1>(transpose_ret),
                is_indices_defined, std::get<2>(transpose_ret), value_contig);
    } else if (!src.layout.is_contiguous()) {
        void* src_contig_ptr = wk_bundle.get(1 + src.layout.ndim);
        TensorLayout src_contig_layout(src.layout);
        src_contig_layout.init_contiguous_stride();

        CnnlTensorDescriptor cnnl_src_desc, cnnl_src_contig_desc;
        cnnl_src_desc.set(src.layout);
        cnnl_src_contig_desc.set(src_contig_layout);

        cnnl_check(cnnlCopy(
                cambricon_handle->cnnl_handle(), cnnl_src_desc.desc(), src.raw_ptr(),
                cnnl_src_contig_desc.desc(), src_contig_ptr));

        TensorND src_contig_tensor(src_contig_ptr, src_contig_layout);
        return std::make_tuple(
                src_contig_tensor, indices, is_indices_defined, std::vector<size_t>(),
                value_contig);
    }

    return std::make_tuple(
            src, indices, is_indices_defined, std::vector<size_t>(), value_contig);
}

template <class Opr>
void exec_index_put(
        Opr& opr, const TensorND& data, const TensorND& value,
        const IndexingMultiAxisVecBase::IndexDesc& index) {
    auto cambricon_handle = concrete_handle(opr.handle());

    WorkspaceBundle wk_bundle =
            opr.get_workspace_bundle(data.layout, value.layout, index);
    size_t wk_size = wk_bundle.total_size_in_bytes();
    void* wk_ptr = cambricon_handle->alloc(wk_size);
    wk_bundle.set(wk_ptr);

    auto src_indices_defined_and_value =
            opr.prepare_src_index_value(data, index, value, wk_bundle);
    auto prepared_src = std::get<0>(src_indices_defined_and_value);
    auto prepared_indices = std::get<1>(src_indices_defined_and_value);
    auto is_indices_defined = std::get<2>(src_indices_defined_and_value);
    auto prepared_value = std::get<4>(src_indices_defined_and_value);

    CnnlTensorDescriptor cnnl_prepared_src_desc, cnnl_prepared_value_desc;
    cnnl_prepared_src_desc.set(&prepared_src);
    cnnl_prepared_value_desc.set(&prepared_value);
    size_t len = data.layout.ndim;
    CnnlTensorDescriptor cnnl_full_index_desc[len];
    for (size_t i = 0; i < len; ++i) {
        if (is_indices_defined[i]) {
            cnnl_full_index_desc[i].set(prepared_indices[i].layout);
        }
    }
    cnnlTensorDescriptor_t cnnl_full_index_desc_t[len];
    for (size_t i = 0; i < len; ++i) {
        cnnl_full_index_desc_t[i] = cnnl_full_index_desc[i].desc();
    }
    void* indices_ptr[len];
    for (size_t i = 0; i < len; ++i) {
        if (!is_indices_defined[i]) {
            indices_ptr[i] = nullptr;
        } else {
            indices_ptr[i] = prepared_indices[i].raw_ptr();
        }
    }

    bool accumulate = false;
    if (std::is_same<Opr, IndexingSetMultiAxisVecImpl>::value) {
        accumulate = false;
    } else if (std::is_same<Opr, IndexingIncrMultiAxisVecImpl>::value) {
        accumulate = true;
    } else {
        megdnn_throw("Unsupported index op");
    }

    size_t index_put_wk_size = 0;
    cnnl_check(cnnlGetIndexPutWorkspaceSize(
            cambricon_handle->cnnl_handle(), cnnl_prepared_src_desc.desc(),
            cnnl_full_index_desc_t, len, cnnl_prepared_value_desc.desc(), accumulate,
            &index_put_wk_size));
    void* index_put_wk_ptr = cambricon_handle->alloc(index_put_wk_size);
    cnnl_check(cnnlIndexPut(
            cambricon_handle->cnnl_handle(), cnnl_prepared_src_desc.desc(),
            prepared_src.raw_ptr(), cnnl_full_index_desc_t, indices_ptr,
            prepared_indices.size(), cnnl_prepared_value_desc.desc(),
            prepared_value.raw_ptr(), index_put_wk_ptr, index_put_wk_size, accumulate,
            true, cnnl_prepared_src_desc.desc(), prepared_src.raw_ptr()));
    if (!is_axis_contiguous(index)) {
        std::vector<size_t> transpose_permute =
                std::get<3>(src_indices_defined_and_value);
        CnnlTransposeDescriptor cnnl_transpose_desc;
        cnnl_transpose_desc.set(data.layout.ndim, transpose_permute.data());
        CnnlTensorDescriptor cnnl_ori_src_desc;
        cnnl_ori_src_desc.set(&data);
        if (data.layout.is_contiguous()) {
            cnnl_check(cnnlTranspose(
                    cambricon_handle->cnnl_handle(), cnnl_transpose_desc.desc(),
                    cnnl_prepared_src_desc.desc(), prepared_src.raw_ptr(),
                    cnnl_ori_src_desc.desc(), data.raw_ptr()));
        } else {
            void* data_contig_ptr = wk_bundle.get(1 + data.layout.ndim);
            TensorLayout data_contig_layout(data.layout);
            data_contig_layout.init_contiguous_stride();
            CnnlTensorDescriptor cnnl_data_contig_desc;
            cnnl_data_contig_desc.set(data_contig_layout);
            cnnl_check(cnnlTranspose(
                    cambricon_handle->cnnl_handle(), cnnl_transpose_desc.desc(),
                    cnnl_prepared_src_desc.desc(), prepared_src.raw_ptr(),
                    cnnl_data_contig_desc.desc(), data_contig_ptr));
            cnnl_check(cnnlCopy(
                    cambricon_handle->cnnl_handle(), cnnl_data_contig_desc.desc(),
                    data_contig_ptr, cnnl_ori_src_desc.desc(), data.raw_ptr()));
        }
    } else if (!data.layout.is_contiguous()) {
        CnnlTensorDescriptor cnnl_ori_src_desc;
        cnnl_ori_src_desc.set(&data);
        cnnl_check(cnnlCopy(
                cambricon_handle->cnnl_handle(), cnnl_prepared_src_desc.desc(),
                prepared_src.raw_ptr(), cnnl_ori_src_desc.desc(), data.raw_ptr()));
    }

    cambricon_handle->free(index_put_wk_ptr);
    cambricon_handle->free(wk_bundle.ptr());
}

WorkspaceBundle IndexingSetMultiAxisVecImpl::get_workspace_bundle(
        const TensorLayout& src_layout, const TensorLayout& value_layout,
        const IndexDesc& index) {
    auto cambricon_handle = concrete_handle(this->handle());
    return {nullptr,
            get_workspace_size_in_prepare_of_index_put(src_layout, index, value_layout),
            cambricon_handle->alignment_requirement()};
}

std::tuple<
        TensorND, std::vector<TensorND>, std::vector<bool>, std::vector<size_t>,
        TensorND>
IndexingSetMultiAxisVecImpl::prepare_src_index_value(
        const TensorND& src, const IndexDesc& index, const TensorND& value,
        const WorkspaceBundle& wk_bundle) {
    return prepare_src_index_value_for_index_put(
            handle(), src, index, value, wk_bundle);
}

size_t IndexingSetMultiAxisVecImpl::get_workspace_in_bytes(size_t value_idx_size) {
    return 0;
}

void IndexingSetMultiAxisVecImpl::exec(
        _megdnn_tensor_inout data, _megdnn_tensor_in value, const IndexDesc& index,
        _megdnn_workspace workspace) {
    megdnn_assert(
            (data.layout.dtype.enumv() == DTypeEnum::Float16 ||
             data.layout.dtype.enumv() == DTypeEnum::BFloat16 ||
             data.layout.dtype.enumv() == DTypeEnum::Bool ||
             data.layout.dtype.enumv() == DTypeEnum::Float32 ||
             data.layout.dtype.enumv() == DTypeEnum::Int32 ||
             data.layout.dtype.enumv() == DTypeEnum::Int16 ||
             data.layout.dtype.enumv() == DTypeEnum::Int8 ||
             data.layout.dtype.enumv() == DTypeEnum::Uint8),
            "Unsupported data type in indexing_set_multi_axis_vec of cambricon");
    exec_index_put(*this, data, value, index);
}

WorkspaceBundle IndexingIncrMultiAxisVecImpl::get_workspace_bundle(
        const TensorLayout& src_layout, const TensorLayout& value_layout,
        const IndexDesc& index) {
    auto cambricon_handle = concrete_handle(this->handle());
    return {nullptr,
            get_workspace_size_in_prepare_of_index_put(src_layout, index, value_layout),
            cambricon_handle->alignment_requirement()};
}

std::tuple<
        TensorND, std::vector<TensorND>, std::vector<bool>, std::vector<size_t>,
        TensorND>
IndexingIncrMultiAxisVecImpl::prepare_src_index_value(
        const TensorND& src, const IndexDesc& index, const TensorND& value,
        const WorkspaceBundle& wk_bundle) {
    return prepare_src_index_value_for_index_put(
            handle(), src, index, value, wk_bundle);
}

// TODO: delete the fallback or modify the workspace size to zero when kernel is not used.
size_t IndexingIncrMultiAxisVecImpl::get_workspace_in_bytes(size_t value_idx_size) {
    return value_idx_size * sizeof(int);
}

void IndexingIncrMultiAxisVecImpl::fallback_to_kernel(
        _megdnn_tensor_inout data, _megdnn_tensor_in value, const IndexDesc& index,
        _megdnn_workspace workspace) {
    auto info = check_exec(data.layout, value.layout, index, workspace.size);
    auto cambricon_handle = concrete_handle(handle());
    ExecImpl<OprIncrCommon>{
            data,
            value,
            index,
            workspace,
            info,
            cambricon_handle->device_info().clusterCount,
            cambricon_handle->device_info().McorePerCluster,
            cambricon_handle->queue()}();
}

void IndexingIncrMultiAxisVecImpl::exec(
        _megdnn_tensor_inout data, _megdnn_tensor_in value, const IndexDesc& index,
        _megdnn_workspace workspace) {
    if (!(data.layout.dtype.enumv() == DTypeEnum::Float16 ||
          data.layout.dtype.enumv() == DTypeEnum::Float32 ||
          data.layout.dtype.enumv() == DTypeEnum::Int32 ||
          data.layout.dtype.enumv() == DTypeEnum::Int16 ||
          data.layout.dtype.enumv() == DTypeEnum::Int8 ||
          data.layout.dtype.enumv() == DTypeEnum::Uint8)) {
        fallback_to_kernel(data, value, index, workspace);
        return;
    }
    exec_index_put(*this, data, value, index);
}

// vim: syntax=cpp.doxygen
