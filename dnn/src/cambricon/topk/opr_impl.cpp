#include "src/cambricon/topk/opr_impl.h"
#include "cnnl.h"
#include "src/cambricon/utils.h"

namespace megdnn {
namespace cambricon {

TopKCnnlDescs::TopKCnnlDescs(
        const TensorLayout& data, const TensorLayout& values, const Mode mode) {
    data_desc.set(data);
    out_value_desc.set(values);
    out_indices_desc.set(values, CNNL_DTYPE_INT32);
    sort_dim = data.ndim - 1;
    switch (mode) {
        case Mode::VALUE_IDX_NOSORT:
            sorted = false;
            break;
        case Mode::VALUE_IDX_SORTED:
            sorted = true;
            break;
        default:
            megdnn_throw("invalid TopK mode in Cambricon");
    }
}

WorkspaceBundle TopKImpl::make_bundle(
        int k, const TensorLayout& data, const TensorLayout& values,
        const TensorLayout& indices) {
    auto handle = concrete_handle(this->handle());
    size_t topk_workspace = 0;
    TopKCnnlDescs descs(data, values, param().mode);
    bool largest = false;
    if (k < 0) {
        largest = true;
        k = std::abs(k);
    }
    cnnl_check(cnnlGetTopKTensorWorkspaceSize(
            /* handle         */ handle->cnnl_handle(),
            /* input_desc     */ descs.data_desc.desc(),
            /* k              */ k,
            /* dim            */ descs.sort_dim,
            /* largest        */ largest,
            /* output_desc    */ descs.out_value_desc.desc(),
            /* index_desc     */ descs.out_indices_desc.desc(),
            /* workspace_size */ &topk_workspace));
    size_t data_workspace = 0;
    if (!data.is_contiguous()) {
        data_workspace = data.access_bytes();
    }
    return {nullptr, {data_workspace, topk_workspace}, handle->alignment_requirement()};
}

size_t TopKImpl::get_workspace_in_bytes(
        int k, const TensorLayout& data, const TensorLayout& values,
        const TensorLayout& indices) {
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(indices);
    return make_bundle(k, data, values, indices).total_size_in_bytes();
}

template <typename ctype>
void TopKImpl::dispatch_with_ctype(
        int k, const ctype* data, ctype* values, int* indices, void* workspace,
        size_t workspace_size, TopKCnnlDescs& descs) {
    auto _handle = cnnl_handle(this->handle());
    bool largest = false;
    if (k < 0) {
        largest = true;
        // cnnl requires k greater than 0
        k = std::abs(k);
    }

    cnnl_check(cnnlTopKTensor_v3(
            /* handle            */ _handle,
            /* input_desc        */ descs.data_desc.desc(),
            /* input             */ data,
            /* k                 */ k,
            /* dim               */ descs.sort_dim,
            /* largest           */ largest,
            /* sorted            */ descs.sorted,
            /* lower_index_first */ descs.lower_index_first,
            /* workspace         */ workspace,
            /* workspace_size    */ workspace_size,
            /* output_desc       */ descs.out_value_desc.desc(),
            /* output            */ values,
            /* index_desc        */ descs.out_indices_desc.desc(),
            /* index             */ indices));
}

void TopKImpl::do_exec(
        int k, _megdnn_tensor_in data, _megdnn_tensor_out values, int32_t* indices,
        _megdnn_workspace workspace) {
    megdnn_assert(
            param().mode != Param::Mode::KTH_ONLY,
            "CNNL topk do not support KTH_ONLY mode");
    auto _handle = cnnl_handle(this->handle());
    TensorLayout indices_layout = values.layout;
    indices_layout.dtype = dtype::Int32{};
    auto ws_bundle = make_bundle(k, data.layout, values.layout, indices_layout);
    ws_bundle.set(workspace.raw_ptr);
    TopKCnnlDescs descs(data.layout, values.layout, param().mode);

    void* target_src = data.raw_ptr();
    if (!data.layout.is_contiguous()) {
        TensorLayout dst;
        dst = data.layout;
        dst.init_contiguous_stride();
        CnnlTensorDescriptor dst_desc;
        dst_desc.set(dst);
        cnnl_check(cnnlCopy(
                _handle, descs.data_desc.desc(), data.raw_ptr(), dst_desc.desc(),
                ws_bundle.get(0)));
        target_src = ws_bundle.get(0);
    }
    switch (data.layout.dtype.enumv()) {
#define cb(t)                                                                   \
    case DTypeTrait<t>::enumv:                                                  \
        do {                                                                    \
            using ct = DTypeTrait<t>::ctype;                                    \
            dispatch_with_ctype<ct>(                                            \
                    k, static_cast<ct*>(target_src), values.ptr<ct>(), indices, \
                    ws_bundle.get(1), ws_bundle.get_size(1), descs);            \
            return;                                                             \
        } while (0);
        cb(::megdnn::dtype::Float32);
        DNN_INC_FLOAT16(cb(::megdnn::dtype::Float16));
        MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb);
        default:
            megdnn_throw(ssprintf(
                    "cambricon topk not support dtype=%s", data.layout.dtype.name()));
#undef cb
    }
}

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
