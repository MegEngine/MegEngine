#include "src/cambricon/argmxx/opr_impl.h"
#include "cnnl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"

#include <numeric>

namespace megdnn {
namespace cambricon {

struct ArgmxxCnnlDescs {
    CnnlTensorDescriptor src_desc, out_value_desc, out_indices_desc;
    int k = 1;
    int sort_dim = 0;
    bool sorted = false;
    bool lower_index_first = true;
    ArgmxxCnnlDescs(const TensorLayout& src, const TensorLayout& dst, int dim) {
        src_desc.set(src);
        out_value_desc.set(dst, convert_to_cnnl_datatype(src.dtype.enumv()));
        out_indices_desc.set(dst, CNNL_DTYPE_INT32);
        sort_dim = dim;
    }
};

WorkspaceBundle ArgmaxForwardImpl::make_bundle(
        const TensorLayout& src, const TensorLayout& dst) {
    auto handle = concrete_handle(this->handle());
    size_t topk_ws = 0;
    ArgmxxCnnlDescs descs(src, dst, param().axis);
    cnnl_check(cnnlGetTopKTensorWorkspaceSize(
            /* handle         */ handle->cnnl_handle(),
            /* input_desc     */ descs.src_desc.desc(),
            /* k              */ descs.k,
            /* dim            */ param().axis,
            /* largest        */ true,
            /* output_desc    */ descs.out_value_desc.desc(),
            /* index_desc     */ descs.out_indices_desc.desc(),
            /* workspace_size */ &topk_ws));
    size_t value_ws = dst.span().dist_elem() * src.dtype.size();
    size_t src_ws = 0;
    if (!src.is_contiguous()) {
        src_ws = src.access_bytes();
    }
    return {nullptr, {topk_ws, value_ws, src_ws}, handle->alignment_requirement()};
}

size_t ArgmaxForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    return make_bundle(src, dst).total_size_in_bytes();
}

template <typename ctype>
void dispatch_argmax_with_ctype(
        cnnlHandle_t handle, const ctype* data, int* indices, WorkspaceBundle ws_bundle,
        ArgmxxCnnlDescs& descs) {
    auto values = ws_bundle.get(1);
    auto ws_ptr = ws_bundle.get(0);
    auto ws_size = ws_bundle.get_size(0);
    cnnl_check(cnnlTopKTensor_v3(
            /* handle            */ handle,
            /* input_desc        */ descs.src_desc.desc(),
            /* input             */ data,
            /* k                 */ 1,
            /* dim               */ descs.sort_dim,
            /* largest           */ true,
            /* sorted            */ descs.sorted,
            /* lower_index_first */ descs.lower_index_first,
            /* workspace         */ ws_ptr,
            /* workspace_size    */ ws_size,
            /* output_desc       */ descs.out_value_desc.desc(),
            /* output            */ values,
            /* index_desc        */ descs.out_indices_desc.desc(),
            /* index             */ indices));
}

void ArgmaxForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    auto _handle = cnnl_handle(this->handle());
    ArgmxxCnnlDescs descs(src.layout, dst.layout, param().axis);
    auto ws_bundle = make_bundle(src.layout, dst.layout);
    ws_bundle.set(workspace.raw_ptr);

    void* target_src = src.raw_ptr();
    if (!src.layout.is_contiguous()) {
        TensorLayout dst;
        dst = src.layout;
        dst.init_contiguous_stride();
        CnnlTensorDescriptor dst_desc;
        dst_desc.set(dst);
        cnnl_check(cnnlCopy(
                _handle, descs.src_desc.desc(), src.raw_ptr(), dst_desc.desc(),
                ws_bundle.get(2)));
        target_src = ws_bundle.get(2);
    }

    switch (src.layout.dtype.enumv()) {
#define cb(t)                                                                  \
    case DTypeTrait<t>::enumv:                                                 \
        do {                                                                   \
            using ct = DTypeTrait<t>::ctype;                                   \
            dispatch_argmax_with_ctype<ct>(                                    \
                    _handle, static_cast<ct*>(target_src), dst.ptr<int32_t>(), \
                    ws_bundle, descs);                                         \
            return;                                                            \
        } while (0);
        cb(::megdnn::dtype::Float32);
        DNN_INC_FLOAT16(cb(::megdnn::dtype::Float16));
        MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb);
        default:
            megdnn_throw("unsupported dtype in cambricon ArgmaxImpl");
    }
}

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
