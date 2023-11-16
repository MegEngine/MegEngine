#include "src/cambricon/masked_fill/opr_impl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"
#include "src/common/elemwise_helper.cuh"

using namespace megdnn;
using namespace cambricon;

void MaskedFillImpl::exec(
        _megdnn_tensor_in origin, _megdnn_tensor_in index, _megdnn_tensor_out dest) {
    MEGDNN_MARK_USED_VAR(origin);
    MEGDNN_MARK_USED_VAR(index);
    MEGDNN_MARK_USED_VAR(dest);
    megdnn_assert("in the Cambrian platform, the MaskedFill op requires a workspace");
}

void MaskedFillImpl::exec(
        _megdnn_tensor_in origin, _megdnn_tensor_in index, _megdnn_tensor_out dest,
        _megdnn_workspace workspace) {
    auto cnnl_handler = concrete_handle(this->handle())->cnnl_handle();
    check_exec(origin.layout, index.layout, dest.layout);

    megdnn_assert(origin.layout.is_contiguous() && index.layout.is_contiguous());
    ElemwiseOpParamN<3> src;
    src[0] = origin;
    src[1] = dest;
    src[2] = index;
    if (src[2].layout.ndim < src[0].layout.ndim) {
        for (size_t n = src[2].layout.ndim; n < src[0].layout.ndim; n++)
            src[2].layout.add_axis_cont_inplace(n);
    }
    src[2].layout = src[2].layout.broadcast(origin.layout);
    // index layout may need add axis
    CnnlTensorDescriptor input_desc, masked_desc, output_desc;
    input_desc.set(src[0].layout);
    masked_desc.set(src[2].layout);
    output_desc.set(src[1].layout);

#define cb(DType)                                                              \
    if (origin.layout.dtype == DType()) {                                      \
        using T = typename DTypeTrait<DType>::ctype;                           \
        auto scale = static_cast<T>(param().value);                            \
        cnnl_check(cnnlMasked_v4(                                              \
                cnnl_handler, cnnlMaskedOp_t::CNNL_MASKED_FILL_HOST,           \
                input_desc.desc(), src[0].raw_ptr(), masked_desc.desc(),       \
                src[2].raw_ptr(), nullptr, /*value_desc*/ nullptr, /*value*/   \
                &scale, workspace.raw_ptr, workspace.size, output_desc.desc(), \
                src[1].raw_ptr(), nullptr));                                   \
        return;                                                                \
    }
    cb(::megdnn::dtype::Float32) cb(::megdnn::dtype::Float16)
            // don't support Uint8
            cb(::megdnn::dtype::Int8) cb(::megdnn::dtype::Int16)
                    cb(::megdnn::dtype::Int32) cb(::megdnn::dtype::Bool)
#undef cb
                            megdnn_throw("bad dtype");
    return;
}

size_t MaskedFillImpl::get_workspace_in_bytes(
        const TensorLayout& origin, const TensorLayout& index,
        const TensorLayout& dest) {
    auto cnnl_handler = concrete_handle(this->handle())->cnnl_handle();
    size_t workspace_size = 0;
    CnnlTensorDescriptor input_desc, masked_desc, output_desc;
    input_desc.set(origin);
    masked_desc.set(index);
    output_desc.set(dest);
    cnnl_check(cnnlGetMaskedWorkspaceSize(
            cnnl_handler, cnnlMaskedOp_t::CNNL_MASKED_FILL_HOST, input_desc.desc(),
            masked_desc.desc(), nullptr, output_desc.desc(), &workspace_size));
    return workspace_size;
}
