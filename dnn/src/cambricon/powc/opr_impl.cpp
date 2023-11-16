#include "src/cambricon/powc/opr_impl.h"
#include "src/cambricon/utils.h"
#include "src/cambricon/utils.mlu.h"

#include "cnnl.h"
#include "cnrt.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"

using namespace megdnn;
using namespace cambricon;

WorkspaceBundle PowCImpl::make_bundle(
        const TensorLayout& src, const TensorLayout& dst) {
    auto handle = concrete_handle(this->handle());
    CnnlTensorDescriptor desc_inp, desc_exp, desc_out;
    auto layout = CNNL_LAYOUT_ARRAY;
    desc_inp.set(src, layout);
    desc_out.set(dst, layout);
    std::vector<size_t> dims = {1};
    desc_exp.set(1, dims, convert_to_cnnl_datatype(src.dtype.enumv()), layout);
    size_t powc_workspace = 0;
    cnnl_check(cnnlGetPowWorkspaceSize(
            handle->cnnl_handle(), desc_inp.desc(), desc_exp.desc(), desc_out.desc(),
            &powc_workspace));
    size_t exp_workspace = sizeof(float);
    return {nullptr, {exp_workspace, powc_workspace}, handle->alignment_requirement()};
}

size_t PowCImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    return make_bundle(src, dst).total_size_in_bytes();
}

template <typename T>
void PowCImpl::do_exec_ct(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    auto handle = concrete_handle(this->handle());
    auto ws_bundle = make_bundle(src.layout, dst.layout);
    ws_bundle.set(workspace.raw_ptr);

    CnnlTensorDescriptor desc_inp, desc_exp, desc_out;
    auto cnnl_layout = CNNL_LAYOUT_ARRAY;
    desc_inp.set(src.layout, cnnl_layout);
    desc_out.set(dst.layout, cnnl_layout);
    std::vector<size_t> dims = {1};
    desc_exp.set(1, dims, convert_to_cnnl_datatype(DTypeTrait<T>::enumv), cnnl_layout);
    T v_exp = static_cast<T>(param().exp);
    T* dev_exp = static_cast<T*>(ws_bundle.get(0));
    cnnl_check(cnnlFill_v3(
            handle->cnnl_handle(), CNNL_POINTER_MODE_HOST, (void*)&v_exp,
            desc_exp.desc(), dev_exp));
    cnnl_check(cnnlPow(
            handle->cnnl_handle(), CNNL_COMPUTATION_HIGH_PRECISION, desc_inp.desc(),
            src.raw_ptr(), desc_exp.desc(), dev_exp, ws_bundle.get(1),
            ws_bundle.get_size(1), desc_out.desc(), dst.raw_ptr()));
}

void PowCImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    megdnn_assert(src.layout.dtype.enumv() == dst.layout.dtype.enumv());
    switch (src.layout.dtype.enumv()) {
#define cb(dt)                  \
    case DTypeTrait<dt>::enumv: \
        return do_exec_ct<DTypeTrait<dt>::ctype>(src, dst, workspace);
        cb(::megdnn::dtype::Float32);
        DNN_INC_FLOAT16(cb(::megdnn::dtype::Float16));
#undef cb
        default:
            megdnn_throw("unsupported dtype for PowC");
    }
}

// vim: syntax=cpp.doxygen
