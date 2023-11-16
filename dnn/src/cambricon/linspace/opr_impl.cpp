#include "src/cambricon/linspace/opr_impl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"

namespace megdnn {
namespace cambricon {

void LinspaceImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto dtype = dst.layout.dtype;
    megdnn_assert(
            dst.layout.dtype == dtype::Int32() ||
                    dst.layout.dtype == dtype::Float16() ||
                    dst.layout.dtype == dtype::Float32(),
            "bad dtype");
    auto handle = concrete_handle(this->handle());
    auto n = dst.layout.total_nr_elems();
    CnnlTensorDescriptor y_desc;
    std::vector<size_t> shape(1, n);
    auto linspace_dtype = convert_to_cnnl_datatype(dtype.enumv());
    bool need_astype = dtype == dtype::Int32();
    if (need_astype) {
        linspace_dtype = CNNL_DTYPE_FLOAT;
    }
    y_desc.set(1, shape, linspace_dtype);
    float start = static_cast<float>(param().start);
    float step = static_cast<float>(
            (param().stop - param().start) /
            std::max(static_cast<double>(param().endpoint ? n - 1 : n), 1.0));
    float end = start + step * (n - 1);
    cnnl_check(cnnlLinspace(
            handle->cnnl_handle(), start, end, y_desc.desc(), dst.raw_ptr()));
    if (need_astype) {
        CnnlTensorDescriptor dst_desc;
        dst_desc.set(1, shape, CNNL_DTYPE_INT32);
        cnnl_check(cnnlCastDataType(
                handle->cnnl_handle(), y_desc.desc(), dst.raw_ptr(),
                CNNL_CAST_FLOAT_TO_INT32, dst_desc.desc(), dst.raw_ptr()));
    }
}

}  // namespace cambricon
}  // namespace megdnn
