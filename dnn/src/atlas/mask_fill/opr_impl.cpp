#include "./opr_impl.h"
#include "aclnnop/aclnn_copy.h"
#include "aclnnop/aclnn_masked_fill_scalar.h"
#include "src/atlas/handle.h"
#include "src/common/utils.h"

namespace megdnn {
namespace atlas {

void MaskedFillImpl::exec(
        _megdnn_tensor_in origin, _megdnn_tensor_in index, _megdnn_tensor_out dest) {
    auto handle = concrete_handle(this->handle());

    TensorLayout formated_layout = index.layout;
    megdnn_assert(origin.layout.is_contiguous() && index.layout.is_contiguous());
    if (formated_layout.ndim < origin.layout.ndim) {
        for (size_t n = formated_layout.ndim; n < origin.layout.ndim; n++)
            formated_layout.add_axis_cont_inplace(n);
    }

    TensorND formated_index(index.raw_ptr(), formated_layout);
    AclTensor acl_origin(origin), acl_index(formated_index), acl_dest(dest);
    uint64_t ws_size;
    aclOpExecutor* executor = nullptr;
    aclnn_check(aclnnInplaceCopyGetWorkspaceSize(
            acl_dest.get(), acl_origin.get(), &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnInplaceCopy(ws.ptr(), ws_size, executor, handle->stream()));

#define cb(DType)                                                           \
    if (origin.layout.dtype == DType()) {                                   \
        using T = typename DTypeTrait<DType>::ctype;                        \
        auto value = static_cast<T>(param().value);                         \
        AclScalar acl_value(value);                                         \
        aclnnInplaceMaskedFillScalarGetWorkspaceSize(                       \
                acl_dest.get(), acl_index.get(), acl_value.get(), &ws_size, \
                &executor);                                                 \
        AclMem ws2(ws_size, handle);                                        \
        aclnn_check(aclnnInplaceMaskedFillScalar(                           \
                ws2.ptr(), ws_size, executor, handle->stream()));           \
        return;                                                             \
    }
    cb(::megdnn::dtype::Int8) cb(::megdnn::dtype::Int32) cb(::megdnn::dtype::Float32)
            cb(::megdnn::dtype::Bool) cb(::megdnn::dtype::Float16)
#undef cb
}
}  // namespace atlas
}  // namespace megdnn