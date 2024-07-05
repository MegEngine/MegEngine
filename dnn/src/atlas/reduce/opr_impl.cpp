#include "src/atlas/reduce/opr_impl.h"
#include "aclnnop/aclnn_amax.h"
#include "aclnnop/aclnn_amin.h"
#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_prod.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

namespace megdnn {
namespace atlas {

using Param = megdnn::param::Reduce;

void ReduceForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    MEGDNN_MARK_USED_VAR(workspace);
    auto&& param = this->param();

    AclIntArray axes({param.axis});
    AclTensor inp(src);
    AclTensor oup(dst);

    bool keepdims = true;
    auto data_type = as_acl_dtype(dst.layout.dtype);
    uint64_t ws_size;
    aclOpExecutor* executor;

    auto handle = concrete_handle(this->handle());

    switch (param.mode) {
        case Param::Mode::SUM: {
            aclnn_check(aclnnReduceSumGetWorkspaceSize(
                    inp.get(), axes.get(), keepdims, data_type, oup.get(), &ws_size,
                    &executor));
            AclMem ws(ws_size, handle);
            aclnn_check(aclnnReduceSum(ws.ptr(), ws_size, executor, handle->stream()));
            break;
        }
        case Param::Mode::MEAN: {
            aclnn_check(aclnnMeanGetWorkspaceSize(
                    inp.get(), axes.get(), keepdims, data_type, oup.get(), &ws_size,
                    &executor));
            AclMem ws(ws_size, handle);
            aclnn_check(aclnnMean(ws.ptr(), ws_size, executor, handle->stream()));
            break;
        }
        case Param::Mode::MIN: {
            aclnn_check(aclnnAminGetWorkspaceSize(
                    inp.get(), axes.get(), keepdims, oup.get(), &ws_size, &executor));
            AclMem ws(ws_size, handle);
            aclnn_check(aclnnAmin(ws.ptr(), ws_size, executor, handle->stream()));
            break;
        }
        case Param::Mode::MAX: {
            aclnn_check(aclnnAmaxGetWorkspaceSize(
                    inp.get(), axes.get(), keepdims, oup.get(), &ws_size, &executor));
            AclMem ws(ws_size, handle);
            aclnn_check(aclnnAmax(ws.ptr(), ws_size, executor, handle->stream()));
            break;
        }
        case Param::Mode::PRODUCT: {
            aclnn_check(aclnnProdDimGetWorkspaceSize(
                    inp.get(), param.axis, keepdims, data_type, oup.get(), &ws_size,
                    &executor));
            AclMem ws(ws_size, handle);
            aclnn_check(aclnnProdDim(ws.ptr(), ws_size, executor, handle->stream()));
            break;
        }
        case Param::Mode::SUM_SQR: {
            AclTempTensor(handle, tmp, src.layout);
            AclScalar acl_scalar_two(2);
            aclnn_call(
                    handle, aclnnPowTensorScalar, inp.get(), acl_scalar_two.get(),
                    tmp.get());
            aclnn_call(
                    handle, aclnnReduceSum, tmp.get(), axes.get(), keepdims, data_type,
                    oup.get());
            break;
        }
        default: {
            megdnn_assert(
                    false, "invalid ReduceMode, mode value: %u\n",
                    static_cast<uint32_t>(param.mode));
        }
    }
}

size_t ReduceForwardImpl::get_workspace_in_bytes(
        const TensorLayout&, const TensorLayout&) {
    return 0;
}

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
