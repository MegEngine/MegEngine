#include "./opr_impl.h"
#include "./kern.cuh"

#include "src/common/reduce_helper_device.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace cumprod;

namespace {

/*!
 * \brief compute cumprod reduction on (A, B, C) tensor to (A, 1, C)
 */
template <typename T, class Op>
void dispatch(
        T* dst, T* workspace, size_t workspace_size, size_t A, size_t B, size_t C,
        bool exclusive, bool reverse, const Op& op, cudaStream_t stream) {
#define IF(exclusive_v, reverse_v)                                    \
    if (exclusive == exclusive_v && reverse == reverse_v) {           \
        run_kern<T, Op, exclusive_v, reverse_v>(                      \
                dst, workspace, workspace_size, A, B, C, op, stream); \
        return;                                                       \
    }
    IF(true, true)
    IF(true, false)
    IF(false, true)
    IF(false, false)
    megdnn_assert_internal(false);
#undef IF
}

}  // anonymous namespace

void CumprodForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    size_t A, B, C;
    reduce::get_ABC(src.layout, A, B, C, param().axis);
    auto stream = cuda_stream(handle());
#define cb(DType)                                                                  \
    if (src.layout.dtype == DType()) {                                             \
        using ctype = DTypeTrait<DType>::ctype;                                    \
        dispatch<ctype, ProdOp<ctype>>(                                            \
                dst.ptr<ctype>(), workspace.ptr<ctype>(), workspace.size, A, B, C, \
                param().exclusive, param().reverse, src.ptr<ctype>(), stream);     \
        return;                                                                    \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(false);
}

size_t CumprodForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout&) {
    size_t A, B, C;
    reduce::get_ABC(src, A, B, C, param().axis);
    cuda_check(cudaSetDevice(concrete_handle(handle())->device_id()));
    return cumprod::get_workspace_in_bytes(A, B, C, src.dtype.size());
}

// vim: syntax=cpp.doxygen
