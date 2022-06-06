#include "src/cuda/norm/opr_impl.h"
#include "helper.h"
#include "src/common/reduce_helper_device.h"
#include "src/common/utils.h"
#include "src/cuda/handle.h"
#include "src/cuda/reduce_helper.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

using namespace device_reduce;
using Mode = Norm::Mode;

template <>
void NormForwardImpl::dispatch_mode<Mode::NEG_INF_NORM>(
        _megdnn_tensor_inout src, _megdnn_tensor_inout dst, _megdnn_workspace workspace,
        size_t A, size_t B, size_t C, cudaStream_t stream) {
#define CASE(dt)                                                                   \
    case DTypeTrait<dt>::enumv: {                                                  \
        using ctype = DTypeTrait<dt>::ctype;                                       \
        auto reduceOp =                                                            \
                MinOp<ctype, ctype, ctype>(src.ptr<ctype>(), dst.ptr<ctype>(), B); \
        run_reduce<MinOp<ctype, ctype, ctype>, false>(                             \
                workspace.ptr<ctype>(), A, B, C, stream, reduceOp);                \
        break;                                                                     \
    };
    switch (src.layout.dtype.enumv()) {
        CASE(::megdnn::dtype::Float32)
#if !MEGDNN_DISABLE_FLOAT16
        CASE(::megdnn::dtype::Float16)
#endif
        default:
            megdnn_assert_internal(false);
    }
#undef CASE
}

template <>
void NormForwardImpl::dispatch_mode<Mode::INF_NORM>(
        _megdnn_tensor_inout src, _megdnn_tensor_inout dst, _megdnn_workspace workspace,
        size_t A, size_t B, size_t C, cudaStream_t stream) {
#define CASE(dt)                                                                   \
    case DTypeTrait<dt>::enumv: {                                                  \
        using ctype = DTypeTrait<dt>::ctype;                                       \
        auto reduceOp =                                                            \
                MaxOp<ctype, ctype, ctype>(src.ptr<ctype>(), dst.ptr<ctype>(), B); \
        run_reduce<MaxOp<ctype, ctype, ctype>, false>(                             \
                workspace.ptr<ctype>(), A, B, C, stream, reduceOp);                \
        break;                                                                     \
    };
    switch (src.layout.dtype.enumv()) {
        CASE(::megdnn::dtype::Float32)
#if !MEGDNN_DISABLE_FLOAT16
        CASE(::megdnn::dtype::Float16)
#endif
        default:
            megdnn_assert_internal(false);
    }
#undef CASE
}

template <>
void NormForwardImpl::dispatch_mode<Mode::P_NORM>(
        _megdnn_tensor_inout src, _megdnn_tensor_inout dst, _megdnn_workspace workspace,
        size_t A, size_t B, size_t C, cudaStream_t stream) {
    typedef dt_float32 p_type;

#define CASE(dt)                                                                \
    case DTypeTrait<dt>::enumv: {                                               \
        using ctype = DTypeTrait<dt>::ctype;                                    \
        p_type epsilon = 0.000001f;                                             \
        if (fabs(param().p - 0.0f) < epsilon) {                                 \
            run_reduce<NormZeroOp<ctype, ctype, ctype>, false>(                 \
                    workspace.ptr<ctype>(), A, B, C, stream,                    \
                    NormZeroOp<ctype, ctype, ctype>(                            \
                            src.ptr<ctype>(), dst.ptr<ctype>(), B));            \
        } else if (fabs(param().p - 1.0f) < epsilon) {                          \
            run_reduce<NormOneOp<ctype, ctype, ctype>, false>(                  \
                    workspace.ptr<ctype>(), A, B, C, stream,                    \
                    NormOneOp<ctype, ctype, ctype>(                             \
                            src.ptr<ctype>(), dst.ptr<ctype>(), B));            \
        } else if (fabs(param().p - 2.0f) < epsilon) {                          \
            run_reduce<NormTwoOp<ctype, ctype, ctype>, false>(                  \
                    workspace.ptr<ctype>(), A, B, C, stream,                    \
                    NormTwoOp<ctype, ctype, ctype>(                             \
                            src.ptr<ctype>(), dst.ptr<ctype>(), B));            \
        } else {                                                                \
            run_reduce<NormOp<ctype, ctype, ctype>, false>(                     \
                    workspace.ptr<ctype>(), A, B, C, stream,                    \
                    NormOp<ctype, ctype, ctype>(                                \
                            src.ptr<ctype>(), dst.ptr<ctype>(), B, param().p)); \
        }                                                                       \
        break;                                                                  \
    };

    switch (src.layout.dtype.enumv()) {
        CASE(::megdnn::dtype::Float32)
#if !MEGDNN_DISABLE_FLOAT16
        CASE(::megdnn::dtype::Float16)
#endif
        default:
            megdnn_assert_internal(false);
    }
#undef CASE
}

}  // namespace cuda

namespace cuda {
void NormForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    size_t A, B, C;
    reduce::get_ABC(src.layout, A, B, C, param().dim);
    auto stream = cuda_stream(this->handle());

#define CASE(mode)                                                 \
    case mode: {                                                   \
        dispatch_mode<mode>(src, dst, workspace, A, B, C, stream); \
        break;                                                     \
    };

    switch (param().mode) {
        CASE(Mode::P_NORM)
        CASE(Mode::INF_NORM)
        CASE(Mode::NEG_INF_NORM)
        default:
            megdnn_assert_internal(false);
    }
#undef CASE

    return;
}

size_t NormForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    using namespace device_reduce;
    size_t A, B, C;
    reduce::get_ABC(src, A, B, C, param().dim);

#define cb(dt, op)                                                              \
    case DTypeTrait<dt>::enumv: {                                               \
        using ctype = DTypeTrait<dt>::ctype;                                    \
        return get_reduce_workspace_in_bytes<op<ctype, ctype, ctype>>(A, B, C); \
        break;                                                                  \
    };

#if !MEGDNN_DISABLE_FLOAT16
#define CASE(mode, op)                                                                \
    case mode: {                                                                      \
        switch (src.dtype.enumv()) {                                                  \
            cb(::megdnn::dtype::Float32, op) cb(::megdnn::dtype::Float16, op) default \
                    : megdnn_assert_internal(false);                                  \
        }                                                                             \
    };
#else
#define CASE(mode, op)                                                                \
    case mode: {                                                                      \
        switch (src.dtype.enumv()) {                                                  \
            cb(::megdnn::dtype::Float32, op) default : megdnn_assert_internal(false); \
        }                                                                             \
    };
#endif

    // XXX: 0/1 norm dispathed to different Op, but workspace size same as
    // NormOp
    switch (param().mode) {
        CASE(Mode::INF_NORM, MaxOp)
        CASE(Mode::NEG_INF_NORM, MinOp)
        CASE(Mode::P_NORM, NormOp)
        default:
            megdnn_assert_internal(false);
    }
#undef CASE
#undef cb
}

}  // namespace cuda
}  // namespace megdnn
