#include "./opr_impl.h"
#include "../convolution/helper.h"

#include "megdnn/dtype.h"
#include "src/common/utils.cuh"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <cstring>

#include "midout.h"
MIDOUT_DECL(megdnn_naive_region_restricted_conv_fwd)

using namespace megdnn;
using namespace naive;

void RegionRestrictedConvolutionForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in rin,
        _megdnn_tensor_in rout, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    MIDOUT_BEGIN(megdnn_naive_region_restricted_conv_fwd) {
        auto filter_meta = check_exec(
                src.layout, filter.layout, rin.layout, rout.layout, dst.layout,
                workspace.size);
        using ComputeMode = Param::ComputeMode;
#define DISPATCH_CMODE(in_dt, r_dt, out_dt, in_ct, r_ct, out_ct, comp_ct, cmode)  \
    do {                                                                          \
        using namespace dtype;                                                    \
        if (src.layout.dtype.enumv() == DTypeTrait<in_dt>::enumv &&               \
            dst.layout.dtype.enumv() == DTypeTrait<out_dt>::enumv &&              \
            rin.layout.dtype.enumv() == DTypeTrait<r_dt>::enumv &&                \
            rout.layout.dtype.enumv() == DTypeTrait<r_dt>::enumv &&               \
            param().compute_mode == cmode) {                                      \
            MEGDNN_DISPATCH_CPU_KERN_OPR((convolution::region_restricted_forward< \
                                          in_ct, in_ct, r_ct, out_ct, comp_ct>(   \
                    src, filter, rin, rout, dst, filter_meta)););                 \
            return;                                                               \
        }                                                                         \
    } while (0);
#define DISPATCH(in_dt, r_dt, out_dt, in_ct, r_ct, out_ct, comp_ct) \
    DISPATCH_CMODE(                                                 \
            in_dt, r_dt, out_dt, in_ct, r_ct, out_ct, comp_ct, ComputeMode::DEFAULT)
#define cb(dt)                                                                     \
    DISPATCH(                                                                      \
            dt, Int32, dt, DTypeTrait<dt>::ctype, dt_int32, DTypeTrait<dt>::ctype, \
            DTypeTrait<dt>::ctype)                                                 \
    DISPATCH(                                                                      \
            dt, Uint8, dt, DTypeTrait<dt>::ctype, dt_uint8, DTypeTrait<dt>::ctype, \
            DTypeTrait<dt>::ctype)
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb
        DNN_INC_FLOAT16(DISPATCH_CMODE(
                Float16, Int32, Float16, dt_float16, dt_int32, dt_float16, dt_float32,
                ComputeMode::FLOAT32));
        DNN_INC_FLOAT16(DISPATCH_CMODE(
                Float16, Uint8, Float16, dt_float16, dt_uint8, dt_float16, dt_float32,
                ComputeMode::FLOAT32));
#undef DISPATCH
        megdnn_throw(ssprintf(
                "unsupported RegionRestrictedConv(%s, %s, %s, %s) -> %s with cmode = "
                "%d",
                src.layout.dtype.name(), filter.layout.dtype.name(),
                rin.layout.dtype.name(), rout.layout.dtype.name(),
                dst.layout.dtype.name(), static_cast<int>(param().compute_mode)));
    }
    MIDOUT_END();
#else
    __builtin_trap();
#endif
}

size_t RegionRestrictedConvolutionBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout& filter, const TensorLayout& diff, const TensorLayout& rin,
        const TensorLayout& rout, const TensorLayout& grad) {
    size_t workspace_size = 0;
    auto flt_dt = filter.dtype.enumv();
    auto grad_dt = grad.dtype.enumv();
    auto diff_dt = diff.dtype.enumv();
    MEGDNN_MARK_USED_VAR(rin);
    MEGDNN_MARK_USED_VAR(rout);
#if !MEGDNN_DISABLE_FLOAT16
    if (flt_dt == DTypeEnum::Float16 || flt_dt == DTypeEnum::BFloat16) {
        megdnn_assert(flt_dt == grad_dt && flt_dt == diff_dt);
        workspace_size = grad.span().dist_elem() * dtype::Float32().size();
    }
#else
    MEGDNN_MARK_USED_VAR(flt_dt);
    MEGDNN_MARK_USED_VAR(grad_dt);
    MEGDNN_MARK_USED_VAR(diff_dt);
#endif

    return workspace_size;
}

void RegionRestrictedConvolutionBackwardDataImpl::exec(
        _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_in rin,
        _megdnn_tensor_in rout, _megdnn_tensor_out grad, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    auto filter_meta = check_exec(
            filter.layout, diff.layout, rin.layout, rout.layout, grad.layout,
            workspace.size);
    using ComputeMode = Param::ComputeMode;
    auto cmode = param().compute_mode;
#define cb(dt)                                                                  \
    do {                                                                        \
        if (filter.layout.dtype == dt() && cmode == ComputeMode::DEFAULT &&     \
            rin.layout.dtype == dtype::Int32() &&                               \
            rout.layout.dtype == dtype::Int32()) {                              \
            using ctype = DTypeTrait<dt>::ctype;                                \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                       \
                    (convolution::region_restricted_backward_data<              \
                            ctype, ctype, dt_int32, ctype>(                     \
                            filter, diff, rin, rout, grad, filter_meta)));      \
            return;                                                             \
        } else if (                                                             \
                filter.layout.dtype == dt() && cmode == ComputeMode::DEFAULT && \
                rin.layout.dtype == dtype::Uint8() &&                           \
                rout.layout.dtype == dtype::Uint8()) {                          \
            using ctype = DTypeTrait<dt>::ctype;                                \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                       \
                    (convolution::region_restricted_backward_data<              \
                            ctype, ctype, dt_uint8, ctype>(                     \
                            filter, diff, rin, rout, grad, filter_meta)));      \
            return;                                                             \
        }                                                                       \
    } while (0);
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb
#if !MEGDNN_DISABLE_FLOAT16
    if (filter.layout.dtype == dtype::Float16() && cmode == ComputeMode::FLOAT32 &&
        rin.layout.dtype == dtype::Int32() && rout.layout.dtype == dtype::Int32()) {
        TensorND grad_fp32{
                workspace.raw_ptr, TensorLayout{grad.layout, dtype::Float32()}};
        auto&& type_cvt = handle()->create_operator<TypeCvt>();
        type_cvt->exec(grad, grad_fp32);
        MEGDNN_DISPATCH_CPU_KERN_OPR((convolution::region_restricted_backward_data<
                                      dt_float16, dt_float16, dt_int32, dt_float32>(
                filter, diff, rin, rout, grad_fp32, filter_meta)));
        type_cvt->exec(grad_fp32, grad);
        return;
    } else if (
            filter.layout.dtype == dtype::Float16() && cmode == ComputeMode::FLOAT32 &&
            rin.layout.dtype == dtype::Uint8() && rout.layout.dtype == dtype::Uint8()) {
        TensorND grad_fp32{
                workspace.raw_ptr, TensorLayout{grad.layout, dtype::Float32()}};
        auto&& type_cvt = handle()->create_operator<TypeCvt>();
        type_cvt->exec(grad, grad_fp32);
        MEGDNN_DISPATCH_CPU_KERN_OPR((convolution::region_restricted_backward_data<
                                      dt_float16, dt_float16, dt_uint8, dt_float32>(
                filter, diff, rin, rout, grad_fp32, filter_meta)));
        type_cvt->exec(grad_fp32, grad);
        return;
    }
#endif
    megdnn_throw(ssprintf(
            "unsupported RegionRestrictedConvolutionBackwardData(%s, %s, %s, %s) -> %s "
            "with cmode = %d",
            filter.layout.dtype.name(), diff.layout.dtype.name(),
            rin.layout.dtype.name(), rout.layout.dtype.name(), grad.layout.dtype.name(),
            static_cast<int>(cmode)));
#else
    __builtin_trap();
#endif
}

size_t RegionRestrictedConvolutionBackwardFilterImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout&,
        const TensorLayout&, const TensorLayout& grad) {
    size_t workspace_size = 0;
#if !MEGDNN_DISABLE_FLOAT16
    auto src_dt = src.dtype.enumv();
    auto grad_dt = grad.dtype.enumv();
    auto diff_dt = diff.dtype.enumv();
    if (src_dt == DTypeEnum::Float16 || src_dt == DTypeEnum::BFloat16) {
        megdnn_assert(src_dt == grad_dt && src_dt == diff_dt);
        workspace_size = grad.span().dist_elem() * dtype::Float32().size();
    }
#endif

    return workspace_size;
}

void RegionRestrictedConvolutionBackwardFilterImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_in rin,
        _megdnn_tensor_in rout, _megdnn_tensor_out grad, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    auto filter_meta = check_exec(
            src.layout, diff.layout, rin.layout, rout.layout, grad.layout,
            workspace.size);
    using ComputeMode = Param::ComputeMode;
    auto cmode = param().compute_mode;
#define cb(dt)                                                               \
    do {                                                                     \
        if (src.layout.dtype == dt() && cmode == ComputeMode::DEFAULT &&     \
            rin.layout.dtype == dtype::Int32() &&                            \
            rout.layout.dtype == dtype::Int32()) {                           \
            using ctype = DTypeTrait<dt>::ctype;                             \
            MEGDNN_DISPATCH_CPU_KERN(                                        \
                    static_cast<HandleImpl*>(handle()),                      \
                    convolution::region_restricted_backward_filter<          \
                            ctype MEGDNN_COMMA ctype MEGDNN_COMMA dt_int32   \
                                    MEGDNN_COMMA ctype>(                     \
                            src, diff, rin, rout, grad, filter_meta););      \
            return;                                                          \
        } else if (                                                          \
                src.layout.dtype == dt() && cmode == ComputeMode::DEFAULT && \
                rin.layout.dtype == dtype::Uint8() &&                        \
                rout.layout.dtype == dtype::Uint8()) {                       \
            using ctype = DTypeTrait<dt>::ctype;                             \
            MEGDNN_DISPATCH_CPU_KERN(                                        \
                    static_cast<HandleImpl*>(handle()),                      \
                    convolution::region_restricted_backward_filter<          \
                            ctype MEGDNN_COMMA ctype MEGDNN_COMMA dt_uint8   \
                                    MEGDNN_COMMA ctype>(                     \
                            src, diff, rin, rout, grad, filter_meta););      \
            return;                                                          \
        }                                                                    \
    } while (0);
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb
#if !MEGDNN_DISABLE_FLOAT16
    if (src.layout.dtype == dtype::Float16() && cmode == ComputeMode::FLOAT32 &&
        rin.layout.dtype == dtype::Int32() && rout.layout.dtype == dtype::Int32()) {
        TensorND grad_fp32{
                workspace.raw_ptr, TensorLayout{grad.layout, dtype::Float32()}};
        auto&& type_cvt = handle()->create_operator<TypeCvt>();
        type_cvt->exec(grad, grad_fp32);
        MEGDNN_DISPATCH_CPU_KERN_OPR((convolution::region_restricted_backward_filter<
                                      dt_float16, dt_float16, dt_int32, dt_float32>(
                src, diff, rin, rout, grad_fp32, filter_meta)););
        type_cvt->exec(grad_fp32, grad);
        return;
    } else if (
            src.layout.dtype == dtype::Float16() && cmode == ComputeMode::FLOAT32 &&
            rin.layout.dtype == dtype::Uint8() && rout.layout.dtype == dtype::Uint8()) {
        TensorND grad_fp32{
                workspace.raw_ptr, TensorLayout{grad.layout, dtype::Float32()}};
        auto&& type_cvt = handle()->create_operator<TypeCvt>();
        type_cvt->exec(grad, grad_fp32);
        MEGDNN_DISPATCH_CPU_KERN_OPR((convolution::region_restricted_backward_filter<
                                      dt_float16, dt_float16, dt_uint8, dt_float32>(
                src, diff, rin, rout, grad_fp32, filter_meta)););
        type_cvt->exec(grad_fp32, grad);
        return;
    }
#endif

    megdnn_assert_internal(0);
#else
    __builtin_trap();
#endif
}

// vim: syntax=cpp.doxygen
