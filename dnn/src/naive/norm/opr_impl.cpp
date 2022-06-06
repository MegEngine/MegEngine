#include "src/naive/norm/opr_impl.h"

#include "helper.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace naive {
using Mode = Norm::Mode;

template <>
void NormForwardImpl::dispatch_mode<Mode::NEG_INF_NORM>(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, size_t A, size_t B, size_t C) {
#define CASE(dt)                                                                 \
    case DTypeTrait<dt>::enumv: {                                                \
        using ctype = DTypeTrait<dt>::ctype;                                     \
        const ctype* __restrict sptr = src.ptr<ctype>();                         \
        ctype* __restrict dptr = dst.ptr<ctype>();                               \
        std::function<ctype(size_t, size_t, size_t, size_t)> func;               \
        func = [&](size_t a, size_t c, size_t bl, size_t br) -> ctype {          \
            if (bl + 1 < br) {                                                   \
                size_t mid = bl + (br - bl) / 2;                                 \
                return Trait<ReduceForward::Mode::MIN, ctype>::apply(            \
                        func(a, c, bl, mid), func(a, c, mid, br));               \
            } else {                                                             \
                return Trait<ReduceForward::Mode::MIN, ctype>::visit(            \
                        sptr[a * B * C + bl * C + c]);                           \
            }                                                                    \
        };                                                                       \
        for (size_t a = 0; a < A; ++a)                                           \
            for (size_t c = 0; c < C; ++c) {                                     \
                dptr[a * C + c] = Trait<ReduceForward::Mode::MIN, ctype>::write( \
                        func(a, c, 0, B), B);                                    \
            }                                                                    \
        break;                                                                   \
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
        _megdnn_tensor_in src, _megdnn_tensor_out dst, size_t A, size_t B, size_t C) {
#define CASE(dt)                                                                 \
    case DTypeTrait<dt>::enumv: {                                                \
        using ctype = DTypeTrait<dt>::ctype;                                     \
        const ctype* __restrict sptr = src.ptr<ctype>();                         \
        ctype* __restrict dptr = dst.ptr<ctype>();                               \
        std::function<ctype(size_t, size_t, size_t, size_t)> func;               \
        func = [&](size_t a, size_t c, size_t bl, size_t br) -> ctype {          \
            if (bl + 1 < br) {                                                   \
                size_t mid = bl + (br - bl) / 2;                                 \
                return Trait<ReduceForward::Mode::MAX, ctype>::apply(            \
                        func(a, c, bl, mid), func(a, c, mid, br));               \
            } else {                                                             \
                return Trait<ReduceForward::Mode::MAX, ctype>::visit(            \
                        sptr[a * B * C + bl * C + c]);                           \
            }                                                                    \
        };                                                                       \
        for (size_t a = 0; a < A; ++a)                                           \
            for (size_t c = 0; c < C; ++c) {                                     \
                dptr[a * C + c] = Trait<ReduceForward::Mode::MAX, ctype>::write( \
                        func(a, c, 0, B), B);                                    \
            }                                                                    \
        break;                                                                   \
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
        _megdnn_tensor_in src, _megdnn_tensor_out dst, size_t A, size_t B, size_t C) {
#define CASE(dt)                                                                     \
    case DTypeTrait<dt>::enumv: {                                                    \
        using ctype = DTypeTrait<dt>::ctype;                                         \
        const ctype* __restrict sptr = src.ptr<ctype>();                             \
        ctype* __restrict dptr = dst.ptr<ctype>();                                   \
        std::function<ctype(size_t, size_t, size_t, size_t)> func;                   \
        if (param().p - 0.f < 0.00001f) {                                            \
            func = [&](size_t a, size_t c, size_t bl, size_t br) -> ctype {          \
                if (bl + 1 < br) {                                                   \
                    size_t mid = bl + (br - bl) / 2;                                 \
                    return NormZeroOp<ctype>::apply(                                 \
                            func(a, c, bl, mid), func(a, c, mid, br));               \
                } else {                                                             \
                    return NormZeroOp<ctype>::visit(sptr[a * B * C + bl * C + c]);   \
                }                                                                    \
            };                                                                       \
            for (size_t a = 0; a < A; ++a) {                                         \
                for (size_t c = 0; c < C; ++c) {                                     \
                    dptr[a * C + c] = NormZeroOp<ctype>::write(func(a, c, 0, B), B); \
                }                                                                    \
            }                                                                        \
        } else {                                                                     \
            func = [&](size_t a, size_t c, size_t bl, size_t br) -> ctype {          \
                if (bl + 1 < br) {                                                   \
                    size_t mid = bl + (br - bl) / 2;                                 \
                    return NormOp<ctype>::apply(                                     \
                            func(a, c, bl, mid), func(a, c, mid, br));               \
                } else {                                                             \
                    return NormOp<ctype>::visit(                                     \
                            sptr[a * B * C + bl * C + c], param().p);                \
                }                                                                    \
            };                                                                       \
            for (size_t a = 0; a < A; ++a) {                                         \
                for (size_t c = 0; c < C; ++c) {                                     \
                    dptr[a * C + c] =                                                \
                            NormOp<ctype>::write(func(a, c, 0, B), B, param().p);    \
                }                                                                    \
            }                                                                        \
        }                                                                            \
        break;                                                                       \
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

void NormForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    using namespace reduce;
    size_t A, B, C;
    reduce::get_ABC(src.layout, A, B, C, param().dim);
    auto make_tensor = [&](DType comp_dtype, _megdnn_tensor_inout tensor,
                           dt_byte*& workspace_ptr) {
        if (comp_dtype == tensor.layout.dtype)
            return tensor;
        auto layout = TensorLayout(tensor.layout, comp_dtype);
        TensorND new_tensor{workspace_ptr, layout};
        workspace_ptr += layout.span().dist_byte();
        return new_tensor;
    };
    auto typecvt = handle()->create_operator<TypeCvt>();

    auto copy_to = [&typecvt](const TensorND& from, const TensorND& to) {
        if (from.raw_ptr() != to.raw_ptr())
            typecvt->exec(from, to);
    };

    auto workspace_ptr = workspace.ptr<dt_byte>();

    auto new_src = make_tensor(src.layout.dtype, src, workspace_ptr);
    auto new_dst = make_tensor(dst.layout.dtype, dst, workspace_ptr);

#define CASE(mode)                                                                   \
    case mode: {                                                                     \
        copy_to(src, new_src);                                                       \
        ::megdnn::naive::HandleImpl* handlePtr = static_cast<HandleImpl*>(handle()); \
        MEGDNN_DISPATCH_CPU_KERN(                                                    \
                handlePtr, dispatch_mode<mode>(new_src, new_dst, A, B, C));          \
        copy_to(new_dst, dst);                                                       \
        break;                                                                       \
    };
    switch (param().mode) {
        CASE(Mode::P_NORM)
        CASE(Mode::INF_NORM)
        CASE(Mode::NEG_INF_NORM)
        default:
            megdnn_assert_internal(false);
    }
#undef CASE
}

size_t NormForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    MEGDNN_MARK_USED_VAR(src);
    MEGDNN_MARK_USED_VAR(dst);
    return 0;
}

}  // namespace naive
}  // namespace megdnn
