#include "src/fallback/reduce/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "midout.h"
#include "reducer.h"
#include "src/common/reduce_helper.h"

MIDOUT_DECL(megdnn_fb_reduce_op)
MIDOUT_DECL(megdnn_fb_reduce_c)
MIDOUT_DECL(megdnn_fb_reduce_dtype)
MIDOUT_DECL(megdnn_fallback_reduce_optimized)

namespace {

using namespace megdnn;

template <typename Op>
void reduce_exec_C1(size_t A, size_t B, Op op) MEGDNN_NOEXCEPT {
    using wtype = typename Op::wtype;
    rep(a, A) {
        std::function<wtype(size_t, size_t)> func;
        func = [&func, B, &op, a](size_t bl, size_t br) -> wtype {
            if (bl + 4096 < br) {
                size_t mid = bl + (br - bl) / 2;
                return op.apply(func(bl, mid), func(mid, br));
            } else {
                wtype res = op.INIT;
                for (size_t b = bl; b < br; ++b) {
                    res = op.apply(res, op.read(a * B + b));
                }
                return res;
            }
        };
        wtype res = func(0, B);
        op.write(a, res);
    }
}

template <typename Op>
void reduce_exec(size_t A, size_t B, size_t C, Op op) MEGDNN_NOEXCEPT {
    using wtype = typename Op::wtype;
    rep(a, A) {
        rep(c, C) {
            std::function<wtype(size_t, size_t)> func;
            func = [&func, B, C, &op, a, c](size_t bl, size_t br) -> wtype {
                if (bl + 4096 < br) {
                    size_t mid = bl + (br - bl) / 2;
                    return op.apply(func(bl, mid), func(mid, br));
                } else {
                    wtype res = op.INIT;
                    for (size_t b = bl; b < br; ++b) {
                        res = op.apply(res, op.read(a * B * C + b * C + c));
                    }
                    return res;
                }
            };
            wtype res = func(0, B);
            op.write(a * C + c, res);
        }
    }
}

}  // anonymous namespace

namespace megdnn {
namespace fallback {

void ReduceImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    if (!exec_optimized(src, dst, workspace)) {
        return exec_fallback(src, dst, workspace);
    }
}

void ReduceImpl::exec_fallback(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    using namespace reduce;
    using Mode = Param::Mode;
    check_exec(src.layout, dst.layout, workspace.size);
    size_t A, B, C;
    get_ABC(src.layout, A, B, C, param().axis);

#define cb_by_op(src_type, dst_type, _wtype, mode_, Op_, kern_func)                   \
    if (param().mode == mode_) {                                                      \
        typedef DTypeTrait<src_type>::ctype src_ctype;                                \
        typedef DTypeTrait<dst_type>::ctype dst_ctype;                                \
        typedef DTypeTrait<_wtype>::ctype wtype;                                      \
        Op_<src_ctype, dst_ctype, wtype> op(src.get_ref_ptr(), dst.get_ref_ptr(), B); \
        MEGDNN_DISPATCH_CPU_KERN_OPR({ kern_func; });                                 \
        return;                                                                       \
    }
#define cb_by_dtype(dtype_, kern_func, type_tuple)                    \
    if (dtype_() == src.layout.dtype) {                               \
        MIDOUT_BEGIN(megdnn_fb_reduce_op, midout_iv(0)) {             \
            cb_by_op(type_tuple, Mode::SUM, SumOp, kern_func);        \
        }                                                             \
        MIDOUT_END();                                                 \
        MIDOUT_BEGIN(megdnn_fb_reduce_op, midout_iv(1)) {             \
            cb_by_op(type_tuple, Mode::SUM_SQR, SumSqrOp, kern_func); \
        }                                                             \
        MIDOUT_END();                                                 \
        MIDOUT_BEGIN(megdnn_fb_reduce_op, midout_iv(2)) {             \
            cb_by_op(type_tuple, Mode::PRODUCT, ProdOp, kern_func);   \
        }                                                             \
        MIDOUT_END();                                                 \
        MIDOUT_BEGIN(megdnn_fb_reduce_op, midout_iv(3)) {             \
            cb_by_op(type_tuple, Mode::MIN, MinOp, kern_func);        \
        }                                                             \
        MIDOUT_END();                                                 \
        MIDOUT_BEGIN(megdnn_fb_reduce_op, midout_iv(4)) {             \
            cb_by_op(type_tuple, Mode::MAX, MaxOp, kern_func);        \
        }                                                             \
        MIDOUT_END();                                                 \
        MIDOUT_BEGIN(megdnn_fb_reduce_op, midout_iv(5)) {             \
            cb_by_op(type_tuple, Mode::MEAN, MeanOp, kern_func);      \
        }                                                             \
        MIDOUT_END();                                                 \
    }

#if !MEGDNN_DISABLE_FLOAT16
#define cb_by_data_type(dtype_, data_type, kern_func)                             \
    if (data_type == DataType::FLOAT_O16xC32) {                                   \
        MIDOUT_BEGIN(megdnn_fb_reduce_dtype, midout_iv(0)){cb_by_dtype(           \
                dtype_, kern_func,                                                \
                dtype_ MEGDNN_COMMA dt_float16 MEGDNN_COMMA float)} MIDOUT_END(); \
    }                                                                             \
    if (data_type == DataType::FLOAT_O32xC32) {                                   \
        MIDOUT_BEGIN(megdnn_fb_reduce_dtype, midout_iv(1)){cb_by_dtype(           \
                dtype_, kern_func,                                                \
                dtype_ MEGDNN_COMMA float MEGDNN_COMMA float)} MIDOUT_END();      \
    }                                                                             \
    if (data_type == DataType::DEFAULT) {                                         \
        MIDOUT_BEGIN(megdnn_fb_reduce_dtype, midout_iv(2)){cb_by_dtype(           \
                dtype_, kern_func,                                                \
                dtype_ MEGDNN_COMMA dtype_ MEGDNN_COMMA dtype_)} MIDOUT_END();    \
    }

#else

#define cb_by_data_type(dtype_, data_type, kern_func)                          \
    if (data_type == DataType::FLOAT_O32xC32) {                                \
        MIDOUT_BEGIN(megdnn_fb_reduce_dtype, midout_iv(0)){cb_by_dtype(        \
                dtype_, kern_func,                                             \
                dtype_ MEGDNN_COMMA float MEGDNN_COMMA float)} MIDOUT_END();   \
    }                                                                          \
    if (data_type == DataType::DEFAULT) {                                      \
        MIDOUT_BEGIN(megdnn_fb_reduce_dtype, midout_iv(1)){cb_by_dtype(        \
                dtype_, kern_func,                                             \
                dtype_ MEGDNN_COMMA dtype_ MEGDNN_COMMA dtype_)} MIDOUT_END(); \
    }
#endif

#define cb_by_c(dtype_, C)                                                       \
    if (C == 1) {                                                                \
        MIDOUT_BEGIN(megdnn_fb_reduce_c, midout_iv(0)){cb_by_data_type(          \
                dtype_, param().data_type,                                       \
                reduce_exec_C1(A MEGDNN_COMMA B MEGDNN_COMMA op))} MIDOUT_END(); \
    } else {                                                                     \
        MIDOUT_BEGIN(megdnn_fb_reduce_c, midout_iv(1)){cb_by_data_type(          \
                dtype_, param().data_type,                                       \
                reduce_exec(A MEGDNN_COMMA B MEGDNN_COMMA C MEGDNN_COMMA         \
                                    op))} MIDOUT_END();                          \
    }

#define cb_all(dtype_) cb_by_c(dtype_, C)

    MEGDNN_FOREACH_COMPUTING_DTYPE(cb_all);

#undef cb_all
#undef cb_by_c
#undef cb_by_data_type
#undef cb_by_op

    naive::ReduceForwardImpl::exec(src, dst, workspace);
}

bool ReduceImpl::exec_optimized(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace) {
    size_t A, B, C;
    reduce::get_ABC(src.layout, A, B, C, param().axis);
    bool execed = false;
    using Mode = param::Reduce::Mode;
#define DISPATCH_FUNC(Reducer, dtype, ctype, comp_type)                           \
    if (C == 1) {                                                                 \
        using _Reducer = Reducer<dtype, ctype, comp_type, true>;                  \
        using _ReducerC1SmallB = Reducer<dtype, ctype, comp_type, false>;         \
        std::function<void(const ctype*, ctype*, DType, size_t, size_t, size_t)>  \
                do_reduce = Exec<_Reducer, true>::do_reduce;                      \
        if (B == 2)                                                               \
            do_reduce = ExecC1SmallB<_ReducerC1SmallB, ctype, 2>::do_reduce;      \
        if (B == 3)                                                               \
            do_reduce = ExecC1SmallB<_ReducerC1SmallB, ctype, 3>::do_reduce;      \
        if (B == 4)                                                               \
            do_reduce = ExecC1SmallB<_ReducerC1SmallB, ctype, 4>::do_reduce;      \
        MIDOUT_BEGIN(                                                             \
                megdnn_fallback_reduce_optimized, ctype, dtype, comp_type,        \
                midout_iv(0)) {                                                   \
            MEGDNN_DISPATCH_CPU_KERN_OPR(do_reduce(                               \
                    reinterpret_cast<ctype*>(src.raw_ptr()),                      \
                    reinterpret_cast<ctype*>(dst.raw_ptr()), src_type, A, B, C)); \
            execed = true;                                                        \
        }                                                                         \
        MIDOUT_END();                                                             \
    } else {                                                                      \
        using _Reducer = Reducer<dtype, ctype, comp_type, false>;                 \
        std::function<void(const ctype*, ctype*, DType, size_t, size_t, size_t)>  \
                do_reduce = Exec<_Reducer, false>::do_reduce;                     \
        MIDOUT_BEGIN(                                                             \
                megdnn_fallback_reduce_optimized, ctype, dtype, comp_type,        \
                midout_iv(1)) {                                                   \
            MEGDNN_DISPATCH_CPU_KERN_OPR(do_reduce(                               \
                    reinterpret_cast<ctype*>(src.raw_ptr()),                      \
                    reinterpret_cast<ctype*>(dst.raw_ptr()), src_type, A, B, C)); \
            execed = true;                                                        \
        }                                                                         \
        MIDOUT_END();                                                             \
    }

#define DISPATCH_MODE_QUANTIZED(dtype, ctype, comp_type)         \
    switch (param().mode) {                                      \
        case Mode::MEAN:                                         \
            DISPATCH_FUNC(MeanReducer, dtype, ctype, comp_type); \
            break;                                               \
        case Mode::MAX:                                          \
            DISPATCH_FUNC(maxReducer, dtype, ctype, ctype);      \
            break;                                               \
        case Mode::MIN:                                          \
            DISPATCH_FUNC(minReducer, dtype, ctype, ctype);      \
            break;                                               \
        default:                                                 \
            break;                                               \
    }

#define DISPATCH_MODE_FLOAT(dtype, ctype, comp_type)             \
    switch (param().mode) {                                      \
        case Mode::MEAN:                                         \
            DISPATCH_FUNC(MeanReducer, dtype, ctype, comp_type); \
            break;                                               \
        case Mode::MAX:                                          \
            DISPATCH_FUNC(maxReducer, dtype, ctype, ctype);      \
            break;                                               \
        case Mode::MIN:                                          \
            DISPATCH_FUNC(minReducer, dtype, ctype, ctype);      \
            break;                                               \
        case Mode::SUM:                                          \
            DISPATCH_FUNC(SumReducer, dtype, ctype, ctype);      \
            break;                                               \
        case Mode::SUM_SQR:                                      \
            DISPATCH_FUNC(SumSqrReducer, dtype, ctype, ctype);   \
            break;                                               \
        case Mode::PRODUCT:                                      \
            DISPATCH_FUNC(ProductReducer, dtype, ctype, ctype);  \
            break;                                               \
        default:                                                 \
            break;                                               \
    }
    if (src.layout.is_contiguous() &&
        src.layout.dtype.category() == DTypeCategory::QUANTIZED &&
        param().data_type == param::Reduce::DataType::DEFAULT) {
        DType src_type = src.layout.dtype;
        if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
            DISPATCH_MODE_QUANTIZED(dt_qint8, int8_t, int32_t)
        }
    } else if (
            src.layout.is_contiguous() &&
            src.layout.dtype.category() == DTypeCategory::FLOAT &&
            param().data_type == param::Reduce::DataType::DEFAULT) {
        DType src_type = src.layout.dtype;
        if (src.layout.dtype.enumv() == DTypeEnum::Float32) {
            DISPATCH_MODE_FLOAT(dt_float32, float, float)
        }
    }
    return execed;
#undef DISPATCH_FUNC
#undef DISPATCH_MODE_QUANTIZED
#undef DISPATCH_MODE_FLOAT
}

}  // namespace fallback
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
