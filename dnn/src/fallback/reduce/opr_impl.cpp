/**
 * \file dnn/src/fallback/reduce/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/reduce/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "midout.h"
#include "src/common/reduce_helper.h"

MIDOUT_DECL(megdnn_fb_reduce_op)
MIDOUT_DECL(megdnn_fb_reduce_c)
MIDOUT_DECL(megdnn_fb_reduce_dtype)

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

void ReduceImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) {
    using namespace reduce;
    using Mode = Param::Mode;
    check_exec(src.layout, dst.layout, workspace.size);
    size_t A, B, C;
    get_ABC(src.layout, A, B, C, param().axis);
#define cb_by_op(src_type, dst_type, _wtype, mode_, Op_, kern_func)   \
    if (param().mode == mode_) {                                      \
        typedef DTypeTrait<src_type>::ctype src_ctype;                \
        typedef DTypeTrait<dst_type>::ctype dst_ctype;                \
        typedef DTypeTrait<_wtype>::ctype wtype;                      \
        Op_<src_ctype, dst_ctype, wtype> op(src.ptr<src_ctype>(),     \
                                            dst.ptr<dst_ctype>(), B); \
        MEGDNN_DISPATCH_CPU_KERN_OPR(kern_func);                      \
        return;                                                       \
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
#define cb_by_data_type(dtype_, data_type, kern_func)                          \
    if (data_type == DataType::FLOAT_O16xC32) {                                \
        MIDOUT_BEGIN(megdnn_fb_reduce_dtype, midout_iv(0)){                    \
                cb_by_dtype(dtype_, kern_func,                                 \
                            dtype_ MEGDNN_COMMA dt_float16                     \
                                    MEGDNN_COMMA float)} MIDOUT_END();         \
    }                                                                          \
    if (data_type == DataType::FLOAT_O32xC32) {                                \
        MIDOUT_BEGIN(megdnn_fb_reduce_dtype, midout_iv(1)){cb_by_dtype(        \
                dtype_, kern_func,                                             \
                dtype_ MEGDNN_COMMA float MEGDNN_COMMA float)} MIDOUT_END();   \
    }                                                                          \
    if (data_type == DataType::DEFAULT) {                                      \
        MIDOUT_BEGIN(megdnn_fb_reduce_dtype, midout_iv(2)){cb_by_dtype(        \
                dtype_, kern_func,                                             \
                dtype_ MEGDNN_COMMA dtype_ MEGDNN_COMMA dtype_)} MIDOUT_END(); \
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

#define cb_by_c(dtype_, C)                                                \
    if (C == 1) {                                                         \
        MIDOUT_BEGIN(megdnn_fb_reduce_c, midout_iv(0)){cb_by_data_type(   \
                dtype_, param().data_type,                                \
                reduce_exec_C1(                                           \
                        A MEGDNN_COMMA B MEGDNN_COMMA op))} MIDOUT_END(); \
    } else {                                                              \
        MIDOUT_BEGIN(megdnn_fb_reduce_c, midout_iv(1)){cb_by_data_type(   \
                dtype_, param().data_type,                                \
                reduce_exec(A MEGDNN_COMMA B MEGDNN_COMMA C MEGDNN_COMMA  \
                                    op))} MIDOUT_END();                   \
    }

#define cb_all(dtype_) cb_by_c(dtype_, C)

    MEGDNN_FOREACH_COMPUTING_DTYPE(cb_all);

#undef cb_all
#undef cb_by_c
#undef cb_by_data_type
#undef cb_by_op

    naive::ReduceForwardImpl::exec(src, dst, workspace);
}

}  // namespace fallback
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
