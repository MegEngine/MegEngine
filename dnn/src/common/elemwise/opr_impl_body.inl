/**
 * \file dnn/src/common/elemwise/opr_impl_body.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef on_arity_dispatched_cb_dtype
#error "on_arity_dispatched_cb_dtype and IMPL_MODE_DISPATCHER must be defined"
#endif

template<int arity>
void ElemwiseForwardImpl::on_arity_dispatched() {
    auto src = make_elemwise_op_param<arity>();
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(on_arity_dispatched_cb_dtype)
    MEGDNN_FOREACH_COMPUTING_DTYPE_INT(on_arity_dispatched_cb_dtype)
    on_arity_dispatched_cb_dtype(::megdnn::dtype::Bool)
    megdnn_throw("bad dtype");
}

template<int arity>
void ElemwiseForwardImpl::on_arity_dispatched_no_bool() {
    auto src = make_elemwise_op_param<arity>();
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(on_arity_dispatched_cb_dtype)
    MEGDNN_FOREACH_COMPUTING_DTYPE_INT(on_arity_dispatched_cb_dtype)
    megdnn_throw("bad dtype");
}

#define FOREACH MEGDNN_FOREACH_ELEMWISE_MODE_UNARY_INT
IMPL_MODE_DISPATCHER(1, DTypeCategory::INT);
#undef FOREACH

#define FOREACH MEGDNN_FOREACH_ELEMWISE_MODE_BINARY_INT
IMPL_MODE_DISPATCHER(2, DTypeCategory::INT);
#undef FOREACH

#define FOREACH MEGDNN_FOREACH_ELEMWISE_MODE_TERNARY_INT
IMPL_MODE_DISPATCHER(3, DTypeCategory::INT);
#undef FOREACH

#define FOREACH MEGDNN_FOREACH_ELEMWISE_MODE_UNARY_FLOAT
IMPL_MODE_DISPATCHER(1, DTypeCategory::FLOAT);
#undef FOREACH

#define FOREACH MEGDNN_FOREACH_ELEMWISE_MODE_BINARY_FLOAT
IMPL_MODE_DISPATCHER(2, DTypeCategory::FLOAT);
#undef FOREACH

#define FOREACH MEGDNN_FOREACH_ELEMWISE_MODE_TERNARY_FLOAT
IMPL_MODE_DISPATCHER(3, DTypeCategory::FLOAT);
#undef FOREACH

#define FOREACH MEGDNN_FOREACH_ELEMWISE_MODE_UNARY_BOOL
IMPL_MODE_DISPATCHER(1, DTypeCategory::BOOL);
#undef FOREACH

#define FOREACH MEGDNN_FOREACH_ELEMWISE_MODE_BINARY_BOOL
IMPL_MODE_DISPATCHER(2, DTypeCategory::BOOL);
#undef FOREACH

void ElemwiseForwardImpl::exec(
        const TensorNDArray &src,
        _megdnn_tensor_out dst) {
    m_src = &src;
    m_dst = &dst;

#define CB_CHK_MODE_ENABLE(_) 1
    if (m_param.mode == Mode::FUSE_MUL_ADD3) {
#if MEGDNN_ELEMWISE_MODE_ENABLE(FUSE_MUL_ADD3, CB_CHK_MODE_ENABLE) +0
        ElemwiseOpParamN<3> param;
        bool c_is_scalar;
        prepare_fma3(param, c_is_scalar);
        switch(m_dst->layout.dtype.enumv()) {
#define cb(_dt) \
            case DTypeTrait<_dt>::enumv: \
            { \
                using ctype = DTypeTrait<_dt>::ctype; \
                if (c_is_scalar) { \
                    return impl_fuse_mul_add3<ctype, true>(param); \
                } else { \
                    return impl_fuse_mul_add3<ctype, false>(param); \
                } \
            }
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
            default:
                megdnn_throw("bad dtype");
        }
#endif	// enable FUSE_MUL_ADD3
    } else if (m_param.mode == Mode::FUSE_MUL_ADD4) {
#if MEGDNN_ELEMWISE_MODE_ENABLE(FUSE_MUL_ADD4, CB_CHK_MODE_ENABLE) +0
        ElemwiseOpParamN<4> param;
        prepare_fma4(param);

        switch(m_dst->layout.dtype.enumv()) {
#define cb(_dt) \
            case DTypeTrait<_dt>::enumv: \
                return impl_fuse_mul_add4<DTypeTrait<_dt>::ctype>(param);
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
            default:
                megdnn_throw("bad dtype");
        }
#endif	// enable FUSE_MUL_ADD4
    }

#undef CB_CHK_MODE_ENABLE

    switch(src.size()) {
#define D(_n) case _n: return on_arity_dispatched<_n>()
        D(1);
        D(2);
#undef D
        case 3: return on_arity_dispatched_no_bool<3>();
        default:
            megdnn_throw("bad size of input tensors");
    }
}

// vim: syntax=cpp.doxygen
