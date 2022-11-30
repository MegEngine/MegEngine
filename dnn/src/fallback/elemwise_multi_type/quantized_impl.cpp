#include "megdnn/tensor_iter.h"
#include "src/fallback/elemwise_helper/elemwise_op.h"
#include "src/fallback/elemwise_multi_type/opr_impl.h"
#include "src/naive/handle.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fallback_elemwise_multi_type_quantized)

using namespace megdnn;
using namespace fallback;
using namespace elemwise;

void ElemwiseMultiTypeImpl::on_quantized_mode(
        const ElemwiseOpParamN<1>& param, const TensorND& dst, Elemwise::Mode mode) {
    megdnn_assert(param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

#define DISPATCH_MODE(_src_dt, _dst_dt)                                           \
    switch (mode) {                                                               \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::RELU, ReluOp)      \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::H_SWISH, HSwishOp) \
        default:                                                                  \
            break;                                                                \
    }

#define DISPATCH_QUANTIZED_MODE(_src_dt, _dst_dt)                                     \
    switch (mode) {                                                                   \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::RELU, ReluOp)          \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::ABS, AbsOp)            \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::SIGMOID, SigmoidOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::EXP, ExpOp)            \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::TANH, TanhOp)          \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FAST_TANH, FastTanhOp) \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::H_SWISH, HSwishOp)     \
        default:                                                                      \
            break;                                                                    \
    }

#define DISPATCH()                                                      \
    if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8 &&      \
        dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {           \
        DISPATCH_QUANTIZED_MODE(dtype::QuantizedS8, dtype::QuantizedS8) \
    } else if (                                                         \
            param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32 && \
            dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {       \
        DISPATCH_MODE(dtype::QuantizedS32, dtype::QuantizedS8)          \
    }

    TensorND src = param[0];

    size_t nr_elems = src.layout.total_nr_elems();
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                            \
    case _mode: {                                                                     \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                        \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                        \
        thin_function<void(const src_ctype*, dst_ctype*, DType, DType, size_t)> run = \
                OpCallerUnary<_op<src_ctype, dst_ctype>, VEC>::run;                   \
        MIDOUT_BEGIN(                                                                 \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(0),          \
                src_ctype, dst_ctype, midout_iv(_mode)) {                             \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                             \
                    run(src.ptr<src_ctype>(), dst.ptr<dst_ctype>(), src.layout.dtype, \
                        dst.layout.dtype, nr_elems));                                 \
            return;                                                                   \
        }                                                                             \
        MIDOUT_END();                                                                 \
    }

    DISPATCH()

    naive::ElemwiseMultiTypeImpl::on_quantized_mode(param, dst, mode);

#undef DISPATCH_SINGLE_MODE
#undef DISPATCH
#undef DISPATCH_QUANTIZED_MODE
#undef DISPATCH_MODE
}

void ElemwiseMultiTypeImpl::on_quantized_mode(
        const ElemwiseOpParamN<2>& param, const TensorND& dst, Elemwise::Mode mode) {
    megdnn_assert(
            param[0].layout.dtype.enumv() == param[1].layout.dtype.enumv() &&
            param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

#define DISPATCH_MODE(_src_dt, _dst_dt)                                              \
    switch (mode) {                                                                  \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::ADD, AddOp)           \
        DISPATCH_SINGLE_MODE(                                                        \
                _src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_RELU, FuseAddReluOp)      \
        DISPATCH_SINGLE_MODE(                                                        \
                _src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_H_SWISH, FuseAddHSwishOp) \
        default:                                                                     \
            break;                                                                   \
    }

#define DISPATCH_QUANTIZED_MODE(_src_dt, _dst_dt)                                     \
    switch (mode) {                                                                   \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::ADD, AddOp)            \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MIN, MinOp)            \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MAX, MaxOp)            \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::SUB, SubOp)            \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MUL, MulOp)            \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::TRUE_DIV, TrueDivOp)   \
        DISPATCH_SINGLE_MODE(                                                         \
                _src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_RELU, FuseAddReluOp)       \
        DISPATCH_SINGLE_MODE(                                                         \
                _src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_SIGMOID, FuseAddSigmoidOp) \
        DISPATCH_SINGLE_MODE(                                                         \
                _src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_TANH, FuseAddTanhOp)       \
        DISPATCH_SINGLE_MODE(                                                         \
                _src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_H_SWISH, FuseAddHSwishOp)  \
        default:                                                                      \
            break;                                                                    \
    }

#define DISPATCH()                                                      \
    if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&     \
        dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {           \
        DISPATCH_MODE(dtype::QuantizedS32, dtype::QuantizedS8)          \
    } else if (                                                         \
            param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8 &&  \
            dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {       \
        DISPATCH_QUANTIZED_MODE(dtype::QuantizedS8, dtype::QuantizedS8) \
    }

    TensorND src0 = param[0];
    TensorND src1 = param[1];

    //! VEC + VEC
    if (is_vector(src0.layout) && is_vector(src1.layout)) {
        size_t nr_elems = src0.layout.total_nr_elems();
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                           \
    case _mode: {                                                                    \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                       \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                       \
        thin_function<void(                                                          \
                const src_ctype*, const src_ctype*, dst_ctype*, DType, DType, DType, \
                size_t)>                                                             \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>, VEC_VEC>::run;       \
        MIDOUT_BEGIN(                                                                \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(1),         \
                src_ctype, dst_ctype, midout_iv(_mode)) {                            \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                            \
                    run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),                \
                        dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype,  \
                        dst.layout.dtype, nr_elems));                                \
            return;                                                                  \
        }                                                                            \
        MIDOUT_END();                                                                \
    }

        DISPATCH()

#undef DISPATCH_SINGLE_MODE
    }

    //! VEC + SCALAR
    {
        bool normal_case = is_vector(src0.layout) && is_broadcasted_scalar(src1.layout);
        bool swap_case = false;
        bool commutable = false;
        if (mode != Elemwise::Mode::SUB && mode != Elemwise::Mode::TRUE_DIV)
            commutable = true;
        if (!normal_case && commutable) {
            swap_case = is_vector(src1.layout) && is_broadcasted_scalar(src0.layout);
        }
        if (normal_case || swap_case) {
            auto &lhs = src0, &rhs = src1;
            if (swap_case) {
                std::swap(lhs, rhs);
            }
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                          \
    case _mode: {                                                                   \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                      \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                      \
        thin_function<void(                                                         \
                const src_ctype*, const src_ctype, dst_ctype*, DType, DType, DType, \
                size_t)>                                                            \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>, VEC_SCALAR>::run;   \
        MIDOUT_BEGIN(                                                               \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(2),        \
                src_ctype, dst_ctype, midout_iv(_mode)) {                           \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                           \
                    run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>()[0],            \
                        dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                        dst.layout.dtype, src0.layout.total_nr_elems()));           \
            return;                                                                 \
        }                                                                           \
        MIDOUT_END();                                                               \
    }

            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }

        //! SCALAR + VEC
        if (!commutable && is_vector(src1.layout) &&
            is_broadcasted_scalar(src0.layout)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                          \
    case _mode: {                                                                   \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                      \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                      \
        thin_function<void(                                                         \
                const src_ctype, const src_ctype*, dst_ctype*, DType, DType, DType, \
                size_t)>                                                            \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>, SCALAR_VEC>::run;   \
        MIDOUT_BEGIN(                                                               \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(3),        \
                src_ctype, dst_ctype, midout_iv(_mode)) {                           \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                           \
                    run(src0.ptr<src_ctype>()[0], src1.ptr<src_ctype>(),            \
                        dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                        dst.layout.dtype, src1.layout.total_nr_elems()));           \
            return;                                                                 \
        }                                                                           \
        MIDOUT_END();                                                               \
    }

            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }
    }

    //! VEC + BCAST101
    {
        BroadcastChannelInfo binfo;
        bool normal_case = is_vector(src0.layout) &&
                           is_broadcasted_channel_like(src1.layout, binfo);
        bool swap_case = false;
        bool commutable = false;
        if (mode != Elemwise::Mode::SUB && mode != Elemwise::Mode::TRUE_DIV)
            commutable = true;
        if (!normal_case && commutable) {
            swap_case = is_vector(src1.layout) &&
                        is_broadcasted_channel_like(src0.layout, binfo);
        }
        if (normal_case || swap_case) {
            auto &lhs = src0, &rhs = src1;
            if (swap_case)
                std::swap(lhs, rhs);
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                           \
    case _mode: {                                                                    \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                       \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                       \
        thin_function<void(                                                          \
                const src_ctype*, const src_ctype*, dst_ctype*, DType, DType, DType, \
                size_t, size_t, size_t)>                                             \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>, VEC_BCAST101>::run;  \
        MIDOUT_BEGIN(                                                                \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(4),         \
                src_ctype, dst_ctype, midout_iv(_mode)) {                            \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                            \
                    run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),                \
                        dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype,  \
                        dst.layout.dtype, binfo.x, binfo.y, binfo.z));               \
            return;                                                                  \
        }                                                                            \
        MIDOUT_END();                                                                \
    }

            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }

        //! BCAST101 + VEC : only for SUB or TRUE_DIV
        if (!commutable && is_vector(src1.layout) &&
            is_broadcasted_channel_like(src0.layout, binfo)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                           \
    case _mode: {                                                                    \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                       \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                       \
        thin_function<void(                                                          \
                const src_ctype*, const src_ctype*, dst_ctype*, DType, DType, DType, \
                size_t, size_t, size_t)>                                             \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>, BCAST101_VEC>::run;  \
        MIDOUT_BEGIN(                                                                \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(5),         \
                src_ctype, dst_ctype, midout_iv(_mode)) {                            \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                            \
                    run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),                \
                        dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype,  \
                        dst.layout.dtype, binfo.x, binfo.y, binfo.z));               \
            return;                                                                  \
        }                                                                            \
        MIDOUT_END();                                                                \
    }

            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }
    }

    //! VEC + BCAST101x4
    {
        BroadcastChannelInfo binfo;
        if (is_vector(src0.layout) &&
            (is_broadcastedx_channel_like<4>(src1.layout, binfo) ||
             is_broadcastedx_channel_like<8>(src1.layout, binfo))) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                            \
    case _mode: {                                                                     \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                        \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                        \
        thin_function<void(                                                           \
                const src_ctype*, const src_ctype*, dst_ctype*, DType, DType, DType,  \
                size_t, size_t, size_t, size_t)>                                      \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>, VEC_BCAST101xX>::run; \
        MIDOUT_BEGIN(                                                                 \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(6), _src_dt, \
                _dst_dt, midout_iv(_mode)) {                                          \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                             \
                    run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),                 \
                        dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype,   \
                        dst.layout.dtype, batch_size, binfo.x, binfo.y, binfo.z));    \
            return;                                                                   \
        }                                                                             \
        MIDOUT_END();                                                                 \
    }
            size_t batch_size = src0.layout.shape[0] / (binfo.x * binfo.y * binfo.z);
            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }

        //! BCAST101x + VEC
        if (is_vector(src1.layout) &&
            is_broadcastedx_channel_like<4>(src0.layout, binfo)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                            \
    case _mode: {                                                                     \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                        \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                        \
        thin_function<void(                                                           \
                const src_ctype*, const src_ctype*, dst_ctype*, DType, DType, DType,  \
                size_t, size_t, size_t, size_t)>                                      \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>, BCAST101xX_VEC>::run; \
        MIDOUT_BEGIN(                                                                 \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(7),          \
                src_ctype, dst_ctype, midout_iv(_mode)) {                             \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                             \
                    run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),                 \
                        dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype,   \
                        dst.layout.dtype, batch_size, binfo.x, binfo.y, binfo.z));    \
            return;                                                                   \
        }                                                                             \
        MIDOUT_END();                                                                 \
    }
            size_t batch_size = src1.layout.shape[0] / (binfo.x * binfo.y * binfo.z);
            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }
    }

    naive::ElemwiseMultiTypeImpl::on_quantized_mode(param, dst, mode);

#undef DISPATCH_MODE
#undef DISPATCH_QUANTIZED_MODE
#undef DISPATCH
}

void ElemwiseMultiTypeImpl::on_quantized_mode(
        const ElemwiseOpParamN<3>& param, const TensorND& dst, Elemwise::Mode mode) {
    megdnn_assert(
            param[0].layout.dtype.enumv() == param[1].layout.dtype.enumv() &&
            param[0].layout.dtype.enumv() == param[2].layout.dtype.enumv() &&
            param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

#define DISPATCH_QUANTIZED_MODE(_src_dt, _dst_dt)                               \
    switch (mode) {                                                             \
        DISPATCH_SINGLE_MODE(                                                   \
                _src_dt, _dst_dt, Elemwise::Mode::FUSE_MUL_ADD3, FuseMulAdd3Op) \
        default:                                                                \
            break;                                                              \
    }

#define DISPATCH()                                                      \
    if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8 &&      \
        dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {           \
        DISPATCH_QUANTIZED_MODE(dtype::QuantizedS8, dtype::QuantizedS8) \
    }

    TensorND src0 = param[0];
    TensorND src1 = param[1];
    TensorND src2 = param[2];

    //! VEC + VEC + VEC
    if (is_vector(src0.layout) && is_vector(src1.layout) && is_vector(src2.layout)) {
        size_t nr_elems = src0.layout.total_nr_elems();
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                          \
    case _mode: {                                                                   \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                      \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                      \
        thin_function<void(                                                         \
                const src_ctype*, const src_ctype*, const src_ctype*, dst_ctype*,   \
                DType, DType, DType, DType, size_t)>                                \
                run = OpCallerTernary<_op<src_ctype, dst_ctype>, VEC_VEC_VEC>::run; \
        MIDOUT_BEGIN(                                                               \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(8),        \
                src_ctype, dst_ctype, midout_iv(_mode)) {                           \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                           \
                    run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),               \
                        src2.ptr<src_ctype>(), dst.ptr<dst_ctype>(),                \
                        src0.layout.dtype, src1.layout.dtype, src2.layout.dtype,    \
                        dst.layout.dtype, nr_elems));                               \
            return;                                                                 \
        }                                                                           \
        MIDOUT_END();                                                               \
    }

        DISPATCH()

#undef DISPATCH_SINGLE_MODE
    }

    //! VEC + VEC + SCALAR
    if (is_vector(src0.layout) && is_vector(src1.layout) &&
        is_broadcasted_scalar(src2.layout)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                             \
    case _mode: {                                                                      \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                         \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                         \
        thin_function<void(                                                            \
                const src_ctype*, const src_ctype*, const src_ctype, dst_ctype*,       \
                DType, DType, DType, DType, size_t)>                                   \
                run = OpCallerTernary<_op<src_ctype, dst_ctype>, VEC_VEC_SCALAR>::run; \
        MIDOUT_BEGIN(                                                                  \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(9),           \
                src_ctype, dst_ctype, midout_iv(_mode)) {                              \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                              \
                    run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),                  \
                        src2.ptr<src_ctype>()[0], dst.ptr<dst_ctype>(),                \
                        src0.layout.dtype, src1.layout.dtype, src2.layout.dtype,       \
                        dst.layout.dtype, src0.layout.total_nr_elems()));              \
            return;                                                                    \
        }                                                                              \
        MIDOUT_END();                                                                  \
    }

        DISPATCH()

#undef DISPATCH_SINGLE_MODE
    }

    //! BCAST101 + VEC + BCAST101
    {
        BroadcastChannelInfo binfo;
        bool normal_case = is_vector(src1.layout) &&
                           is_broadcasted_channel_like(src0.layout, binfo) &&
                           src0.layout.eq_shape(src2.layout);
        if (normal_case) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                           \
    case _mode: {                                                                    \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                       \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                       \
        thin_function<void(                                                          \
                const src_ctype*, const src_ctype*, const src_ctype*, dst_ctype*,    \
                DType, DType, DType, DType, size_t, size_t, size_t, size_t)>         \
                run = OpCallerTernary<                                               \
                        _op<src_ctype, dst_ctype>, BCAST101_VEC_BCAST101>::run;      \
        MIDOUT_BEGIN(                                                                \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(10),        \
                src_ctype, dst_ctype, midout_iv(_mode)) {                            \
            MEGDNN_DISPATCH_CPU_KERN_OPR(run(                                        \
                    src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),                    \
                    src2.ptr<src_ctype>(), dst.ptr<dst_ctype>(), src0.layout.dtype,  \
                    src1.layout.dtype, src2.layout.dtype, dst.layout.dtype, binfo.x, \
                    binfo.y, binfo.z, binfo.y* binfo.z));                            \
            return;                                                                  \
        }                                                                            \
        MIDOUT_END();                                                                \
    }

            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }
    }

    //! VEC + BCAST101x4 + VEC
    {
        BroadcastChannelInfo binfo;
        if (is_vector(src0.layout) &&
            (is_broadcastedx_channel_like<4>(src1.layout, binfo) ||
             is_broadcastedx_channel_like<8>(src1.layout, binfo)) &&
            src0.layout.eq_shape(src2.layout)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                         \
    case _mode: {                                                                  \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                     \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                     \
        thin_function<void(                                                        \
                const src_ctype*, const src_ctype*, const src_ctype*, dst_ctype*,  \
                DType, DType, DType, DType, size_t, size_t, size_t, size_t)>       \
                run = OpCallerTernary<                                             \
                        _op<src_ctype, dst_ctype>, VEC_BCAST101xX_VEC>::run;       \
        MIDOUT_BEGIN(                                                              \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(11),      \
                src_ctype, dst_ctype, midout_iv(_mode)) {                          \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                          \
                    run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),              \
                        src2.ptr<src_ctype>(), dst.ptr<dst_ctype>(),               \
                        src0.layout.dtype, src1.layout.dtype, src2.layout.dtype,   \
                        dst.layout.dtype, batch_size, binfo.x, binfo.y, binfo.z)); \
            return;                                                                \
        }                                                                          \
        MIDOUT_END();                                                              \
    }

            size_t batch_size = src0.layout.shape[0] / (binfo.x * binfo.y * binfo.z);
            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }

        //! BCAST101x + VEC +BCAST101x
        if (is_vector(src1.layout) &&
            (is_broadcastedx_channel_like<4>(src0.layout, binfo) ||
             is_broadcastedx_channel_like<8>(src0.layout, binfo)) &&
            src0.layout.eq_shape(src2.layout)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                          \
    case _mode: {                                                                   \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                      \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                      \
        thin_function<void(                                                         \
                const src_ctype*, const src_ctype*, const src_ctype*, dst_ctype*,   \
                DType, DType, DType, DType, size_t, size_t, size_t, size_t)>        \
                run = OpCallerTernary<                                              \
                        _op<src_ctype, dst_ctype>, BCAST101xX_VEC_BCAST101xX>::run; \
        MIDOUT_BEGIN(                                                               \
                megdnn_fallback_elemwise_multi_type_quantized, midout_iv(12),       \
                src_ctype, dst_ctype, midout_iv(_mode)) {                           \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                           \
                    run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),               \
                        src2.ptr<src_ctype>(), dst.ptr<dst_ctype>(),                \
                        src0.layout.dtype, src1.layout.dtype, src2.layout.dtype,    \
                        dst.layout.dtype, batch_size, binfo.x, binfo.y, binfo.z));  \
            return;                                                                 \
        }                                                                           \
        MIDOUT_END();                                                               \
    }

            size_t batch_size = src1.layout.shape[0] / (binfo.x * binfo.y * binfo.z);
            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }
    }

    naive::ElemwiseMultiTypeImpl::on_quantized_mode(param, dst, mode);
#undef DISPATCH
#undef DISPATCH_QUANTIZED_MODE
}

// vim: syntax=cpp.doxygen
