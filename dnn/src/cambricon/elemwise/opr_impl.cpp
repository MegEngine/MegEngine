#include "opr_impl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_op_descriptor.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/cnnl_wrapper/cnnl_types.h"
#include "src/cambricon/utils.h"

using namespace megdnn;
using namespace cambricon;
using Mode = param::Elemwise::Mode;

namespace {

void check_unary(
        const TensorNDArray& src, _megdnn_tensor_out dst,
        const param::Elemwise::Mode& mode) {
    bool mode_ok_unary = mode == Mode::RELU || mode == Mode::EXP ||
                         mode == Mode::NEGATE || mode == Mode::LOG ||
                         mode == Mode::SIGMOID || mode == Mode::LOGSIGMOID ||
                         mode == Mode::SQRT || mode == Mode::NOT;
    megdnn_assert(mode_ok_unary, "Elemwise unsupport mode:%d", static_cast<int>(mode));
    megdnn_assert(
            src.at(0).layout.eq_shape(dst.layout) && dst.layout.ndim < 8,
            "Elemwise Tensor nDim mismatch with src : %zu -> dst : %zu",
            src.at(0).layout.ndim, dst.layout.ndim);
    auto dtype_src0 = src.at(0).layout.dtype.enumv();
    auto dtype_dest = dst.layout.dtype.enumv();
    megdnn_assert(
            dtype_src0 == dtype_dest, "Elemwise binary dtype mismatch : %d vs %d",
            static_cast<int>(dtype_src0), static_cast<int>(dtype_dest));
}
void exec_unary(
        HandleImpl* handle, const TensorND& src, _megdnn_tensor_out dst,
        const param::Elemwise::Mode& mode, const WorkspaceBundle& wk_bundle) {
    auto cnnl_handler = handle->cnnl_handle();
    auto dtype_dest = dst.layout.dtype.enumv();
    CnnlTensorDescriptor input_desc, output_desc;
    input_desc.set(src.layout);
    output_desc.set(dst.layout);
    switch (mode) {
        case Mode::RELU: {  // float or half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest),
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            CnnlActivationDescriptor relu_desc;
            relu_desc.set(
                    cnnlActivationMode_t::CNNL_ACTIVATION_RELU,
                    cnnlActivationPreference_t::CNNL_ACTIVATION_HIGH_PRECISION,
                    cnnlNanPropagation_t::CNNL_NOT_PROPAGATE_NAN, 1.0);
            cnnl_check(cnnlActivationForward(
                    cnnl_handler, relu_desc.desc(), nullptr, input_desc.desc(),
                    src.raw_ptr(), nullptr, output_desc.desc(), dst.raw_ptr()));
            break;
        }
        case Mode::EXP: {  // float or half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest),
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            cnnl_check(cnnlExp_v2(
                    cnnl_handler,
                    cnnlComputationPreference_t::CNNL_COMPUTATION_HIGH_PRECISION,
                    input_desc.desc(), src.raw_ptr(), output_desc.desc(),
                    dst.raw_ptr()));
            break;
        }
        case Mode::LOG: {  // float or half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest),
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            cnnl_check(cnnlLog_v2(
                    cnnl_handler,
                    cnnlComputationPreference_t::CNNL_COMPUTATION_HIGH_PRECISION,
                    cnnlLogBase_t::CNNL_LOG_E, input_desc.desc(), src.raw_ptr(),
                    output_desc.desc(), dst.raw_ptr()));
            break;
        }
        case Mode::NEGATE: {  // int32, float or half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            cnnl_check(cnnlNegTensor(
                    cnnl_handler, input_desc.desc(), src.raw_ptr(), output_desc.desc(),
                    dst.raw_ptr()));
            break;
        }
        case Mode::SIGMOID: {  // float or half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest),
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            CnnlActivationDescriptor sigmoid_desc;
            sigmoid_desc.set(
                    cnnlActivationMode_t::CNNL_ACTIVATION_SIGMOID,
                    cnnlActivationPreference_t::CNNL_ACTIVATION_HIGH_PRECISION,
                    cnnlNanPropagation_t::CNNL_NOT_PROPAGATE_NAN, 1.0);
            cnnl_check(cnnlActivationForward(
                    cnnl_handler, sigmoid_desc.desc(), nullptr, input_desc.desc(),
                    src.raw_ptr(), nullptr, output_desc.desc(), dst.raw_ptr()));
            break;
        }
        case Mode::LOGSIGMOID: {  // float or half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest),
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            CnnlActivationDescriptor logsigmoid_desc;
            logsigmoid_desc.set(
                    cnnlActivationMode_t::CNNL_ACTIVATION_LOGSIGMOID,
                    cnnlActivationPreference_t::CNNL_ACTIVATION_HIGH_PRECISION,
                    cnnlNanPropagation_t::CNNL_NOT_PROPAGATE_NAN, 1.0);
            cnnl_check(cnnlActivationForward(
                    cnnl_handler, logsigmoid_desc.desc(), nullptr, input_desc.desc(),
                    src.raw_ptr(), nullptr, output_desc.desc(), dst.raw_ptr()));
            break;
        }
        case Mode::SQRT: {  // half,float
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest),
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            cnnl_check(cnnlSqrt_v2(
                    cnnl_handler,
                    cnnlComputationPreference_t::CNNL_COMPUTATION_HIGH_PRECISION,
                    input_desc.desc(), src.raw_ptr(), output_desc.desc(),
                    dst.raw_ptr()));
            break;
        }
        case Mode::NOT: {  // bool
            megdnn_assert(
                    dtype_dest == megdnn::DTypeEnum::Bool,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            Workspace cnnl_wk = wk_bundle.get_workspace(0);
            cnnl_check(cnnlLogicOp(
                    cnnl_handler, cnnlLogicOp_t::CNNL_LOGIC_OP_NOT, input_desc.desc(),
                    src.raw_ptr(), input_desc.desc(), src.raw_ptr(), cnnl_wk.raw_ptr,
                    cnnl_wk.size, output_desc.desc(), dst.raw_ptr()));
            break;
        }
        default:
            megdnn_throw("elemwise unsupport mode");
    }
}

void check_binary(
        const TensorNDArray& src, _megdnn_tensor_out dst,
        const param::Elemwise::Mode& mode) {
    bool mode_ok_binary = mode == Mode::ADD || mode == Mode::SUB || mode == Mode::MUL ||
                          mode == Mode::TRUE_DIV || mode == Mode::SWITCH_GT0 ||
                          mode == Mode::AND || mode == Mode::MAX || mode == Mode::MIN ||
                          mode == Mode::SIGMOID_GRAD || mode == Mode::SOFTPLUS_GRAD ||
                          mode == Mode::POW || mode == Mode::MOD ||
                          mode == Mode::FLOOR_DIV;
    megdnn_assert(mode_ok_binary, "Elemwise unsupport mode:%d", static_cast<int>(mode));
    auto dtype_src0 = src.at(0).layout.dtype.enumv();
    auto dtype_src1 = src.at(1).layout.dtype.enumv();
    auto dtype_dest = dst.layout.dtype.enumv();
    megdnn_assert(
            dtype_src0 == dtype_src1, "Elemwise binary dtype mismatch : %d vs %d",
            static_cast<int>(dtype_src0), static_cast<int>(dtype_src1));
    megdnn_assert(
            dtype_src0 == dtype_dest, "Elemwise binary dtype mismatch : %d vs %d",
            static_cast<int>(dtype_src0), static_cast<int>(dtype_dest));
}

void* to_contiguous(
        cnnlHandle_t cnnl_handler, const TensorND& in,
        const CnnlTensorDescriptor& src_desc, CnnlTensorDescriptor& dst_desc,
        Workspace wk, size_t offset = 0) {
    TensorLayout dst_layout = in.layout;
    dst_layout.init_contiguous_stride();
    dst_desc.set(dst_layout);
    if (in.layout.is_contiguous()) {
        return in.raw_ptr();
    }
    megdnn_assert(dst_layout.access_bytes() <= wk.size);
    cnnl_check(cnnlCopy(
            cnnl_handler, src_desc.desc(), in.raw_ptr(), dst_desc.desc(),
            wk.ptr<void>(offset)));
    return wk.ptr<void>(offset);
}

void* to_broadcast(
        cnnlHandle_t cnnl_handler, const TensorND& in, const TensorLayout& dst_layout,
        const CnnlTensorDescriptor& src_desc, Workspace wk, size_t offset = 0) {
    CnnlTensorDescriptor dst_desc;
    dst_desc.set(dst_layout);
    if (in.layout.eq_layout(dst_layout)) {
        return in.raw_ptr();
    }
    megdnn_assert(dst_layout.access_bytes() <= wk.size);
    cnnl_check(cnnlExpand(
            cnnl_handler, src_desc.desc(), in.raw_ptr(), dst_desc.desc(),
            wk.ptr<void>(offset)));
    return wk.ptr<void>(offset);
}

template <typename T>
void opTensorRun(
        cnnlHandle_t cnnl_handler, cnnlOpTensorDesc_t op_mode,
        const CnnlTensorDescriptor& lhs_desc, const void* lhs,
        const CnnlTensorDescriptor& rhs_desc, const void* rhs,
        const CnnlTensorDescriptor& output_desc, void* dst, megdnn::DTypeEnum dst_dtype,
        const Workspace& cnnl_wk) {
    CnnlOpTensorDescriptor optensor_desc;
    auto cnnl_type = convert_to_cnnl_datatype(dst_dtype);
    optensor_desc.set(op_mode, cnnl_type, cnnlNanPropagation_t::CNNL_NOT_PROPAGATE_NAN);
    T alpha1{1}, alpha2{1}, beta{0};
    cnnl_check(cnnlOpTensor(
            cnnl_handler, optensor_desc.desc(), &alpha1, lhs_desc.desc(), lhs, &alpha2,
            rhs_desc.desc(), rhs, cnnl_wk.raw_ptr, cnnl_wk.size, &beta,
            output_desc.desc(), dst));
}

void exec_binary(
        HandleImpl* handle, const TensorND& lhs, const TensorND& rhs,
        _megdnn_tensor_out dst, const param::Elemwise::Mode& mode,
        const WorkspaceBundle& wk_bundle) {
    auto cnnl_handler = handle->cnnl_handle();
    auto dtype_dest = dst.layout.dtype.enumv();
    CnnlTensorDescriptor lhs_desc, rhs_desc, output_desc;
    lhs_desc.set(lhs.layout);
    rhs_desc.set(rhs.layout);
    output_desc.set(dst.layout);

    Workspace cnnl_wk = wk_bundle.get_workspace(0);
    Workspace lhs_contig_wk, rhs_contig_wk;
    lhs_contig_wk = wk_bundle.get_workspace(1);
    rhs_contig_wk = wk_bundle.get_workspace(2);
    void *lhs_ptr, *rhs_ptr;
    CnnlTensorDescriptor lhs_contig_desc, rhs_contig_desc;
    auto contiguous_src = [&]() {
        lhs_ptr = to_contiguous(
                cnnl_handler, lhs, lhs_desc, lhs_contig_desc, lhs_contig_wk);
        rhs_ptr = to_contiguous(
                cnnl_handler, rhs, rhs_desc, rhs_contig_desc, rhs_contig_wk);
    };
    switch (mode) {
        case Mode::ADD: {  // float, half, int32
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            contiguous_src();
            if (dtype_dest == megdnn::DTypeEnum::Int32) {
                opTensorRun<int32_t>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_ADD,
                        lhs_contig_desc, lhs_ptr, rhs_contig_desc, rhs_ptr, output_desc,
                        dst.raw_ptr(), dst.layout.dtype.enumv(), cnnl_wk);
            } else {
                opTensorRun<float>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_ADD,
                        lhs_contig_desc, lhs_ptr, rhs_contig_desc, rhs_ptr, output_desc,
                        dst.raw_ptr(), dst.layout.dtype.enumv(), cnnl_wk);
            }
            break;
        }
        case Mode::MUL: {  // float, half, int32
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            contiguous_src();
            if (dtype_dest == megdnn::DTypeEnum::Int32) {
                opTensorRun<int32_t>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_MUL,
                        lhs_contig_desc, lhs_ptr, rhs_contig_desc, rhs_ptr, output_desc,
                        dst.raw_ptr(), dst.layout.dtype.enumv(), cnnl_wk);
            } else {
                opTensorRun<float>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_MUL,
                        lhs_contig_desc, lhs_ptr, rhs_contig_desc, rhs_ptr, output_desc,
                        dst.raw_ptr(), dst.layout.dtype.enumv(), cnnl_wk);
            }
            break;
        }
        case Mode::SUB: {  // float, half, int32
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            contiguous_src();
            if (dtype_dest == megdnn::DTypeEnum::Int32) {
                opTensorRun<int32_t>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_SUB,
                        lhs_contig_desc, lhs_ptr, rhs_contig_desc, rhs_ptr, output_desc,
                        dst.raw_ptr(), dst.layout.dtype.enumv(), cnnl_wk);
            } else {
                opTensorRun<float>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_SUB,
                        lhs_contig_desc, lhs_ptr, rhs_contig_desc, rhs_ptr, output_desc,
                        dst.raw_ptr(), dst.layout.dtype.enumv(), cnnl_wk);
            }
            break;
        }
        case Mode::TRUE_DIV: {  // float, half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest),
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            cnnl_check(cnnlDiv_v2(
                    cnnl_handler,
                    cnnlComputationPreference_t::CNNL_COMPUTATION_HIGH_PRECISION,
                    lhs_desc.desc(), lhs.raw_ptr(), rhs_desc.desc(), rhs.raw_ptr(),
                    cnnl_wk.raw_ptr, cnnl_wk.size, output_desc.desc(), dst.raw_ptr()));
            break;
        }
        case Mode::SWITCH_GT0: {  // float, half, int32
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            auto lhs_bcast_wk = wk_bundle.get_workspace(3);
            auto rhs_bcast_wk = wk_bundle.get_workspace(4);
            void* lhs_bcast_ptr =
                    to_broadcast(cnnl_handler, lhs, dst.layout, lhs_desc, lhs_bcast_wk);
            void* rhs_bcast_ptr =
                    to_broadcast(cnnl_handler, rhs, dst.layout, rhs_desc, rhs_bcast_wk);
            CnnlTensorDescriptor lhs_bcast_desc, rhs_bcast_desc;
            lhs_bcast_desc.set(dst.layout);
            rhs_bcast_desc.set(dst.layout);

#define cb(DType)                                                                  \
    if (dst.layout.dtype == DType()) {                                             \
        using ctype = typename DTypeTrait<DType>::ctype;                           \
        const ctype thresh(0);                                                     \
        cnnl_check(cnnlThresholdBackward(                                          \
                cnnl_handler, lhs_bcast_desc.desc(), lhs_bcast_ptr,                \
                rhs_bcast_desc.desc(), rhs_bcast_ptr, &thresh, output_desc.desc(), \
                dst.raw_ptr()));                                                   \
    }
            cb(::megdnn::dtype::Float16) cb(::megdnn::dtype::Float32)
                    cb(::megdnn::dtype::Int32)
#undef cb

                            break;
        }
        case Mode::AND: {  // bool
            megdnn_assert(
                    dtype_dest == megdnn::DTypeEnum::Bool,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            cnnl_check(cnnlLogicOp(
                    cnnl_handler, cnnlLogicOp_t::CNNL_LOGIC_OP_AND, lhs_desc.desc(),
                    lhs.raw_ptr(), rhs_desc.desc(), rhs.raw_ptr(), cnnl_wk.raw_ptr,
                    cnnl_wk.size, output_desc.desc(), dst.raw_ptr()));
            break;
        }
        case Mode::MAX: {  // int32, float, half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            contiguous_src();
            cnnl_check(cnnlMaximum(
                    cnnl_handler, lhs_contig_desc.desc(), lhs_ptr,
                    rhs_contig_desc.desc(), rhs_ptr, output_desc.desc(), dst.raw_ptr(),
                    cnnl_wk.raw_ptr, cnnl_wk.size));
            break;
        }
        case Mode::MIN: {  // int32, float, half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            contiguous_src();
            cnnl_check(cnnlMinimum(
                    cnnl_handler, lhs_contig_desc.desc(), lhs_ptr,
                    rhs_contig_desc.desc(), rhs_ptr, output_desc.desc(), dst.raw_ptr(),
                    cnnl_wk.raw_ptr, cnnl_wk.size));
            break;
        }
        case Mode::POW: {  // float, half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            contiguous_src();
            cnnl_check(cnnlPow(
                    cnnl_handler,
                    cnnlComputationPreference_t::CNNL_COMPUTATION_HIGH_PRECISION,
                    lhs_contig_desc.desc(), lhs_ptr, rhs_contig_desc.desc(), rhs_ptr,
                    cnnl_wk.raw_ptr, cnnl_wk.size, output_desc.desc(), dst.raw_ptr()));
            break;
        }
        case Mode::SOFTPLUS_GRAD: {  // float, half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest),
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            Workspace lhs_bcast_wk, rhs_bcast_wk;
            lhs_bcast_wk = wk_bundle.get_workspace(3);
            rhs_bcast_wk = wk_bundle.get_workspace(4);
            void *lhs_bcast_ptr, *rhs_bcast_ptr;
            lhs_bcast_ptr =
                    to_broadcast(cnnl_handler, lhs, dst.layout, lhs_desc, lhs_bcast_wk);
            rhs_bcast_ptr =
                    to_broadcast(cnnl_handler, rhs, dst.layout, rhs_desc, rhs_bcast_wk);
            CnnlTensorDescriptor lhs_bcast_desc, rhs_bcast_desc;
            lhs_bcast_desc.set(dst.layout);
            rhs_bcast_desc.set(dst.layout);
            cnnl_check(cnnlSoftplusBackward(
                    cnnl_handler, lhs_bcast_desc.desc(), lhs_bcast_ptr,
                    rhs_bcast_desc.desc(), rhs_bcast_ptr, output_desc.desc(),
                    dst.raw_ptr(), 1, INT_MAX));
            break;
        }
        case Mode::SIGMOID_GRAD: {  // float, half
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest),
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            Workspace lhs_bcast_wk, rhs_bcast_wk;
            lhs_bcast_wk = wk_bundle.get_workspace(3);
            rhs_bcast_wk = wk_bundle.get_workspace(4);
            void *lhs_bcast_ptr, *rhs_bcast_ptr;
            lhs_bcast_ptr =
                    to_broadcast(cnnl_handler, lhs, dst.layout, lhs_desc, lhs_bcast_wk);
            rhs_bcast_ptr =
                    to_broadcast(cnnl_handler, rhs, dst.layout, rhs_desc, rhs_bcast_wk);
            CnnlTensorDescriptor lhs_bcast_desc, rhs_bcast_desc;
            lhs_bcast_desc.set(dst.layout);
            rhs_bcast_desc.set(dst.layout);
            CnnlActivationDescriptor sigmoid_desc;
            sigmoid_desc.set(
                    cnnlActivationMode_t::CNNL_ACTIVATION_SIGMOID,
                    cnnlActivationPreference_t::CNNL_ACTIVATION_HIGH_PRECISION,
                    cnnlNanPropagation_t::CNNL_NOT_PROPAGATE_NAN, 1.0);
            cnnl_check(cnnlActivationBackward(
                    cnnl_handler, sigmoid_desc.desc(), nullptr, lhs_bcast_desc.desc(),
                    lhs_bcast_ptr, rhs_bcast_desc.desc(), rhs_bcast_ptr, nullptr,
                    nullptr, nullptr, output_desc.desc(), dst.raw_ptr()));

            break;
        }
        case Mode::MOD: {
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            contiguous_src();
            if (dtype_dest == megdnn::DTypeEnum::Int32) {
                cnnl_check(cnnlFloorMod(
                        cnnl_handler, lhs_contig_desc.desc(), lhs_ptr,
                        rhs_contig_desc.desc(), rhs_ptr, output_desc.desc(),
                        dst.raw_ptr(), cnnl_wk.raw_ptr, cnnl_wk.size));
            } else {
                cnnl_check(cnnlFloorModTrunc(
                        cnnl_handler, lhs_contig_desc.desc(), lhs_ptr,
                        rhs_contig_desc.desc(), rhs_ptr, output_desc.desc(),
                        dst.raw_ptr(), cnnl_wk.raw_ptr, cnnl_wk.size));
            }
            break;
        }
        case Mode::FLOOR_DIV: {
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            contiguous_src();
            cnnl_check(cnnlFloorDiv_v2(
                    cnnl_handler,
                    cnnlComputationPreference_t::CNNL_COMPUTATION_ULTRAHIGH_PRECISION,
                    lhs_contig_desc.desc(), lhs_ptr, rhs_contig_desc.desc(), rhs_ptr,
                    output_desc.desc(), dst.raw_ptr(), cnnl_wk.raw_ptr, cnnl_wk.size));
            break;
        }
        default:
            megdnn_throw("elemwise unsupport mode");
    }
}

void check_ternary(
        const TensorNDArray& src, _megdnn_tensor_out dst,
        const param::Elemwise::Mode& mode) {
    bool mode_ok_ternary = mode == Mode::COND_LEQ_MOV || mode == Mode::COND_LT_MOV ||
                           mode == Mode::CLIP || mode == Mode::FUSE_MUL_ADD3;

    megdnn_assert(
            mode_ok_ternary, "Elemwise unsupport mode:%d", static_cast<int>(mode));
    auto dtype_src0 = src.at(0).layout.dtype.enumv();
    auto dtype_src1 = src.at(1).layout.dtype.enumv();
    auto dtype_src2 = src.at(2).layout.dtype.enumv();
    auto dtype_dest = dst.layout.dtype.enumv();
    megdnn_assert(
            dtype_src0 == dtype_src1 && dtype_src0 == dtype_src2,
            "Elemwise ternary dtype mismatch : %d vs %d vs %d",
            static_cast<int>(dtype_src0), static_cast<int>(dtype_src1),
            static_cast<int>(dtype_src2));
    megdnn_assert(
            dtype_src0 == dtype_dest, "Elemwise ternary dtype mismatch : %d vs %d",
            static_cast<int>(dtype_src0), static_cast<int>(dtype_dest));
}

void exec_ternary(
        HandleImpl* handle, const TensorND& src0, const TensorND& src1,
        const TensorND& src2, _megdnn_tensor_out dst, const param::Elemwise::Mode& mode,
        const WorkspaceBundle& wk_bundle) {
    auto cnnl_handler = handle->cnnl_handle();
    auto dtype_dest = dst.layout.dtype.enumv();

    CnnlTensorDescriptor src0_desc, src1_desc, src2_desc, output_desc;
    src0_desc.set(src0.layout);
    src1_desc.set(src1.layout);
    src2_desc.set(src2.layout);
    output_desc.set(dst.layout);

    auto convert_mode_to_logicOp = [mode]() {
        if (mode == Mode::COND_LEQ_MOV) {
            return cnnlLogicOp_t::CNNL_LOGIC_OP_LE;
        } else if (mode == Mode::COND_LT_MOV) {
            return cnnlLogicOp_t::CNNL_LOGIC_OP_LT;
        } else {
            megdnn_throw("unsupport elemwise ternary opr");
        }
    };

    switch (mode) {
        case Mode::COND_LT_MOV:
        case Mode::COND_LEQ_MOV: {  // float, half, int32
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            Workspace logic_wk, src0_wk, logic_res_wk, optensor_wk;
            logic_wk = wk_bundle.get_workspace(0);
            src0_wk = wk_bundle.get_workspace(1);
            logic_res_wk = wk_bundle.get_workspace(2);
            optensor_wk = wk_bundle.get_workspace(3);
            TensorShapeArray src0_1;
            src0_1.push_back(src0.layout);
            src0_1.push_back(src1.layout);
            TensorShape logic_res_shape;
            Elemwise::deduce_shape(src0_1, logic_res_shape);
            TensorLayout logic_res_layout(logic_res_shape, src0.layout.dtype);
            void* src0_ptr = to_broadcast(
                    cnnl_handler, src0, logic_res_layout, src0_desc, src0_wk);
            CnnlTensorDescriptor logic_res_desc;
            logic_res_desc.set(logic_res_layout);
            cnnl_check(cnnlLogicOp(
                    cnnl_handler, convert_mode_to_logicOp(), logic_res_desc.desc(),
                    src0_ptr, src1_desc.desc(), src1.raw_ptr(), logic_wk.raw_ptr,
                    logic_wk.size, logic_res_desc.desc(), logic_res_wk.raw_ptr));
            if (dtype_dest == megdnn::DTypeEnum::Int32) {
                opTensorRun<int32_t>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_MUL,
                        logic_res_desc, logic_res_wk.raw_ptr, src2_desc, src2.raw_ptr(),
                        output_desc, dst.raw_ptr(), dst.layout.dtype.enumv(),
                        optensor_wk);
            } else {
                opTensorRun<float>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_MUL,
                        logic_res_desc, logic_res_wk.raw_ptr, src2_desc, src2.raw_ptr(),
                        output_desc, dst.raw_ptr(), dst.layout.dtype.enumv(),
                        optensor_wk);
            }
            break;
        }
        case Mode::CLIP: {
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            // run cnnl
            if (src1.layout.access_bytes() == src1.layout.dtype.size(1) ||
                src2.layout.access_bytes() == src2.layout.dtype.size(1)) {
                CnnlTensorDescriptor src0_desc, output_desc;
                src0_desc.set(src0.layout);
                output_desc.set(dst.layout);
                Workspace src0_wk = wk_bundle.get_workspace(1);
                CnnlTensorDescriptor src0_contig_desc;
                void* src0_ptr;
                src0_ptr = to_contiguous(
                        cnnl_handler, src0, src0_desc, src0_contig_desc, src0_wk, 0);
                cnnl_check(cnnlClip_v2(
                        cnnl_handler, CNNL_POINTER_MODE_DEVICE, src0_contig_desc.desc(),
                        src0_ptr, src1.raw_ptr(), src2.raw_ptr(), output_desc.desc(),
                        dst.raw_ptr()));
            } else {
                megdnn_throw(
                        "elemwise clip mode don't support non scaler lower and upper");
            }
            break;
        }
        case Mode::FUSE_MUL_ADD3: {
            megdnn_assert(
                    check_dtype_float_ieee(dtype_dest) ||
                            dtype_dest == megdnn::DTypeEnum::Int32,
                    "Cambricon unsupport elemwise mode:%d with dtype:%d",
                    static_cast<int>(mode), static_cast<int>(dtype_dest));
            Workspace mul_wk, add_wk, mul_res_wk;
            mul_wk = wk_bundle.get_workspace(0);
            add_wk = wk_bundle.get_workspace(1);
            mul_res_wk = wk_bundle.get_workspace(2);
            TensorShapeArray src0_1;
            src0_1.push_back(src0.layout);
            src0_1.push_back(src1.layout);
            TensorShape mul_res_shape;
            Elemwise::deduce_shape(src0_1, mul_res_shape);
            TensorLayout mul_res_layout(mul_res_shape, src0.layout.dtype);
            CnnlTensorDescriptor mul_res_desc;
            mul_res_desc.set(mul_res_layout);
            if (dtype_dest == megdnn::DTypeEnum::Int32) {
                opTensorRun<int32_t>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_MUL, src0_desc,
                        src0.raw_ptr(), src1_desc, src1.raw_ptr(), mul_res_desc,
                        mul_res_wk.raw_ptr, src0.layout.dtype.enumv(), mul_wk);
                opTensorRun<int32_t>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_ADD,
                        mul_res_desc, mul_res_wk.raw_ptr, src2_desc, src2.raw_ptr(),
                        output_desc, dst.raw_ptr(), dst.layout.dtype.enumv(), add_wk);
            } else {
                opTensorRun<float>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_MUL, src0_desc,
                        src0.raw_ptr(), src1_desc, src1.raw_ptr(), mul_res_desc,
                        mul_res_wk.raw_ptr, src0.layout.dtype.enumv(), mul_wk);
                opTensorRun<float>(
                        cnnl_handler, cnnlOpTensorDesc_t::CNNL_OP_TENSOR_ADD,
                        mul_res_desc, mul_res_wk.raw_ptr, src2_desc, src2.raw_ptr(),
                        output_desc, dst.raw_ptr(), dst.layout.dtype.enumv(), add_wk);
            }
            break;
        }
        default:
            break;
    }
}

}  // anonymous namespace

void ElemwiseForwardImpl::exec(const TensorNDArray& src, _megdnn_tensor_out dst) {
    WorkspaceBundle wk_bundle = alloc_cnnl_workspace(src, dst);
    auto handle = concrete_handle(this->handle());
    auto mode = m_param.mode;
    int nr_operands = src.size();
    if (nr_operands == 1) {  // unary mode
        check_unary(src, dst, mode);
        exec_unary(handle, src.at(0), dst, mode, wk_bundle);
    } else if (nr_operands == 2) {  // binary mode
        check_binary(src, dst, mode);
        exec_binary(handle, src.at(0), src.at(1), dst, mode, wk_bundle);
    } else if (nr_operands == 3) {
        check_ternary(src, dst, mode);
        exec_ternary(handle, src.at(0), src.at(1), src.at(2), dst, mode, wk_bundle);
    } else {
        megdnn_throw("elemwise unsupport mode");
    }
    free_cnnl_workspace(wk_bundle);
}

namespace {

WorkspaceBundle make_bundle(
        HandleImpl* handle, const SmallVector<size_t>& sizes_in_bytes) {
    WorkspaceBundle wk_bundle{nullptr, sizes_in_bytes, handle->alignment_requirement()};
    void* wk_ptr = handle->alloc(wk_bundle.total_size_in_bytes());
    wk_bundle.set(wk_ptr);
    return wk_bundle;
}

WorkspaceBundle get_unary_ws(
        HandleImpl* handle, const TensorLayoutArray& src, const TensorLayout& dst,
        const param::Elemwise::Mode& mode) {
    auto cnnl_handler = handle->cnnl_handle();
    bool mode_need_ws = mode == Mode::NOT;
    if (!mode_need_ws)
        return {nullptr, {}, handle->alignment_requirement()};
    CnnlTensorDescriptor input_desc, output_desc;
    input_desc.set(src[0]);
    output_desc.set(dst);
    // 1st workspace is cnnl workspace
    SmallVector<size_t> sizes_in_bytes(1, 0);
    switch (mode) {
        case Mode::NOT:
            cnnl_check(cnnlGetLogicOpWorkspaceSize(
                    cnnl_handler, input_desc.desc(), input_desc.desc(),
                    output_desc.desc(), &sizes_in_bytes[0]));
            break;
        default:
            break;
    }

    return make_bundle(handle, sizes_in_bytes);
}
WorkspaceBundle get_binary_ws(
        HandleImpl* handle, const TensorLayoutArray& src, const TensorLayout& dst,
        const param::Elemwise::Mode& mode) {
    auto cnnl_handler = handle->cnnl_handle();
    bool mode_need_ws = mode == Mode::TRUE_DIV || mode == Mode::FLOOR_DIV ||
                        mode == Mode::MOD || mode == Mode::MAX || mode == Mode::MIN ||
                        mode == Mode::POW || mode == Mode::AND || mode == Mode::ADD ||
                        mode == Mode::SUB || mode == Mode::MUL ||
                        mode == Mode::SWITCH_GT0 || mode == Mode::SOFTPLUS_GRAD ||
                        mode == Mode::SIGMOID_GRAD;
    if (!mode_need_ws)
        return {nullptr, {}, handle->alignment_requirement()};
    CnnlTensorDescriptor lhs_desc, rhs_desc, output_desc;
    lhs_desc.set(src[0]);
    rhs_desc.set(src[1]);
    output_desc.set(dst);
    // 1st workspace is cnnl workspace, 2st and 3st are handle un-contiguous
    SmallVector<size_t> sizes_in_bytes(3, 0);
    // ADD,SUB,MUL,MOD,FLOOR_DIV,MIN,MAX,POW,SWITCH_GT0 need handle uncontig
    auto handle_uncontig_wk = [&]() {
        size_t lhs_wk = !src[0].is_contiguous() ? src[0].access_bytes() : 0;
        size_t rhs_wk = !src[1].is_contiguous() ? src[1].access_bytes() : 0;
        sizes_in_bytes[1] = lhs_wk;
        sizes_in_bytes[2] = rhs_wk;
    };

    switch (mode) {
        case Mode::TRUE_DIV:
            cnnl_check(cnnlGetDivWorkspaceSize(
                    cnnl_handler, lhs_desc.desc(), rhs_desc.desc(), output_desc.desc(),
                    &sizes_in_bytes[0]));
            break;
        case Mode::FLOOR_DIV:
            cnnl_check(cnnlGetFloorDivWorkspaceSize(
                    cnnl_handler, lhs_desc.desc(), rhs_desc.desc(), output_desc.desc(),
                    &sizes_in_bytes[0]));
            handle_uncontig_wk();
            break;
        case Mode::MOD:
            if (dst.dtype.enumv() == megdnn::DTypeEnum::Int32) {
                cnnl_check(cnnlGetFloorModWorkspaceSize(
                        cnnl_handler, lhs_desc.desc(), rhs_desc.desc(),
                        output_desc.desc(), &sizes_in_bytes[0]));
            } else {
                cnnl_check(cnnlGetFloorModTruncWorkspaceSize(
                        cnnl_handler, lhs_desc.desc(), rhs_desc.desc(),
                        output_desc.desc(), &sizes_in_bytes[0]));
            }
            handle_uncontig_wk();
            break;
        case Mode::POW:
            cnnl_check(cnnlGetPowWorkspaceSize(
                    cnnl_handler, lhs_desc.desc(), rhs_desc.desc(), output_desc.desc(),
                    &sizes_in_bytes[0]));
            handle_uncontig_wk();
            break;
        case Mode::AND:
            cnnl_check(cnnlGetLogicOpWorkspaceSize(
                    cnnl_handler, lhs_desc.desc(), rhs_desc.desc(), output_desc.desc(),
                    &sizes_in_bytes[0]));
            break;
        case Mode::MAX:
        case Mode::MIN: {
            cnnl_check(cnnlGetMaximumWorkspaceSize(  // equal between min and max
                    cnnl_handler, output_desc.desc(), &sizes_in_bytes[0]));
            handle_uncontig_wk();
            break;
        }
        case Mode::ADD:
        case Mode::SUB:
        case Mode::MUL: {
            cnnl_check(cnnlGetOpTensorWorkspaceSize(
                    cnnl_handler, lhs_desc.desc(), rhs_desc.desc(), output_desc.desc(),
                    &sizes_in_bytes[0]));
            handle_uncontig_wk();
            break;
        }
        case Mode::SWITCH_GT0:
        case Mode::SIGMOID_GRAD:
        case Mode::SOFTPLUS_GRAD: {
            size_t src0_bcast, src1_bcast;
            src0_bcast = !src[0].eq_layout(dst) ? dst.access_bytes() : 0;
            src1_bcast = !src[1].eq_layout(dst) ? dst.access_bytes() : 0;
            sizes_in_bytes.push_back(src0_bcast);
            sizes_in_bytes.push_back(src1_bcast);
            break;
        }
        default:
            break;
    }

    return make_bundle(handle, sizes_in_bytes);
}

WorkspaceBundle get_ternary_ws(
        HandleImpl* handle, const TensorLayoutArray& src, const TensorLayout& dst,
        const param::Elemwise::Mode& mode) {
    auto cnnl_handler = handle->cnnl_handle();
    bool mode_need_ws = mode == Mode::CLIP || mode == Mode::COND_LEQ_MOV ||
                        mode == Mode::COND_LT_MOV || mode == Mode::FUSE_MUL_ADD3;
    if (!mode_need_ws)
        return {nullptr, {}, handle->alignment_requirement()};
    CnnlTensorDescriptor src0_desc, src1_desc, src2_desc, output_desc;
    src0_desc.set(src[0]);
    src1_desc.set(src[1]);
    src2_desc.set(src[2]);
    output_desc.set(dst);
    // 1st workspace is cnnl workspace
    SmallVector<size_t> sizes_in_bytes(1, 0);
    switch (mode) {
        case Mode::CLIP: {
            size_t dtype_size = src[0].dtype.size(1);
            size_t src_wk =
                    !src[0].is_contiguous() ? src[0].total_nr_elems() * dtype_size : 0;
            sizes_in_bytes.push_back(src_wk);
            break;
        }
        case Mode::COND_LT_MOV:
        case Mode::COND_LEQ_MOV: {
            TensorShapeArray src0_1;
            src0_1.push_back(src[0]);
            src0_1.push_back(src[1]);
            TensorShape logic_res_shape;
            Elemwise::deduce_shape(src0_1, logic_res_shape);
            TensorLayout logic_res_layout(logic_res_shape, src[0].dtype);
            CnnlTensorDescriptor logic_res_desc;
            logic_res_desc.set(logic_res_layout);
            cnnl_check(cnnlGetLogicOpWorkspaceSize(
                    cnnl_handler, logic_res_desc.desc(), src1_desc.desc(),
                    logic_res_desc.desc(), &sizes_in_bytes[0]));
            size_t src0_wk = 0, logic_res_wk = 0;
            if (!src[0].eq_layout(dst)) {
                src0_wk = logic_res_layout.access_bytes();
            }
            logic_res_wk = logic_res_layout.access_bytes();
            sizes_in_bytes.push_back(src0_wk);
            sizes_in_bytes.push_back(logic_res_wk);
            size_t optensor_wk = 0;
            cnnl_check(cnnlGetOpTensorWorkspaceSize(
                    cnnl_handler, logic_res_desc.desc(), src2_desc.desc(),
                    output_desc.desc(), &optensor_wk));
            sizes_in_bytes.push_back(optensor_wk);
            break;
        }
        case Mode::FUSE_MUL_ADD3: {
            TensorShapeArray src0_1;
            src0_1.push_back(src[0]);
            src0_1.push_back(src[1]);
            TensorShape mul_res_shape;
            Elemwise::deduce_shape(src0_1, mul_res_shape);
            TensorLayout mul_res_layout(mul_res_shape, src[0].dtype);
            CnnlTensorDescriptor mul_res_desc;
            mul_res_desc.set(mul_res_layout);
            cnnl_check(cnnlGetOpTensorWorkspaceSize(
                    cnnl_handler, src0_desc.desc(), src1_desc.desc(),
                    mul_res_desc.desc(), &sizes_in_bytes[0]));
            size_t add_wk = 0;
            cnnl_check(cnnlGetOpTensorWorkspaceSize(
                    cnnl_handler, mul_res_desc.desc(), src2_desc.desc(),
                    output_desc.desc(), &add_wk));
            sizes_in_bytes.push_back(add_wk);
            sizes_in_bytes.push_back(mul_res_layout.access_bytes());
        }
        default:
            break;
    }
    return make_bundle(handle, sizes_in_bytes);
}

}  // anonymous namespace

WorkspaceBundle ElemwiseForwardImpl::alloc_cnnl_workspace(
        const TensorNDArray& src, const TensorND& dst) {
    TensorLayoutArray src_layouts(src.size());
    std::transform(
            src.begin(), src.end(), src_layouts.begin(),
            [](const TensorND& tensor) { return tensor.layout; });
    auto mode = m_param.mode;
    auto handle = concrete_handle(this->handle());
    int nr_operands = src.size();
    if (nr_operands == 1) {  // unary mode
        return get_unary_ws(handle, src_layouts, dst.layout, mode);
    } else if (nr_operands == 2) {  // binary mode
        return get_binary_ws(handle, src_layouts, dst.layout, mode);
    } else if (nr_operands == 3) {
        return get_ternary_ws(handle, src_layouts, dst.layout, mode);
    } else {
        megdnn_throw("elemwise unsupport mode");
    }
}

void ElemwiseForwardImpl::free_cnnl_workspace(const WorkspaceBundle& wk_bundle) {
    concrete_handle(handle())->free(wk_bundle.ptr());
}