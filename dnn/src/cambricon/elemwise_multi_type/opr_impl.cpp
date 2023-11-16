#include "src/cambricon/elemwise_multi_type/opr_impl.h"
#include <vector>
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"

using namespace megdnn;
using namespace cambricon;

void check_param_logic_op(
        const TensorLayout& src0, const TensorLayout& src1, const TensorLayout& dst,
        megdnn::param::ElemwiseMultiType::Mode mode) {
    using Mode = megdnn::param::ElemwiseMultiType::Mode;
    auto src0_dtype = src0.dtype.enumv();
    auto src1_dtype = src1.dtype.enumv();
    auto dst_dtype = dst.dtype.enumv();
    megdnn_assert(
            (src0_dtype == DTypeEnum::Float32 && src1_dtype == DTypeEnum::Float32 &&
             dst_dtype == DTypeEnum::Float32) ||
                    (src0_dtype == DTypeEnum::Float32 &&
                     src1_dtype == DTypeEnum::Float32 &&
                     dst_dtype == DTypeEnum::Bool) ||
                    (src0_dtype == DTypeEnum::Float32 &&
                     src1_dtype == DTypeEnum::Int32 && dst_dtype == DTypeEnum::Bool) ||
                    (src0_dtype == DTypeEnum::Float16 &&
                     src1_dtype == DTypeEnum::Float16 &&
                     dst_dtype == DTypeEnum::Float16) ||
                    (src0_dtype == DTypeEnum::Float16 &&
                     src1_dtype == DTypeEnum::Float16 &&
                     dst_dtype == DTypeEnum::Bool) ||
                    (src0_dtype == DTypeEnum::Int32 && src1_dtype == DTypeEnum::Int32 &&
                     dst_dtype == DTypeEnum::Int32) ||
                    (src0_dtype == DTypeEnum::Int32 && src1_dtype == DTypeEnum::Int32 &&
                     dst_dtype == DTypeEnum::Bool) ||
                    (src0_dtype == DTypeEnum::Int32 &&
                     src1_dtype == DTypeEnum::Float32 &&
                     dst_dtype == DTypeEnum::Bool) ||
                    (src0_dtype == DTypeEnum::Int16 && src1_dtype == DTypeEnum::Int16 &&
                     dst_dtype == DTypeEnum::Int16) ||
                    (src0_dtype == DTypeEnum::Int16 && src1_dtype == DTypeEnum::Int16 &&
                     dst_dtype == DTypeEnum::Bool) ||
                    (src0_dtype == DTypeEnum::Int8 && src1_dtype == DTypeEnum::Int8 &&
                     dst_dtype == DTypeEnum::Int8) ||
                    (src0_dtype == DTypeEnum::Int8 && src1_dtype == DTypeEnum::Int8 &&
                     dst_dtype == DTypeEnum::Bool) ||
                    (src0_dtype == DTypeEnum::Uint8 && src1_dtype == DTypeEnum::Uint8 &&
                     dst_dtype == DTypeEnum::Uint8) ||
                    (src0_dtype == DTypeEnum::Uint8 && src1_dtype == DTypeEnum::Uint8 &&
                     dst_dtype == DTypeEnum::Bool) ||
                    (src0_dtype == DTypeEnum::Bool && src1_dtype == DTypeEnum::Bool &&
                     dst_dtype == DTypeEnum::Bool),
            "Elemwise_multi_type Opr Requires data type of input0, input1 and output "
            "must satisfy the requirments, but got %d, %d and %d\n",
            static_cast<int>(src0_dtype), static_cast<int>(src1_dtype),
            static_cast<int>(dst_dtype));
    for (size_t i = 0; i < src0.ndim && i < src1.ndim && i < dst.ndim; ++i) {
        megdnn_assert(
                src0.shape[src0.ndim - 1 - i] == src1.shape[src1.ndim - 1 - i] ||
                        src0.shape[src0.ndim - 1 - i] == 1 ||
                        src1.shape[src1.ndim - 1 - i] == 1,
                "Elemwise_multi_type Opr Requires the length of each dimension of "
                "input0 and input1 must be the same or one of them equal to 1, but got "
                "%zu and %zu",
                src0.shape[src0.ndim - 1 - i], src1.shape[src1.ndim - 1 - i]);
        megdnn_assert(
                dst.shape[dst.ndim - 1 - i] == std::max(
                                                       src0.shape[src0.ndim - 1 - i],
                                                       src1.shape[src1.ndim - 1 - i]),
                "Elemwise_multi_type Opr Requires the length of each dimension of "
                "output should equal to the larger one between corresponding "
                "dimensions of input0 and input1, but got %zu, %zu and %zu",
                dst.shape[dst.ndim - 1 - i], src0.shape[src0.ndim - 1 - i],
                src1.shape[src1.ndim - 1 - i]);
    }
    megdnn_assert(
            mode == Mode::EQ || mode == Mode::NEQ || mode == Mode::LEQ ||
                    mode == Mode::LT,
            "Elemwise_multi_type Opr Supports EQ, NEQ, LEQ and LT, but got %d",
            static_cast<int>(mode));
}

cnnlLogicOp_t convert_mode_to_logic_op(megdnn::param::ElemwiseMultiType::Mode mode) {
    using Mode = megdnn::param::ElemwiseMultiType::Mode;
    switch (mode) {
        case Mode::EQ:
            return CNNL_LOGIC_OP_EQ;
        case Mode::NEQ:
            return CNNL_LOGIC_OP_NE;
        case Mode::LEQ:
            return CNNL_LOGIC_OP_LE;
        case Mode::LT:
            return CNNL_LOGIC_OP_LT;
        default:
            megdnn_throw("bad mode");
    }
}

size_t get_workspace_in_bytes_bool_mode_with_two_inputs(
        Handle* handle, const TensorLayout& src0, const TensorLayout& src1,
        const TensorLayout& dst) {
    auto cambricon_handle = concrete_handle(handle);
    CnnlTensorDescriptor src0_dsc, src1_dsc, dst_dsc;
    src0_dsc.set(src0);
    src1_dsc.set(src1);
    dst_dsc.set(dst);
    size_t workspace_size = 0;
    cnnl_check(cnnlGetLogicOpWorkspaceSize(
            cambricon_handle->cnnl_handle(), src0_dsc.desc(), src1_dsc.desc(),
            dst_dsc.desc(), &workspace_size));
    return workspace_size;
}

size_t get_workspace_in_bytes_bool_mode(
        Handle* handle, const TensorNDArray& src, const TensorND& dst,
        megdnn::param::ElemwiseMultiType::Mode mode) {
    if (src.size() == 2) {
        return get_workspace_in_bytes_bool_mode_with_two_inputs(
                handle, src[0].layout, src[1].layout, dst.layout);
    }
    megdnn_throw(ssprintf("Unsupported mode %d", static_cast<int>(mode)));
}

#define ALLOC_CNNL_WORKSPACE_ON_BOOL_MODE(_MODE)                                    \
    case Mode::_MODE:                                                               \
        return get_workspace_in_bytes_bool_mode(                                    \
                handle(), src, dst, megdnn::param::ElemwiseMultiType::Mode::_MODE); \
        break

size_t ElemwiseMultiTypeImpl::get_workspace_in_bytes(
        const TensorNDArray& src, const TensorND& dst) {
    switch (m_param.mode) {
        ALLOC_CNNL_WORKSPACE_ON_BOOL_MODE(LT);
        ALLOC_CNNL_WORKSPACE_ON_BOOL_MODE(LEQ);
        ALLOC_CNNL_WORKSPACE_ON_BOOL_MODE(EQ);
        ALLOC_CNNL_WORKSPACE_ON_BOOL_MODE(NEQ);
        default:
            megdnn_throw("invalid mode");
    }
}

void dest_type_bool_mode_with_two_inputs(
        Handle* handle, const TensorND& src0, const TensorND& src1, const TensorND& dst,
        megdnn::param::ElemwiseMultiType::Mode mode, Workspace* workspace) {
    megdnn_assert(src0.layout.dtype.enumv() == src1.layout.dtype.enumv());
    check_param_logic_op(src0.layout, src1.layout, dst.layout, mode);
    auto cambricon_handle = concrete_handle(handle);

    auto cnnl_mode = convert_mode_to_logic_op(mode);
    CnnlTensorDescriptor src0_dsc, src1_dsc, dst_dsc;
    src0_dsc.set(&src0);
    src1_dsc.set(&src1);
    dst_dsc.set(&dst);

    cnnl_check(cnnlLogicOp(
            cambricon_handle->cnnl_handle(), cnnl_mode, src0_dsc.desc(), src0.raw_ptr(),
            src1_dsc.desc(), src1.raw_ptr(), static_cast<void*>(workspace->raw_ptr),
            workspace->size, dst_dsc.desc(), dst.raw_ptr()));
}

void ElemwiseMultiTypeImpl::dest_type_bool_mode(
        const TensorNDArray& src, const TensorND& dst_tensor,
        megdnn::param::ElemwiseMultiType::Mode mode, Workspace* workspace) {
    if (src.size() == 2) {
        dest_type_bool_mode_with_two_inputs(
                handle(), src[0], src[1], dst_tensor, mode, workspace);
    } else {
        megdnn_throw(ssprintf("Unsupported mode %d", static_cast<int>(mode)));
    }
}

#define ON_BOOL_MODE(_MODE)                                                    \
    case Mode::_MODE:                                                          \
        dest_type_bool_mode(                                                   \
                src, dst, megdnn::param::ElemwiseMultiType::Mode::_MODE, &wk); \
        break

void ElemwiseMultiTypeImpl::exec(
        _megdnn_in const TensorNDArray& src, _megdnn_tensor_out dst) {
    auto cambricon_handle = concrete_handle(this->handle());
    size_t wk_size = get_workspace_in_bytes(src, dst);
    void* wk_ptr = cambricon_handle->alloc(wk_size);
    Workspace wk(static_cast<dt_byte*>(wk_ptr), wk_size);
    switch (m_param.mode) {
        ON_BOOL_MODE(LT);
        ON_BOOL_MODE(LEQ);
        ON_BOOL_MODE(EQ);
        ON_BOOL_MODE(NEQ);
        default:
            megdnn_throw("invalid mode");
    }
    cambricon_handle->free(wk_ptr);
}
// vim: syntax=cpp.doxygen
