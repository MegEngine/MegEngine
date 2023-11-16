#include "src/cambricon/reduce/opr_impl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_op_descriptor.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"

namespace megdnn {
namespace cambricon {

using Param = megdnn::param::Reduce;

namespace {

cnnlReduceOp_t convert_to_cnnl_reduce_op(const Param::Mode mode) {
    switch (mode) {
        case Param::Mode::MAX:
            return CNNL_REDUCE_MAX;
        case Param::Mode::MEAN:
            return CNNL_REDUCE_AVG;
        case Param::Mode::MIN:
            return CNNL_REDUCE_MIN;
        case Param::Mode::PRODUCT:
            return CNNL_REDUCE_MUL;
        case Param::Mode::SUM:
            return CNNL_REDUCE_ADD;
        case Param::Mode::SUM_SQR:
            return CNNL_REDUCE_SUMSQ;
        default:
            megdnn_assert(
                    false, "Invalid Reduce Mode for Cambricon: %d",
                    static_cast<int>(mode));
    }
}

cnnlDataType_t convert_reduce_computation_dtype_to_cnnl_data_type(
        Param::Reduce::DataType data_type, DTypeEnum src_type) {
    using DataType = Param::Reduce::DataType;
    switch (data_type) {
        case DataType::DEFAULT:
            // TODO: in case of DEFAULT, the computation type is chosen by the input and
            // output type and mode, not only input type, but it can be directly chosen
            // by the input type currently on cambricon.
            return convert_to_cnnl_datatype(src_type);
        case DataType::FLOAT_IO16xC32:
        case DataType::FLOAT_O32xC32:
        case DataType::FLOAT_O16xC32:
            return CNNL_DTYPE_FLOAT;
        default:
            megdnn_assert(
                    false, "Invalid Reduce Computation Type for Cambricon: %d",
                    static_cast<int>(data_type));
    }
}
struct ReduceForwardCnnlDescs {
    CnnlReduceDescriptor cnnl_reduce_dsc;
    CnnlTensorDescriptor cnnl_src_dsc, cnnl_dst_dsc;
    ReduceForwardCnnlDescs(
            const TensorLayout& src_layout, const TensorLayout& dst_layout,
            const Param& param) {
        int axis[] = {static_cast<int>(param.axis)};
        cnnlReduceOp_t reduce_op = convert_to_cnnl_reduce_op(param.mode);
        auto computation_dtype = convert_reduce_computation_dtype_to_cnnl_data_type(
                param.data_type, src_layout.dtype.enumv());
        cnnl_reduce_dsc.set(
                axis, 1, reduce_op, computation_dtype, CNNL_NOT_PROPAGATE_NAN,
                CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);

        cnnl_src_dsc.set(src_layout);
        cnnl_dst_dsc.set(dst_layout);
    }
};

void check_param_reduce_forward(
        Handle* handle, const TensorLayout& src_layout, const TensorLayout& dst_layout,
        const Param& param) {
    megdnn_assert(
            src_layout.is_contiguous(), "Reduce Opr Requires input is contiguous\n");
    megdnn_assert(
            dst_layout.is_contiguous(), "Reduce Opr Requires output is contiguous\n");
    auto src_dtype = src_layout.dtype.enumv();
    auto dst_dtype = dst_layout.dtype.enumv();
    megdnn_assert(
            (src_dtype == dst_dtype),
            "Reduce Opr Requires data type of input and output must be the "
            "same, but got %d and %d\n",
            static_cast<int>(src_dtype), static_cast<int>(dst_dtype));
    auto device_name = concrete_handle(handle)->device_info().name;
    // TODO: BFloat16 is not supported in case of DEFAULT
    if (param.mode == Param::Mode::MAX || param.mode == Param::Mode::MIN ||
        param.mode == Param::Mode::PRODUCT || param.mode == Param::Mode::MEAN ||
        param.mode == Param::Mode::SUM) {
        megdnn_assert(
                src_dtype == DTypeEnum::Float32 || src_dtype == DTypeEnum::Float16 ||
                        (src_dtype == DTypeEnum::BFloat16 &&
                         0 == strcmp(device_name, "MLU590")) ||
                        src_dtype == DTypeEnum::Int32,
                "Reduce Opr Requires data type of input and output is float32, "
                "float16, bfloat16(on MLU590) or int32 when mode is MAX, MIN, PRODUCT, "
                "MEAN or "
                "SUM, but got %d\n",
                static_cast<int>(src_dtype));
    } else if (param.mode == Param::Mode::SUM_SQR) {
        megdnn_assert(
                src_dtype == DTypeEnum::Float32 || src_dtype == DTypeEnum::Float16 ||
                        (src_dtype == DTypeEnum::BFloat16 &&
                         0 == strcmp(device_name, "MLU590")),
                "Reduce Opr Requires data type of input and output is float32, "
                "float16 or bfloat16(on MLU590) when mode is SUM_SQR, but got %d\n",
                static_cast<int>(src_dtype));
    } else {
        // TODO: CNNL_REDUCE_SUMSQ only supports Caffe framework.
        megdnn_assert(
                false,
                "Reduce Opr Requires mode is MAX, MIN, PRODUCT, MEAN, SUM or SUM_SQR, "
                "but got %d\n",
                static_cast<int>(param.mode));
    }
}

}  // namespace

void ReduceForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_param_reduce_forward(this->handle(), src.layout, dst.layout, param());
    auto handle = concrete_handle(this->handle());

    ReduceForwardCnnlDescs decs(src.layout, dst.layout, param());
    cnnl_check(cnnlReduce(
            handle->cnnl_handle(), decs.cnnl_reduce_dsc.desc(), workspace.ptr<void>(),
            workspace.size, nullptr, decs.cnnl_src_dsc.desc(), src.raw_ptr(), 0,
            nullptr, nullptr, decs.cnnl_dst_dsc.desc(), dst.raw_ptr()));
}

size_t ReduceForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    auto handle = concrete_handle(this->handle());
    size_t workspace_size = 0;

    ReduceForwardCnnlDescs decs(src, dst, param());
    cnnl_check(cnnlGetReduceOpWorkspaceSize(
            handle->cnnl_handle(), decs.cnnl_src_dsc.desc(), decs.cnnl_dst_dsc.desc(),
            decs.cnnl_reduce_dsc.mut_desc(), &workspace_size));
    return workspace_size;
}

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
