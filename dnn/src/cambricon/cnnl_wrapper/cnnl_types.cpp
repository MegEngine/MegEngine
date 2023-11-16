#include "src/cambricon/cnnl_wrapper/cnnl_types.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cambricon {

cnnlDataType_t convert_to_cnnl_datatype(::megdnn::DTypeEnum dtype) {
    switch (dtype) {
        case ::megdnn::DTypeEnum::Float32:
            return CNNL_DTYPE_FLOAT;
#if !MEGDNN_DISABLE_FLOAT16
        case ::megdnn::DTypeEnum::Float16:
            return CNNL_DTYPE_HALF;

        case ::megdnn::DTypeEnum::BFloat16:
            return CNNL_DTYPE_BFLOAT16;
#endif
        case ::megdnn::DTypeEnum::Int32:
            return CNNL_DTYPE_INT32;
        case ::megdnn::DTypeEnum::Int16:
            return CNNL_DTYPE_INT16;
        case ::megdnn::DTypeEnum::Int8:
            return CNNL_DTYPE_INT8;
        case ::megdnn::DTypeEnum::Uint16:
            return CNNL_DTYPE_UINT16;
        case ::megdnn::DTypeEnum::Uint8:
            return CNNL_DTYPE_UINT8;
        case ::megdnn::DTypeEnum::Bool:
            return CNNL_DTYPE_BOOL;
        default:
            megdnn_assert(
                    false, "Unspport DTypeEnum in Cambricon {%d}",
                    static_cast<int>(dtype));
    };
}

}  // namespace cambricon
}  // namespace megdnn
