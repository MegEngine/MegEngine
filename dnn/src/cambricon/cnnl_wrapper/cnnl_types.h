#pragma once

#include "cnnl.h"
#include "megdnn/basic_types.h"

namespace megdnn {
namespace cambricon {

cnnlDataType_t convert_to_cnnl_datatype(::megdnn::DTypeEnum dtype);

}  // namespace cambricon
}  // namespace megdnn
