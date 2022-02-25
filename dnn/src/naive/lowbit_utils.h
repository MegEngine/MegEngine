#pragma once
#include "megdnn/basic_types.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

void uint4_to_uint8(const TensorND& in, const TensorND& out);

void uint8_to_uint4(const TensorND& in, const TensorND& out);

void uint4_to_int8(const TensorND& in, const TensorND& out);

void int8_to_uint4(const TensorND& in, const TensorND& out);

void int4_to_int8(const TensorND& in, const TensorND& out);

void int8_to_int4(const TensorND& in, const TensorND& out);

}  // namespace naive
}  // namespace megdnn
