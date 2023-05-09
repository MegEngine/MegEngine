#include "./dlpack.h"
#include "./tensor.h"

#include "megbrain/imperative/value.h"
#include "megbrain/tensor.h"

namespace mgb {
namespace imperative {

DLManagedTensor* to_dlpack(const ValueRef src);

DLDevice get_dl_device(const DeviceTensorND& dv);

DLDataType get_dl_datatype(const DeviceTensorND& dv);

ValueRef from_dlpack(DLManagedTensor* dlMTensor, int stream);

CompNode get_tensor_device(const DLDevice& ctx, int stream);

mgb::DType get_tensor_type(const DLDataType& dtype);

}  // namespace imperative

}  // namespace mgb