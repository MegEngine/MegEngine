#pragma once

#include "megcore_cdefs.h"
#include "megdnn/handle.h"
#include "src/cambricon/utils.mlu.h"
#include "src/common/utils.h"

#include "src/cambricon/handle.h"

#include <cnnl.h>
#include <cnrt.h>
#include "megdnn/dtype.h"

namespace megdnn {
namespace cambricon {

static inline HandleImpl* concrete_handle(Handle* handle) {
    return static_cast<cambricon::HandleImpl*>(handle);
}

static inline cnnlHandle_t cnnl_handle(Handle* handle) {
    return concrete_handle(handle)->cnnl_handle();
}

static inline cnrtQueue_t cnrt_queue(Handle* handle) {
    return concrete_handle(handle)->queue();
}

inline BangHandle concrete_banghandle(Handle* handle) {
    BangHandle bang_handle(
            concrete_handle(handle)->queue(),
            concrete_handle(handle)->device_info().clusterCount,
            concrete_handle(handle)->device_info().McorePerCluster);
    return bang_handle;
}
//! get device info of current active device
cnrtDeviceProp_t current_device_info();

bool check_dtype_int(megdnn::DTypeEnum dtype);
bool check_dtype_int_all(megdnn::DTypeEnum dtype);
bool check_dtype_float(megdnn::DTypeEnum dtype);
bool check_dtype_float_ieee(megdnn::DTypeEnum dtype);
bool check_dtype(megdnn::DTypeEnum dtype);

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
