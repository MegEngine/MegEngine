#include "src/cambricon/utils.h"
#include "src/cambricon/utils.mlu.h"

#include "src/cambricon/handle.h"
#include "src/common/utils.h"

#include <mutex>
#include <unordered_map>

using namespace megdnn;
using namespace cambricon;

void cambricon::__throw_cndrv_error__(CNresult err, const char* msg) {
    const char* ret = nullptr;
    cnGetErrorString(err, &ret);
    if (!ret) {
        ret = "invalid_stub_call";
    }
    auto s = ssprintf("cndrv return %s(%d) occurred; expr: %s", ret, int(err), msg);
    megdnn_throw(s.c_str());
}

void cambricon::__throw_cnrt_error__(cnrtRet_t err, const char* msg) {
    auto s = ssprintf(
            "cnrt return %s(%d) occurred; expr: %s", cnrtGetErrorStr(err), int(err),
            msg);
    megdnn_throw(s.c_str());
}

void cambricon::__throw_cnnl_error__(cnnlStatus_t err, const char* msg) {
    auto s = ssprintf(
            "cnnl return %s(%d) occurred; expr: %s", cnnlGetErrorString(err), int(err),
            msg);
    megdnn_throw(s.c_str());
}

cnrtDeviceProp_t cambricon::current_device_info() {
    int dev_id = -1;
    cnrtDeviceProp_t device_info;
    cnrt_check(cnrtGetDevice(&dev_id));
    cnrtGetDeviceProperties(&device_info, dev_id);
    return device_info;
}

bool cambricon::check_dtype_float_ieee(megdnn::DTypeEnum dtype) {
    return dtype == megdnn::DTypeEnum::Float32 || dtype == megdnn::DTypeEnum::Float16;
}
bool cambricon::check_dtype_int_all(megdnn::DTypeEnum dtype) {
    return dtype == megdnn::DTypeEnum::Int32 || dtype == megdnn::DTypeEnum::Int16 ||
           dtype == megdnn::DTypeEnum::Int8 || dtype == megdnn::DTypeEnum::Uint8;
}

bool cambricon::check_dtype_int(megdnn::DTypeEnum dtype) {
    return dtype == megdnn::DTypeEnum::Int16 || dtype == megdnn::DTypeEnum::Int32;
}
bool cambricon::check_dtype_float(megdnn::DTypeEnum dtype) {
    return dtype == megdnn::DTypeEnum::Float32 || dtype == megdnn::DTypeEnum::Float16 ||
           dtype == megdnn::DTypeEnum::BFloat16;
}
bool cambricon::check_dtype(megdnn::DTypeEnum dtype) {
    return check_dtype_int(dtype) || check_dtype_float(dtype);
}
// vim: syntax=cpp.doxygen
