#include "src/cambricon/checksum/opr_impl.h"
#include "src/cambricon/checksum/checksum.mlu.h"

#include "src/cambricon/utils.h"

#include <algorithm>

using namespace megdnn;
using namespace cambricon;

namespace {
void bang_c_wrapper(
        uint32_t* dst, const uint32_t* src, int nr_elems, cnrtQueue_t queue,
        int core_version) {
    if (core_version == 220) {
        checksum_kernel_union1_wrapper(dst, src, nr_elems, queue);
    } else {
        checksum_kernel_union4_wrapper(dst, src, nr_elems, queue);
    }
    after_kernel_launch();
}
}  // namespace

size_t ChecksumForwardImpl::get_workspace_in_bytes(const TensorLayout& /* data */) {
    size_t ws_size = sizeof(ChecksumForward::Result::checksum);
    return ws_size;
}

ChecksumForward::Result ChecksumForwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_workspace workspace) {
    Result result;
    memset(&result, 0, sizeof(result));
    check_exec(data.layout, workspace.size);
    auto queue = cnrt_queue(handle());

    auto ptr = static_cast<uint8_t*>(data.raw_ptr());
    size_t size_all = data.layout.shape[0], size_ints = size_all / sizeof(uint32_t);
    auto last_val_size = std::min<size_t>(size_all, 4);
    cnrt_check(cnrtMemcpyAsync(
            &result.last_val, ptr + size_all - last_val_size, last_val_size, queue,
            CNRT_MEM_TRANS_DIR_DEV2HOST));
    if (size_ints) {
        auto&& device_info = current_device_info();
        bang_c_wrapper(
                reinterpret_cast<uint32_t*>(workspace.raw_ptr),
                static_cast<uint32_t*>(data.raw_ptr()), size_ints, queue,
                device_info.ISAVersion);
        cnrt_check(cnrtMemcpyAsync(
                &result.checksum, workspace.raw_ptr, sizeof(result.checksum), queue,
                CNRT_MEM_TRANS_DIR_DEV2HOST));
    }
    cnrt_check(cnrtQueueSync(queue));
    return result;
}

// vim: syntax=cpp.doxygen
