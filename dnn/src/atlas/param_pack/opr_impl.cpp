#include "src/atlas/param_pack/opr_impl.h"
#include "megdnn/dtype.h"
#include "src/atlas/utils.h"

namespace megdnn {
namespace atlas {

size_t ParamPackConcatImpl::get_workspace_in_bytes(
        const TensorShape&, const TensorShape&, const TensorShape&) {
    return 0;
}

void ParamPackConcatImpl::exec(
        _megdnn_tensor_in srcs, _megdnn_tensor_in offsets, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(dst.layout, offsets.layout, srcs.layout);

    size_t inp_size = srcs.layout.shape[0], out_size = dst.layout.total_nr_elems();
    int32_t* offset_vals = offsets.ptr<int32_t>();
    int8_t* addr = static_cast<int8_t*>(dst.raw_ptr());
    void** srcs_ptr = static_cast<void**>(srcs.raw_ptr());
    size_t dtype_size = dst.layout.dtype.size();

    auto stream = concrete_handle(this->handle())->stream();
    acl_check(aclrtMemsetAsync(
            addr, out_size * dtype_size, 0, out_size * dtype_size, stream));
    for (size_t i = 0; i < inp_size; i++) {
        void* src = srcs_ptr[i];
        size_t data_size = (offset_vals[i * 2 + 1] - offset_vals[i * 2]) * dtype_size;
        acl_safe_memcpy_async(
                addr + offset_vals[i * 2] * dtype_size, data_size, src, data_size,
                ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
}

}  // namespace atlas
}  // namespace megdnn
