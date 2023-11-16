#include "src/cambricon/param_pack/opr_impl.h"
#include "./param_pack.mlu.h"
#include "megdnn/dtype.h"
#include "src/cambricon/cnnl_wrapper/cnnl_types.h"
#include "src/cambricon/utils.h"
#include "src/cambricon/utils.mlu.h"

namespace megdnn {
namespace cambricon {

size_t ParamPackConcatImpl::get_workspace_in_bytes(
        const TensorShape&, const TensorShape& offsets, const TensorShape&) {
    return sizeof(size_t) * (offsets.shape[0] / 2);
}

void ParamPackConcatImpl::exec(
        _megdnn_tensor_in srcs, _megdnn_tensor_in offsets, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(dst.layout, offsets.layout, srcs.layout);
    auto queue = cnrt_queue(this->handle());

    size_t inp_size = srcs.layout.shape[0], out_size = dst.layout.total_nr_elems();
    size_t dtype_size = dst.layout.dtype.size();

    const void** src_cpu = static_cast<const void**>(srcs.raw_ptr());
    megdnn_assert_internal(src_cpu);
    const void** src_gpu = reinterpret_cast<const void**>(workspace.raw_ptr);

    int32_t* offsets_gpu = offsets.ptr<int32_t>();
    cnrt_check(cnrtMemcpyAsync(
            src_gpu, src_cpu, sizeof(void*) * inp_size, queue, cnrtMemcpyHostToDev));
    cnrt_check(cnrtMemsetAsync(dst.raw_ptr(), 0, out_size * dtype_size, queue));

    auto bang_handle = concrete_banghandle(this->handle());
    param_pack::concat_proxy(
            bang_handle, src_gpu, dst.raw_ptr(), offsets_gpu, dtype_size, inp_size);
}

}  // namespace cambricon
}  // namespace megdnn
