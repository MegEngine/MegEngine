#include "src/naive/param_pack/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

template <typename T>
void exec_internal(
        _megdnn_tensor_in srcs, int32_t* offsets, _megdnn_tensor_out dst,
        _megdnn_workspace) {
    auto srcs_ptr = static_cast<const T**>(srcs.raw_ptr());
    auto dst_ptr = dst.ptr<T>();

    int32_t last_pos = 0;
    for (size_t i = 0; i < srcs.layout[0]; i++) {
        int32_t begin = offsets[i * 2], end = offsets[i * 2 + 1];
        while (last_pos < begin) {
            dst_ptr[last_pos] = 0;
            last_pos++;
        }
        for (int32_t j = 0; j < end - begin; j++) {
            dst_ptr[begin + j] = srcs_ptr[i][j];
        }
        last_pos = end;
    }
}

void ParamPackConcatImpl::exec(
        _megdnn_tensor_in srcs, _megdnn_tensor_in offsets, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(dst.layout, offsets.layout, srcs.layout);
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC

#define cb(DType)                                                                    \
    if (dst.layout.dtype == DType()) {                                               \
        using ctype = typename DTypeTrait<DType>::ctype;                             \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                                \
                exec_internal<ctype>(srcs, offsets.ptr<int32_t>(), dst, workspace)); \
        return;                                                                      \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    megdnn_throw("bad type");
#undef cb

#else
    __builtin_trap();
#endif
}
