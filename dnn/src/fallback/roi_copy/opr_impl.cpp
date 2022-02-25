#include "src/fallback/roi_copy/opr_impl.h"
#include "src/fallback/handle.h"

#include "src/common/cv/common.h"
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"

#include <cstring>

namespace megdnn {
namespace fallback {

void ROICopyImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    size_t N = dst.layout.shape[0], OH = dst.layout.shape[1], OW = dst.layout.shape[2],
           OC = dst.layout.shape[3];
    ptrdiff_t istride0 = src.layout.stride[0], istride1 = src.layout.stride[1],
              istride2 = src.layout.stride[2], istride3 = src.layout.stride[3];
    auto row_from = param().row_from;
    auto col_from = param().col_from;

    auto kern = [=]() {
        TensorLayout relayout_src_layout(
                {N, OH, OW, OC}, {istride0, istride1, istride2, istride3},
                src.layout.dtype);
        TensorND relayout_src(
                static_cast<char*>(src.raw_ptr()) +
                        (row_from * istride1 + col_from * istride2) *
                                src.layout.dtype.size(),
                relayout_src_layout);

        auto relayout = inplace_cpu_handle(0)->create_operator<Relayout>();
        relayout->exec(relayout_src, dst);
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(kern());
}

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
