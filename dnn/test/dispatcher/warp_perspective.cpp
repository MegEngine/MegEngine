#include "test/cpu/fixture.h"

#include "megdnn/oprs/imgproc.h"
#include "test/common/null_dispatcher.h"
#include "test/common/utils.h"

namespace megdnn {
namespace test {

#if !MEGDNN_NO_THREAD
TEST(DISPATCHER, WARP_PERSPECTIVE) {
    std::shared_ptr<MegcoreCPUDispatcher> dispatcher =
            std::make_shared<NullDispatcher>();
    auto handle = create_cpu_handle_with_dispatcher(0, dispatcher);

    auto opr = handle->create_operator<WarpPerspective>();
    auto src_layout = TensorLayout({2, 3, 10, 10}, dtype::Float32()),
         mat_layout = TensorLayout({2, 3, 3}, dtype::Float32()),
         dst_layout = TensorLayout({2, 3, 10, 10}, dtype::Float32());
    TensorND src(nullptr, src_layout), mat(nullptr, mat_layout),
            dst(nullptr, dst_layout);
    opr->param().imode = param::WarpPerspective::InterpolationMode::LINEAR;
    auto wsize = opr->get_workspace_in_bytes(src_layout, mat_layout, dst_layout);
    Workspace workspace(nullptr, wsize);

    opr->exec(src, mat, dst, workspace);
}
#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
