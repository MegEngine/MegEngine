#include <gtest/gtest.h>

#include "megdnn/oprs.h"
#include "test/common/null_dispatcher.h"
#include "test/common/tensor.h"
#include "test/common/utils.h"
#include "test/common/workspace_wrapper.h"

namespace megdnn {
namespace test {

#if !MEGDNN_NO_THREAD
TEST(DISPATCHER, NULL_DISPATCHER) {
    std::shared_ptr<MegcoreCPUDispatcher> dispatcher =
            std::make_shared<NullDispatcher>();
    auto handle = create_cpu_handle_with_dispatcher(0, dispatcher);

    auto opr = handle->create_operator<Convolution>();

    auto layout = TensorLayout({1, 1, 1, 1}, dtype::Float32());
    TensorND src(nullptr, layout), filter(nullptr, layout), dst(nullptr, layout);
    auto wsize = opr->get_workspace_in_bytes(layout, layout, layout, nullptr);
    Workspace workspace(nullptr, wsize);

    opr->exec(src, filter, dst, nullptr, workspace);
}
#endif

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
