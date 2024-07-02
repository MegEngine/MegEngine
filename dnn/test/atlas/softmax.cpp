#include "test/common/softmax.h"
#include "test/atlas/fixture.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(ATLAS, SOFTMAX_FORWARD) {
    auto args = softmax::get_args();
    std::vector<DType> dtypes{dtype::Float16(), dtype::Float32()};

    for (auto dtype : dtypes)
        for (auto&& arg : args) {
            auto param = arg.param;
            auto src = arg.ishape;
            Checker<Softmax> checker(handle_atlas());
            checker.set_param(param).set_dtype(0, dtype).set_dtype(1, dtype).exec(
                    TensorShapeArray{src, {}});
        }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
