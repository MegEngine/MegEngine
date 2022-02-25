#include "test/cuda/fixture.h"

#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, CONCAT) {
    Checker<Concat> checker(handle_cuda());
    using Param = Concat::Param;
    for (auto dtype : std::vector<DType>{dtype::Float32(), dtype::Float16()})
        for (size_t axis = 0; axis < 4; ++axis) {
            Param param;
            param.axis = axis;
            TensorShapeArray shapes(4, TensorShape({2, 3, 4, 5}));
            for (size_t i = 0; i < 4; ++i) {
                shapes[i].shape[axis] = i + 1;
            }
            shapes.emplace_back();
            for (size_t i = 0; i < shapes.size(); ++i)
                checker.set_dtype(i, dtype);
            checker.set_param(param).exec(shapes);
        }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
