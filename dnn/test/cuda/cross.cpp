#include "test/cuda/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, CROSS) {
    Checker<Cross> checker(handle_cuda());
    for (DType dtype :
         std::vector<DType>{dtype::Float16(), dtype::Int32(), dtype::Float32()}) {
        checker.set_param({-2, 1, -1})
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype);
        checker.exec(TensorShapeArray{{2, 3, 4}, {2, 3, 4}, {2, 4, 3}});

        checker.set_param({0, -1, 2})
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype);
        checker.exec(TensorShapeArray{{3, 2, 3, 4}, {2, 3, 4, 3}, {2, 3, 3, 4}});
    }
}

}  // namespace test
}  // namespace megdnn
   // vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}