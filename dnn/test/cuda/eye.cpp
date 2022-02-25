#include "test/cuda/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, EYE) {
    Checker<Eye> checker(handle_cuda());
    for (DType dtype :
         std::vector<DType>{dtype::Float16(), dtype::Int32(), dtype::Float32()})
        for (int k = -20; k < 20; ++k) {
            checker.set_param({k, dtype.enumv()});
            checker.set_dtype(0, dtype);
            checker.exec(TensorShapeArray{{3, 4}});
            checker.exec(TensorShapeArray{{4, 3}});
        }
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
