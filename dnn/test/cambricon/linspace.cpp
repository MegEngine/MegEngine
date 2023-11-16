#include "test/cambricon/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CAMBRICON, LINSPACE) {
    Checker<Linspace> checker(handle_cambricon());
    Linspace::Param param;
    param.start = 0.5;
    param.stop = 1.5;
    param.endpoint = true;
    for (DType dtype :
         std::vector<DType>{dtype::Float32(), dtype::Float16(), dtype::Int32()}) {
        checker.set_dtype(0, dtype).set_param(param).exec(TensorShapeArray{{11}});
    }
    param.endpoint = false;
    for (DType dtype :
         std::vector<DType>{dtype::Float32(), dtype::Float16(), dtype::Int32()}) {
        checker.set_dtype(0, dtype).set_param(param).exec(TensorShapeArray{{11}});
    }
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
