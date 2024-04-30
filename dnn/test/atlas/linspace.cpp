#include "test/common/fill.h"

#include "test/atlas/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(ATLAS, LINSPACE) {
    Checker<Linspace> checker(handle_atlas());
    Linspace::Param param;
    param.start = 0.5;
    param.stop = 1.5;
    for (bool endpoint : std::vector<bool>{false, true}) {
        param.endpoint = endpoint;
        for (DType dtype : std::vector<DType>{dtype::Float16(), dtype::Float32()}) {
            checker.set_dtype(0, dtype).set_param(param).exec(TensorShapeArray{{11}});
        }
    }

    param.start = 0;
    param.stop = 10;
    for (bool endpoint : std::vector<bool>{false, true}) {
        param.endpoint = endpoint;
        for (DType dtype : std::vector<DType>{dtype::Int32(), dtype::Int16()}) {
            checker.set_dtype(0, dtype).set_param(param).exec(TensorShapeArray{{11}});
        }
    }
}