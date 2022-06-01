#include "test/fallback/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

namespace megdnn {
namespace test {

TEST_F(FALLBACK, SOFTMAX_FORWARD) {
    Checker<Softmax> checker(handle());

    Softmax::Param param0{0};
    checker.set_param(param0).exec(TensorShapeArray{{11}, {}});
    checker.set_param(param0).exec(TensorShapeArray{{11, 11}, {}});
    checker.set_param(param0).exec(TensorShapeArray{{11, 11, 11}, {}});
    checker.set_param(param0).exec(TensorShapeArray{{11, 11, 11, 11}, {}});
    checker.set_param(param0).exec(TensorShapeArray{{11, 11, 11, 11, 11}, {}});
    checker.set_param(param0).exec(TensorShapeArray{{11, 7, 5, 5, 5, 11}, {}});
    checker.set_param(param0).exec(TensorShapeArray{{11, 7, 5, 7, 5, 7, 7}, {}});
    Softmax::Param param1{1};
    checker.set_param(param1).exec(TensorShapeArray{{11, 11}, {}});
    checker.set_param(param1).exec(TensorShapeArray{{11, 11, 11}, {}});
    checker.set_param(param1).exec(TensorShapeArray{{11, 11, 11, 11}, {}});
    checker.set_param(param1).exec(TensorShapeArray{{11, 11, 11, 11, 11}, {}});
    checker.set_param(param1).exec(TensorShapeArray{{11, 5, 5, 5, 5, 11}, {}});
    checker.set_param(param1).exec(TensorShapeArray{{11, 7, 5, 7, 5, 7, 7}, {}});
    Softmax::Param param2{2};
    checker.set_param(param2).exec(TensorShapeArray{{11, 11, 11}, {}});
    checker.set_param(param2).exec(TensorShapeArray{{11, 11, 11, 11}, {}});
    checker.set_param(param2).exec(TensorShapeArray{{11, 11, 11, 11, 11}, {}});
    checker.set_param(param2).exec(TensorShapeArray{{11, 5, 5, 5, 5, 11}, {}});
    checker.set_param(param2).exec(TensorShapeArray{{11, 5, 5, 5, 5, 7, 7}, {}});
    Softmax::Param param3{3};
    checker.set_param(param3).exec(TensorShapeArray{{11, 11, 11, 11}, {}});
    checker.set_param(param3).exec(TensorShapeArray{{11, 11, 11, 11, 11}, {}});
    checker.set_param(param3).exec(TensorShapeArray{{11, 5, 5, 5, 5, 11}, {}});
    checker.set_param(param3).exec(TensorShapeArray{{11, 5, 5, 5, 5, 7, 7}, {}});
    Softmax::Param param4{4};
    checker.set_param(param4).exec(TensorShapeArray{{11, 11, 11, 11, 11}, {}});
    checker.set_param(param4).exec(TensorShapeArray{{11, 5, 5, 5, 5, 11}, {}});
    checker.set_param(param4).exec(TensorShapeArray{{11, 5, 5, 5, 5, 7, 7}, {}});
    Softmax::Param param5{5};
    checker.set_param(param5).exec(TensorShapeArray{{11, 5, 5, 5, 5, 11}, {}});
    checker.set_param(param5).exec(TensorShapeArray{{11, 5, 5, 5, 5, 7, 7}, {}});
    Softmax::Param param6{6};
    checker.set_param(param6).exec(TensorShapeArray{{11, 5, 5, 5, 5, 7, 7}, {}});
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
