#include "test/common/rotate.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"

#include "test/aarch64/fixture.h"

namespace megdnn {
namespace test {

TEST_F(AARCH64, ROTATE) {
    using namespace rotate;
    std::vector<TestArg> args = get_args();
    Checker<Rotate> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, arg.dtype)
                .set_dtype(1, arg.dtype)
                .execs({arg.src, {}});
    }
}

TEST_F(AARCH64, ROTATE_RECORD) {
    using namespace rotate;
    std::vector<TestArg> args = get_args();
    TaskRecordChecker<Rotate> checker(0);

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, arg.dtype)
                .set_dtype(1, arg.dtype)
                .execs({arg.src, {}});
    }
}

TEST_F(AARCH64, BENCHMARK_ROTATE) {
    using namespace rotate;
    using Param = param::Rotate;

#define BENCHMARK_PARAM(benchmarker) \
    benchmarker.set_param(param);    \
    benchmarker.set_dtype(0, dtype::Uint8());

    auto run = [&](const TensorShapeArray& shapes, Param param) {
        auto handle_naive = create_cpu_handle(2);
        Benchmarker<Rotate> benchmarker(handle());
        Benchmarker<Rotate> benchmarker_naive(handle_naive.get());

        BENCHMARK_PARAM(benchmarker);
        BENCHMARK_PARAM(benchmarker_naive);
        for (auto&& shape : shapes) {
            printf("execute %s: current---naive\n", shape.to_string().c_str());
            benchmarker.execs({shape, {}});
            benchmarker_naive.execs({shape, {}});
        }
    };

    Param param;
    TensorShapeArray shapes = {
            {1, 100, 100, 1},
            {2, 100, 100, 3},
    };

    param.clockwise = true;
    run(shapes, param);

    param.clockwise = false;
    run(shapes, param);
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
