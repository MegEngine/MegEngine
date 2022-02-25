#include "test/common/cvt_color.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rotate.h"

#include "test/x86/fixture.h"

#include "test/common/task_record_check.h"
namespace megdnn {
namespace test {

using Mode = param::CvtColor::Mode;

TEST_F(X86, CVTCOLOR) {
    using namespace cvt_color;
    std::vector<TestArg> args = get_args();
    Checker<CvtColor> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, arg.dtype)
                .set_dtype(1, arg.dtype)
                .execs({arg.src, {}});
    }
}

TEST_F(X86, CVTCOLOR_RECORD) {
    using namespace cvt_color;
    std::vector<TestArg> args = get_args();
    TaskRecordChecker<CvtColor> checker(0);
    checker.set_param(args[0].param)
            .set_dtype(0, args[0].dtype)
            .set_dtype(1, args[0].dtype)
            .execs({args[0].src, {}});
}

#ifdef MEGDNN_WITH_BENCHMARK
TEST_F(X86, BENCHMARK_CVTCOLOR_RGB2GRAY) {
    using namespace cvt_color;
    using Param = param::CvtColor;

#define BENCHMARK_PARAM(benchmarker, dtype) \
    benchmarker.set_param(param);           \
    benchmarker.set_dtype(0, dtype);

    auto run = [&](const TensorShapeArray& shapes, Param param) {
        auto handle_naive = create_cpu_handle(2);
        Benchmarker<CvtColor> benchmarker(handle());
        Benchmarker<CvtColor> benchmarker_naive(handle_naive.get());

        BENCHMARK_PARAM(benchmarker, dtype::Uint8());
        BENCHMARK_PARAM(benchmarker_naive, dtype::Uint8());
        for (auto&& shape : shapes) {
            printf("execute %s: current---naive\n", shape.to_string().c_str());
            benchmarker.execs({shape, {}});
            benchmarker_naive.execs({shape, {}});
        }

        BENCHMARK_PARAM(benchmarker, dtype::Float32());
        BENCHMARK_PARAM(benchmarker_naive, dtype::Float32());
        for (auto&& shape : shapes) {
            printf("execute %s: current---naive\n", shape.to_string().c_str());
            benchmarker.execs({shape, {}});
            benchmarker_naive.execs({shape, {}});
        }
    };

    Param param;
    TensorShapeArray shapes = {
            {1, 500, 512, 3},
            {2, 500, 512, 3},
    };

    param.mode = Param::Mode::RGB2GRAY;
    run(shapes, param);

#undef BENCHMARK_PARAM
}

TEST_F(X86, BENCHMARK_CVTCOLOR_BT601_YUV) {
    using namespace cvt_color;
    using Param = param::CvtColor;

#define BENCHMARK_PARAM(benchmarker, dtype) \
    benchmarker.set_param(param);           \
    benchmarker.set_dtype(0, dtype);

    auto run = [&](const TensorShapeArray& shapes, Param param) {
        auto handle_naive = create_cpu_handle(2);
        Benchmarker<CvtColor> benchmarker(handle());
        Benchmarker<CvtColor> benchmarker_naive(handle_naive.get());

        BENCHMARK_PARAM(benchmarker, dtype::Uint8());
        BENCHMARK_PARAM(benchmarker_naive, dtype::Uint8());
        for (auto&& shape : shapes) {
            printf("execute %s: current---naive\n", shape.to_string().c_str());
            benchmarker.execs({shape, {}});
            benchmarker_naive.execs({shape, {}});
        }
    };

    Param param;
    TensorShapeArray shapes = {
            {1, 300, 512, 1},
    };

    param.mode = Param::Mode::BT601_YUV2RGB_NV21;
    run(shapes, param);

#undef BENCHMARK_PARAM
}
#endif

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
