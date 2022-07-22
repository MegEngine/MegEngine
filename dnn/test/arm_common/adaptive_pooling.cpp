#include "test/arm_common/fixture.h"

#include "megdnn/tensor_iter.h"
#include "src/common/utils.h"
#include "test/common/adaptive_pooling.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(ARM_COMMON, ADAPTIVE_POOLING_FORWARD_NCHW44) {
    auto args = adaptive_pooling::get_args_nchw44();
    Checker<AdaptivePooling> checker(handle());
    checker.set_epsilon(1e-4);
    for (DType dtype : {(DType)dtype::Float32(), (DType)dtype::QuantizedS8(1.0)})
        for (auto&& arg : args) {
            auto param = arg.param;
            auto src = arg.ishape;
            auto dst = arg.oshape;

            checker.set_param(param).set_dtype(0, dtype).set_dtype(1, dtype).exec(
                    TensorShapeArray{src, dst, {}});
        }
}

TEST_F(ARM_COMMON, ADAPTIVE_POOLING_FORWARD) {
    auto args = adaptive_pooling::get_args();
    Checker<AdaptivePooling> checker(handle());
    checker.set_epsilon(1e-4);
    for (DType dtype : {(DType)dtype::Float32(), (DType)dtype::QuantizedS8(1.0)})
        for (auto&& arg : args) {
            auto param = arg.param;
            auto src = arg.ishape;
            auto dst = arg.oshape;

            checker.set_param(param).set_dtype(0, dtype).set_dtype(1, dtype).exec(
                    TensorShapeArray{src, dst, {}});
        }
}

#if MEGDNN_WITH_BENCHMARK
namespace {
void benchmark_globalpooling_nchw44_fp32(Handle* handle) {
    using Param = param::AdaptivePooling;
    auto run = [&](size_t n, size_t c, size_t h, size_t w, Param::Mode mode) {
        Param param;
        param.format = Param::Format::NCHW;
        param.mode = mode;
        TensorShape nchw_shape = {n, c, h, w};
        TensorShape nchw_dst_shape = {n, c, 1, 1};
        TensorShape nchw44_shape = {n, c / 4, h, w, 4};
        TensorShape nchw44_dst_shape = {n, c / 4, 1, 1, 4};
        TensorLayout dst_layout;
        float calc_amount = n * c * h * w;

        Benchmarker<AdaptivePooling> benchmarker_float_nchw(handle);
        Benchmarker<AdaptivePooling> benchmarker_float_nchw44(handle);
        Benchmarker<AdaptivePooling> benchmarker_int_nchw44(handle);
        size_t RUN = 500;
        auto t1 = benchmarker_float_nchw.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .exec({nchw_shape, nchw_dst_shape});

        param.format = Param::Format::NCHW44;
        auto t2 = benchmarker_int_nchw44.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .execl({{nchw44_shape, dtype::QuantizedS8(1.0)},
                                  {nchw44_dst_shape, dtype::QuantizedS8(1.0)}});

        auto t3 = benchmarker_float_nchw44.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .exec({nchw44_shape, nchw44_dst_shape});

        printf("{%zu %zu %zu %zu} \n"
               "nchw_fp32={%.3f ms, %.3f Mflops},  "
               "nchw44_int={%.3f ms, %.3f Mflops},  "
               "nchw44_fp32={%.3f ms, %.3f Mflops, speed_up %f}\n\n",
               n, c, h, w, t1 / RUN, calc_amount / (t1 / RUN * 1000), t2 / RUN,
               calc_amount / (t2 / RUN * 1000), t3 / RUN,
               calc_amount / (t3 / RUN * 1000), t1 / t3);
    };

    run(1, 128, 25, 25, param::AdaptivePooling::Mode::AVERAGE);
}
}  // namespace
TEST_F(ARM_COMMON, BENCHMARK_GLOBAL_POOLING_NCHW44_FP32) {
    benchmark_globalpooling_nchw44_fp32(handle());
}

#endif
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
