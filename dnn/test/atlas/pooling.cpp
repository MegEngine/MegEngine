#include "test/common/pooling.h"
#include "test/atlas/fixture.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(ATLAS, POOLING_FORWARD) {
    using namespace pooling;
    using Param = param::Pooling;
    using Mode = param::Pooling::Mode;
    using Format = param::Pooling::Format;
    std::vector<TestArg> args;

    Checker<Pooling> checker(handle_atlas());

    auto run_pooling = [&](size_t N, size_t C, size_t H, size_t W, size_t pad_h,
                           size_t pad_w, size_t stride_h, size_t stride_w,
                           size_t window_h, size_t window_w, Mode mode, DType dtype0,
                           DType dtype1) {
        Param param1;
        param1.mode = mode;
        param1.format = Format::NCHW;
        param1.pad_h = pad_h;
        param1.pad_w = pad_w;
        param1.stride_h = stride_h;
        param1.stride_w = stride_w;
        param1.window_h = window_h;
        param1.window_w = window_w;
        checker.set_param(param1).set_dtype(0, dtype0).set_dtype(1, dtype1);
        if (dtype0 == dtype::Float16() || dtype1 == dtype::Float16()) {
            checker.set_epsilon(1e-2);
        }
        checker.execs({{N, C, H, W}, {}});
    };

#define RUN_MAX_POOLING(_dt0, _dt1)                                        \
    run_pooling(1, 64, 112, 112, 1, 1, 2, 2, 3, 3, Mode::MAX, _dt0, _dt1); \
    run_pooling(2, 64, 112, 112, 1, 1, 2, 2, 3, 3, Mode::MAX, _dt0, _dt1);
    RUN_MAX_POOLING(dtype::Float32(), dtype::Float32());
    RUN_MAX_POOLING(dtype::Float16(), dtype::Float16());
#undef RUN_MAX_POOLING

#define RUN_AVG_POOLING(_dt)                                               \
    run_pooling(1, 2048, 7, 7, 1, 1, 1, 1, 7, 7, Mode::AVERAGE, _dt, _dt); \
    run_pooling(2, 2048, 7, 7, 0, 0, 1, 1, 7, 7, Mode::AVERAGE, _dt, _dt);
    RUN_AVG_POOLING(dtype::Float32());
    RUN_AVG_POOLING(dtype::Float16());
#undef RUN_AVG_POOLING

#define RUN_AVG_EXCL_POOLING(_dt)                                                   \
    run_pooling(                                                                    \
            1, 64, 112, 112, 0, 0, 2, 2, 3, 3, Mode::AVERAGE_COUNT_EXCLUDE_PADDING, \
            _dt, _dt);                                                              \
    run_pooling(                                                                    \
            2, 64, 112, 112, 1, 1, 2, 2, 3, 3, Mode::AVERAGE_COUNT_EXCLUDE_PADDING, \
            _dt, _dt);
    RUN_AVG_EXCL_POOLING(dtype::Float32());
    RUN_AVG_EXCL_POOLING(dtype::Float16());
#undef RUN_AVG_EXCL_POOLING
}

TEST_F(ATLAS, POOLING_BACKWARD) {
    using namespace pooling;
    using Param = param::Pooling;
    using Mode = param::Pooling::Mode;
    using Format = param::Pooling::Format;
    std::vector<TestArg> args;

    Checker<PoolingBackward> checker(handle_atlas());

    auto run_pooling = [&](size_t N, size_t C, size_t H, size_t W, size_t pad_h,
                           size_t pad_w, size_t stride_h, size_t stride_w,
                           size_t window_h, size_t window_w, Mode mode, DType dtype) {
        Param param;
        param.mode = mode;
        param.format = Format::NCHW;
        param.pad_h = pad_h;
        param.pad_w = pad_w;
        param.stride_h = stride_h;
        param.stride_w = stride_w;
        param.window_h = window_h;
        param.window_w = window_w;
        checker.set_param(param)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype);
        size_t OH = (H + 2 * pad_h - window_h) / stride_h + 1;
        size_t OW = (W + 2 * pad_w - window_w) / stride_w + 1;

        auto constraint = [&](CheckerHelper::TensorValueArray& tensors_orig) {
            megdnn_assert(tensors_orig.size() == 4);
            auto opr = handle_naive()->create_operator<PoolingForward>();
            opr->param() = param;
            opr->exec(tensors_orig[0], tensors_orig[1], {nullptr, 0});
        };
        checker.set_tensors_constraint(constraint);
        if (dtype == dtype::Float16()) {
            checker.set_epsilon(1e-2);
        }
        checker.execs({{N, C, H, W}, {N, C, OH, OW}, {N, C, OH, OW}, {N, C, H, W}});
    };

#define RUN_MAX_POOLING(_dt)                                        \
    run_pooling(1, 64, 112, 112, 1, 1, 2, 2, 3, 3, Mode::MAX, _dt); \
    run_pooling(2, 64, 112, 112, 1, 1, 2, 2, 3, 3, Mode::MAX, _dt);
    RUN_MAX_POOLING(dtype::Float32());
#undef RUN_MAX_POOLING

#define RUN_AVG_POOLING(_dt)                                          \
    run_pooling(1, 2048, 7, 7, 0, 0, 1, 1, 7, 7, Mode::AVERAGE, _dt); \
    run_pooling(2, 2048, 7, 7, 0, 0, 1, 1, 7, 7, Mode::AVERAGE, _dt);
    RUN_AVG_POOLING(dtype::Float32());
#undef RUN_AVG_POOLING

#define RUN_AVG_EXCL_POOLING(_dt)                                                   \
    run_pooling(                                                                    \
            1, 64, 112, 112, 1, 1, 2, 2, 3, 3, Mode::AVERAGE_COUNT_EXCLUDE_PADDING, \
            _dt);                                                                   \
    run_pooling(                                                                    \
            2, 64, 112, 112, 1, 1, 2, 2, 3, 3, Mode::AVERAGE_COUNT_EXCLUDE_PADDING, \
            _dt);
    RUN_AVG_EXCL_POOLING(dtype::Float32());
#undef RUN_AVG_EXCL_POOLING
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
