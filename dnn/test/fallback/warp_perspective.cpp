#include "test/fallback/fixture.h"

#include "test/common/checker.h"
#include "test/common/random_state.h"
#include "test/common/rng.h"
#include "test/common/task_record_check.h"
#include "test/common/warp_perspective.h"
namespace megdnn {
namespace test {

TEST_F(FALLBACK, WARP_PERSPECTIVE) {
    Checker<WarpPerspective> checker(handle());
    param::WarpPerspective param;
    class ResizeMatRNG : public RNG {
        void gen(const TensorND& tensor_) override {
            auto& gen = RandomState::generator();
            std::uniform_real_distribution<dt_float32> pdist3(1.9f, 3.1f);
            std::uniform_real_distribution<dt_float32> pdist(0.9f, 1.1f);
            std::uniform_real_distribution<dt_float32> pdisth(0.4f, 0.6f);
            std::uniform_real_distribution<dt_float32> ndist(-1.1f, -0.9f);
            std::uniform_real_distribution<dt_float32> ndist3(-3.1f, -1.9f);
            std::uniform_real_distribution<dt_float32> ndisth(-0.6f, -0.4f);
            std::uniform_int_distribution<int> dice(0, 5);
            float* ptr = tensor_.ptr<dt_float32>();
            auto N = tensor_.layout.shape[0];
            for (size_t n = 0; n < N; ++n) {
                for (size_t i = 0; i < 9; ++i) {
                    switch (dice(gen)) {
                        case 0:
                            ptr[i] = pdist3(gen);
                            break;
                        case 1:
                            ptr[i] = pdist(gen);
                            break;
                        case 2:
                            ptr[i] = pdisth(gen);
                            break;
                        case 3:
                            ptr[i] = ndist(gen);
                            break;
                        case 4:
                            ptr[i] = ndist3(gen);
                            break;
                        case 5:
                            ptr[i] = ndisth(gen);
                            break;
                    }
                }
                // is resize?
                if (n & 1) {
                    ptr[1] = 0;
                    ptr[3] = 0;
                    ptr[6] = ptr[7] = 0;
                }
                ptr += 9;
            }
        }
    } rng;
    checker.set_rng(1, &rng);
    using BMode = param::WarpPerspective::BorderMode;
    param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
    for (auto mode :
         {BMode::REFLECT_101, BMode::REPLICATE, BMode::REFLECT, BMode::WRAP,
          BMode::CONSTANT}) {
        param.bmode = mode;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{1000, 2, 10, 11}, {1000, 3, 3}, {1000, 2, 12, 13}});
    }
#if MEGDNN_TEST_ASAN
//! asan detect nan will make test failed
#else
    // resize nan case
    UniformFloatRNG rng_zero(0, 0);
    checker.set_rng(1, &rng_zero);
    {
        param.bmode = BMode::CONSTANT;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{1000, 2, 10, 11}, {1000, 3, 3}, {1000, 2, 12, 13}});
    }
#endif
}

TEST_F(FALLBACK, WARP_PERSPECTIVE_RECORD) {
    TaskRecordChecker<WarpPerspective> checker(1);
    param::WarpPerspective param;
    class ResizeMatRNG : public RNG {
        void gen(const TensorND& tensor_) override {
            auto& gen = RandomState::generator();
            std::uniform_real_distribution<dt_float32> pdist3(1.9f, 3.1f);
            std::uniform_real_distribution<dt_float32> pdist(0.9f, 1.1f);
            std::uniform_real_distribution<dt_float32> pdisth(0.4f, 0.6f);
            std::uniform_real_distribution<dt_float32> ndist(-1.1f, -0.9f);
            std::uniform_real_distribution<dt_float32> ndist3(-3.1f, -1.9f);
            std::uniform_real_distribution<dt_float32> ndisth(-0.6f, -0.4f);
            std::uniform_int_distribution<int> dice(0, 5);
            float* ptr = tensor_.ptr<dt_float32>();
            auto N = tensor_.layout.shape[0];
            for (size_t n = 0; n < N; ++n) {
                for (size_t i = 0; i < 9; ++i) {
                    switch (dice(gen)) {
                        case 0:
                            ptr[i] = pdist3(gen);
                            break;
                        case 1:
                            ptr[i] = pdist(gen);
                            break;
                        case 2:
                            ptr[i] = pdisth(gen);
                            break;
                        case 3:
                            ptr[i] = ndist(gen);
                            break;
                        case 4:
                            ptr[i] = ndist3(gen);
                            break;
                        case 5:
                            ptr[i] = ndisth(gen);
                            break;
                    }
                }
                // is resize?
                if (n & 1) {
                    ptr[1] = 0;
                    ptr[3] = 0;
                    ptr[6] = ptr[7] = 0;
                }
                ptr += 9;
            }
        }
    } rng;
    checker.set_rng(1, &rng);
    using BMode = param::WarpPerspective::BorderMode;
    param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
    // for (auto mode :
    //      {BMode::REFLECT_101, BMode::REPLICATE, BMode::REFLECT, BMode::WRAP,
    //       BMode::CONSTANT}) {
    param.bmode = BMode::REFLECT_101;
    param.border_val = 1.737;
    checker.set_param(param);
    checker.exec({{1, 2, 10, 11}, {1, 3, 3}, {1, 2, 12, 13}});
    // }
#if MEGDNN_TEST_ASAN
//! asan detect nan will make test failed
#else
    // resize nan case
    UniformFloatRNG rng_zero(0, 0);
    checker.set_rng(1, &rng_zero);
    {
        param.bmode = BMode::CONSTANT;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{1000, 2, 10, 11}, {1000, 3, 3}, {1000, 2, 12, 13}});
    }
#endif
}

TEST_F(FALLBACK, WARP_PERSPECTIVE_MAT_IDX) {
    warp_perspective::run_mat_idx_test(handle());
}

TEST_F(FALLBACK, WARP_PERSPECTIFVE_NCHW_INT8) {
    warp_perspective::run_int8_test(handle());
}

TEST_F(FALLBACK, WARP_PERSPECTIFVE_NCHW_INT8_RECORD) {
    warp_perspective::run_int8_test_record(1);
}

TEST_F(FALLBACK, WARP_PERSPECTIFVE_NCHW_QUINT8) {
    warp_perspective::run_quint8_test(handle());
}

TEST_F(FALLBACK, WARP_PERSPECTIVE_MULTI_SRC_NCHW) {
    using Param = WarpPerspective::Param;
    Param param;
    WarpPerspectiveMatRNG rng;

    for (auto bmode :
         {WarpPerspective::BorderMode::WRAP, WarpPerspective::BorderMode::REFLECT,
          WarpPerspective::BorderMode::REPLICATE,
          WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;
        param.format = Param::Format::NCHW;

        auto run = [&param, &rng, this](
                           size_t bs, size_t ih, size_t iw, size_t c, size_t oh,
                           size_t ow, DType dtype) {
            Checker<WarpPerspectiveForward, WarpPerspectiveMultiSrcProxy> checker(
                    handle());
            checker.set_param(param);
            TensorShapeArray shapes;
            // src
            for (size_t i = 0; i < bs; i++) {
                shapes.emplace_back(TensorShape{{1, c, ih, iw}});
                checker.set_dtype(i, dtype);
            }
            // mat
            shapes.emplace_back(TensorShape{{bs, 3, 3}});
            checker.set_rng(bs, &rng);
            // dst
            shapes.emplace_back(TensorShape{{bs, c, oh, ow}});
            checker.set_dtype(bs + 1, dtype);
            checker.execs(shapes);
        };

        for (auto dtype : std::vector<DType>{dtype::Float32(), dtype::Float16()}) {
            run(1, 20, 18, 4, 6, 6, dtype);
            run(20, 10, 11, 123, 15, 16, dtype);
            run(100, 10, 11, 3, 11, 12, dtype);
        }
    }
}

TEST_F(FALLBACK, WARP_PERSPECTIVE_MULTI_SRC_NHWC) {
    using Param = WarpPerspective::Param;
    Param param;
    WarpPerspectiveMatRNG rng;

    for (auto bmode :
         {WarpPerspective::BorderMode::WRAP, WarpPerspective::BorderMode::REFLECT,
          WarpPerspective::BorderMode::REPLICATE,
          WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;
        param.format = Param::Format::NHWC;

        auto run = [&param, &rng, this](
                           size_t bs, size_t ih, size_t iw, size_t c, size_t oh,
                           size_t ow, DType dtype) {
            Checker<WarpPerspectiveForward, WarpPerspectiveMultiSrcProxy> checker(
                    handle());
            checker.set_param(param);
            TensorShapeArray shapes;
            // src
            for (size_t i = 0; i < bs; i++) {
                shapes.emplace_back(TensorShape{{1, ih, iw, c}});
                checker.set_dtype(i, dtype);
            }
            // mat
            shapes.emplace_back(TensorShape{{bs, 3, 3}});
            checker.set_rng(bs, &rng);
            // dst
            shapes.emplace_back(TensorShape{{bs, oh, ow, c}});
            checker.set_dtype(bs + 1, dtype);
            checker.execs(shapes);
        };

        for (auto dtype : std::vector<DType>{dtype::Float32(), dtype::Float16()}) {
            run(1, 20, 18, 4, 6, 6, dtype);
            run(20, 10, 11, 123, 15, 16, dtype);
            run(100, 10, 11, 3, 11, 12, dtype);
        }
    }
}

TEST_F(FALLBACK, WARP_PERSPECTIVE_MULTI_SRC_WITH_IDX_NCHW) {
    using Param = WarpPerspective::Param;
    Param param;
    WarpPerspectiveMatRNG rng;
    UniformIntRNG idx_rng{0, 0};

    for (auto bmode :
         {WarpPerspective::BorderMode::WRAP, WarpPerspective::BorderMode::REFLECT,
          WarpPerspective::BorderMode::REPLICATE,
          WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;
        param.format = Param::Format::NCHW;

        auto run = [&param, &rng, &idx_rng, this](
                           size_t bs, size_t ih, size_t iw, size_t c, size_t oh,
                           size_t ow, size_t idx, DType dtype) {
            Checker<WarpPerspectiveForward, WarpPerspectiveMultiSrcProxy> checker(
                    handle());
            checker.set_param(param);
            TensorShapeArray shapes;
            // src
            for (size_t i = 0; i < bs; i++) {
                shapes.emplace_back(TensorShape{{1, c, ih, iw}});
                checker.set_dtype(i, dtype);
            }
            // mat
            shapes.emplace_back(TensorShape{{idx, 3, 3}});
            checker.set_rng(bs, &rng);
            // mat_idx
            shapes.emplace_back(TensorShape{{idx}});
            checker.set_dtype(bs + 1, dtype::Int32());
            idx_rng = UniformIntRNG{0, (int)bs - 1};
            checker.set_rng(bs + 1, &idx_rng);
            // dst
            shapes.emplace_back(TensorShape{{idx, c, oh, ow}});
            checker.set_dtype(bs + 2, dtype);
            checker.execs(shapes);
        };

        for (auto dtype : std::vector<DType>{dtype::Float32(), dtype::Float16()}) {
            run(1, 20, 18, 4, 6, 6, 1, dtype);
            run(20, 10, 11, 123, 15, 16, 10, dtype);
            run(100, 10, 11, 3, 11, 12, 100, dtype);
        }
    }
}

TEST_F(FALLBACK, WARP_PERSPECTIVE_MULTI_SRC_WITH_IDX_NHWC) {
    using Param = WarpPerspective::Param;
    Param param;
    WarpPerspectiveMatRNG rng;
    UniformIntRNG idx_rng{0, 0};

    for (auto bmode :
         {WarpPerspective::BorderMode::WRAP, WarpPerspective::BorderMode::REFLECT,
          WarpPerspective::BorderMode::REPLICATE,
          WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;
        param.format = Param::Format::NHWC;

        auto run = [&param, &rng, &idx_rng, this](
                           size_t bs, size_t ih, size_t iw, size_t c, size_t oh,
                           size_t ow, size_t idx, DType dtype) {
            Checker<WarpPerspectiveForward, WarpPerspectiveMultiSrcProxy> checker(
                    handle());
            checker.set_param(param);
            TensorShapeArray shapes;
            // src
            for (size_t i = 0; i < bs; i++) {
                shapes.emplace_back(TensorShape{{1, ih, iw, c}});
                checker.set_dtype(i, dtype);
            }
            // mat
            shapes.emplace_back(TensorShape{{idx, 3, 3}});
            checker.set_rng(bs, &rng);
            // mat_idx
            shapes.emplace_back(TensorShape{{idx}});
            checker.set_dtype(bs + 1, dtype::Int32());
            idx_rng = UniformIntRNG{0, (int)bs - 1};
            checker.set_rng(bs + 1, &idx_rng);
            // dst
            shapes.emplace_back(TensorShape{{idx, oh, ow, c}});
            checker.set_dtype(bs + 2, dtype);
            checker.execs(shapes);
        };

        for (auto dtype : std::vector<DType>{dtype::Float32(), dtype::Float16()}) {
            run(1, 20, 18, 4, 6, 6, 1, dtype);
            run(20, 10, 11, 123, 15, 16, 10, dtype);
            run(100, 10, 11, 3, 11, 12, 100, dtype);
        }
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
