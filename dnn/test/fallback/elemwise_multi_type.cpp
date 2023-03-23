#include "test/common/elemwise_multi_type.h"
#include "test/common/benchmarker.h"
#include "test/common/task_record_check.h"
#include "test/fallback/fixture.h"
using namespace megdnn;
using namespace test;

namespace {
template <typename tag>
class FALLBACK_ELEMWISE_MULTI_TYPE : public FALLBACK {};
TYPED_TEST_CASE(FALLBACK_ELEMWISE_MULTI_TYPE, elemwise_multi_type::test_types);
}  // anonymous namespace

TYPED_TEST(FALLBACK_ELEMWISE_MULTI_TYPE, run) {
    elemwise_multi_type::run_test<TypeParam>(this->handle());
}

TEST_F(FALLBACK, ELEMWISE_QUANTIZED_MODE_UNARY) {
    using Mode = ElemwiseMultiType::Param::Mode;
    Checker<ElemwiseMultiType> checker(handle());

    std::unique_ptr<RNG> rng;
    for (auto mode :
         {Mode::QRELU, Mode::QABS, Mode::QSIGMOID, Mode::QEXP, Mode::QTANH,
          Mode::QFAST_TANH, Mode::QH_SWISH}) {
        checker.set_param({mode});

        for (DType src_type :
             std::vector<DType>{dtype::QuantizedS8(1.4f), dtype::QuantizedS32(1.3f)}) {
            checker.set_dtype(0, src_type);
            if (src_type.enumv() == DTypeEnum::QuantizedS8) {
                rng = std::make_unique<UniformIntRNG>(-127, 127);
                checker.set_dtype(1, dtype::QuantizedS8(1.7f));
            } else {
                rng = std::make_unique<UniformIntRNG>(INT16_MIN >> 1, INT16_MAX >> 1);
            }

            checker.set_rng(0, rng.get());
            auto run = [&]() {
                checker.execs({{3, 4, 5, 6}, {}});

                checker.execs({{3}, {}});
                checker.execs({{9}, {}});
                checker.execs({{17}, {}});
            };

            if (src_type.enumv() == DTypeEnum::QuantizedS32) {
                for (DType dst_type :
                     std::vector<DType>{dtype::QuantizedS8(32718.6f)}) {
                    checker.set_dtype(1, dst_type);
                    run();
                }
            } else {
                run();
            }
        }
    }
}

TEST_F(FALLBACK, ELEMWISE_QUANTIZED_MODE_BINARY) {
    using Mode = ElemwiseMultiType::Param::Mode;
    Checker<ElemwiseMultiType> checker(handle());
    auto run = [&]() {
        //! nchw44
        checker.execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});
        checker.execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
        checker.execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});
        //! VEC + SCALAR
        checker.execs({{3, 4, 5, 6}, {1, 1, 1, 1}, {}});
        checker.execs({{1, 1, 1, 1}, {3, 4, 5, 6}, {}});
        checker.execs({{3, 4, 5, 6}, {1}, {}});
        checker.execs({{1}, {3, 4, 5, 6}, {}});

        //! VEC + 1C11
        checker.execs({{3, 4, 5, 6}, {1, 4, 1, 1}, {}});
        checker.execs({{1, 4, 1, 1}, {3, 4, 5, 6}, {}});

        //! VEC + VEC
        checker.execs({{3}, {3}, {}});
        checker.execs({{9}, {9}, {}});
        checker.execs({{17}, {17}, {}});
    };

    // qint32 to qint8/quint8
    for (auto mode : {Mode::QADD, Mode::QFUSE_ADD_RELU, Mode::QFUSE_ADD_H_SWISH}) {
        checker.set_param({mode});
        UniformIntRNG rng{INT16_MIN >> 1, INT16_MAX >> 1};
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_dtype(0, dtype::QuantizedS32(1.3f))
                .set_dtype(1, dtype::QuantizedS32(1.2f));

        for (DType dst_type : std::vector<DType>{dtype::QuantizedS8(32718.6f)}) {
            checker.set_dtype(2, dst_type);
            run();
        }
    }

    for (auto mode :
         {Mode::QMUL, Mode::QADD, Mode::QMIN, Mode::QMAX, Mode::QSUB,
          Mode::QFUSE_ADD_RELU, Mode::QFUSE_ADD_SIGMOID, Mode::QFUSE_ADD_H_SWISH}) {
        checker.set_param({mode});

        // qint8 to qint8
        UniformIntRNG rng_int8{-127, 127};
        checker.set_rng(0, &rng_int8)
                .set_rng(1, &rng_int8)
                .set_dtype(0, dtype::QuantizedS8(1.35f))
                .set_dtype(1, dtype::QuantizedS8(1.15f))
                .set_dtype(2, dtype::QuantizedS8(1.75f));
        run();
    }

    //! TRUE_DIV : 0.0 / 0.0 will fail
    checker.set_param({Mode::QTRUE_DIV});
    UniformIntRNG rng_int8_1{-127, 127};
    UniformIntRNG rng_int8_2{-127, -1};
    checker.set_rng(0, &rng_int8_1)
            .set_rng(1, &rng_int8_2)
            .set_dtype(0, dtype::QuantizedS8(1.4f))
            .set_dtype(1, dtype::QuantizedS8(1.1f))
            .set_dtype(2, dtype::QuantizedS8(1.7f));

    run();
    //! TANH
    checker.set_param({Mode::QFUSE_ADD_TANH});
    UniformIntRNG rng_int8{-5, 5};
    checker.set_rng(0, &rng_int8)
            .set_rng(1, &rng_int8)
            .set_dtype(0, dtype::QuantizedS8(1.1f))
            .set_dtype(1, dtype::QuantizedS8(1.4f))
            .set_dtype(2, dtype::QuantizedS8(1.7f));

    run();
}

TEST_F(FALLBACK, ELEMWISE_QUANTIZED_MODE_TERNARY) {
    using Mode = ElemwiseMultiType::Param::Mode;
    Checker<ElemwiseMultiType> checker(handle());

    auto run = [&]() {
        //! nchw44
        checker.execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});

        //! nchw44
        checker.execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
        checker.execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
        checker.execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});

        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {1, 1, 1, 1}, {}});
        checker.execs({{1, 4, 1, 1}, {3, 4, 5, 6}, {1, 4, 1, 1}, {}});

        checker.execs({{3}, {3}, {3}, {}});
        checker.execs({{9}, {9}, {9}, {}});
        checker.execs({{17}, {17}, {17}, {}});
        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {3, 4, 5, 6}, {}});
    };

    for (auto mode : {Mode::QFUSE_MUL_ADD3}) {
        checker.set_param({mode});

        // qint8 to qint8
        UniformIntRNG rng_int8{-127, 127};
        checker.set_rng(0, &rng_int8)
                .set_rng(1, &rng_int8)
                .set_rng(2, &rng_int8)
                .set_dtype(0, dtype::QuantizedS8(1.45f))
                .set_dtype(1, dtype::QuantizedS8(1.15f))
                .set_dtype(2, dtype::QuantizedS8(1.75f))
                .set_dtype(3, dtype::QuantizedS8(1.35f));
        run();
    }
}

TEST_F(FALLBACK, ELEMWISE_MULTI_TYPE_RECORD_FMA3_INT16x32x32x32) {
    TaskRecordChecker<ElemwiseMultiType> checker{1};
    checker.set_param({ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32});
    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Int32());
    checker.set_dtype(2, dtype::Int32());
    UniformIntRNG rng{-10, 10};
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);
    constexpr size_t A = 32, B = 602, C = 103;
    checker.execs({{A, B, C}, {1, B, 1}, {1, B, 1}, {}});
}

TEST_F(FALLBACK, ELEMWISE_MULTI_TYPE_RECORD_FMA3_INT16xF32xF32xF32) {
    TaskRecordChecker<ElemwiseMultiType> checker{1};
    checker.set_param({ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16xF32xF32xF32});
    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Float32());
    checker.set_dtype(2, dtype::Float32());
    UniformIntRNG rng{-10, 10};
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);
    checker.execs({{5, 7, 16}, {1, 1, 16}, {1, 1, 16}, {}})
            .execs({{2, 700, 600}, {1, 1, 600}, {1, 1, 600}, {}})
            .execs({{2, 700, 600}, {2, 700, 600}, {2, 700, 600}, {}})
            .execs({{16, 16, 128}, {16, 16, 128}, {16, 16, 128}, {}})
            .execs({{16, 128, 16, 16}, {1, 128, 1, 1}, {1, 128, 1, 1}, {}})
            .execs({{16, 128, 16, 16}, {1, 1, 1, 1}, {1, 1, 1, 1}, {}});
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(FALLBACK, ELEMWISE_MULTI_TYPE_BENCHMARK_FMA3_INT16x32x32x32) {
    Benchmarker<ElemwiseMultiType> bench{handle()};
    bench.set_param({ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32});
    bench.set_dtype(0, dtype::Int16());
    bench.set_dtype(1, dtype::Int32());
    bench.set_dtype(2, dtype::Int32());
    UniformIntRNG rng{-100, 100};
    bench.set_rng(0, &rng);
    bench.set_rng(1, &rng);
    bench.set_rng(2, &rng);
    bench.set_adaptive_benchmark(0.8);
    constexpr size_t A = 32, B = 602, C = 103;
    auto time = bench.execs({{A, B, C}, {1, B, 1}, {1, B, 1}, {}}) * 1e-3;
    printf("computation: %.2fGFLOPS/s memory: %.2fGiB/s\n", A * B * C * 2 / time * 1e-9,
           (A * B * C * (2 + 4) + B * 8) / time / (1024.0 * 1024.0 * 1024.0));
}

TEST_F(FALLBACK, ELEMWISE_MULTI_TYPE_BENCHMARK_FMA3_IXxf32xf32xI8) {
    Benchmarker<ElemwiseMultiType> bench{handle()};
    bench.set_param({ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8});
    std::array<DType, 3> src_types{{dtype::Int8{}, dtype::Int16{}, dtype::Int32{}}};
    bench.set_dtype(1, dtype::Float32());
    bench.set_dtype(2, dtype::Float32());
    UniformIntRNG rng{-100, 100};
    bench.set_rng(0, &rng);
    bench.set_adaptive_benchmark(0.8);
    constexpr size_t A = 328, B = 602;
    for (auto stype : src_types) {
        bench.set_dtype(0, stype);
        auto time = bench.execs({{A, B}, {1, B}, {1, B}, {}}) * 1e-3;
        printf("stype: %s, computation: %.2fGFLOPS/s memory: %.2fGiB/s\n", stype.name(),
               A * B * 2 / time * 1e-9,
               (A * B * (stype.size() + 1) + B * 8) / time /
                       (1024.0 * 1024.0 * 1024.0));
    }
}
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
