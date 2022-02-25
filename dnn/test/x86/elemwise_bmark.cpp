#include "test/x86/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"

using namespace megdnn;
using namespace test;

#define TEST_IN_DIFF_DISTRUBUTION(proportion_of_inf, dataset_number) \
    max_val = 88.3762626647949f / (1 - proportion_of_inf);           \
    UniformFloatRNG rng##dataset_number(0.f, max_val);               \
    B.set_rng(0, &rng##dataset_number);                              \
    B.execs({{355600}, {}});

TEST_F(X86, BENCHMARK_ELEM_EXP_BASED_OPTRS) {
    Benchmarker<ElemwiseForward> B(handle());
    using Mode = ElemwiseForward::Param::Mode;
    // UniformFloatWithZeroRNG rng(80, 100, 0.1);
    printf("Test Optr exp(x)\n");
    B.set_param(Mode::EXP);
    B.execs({{355600}, {}});
    float max_val = 0;
    TEST_IN_DIFF_DISTRUBUTION(0.25, 1)
    TEST_IN_DIFF_DISTRUBUTION(0.5, 2)
    TEST_IN_DIFF_DISTRUBUTION(0.75, 3)
    TEST_IN_DIFF_DISTRUBUTION(0.9999, 4)

    printf("Test Optr tanh(x)\n");
    B.set_param(Mode::TANH);
    B.execs({{355600}, {}});
    max_val = 0;
    TEST_IN_DIFF_DISTRUBUTION(0.25, 5)
    TEST_IN_DIFF_DISTRUBUTION(0.5, 6)
    TEST_IN_DIFF_DISTRUBUTION(0.75, 7)
    TEST_IN_DIFF_DISTRUBUTION(0.9999, 8)

    printf("Test Optr fast_tanh(x)\n");
    B.set_param(Mode::FAST_TANH);
    B.execs({{355600}, {}});

    printf("Test Optr sigmoid(x)\n");
    B.set_param(Mode::SIGMOID);
    B.execs({{355600}, {}});
    max_val = 0;
    TEST_IN_DIFF_DISTRUBUTION(0.25, 13)
    TEST_IN_DIFF_DISTRUBUTION(0.5, 14)
    TEST_IN_DIFF_DISTRUBUTION(0.75, 15)
    TEST_IN_DIFF_DISTRUBUTION(0.9999, 16)

    printf("Test Optr tanh_grad(x)\n");
    B.set_param(Mode::TANH_GRAD);
    B.execs({{355600}, {355600}, {}});

    printf("Test Optr fast_tanh_grad(x)\n");
    B.set_param(Mode::FAST_TANH_GRAD);
    B.execs({{355600}, {355600}, {}});
}

// 1. Unary
#define BENCHMARK_UNARY(Optr, size)  \
    printf("Test for %s \n", #Optr); \
    B.set_param(Mode::Optr);         \
    B.execs(                         \
            {{                       \
                     4,              \
                     4,              \
                     4,              \
                     1 + size / 64,  \
             },                      \
             {}});

// 2. Binary
#define BENCHMARK_BINARY(Optr, size) \
    B.set_param(Mode::Optr);         \
    B.execs({{size}, {size}, {}});

#define BENCHMARK_BINARY_SCALAR(Optr, size) \
    B.set_param(Mode::Optr);                \
    B.execs({{size}, {1}, {}});

#define BENCHMARK_BINARY_1C11(Optr, chan) \
    B.set_param(Mode::Optr);              \
    B.execs({{9, chan, 33, 127}, {1, chan, 1, 1}, {}});

#define BENCHMARK_BINARY_ALL_KINDS(Optr, size) \
    printf("Test for %s \n", #Optr);           \
    BENCHMARK_BINARY(Optr, size)               \
    BENCHMARK_BINARY_SCALAR(Optr, size)        \
    BENCHMARK_BINARY_1C11(Optr, (1 + size / 37719))

// 3. Ternary
#define BENCHMARK_TERNARY(Optr, size) \
    B.set_param(Mode::Optr);          \
    B.execs({{size}, {size}, {size}, {}});

#define BENCHMARK_TERNARY_SCALAR(Optr, size) \
    B.set_param(Mode::Optr);                 \
    B.execs({{size}, {size}, {1}, {}});

#define BENCHMARK_TERNARY_1C11(Optr, chan) \
    B.set_param(Mode::Optr);               \
    B.execs({{1, chan, 1, 1}, {9, chan, 33, 127}, {1, chan, 1, 1}, {}});

#define BENCHMARK_TERNARY_ALL_KINDS(Optr, size) \
    printf("Test for %s \n", #Optr);            \
    BENCHMARK_TERNARY(Optr, size)               \
    BENCHMARK_TERNARY_SCALAR(Optr, size)        \
    BENCHMARK_TERNARY_1C11(Optr, (size / 37719))

#define BENCHMARK_CASE_INT(size)                    \
    BENCHMARK_BINARY_ALL_KINDS(ADD, size)           \
    BENCHMARK_BINARY_ALL_KINDS(SUB, size)           \
    BENCHMARK_BINARY_ALL_KINDS(MUL, size)           \
    BENCHMARK_BINARY_ALL_KINDS(TRUE_DIV, size)      \
    BENCHMARK_BINARY_ALL_KINDS(MIN, size)           \
    BENCHMARK_BINARY_ALL_KINDS(MAX, size)           \
    BENCHMARK_UNARY(RELU, size)                     \
    BENCHMARK_UNARY(ABS, size)                      \
    BENCHMARK_BINARY_ALL_KINDS(FUSE_ADD_RELU, size) \
    BENCHMARK_TERNARY_ALL_KINDS(FUSE_MUL_ADD3, size)

#define BENCHMARK_CASE_FLOAT(size)                  \
    BENCHMARK_CASE_INT(size)                        \
    BENCHMARK_BINARY_ALL_KINDS(FUSE_ADD_TANH, size) \
    BENCHMARK_BINARY_ALL_KINDS(FUSE_ADD_SIGMOID, size)

TEST_F(X86, BENCHMARK_ELEM_EVERY_DTYPE) {
    Benchmarker<ElemwiseForward> B(handle());
    using Mode = ElemwiseForward::Param::Mode;

    printf("\nTest case float32:\n");
    B.set_dtype(0, dtype::Float32());
    B.set_dtype(1, dtype::Float32());
    B.set_dtype(2, dtype::Float32());
    BENCHMARK_CASE_FLOAT(1556011)

    // printf("\nTest case int32:\n");
    // B.set_dtype(0, dtype::Int32());
    // B.set_dtype(1, dtype::Int32());
    // B.set_dtype(2, dtype::Int32());
    // BENCHMARK_CASE_INT(1556011)

    // printf("\nTest case int16:\n");
    // B.set_dtype(0, dtype::Int16());
    // B.set_dtype(1, dtype::Int16());
    // B.set_dtype(2, dtype::Int16());
    // BENCHMARK_CASE_INT(1556011)

    // printf("\nTest case int8:\n");
    // B.set_dtype(0, dtype::Int8());
    // B.set_dtype(1, dtype::Int8());
    // B.set_dtype(2, dtype::Int8());
    // BENCHMARK_CASE_INT(1556011)
}

#if MEGDNN_WITH_BENCHMARK
namespace {
void run_elemwise_benchmark(
        const TensorShapeArray& shapes, param::Elemwise::Mode mode,
        const char* mode_str, DType type, Handle* handle_bench) {
    auto handle_fallback = create_cpu_handle(1);
    Benchmarker<Elemwise> benchmarker_bench(handle_bench);
    Benchmarker<Elemwise> benchmarker_fallback(handle_fallback.get());

    float throughput = 0;
    SmallVector<TensorLayout> layouts;
    std::string src_strs;
    for (size_t i = 0; i < shapes.size(); i++) {
        layouts.emplace_back(shapes[i], type);
        throughput += layouts.back().span().dist_byte();
        src_strs += layouts.back().to_string();
        if (i != shapes.size() - 1) {
            src_strs += ",";
        }
    }
    constexpr size_t RUN = 50;
    benchmarker_fallback.set_times(RUN).set_display(false);
    benchmarker_bench.set_times(RUN).set_display(false);

    benchmarker_fallback.set_param(mode);
    benchmarker_bench.set_param(mode);

    TensorLayout dst_layout;
    auto opr = handle_bench->create_operator<Elemwise>();
    opr->param() = mode;
    opr->deduce_layout(layouts, dst_layout);

    float computations =
            dst_layout.total_nr_elems() * (std::max<size_t>(shapes.size(), 2) - 1);
    throughput += dst_layout.span().dist_byte();
    computations *= (1e3 / (1024.0 * 1024));
    throughput *= (1e3 / (1024.0 * 1024));

    layouts.emplace_back(dst_layout);
    auto fallback_time = benchmarker_fallback.execl(layouts) / RUN;
    auto bench_time = benchmarker_bench.execl(layouts) / RUN;

    float fallback_flops = computations / fallback_time;
    float bench_flops = computations / bench_time;
    float fallback_thr = throughput / fallback_time;
    float bench_thr = throughput / bench_time;

    printf("%s = %s (type: %s, mode: %s) cpu=%fMFLOPS %fMB/s, bench=%fMFLOPS "
           "%fMB/s "
           "computations: %fx, throughput: %fx\n",
           src_strs.c_str(), dst_layout.to_string().c_str(), type.name(), mode_str,
           fallback_flops, fallback_thr, bench_flops, bench_thr,
           bench_flops / fallback_flops, bench_thr / fallback_thr);
}
}  // namespace

#define INT_RUN(shape, mode)                                              \
    run_elemwise_benchmark(shape, mode, #mode, dtype::Int8{}, handle());  \
    run_elemwise_benchmark(shape, mode, #mode, dtype::Int16{}, handle()); \
    run_elemwise_benchmark(shape, mode, #mode, dtype::Int32{}, handle());

#define FLOAT_RUN(shape, mode)                                              \
    run_elemwise_benchmark(shape, mode, #mode, dtype::Float32{}, handle()); \
    run_elemwise_benchmark(shape, mode, #mode, dtype::Float16{}, handle());

#define BENCHMARK_CASES(shape) \
    INT_BENCHMARK_CASES(shape) \
    FLOAT_BENCHMARK_CASES(shape)

TEST_F(X86, BENCHMARK_UNARY) {
#define INT_BENCHMARK_CASES(shape) \
    INT_RUN(shape, Mode::RELU);    \
    INT_RUN(shape, Mode::ABS);

#define FLOAT_BENCHMARK_CASES(shape) \
    FLOAT_RUN(shape, Mode::RELU);    \
    FLOAT_RUN(shape, Mode::ABS);     \
    FLOAT_RUN(shape, Mode::SIGMOID); \
    FLOAT_RUN(shape, Mode::EXP);     \
    FLOAT_RUN(shape, Mode::TANH);    \
    FLOAT_RUN(shape, Mode::FAST_TANH);

    using Mode = param::Elemwise::Mode;
    BENCHMARK_CASES({{10000}});
    BENCHMARK_CASES({{50000}});

#undef INT_BENCHMARK_CASES
#undef FLOAT_BENCHMARK_CASES
}

TEST_F(X86, BENCHMARK_BINARY) {
#define INT_BENCHMARK_CASES(shape) \
    INT_RUN(shape, Mode::MIN);     \
    INT_RUN(shape, Mode::MAX);     \
    INT_RUN(shape, Mode::ADD);     \
    INT_RUN(shape, Mode::SUB);     \
    INT_RUN(shape, Mode::MUL);     \
    INT_RUN(shape, Mode::RMULH);   \
    INT_RUN(shape, Mode::FUSE_ADD_RELU);

#define FLOAT_BENCHMARK_CASES(shape)  \
    FLOAT_RUN(shape, Mode::MIN);      \
    FLOAT_RUN(shape, Mode::MAX);      \
    FLOAT_RUN(shape, Mode::ADD);      \
    FLOAT_RUN(shape, Mode::SUB);      \
    FLOAT_RUN(shape, Mode::MUL);      \
    FLOAT_RUN(shape, Mode::POW);      \
    FLOAT_RUN(shape, Mode::TRUE_DIV); \
    FLOAT_RUN(shape, Mode::FUSE_ADD_RELU);

    using Mode = param::Elemwise::Mode;
    TensorShapeArray shapes = {{1, 112, 28, 28}, {1, 112, 28, 28}};
    BENCHMARK_CASES(shapes);
    shapes = {{1, 16, 1, 1}, {1, 16, 112, 112}};
    BENCHMARK_CASES(shapes);
    shapes = {{1, 448, 7, 7}, {1, 448, 7, 7}};
    BENCHMARK_CASES(shapes);

#undef INT_BENCHMARK_CASES
#undef FLOAT_BENCHMARK_CASES
}

TEST_F(X86, BENCHMARK_TERNARY_FMA3) {
#define INT_BENCHMARK_CASES(shape) INT_RUN(shape, Mode::FUSE_MUL_ADD3);

#define FLOAT_BENCHMARK_CASES(shape) FLOAT_RUN(shape, Mode::FUSE_MUL_ADD3);

    using Mode = param::Elemwise::Mode;
    TensorShapeArray shapes = {{30, 40, 70}, {30, 40, 70}, {30, 40, 70}};
    BENCHMARK_CASES(shapes);
    shapes = {{1, 4, 1, 1}, {3, 4, 5, 7}, {1, 4, 1, 1}};
    BENCHMARK_CASES(shapes);
    shapes = {{3, 4, 5, 7}, {3, 4, 5, 7}, {1, 1, 1, 1}};
    BENCHMARK_CASES(shapes);

#undef INT_BENCHMARK_CASES
#undef FLOAT_BENCHMARK_CASES
}

#undef BENCHMARK_CASES
#undef INT_RUN
#undef FLOAT_RUN

#endif
