#include "test/aarch64/fixture.h"
#include "test/common/benchmarker.h"

#include "test/common/checker.h"
#include "test/common/relayout.h"
#include "test/common/rng.h"
#include "test/common/task_record_check.h"

namespace megdnn {
namespace test {

namespace {
template <typename tag>
class AARCH64_RELAYOUT : public AARCH64 {};
TYPED_TEST_CASE(AARCH64_RELAYOUT, relayout::test_types);
TYPED_TEST(AARCH64_RELAYOUT, run) {
    relayout::run_test<TypeParam>(this->handle());
}
}  // namespace

TEST_F(AARCH64, Relayout) {
    Checker<Relayout> checker(handle());
    std::vector<::megdnn::DType> dtype_vec;
    dtype_vec.push_back(dtype::Float32());
    dtype_vec.push_back(dtype::Int16());
    dtype_vec.push_back(dtype::Uint16());
    dtype_vec.push_back(dtype::Int8());
    for (auto dtype : dtype_vec) {
        TensorLayout src({1, 54, 112, 256}, {54, 1, 16384, 64}, dtype);
        TensorLayout dst({1, 54, 112, 256}, {1548288, 28672, 256, 1}, dtype);
        checker.execl({src, dst});
    }
    TensorLayout src_4_3({1, 3, 112, 256}, {3, 1, 1024, 4}, dtype::Uint16());
    TensorLayout dst_4_3({1, 3, 112, 256}, {86016, 28672, 256, 1}, dtype::Uint16());
    checker.execl({src_4_3, dst_4_3});
}

TEST_F(AARCH64, RelayoutNonContig) {
    Checker<Relayout> checker(handle());
    std::vector<::megdnn::DType> dtype_vec;
    dtype_vec.push_back(dtype::Float32());
    dtype_vec.push_back(dtype::Int16());
    dtype_vec.push_back(dtype::Uint16());
    dtype_vec.push_back(dtype::Int8());
    for (auto dtype : dtype_vec) {
        TensorLayout src({4, 90, 15, 29}, {41760, 1, 2784, 96}, dtype);
        TensorLayout dst({4, 90, 15, 29}, {39150, 435, 29, 1}, dtype);
        checker.execl({src, dst});
    }
}

TEST_F(AARCH64, RelayoutBig) {
    Checker<Relayout> checker(handle());
    ConsecutiveRNG rng;
    checker.set_rng(0, &rng);
    int m = 512;
    int n = 512;
    TensorLayout src({(size_t)m, (size_t)n}, {1, n}, dtype::Float32());
    TensorLayout dst({(size_t)m, (size_t)n}, {n, 1}, dtype::Float32());
    checker.execl({src, dst});
}

TEST_F(AARCH64, RelayoutSplict) {
    Checker<Relayout> checker(handle());
    ConsecutiveRNG rng;
    checker.set_rng(0, &rng);
    int m = 4;
    for (int n : {4, 28}) {
        TensorLayout src({(size_t)m, (size_t)n}, {1, m}, dtype::Uint16());
        TensorLayout dst({(size_t)m, (size_t)n}, {n, 1}, dtype::Uint16());
        checker.execl({src, dst});
    }
}

TEST_F(AARCH64, RelayoutRecord) {
    TaskRecordChecker<Relayout> checker(0);
    std::vector<::megdnn::DType> dtype_vec;
    dtype_vec.push_back(dtype::Float32());
    dtype_vec.push_back(dtype::Int16());
    dtype_vec.push_back(dtype::Uint16());
    dtype_vec.push_back(dtype::Int8());
    for (auto dtype : dtype_vec) {
        TensorLayout src({1, 54, 112, 256}, {54, 1, 16384, 64}, dtype);
        TensorLayout dst({1, 54, 112, 256}, {1548288, 28672, 256, 1}, dtype);
        checker.execl({src, dst});
    }
}

#if MEGDNN_WITH_BENCHMARK

TEST_F(AARCH64, BENCHMARK_Relayout) {
    constexpr size_t WARM_RUNS = 100;
    constexpr size_t RUNS = 600;
    auto dtype = dtype::Float32();
    Benchmarker<Relayout> benchmarker_relayout(handle());
    Benchmarker<Relayout> benchmarker_fbk_relayout(fallback_handle());
    benchmarker_relayout.set_times(WARM_RUNS);
    benchmarker_fbk_relayout.set_times(WARM_RUNS);
    int m = 512;
    int n = 512;
    TensorLayout src({(size_t)m, (size_t)n}, {1, n}, dtype);
    TensorLayout dst({(size_t)m, (size_t)n}, {n, 1}, dtype);
    TensorLayoutArray tensor_case;
    tensor_case.push_back(src);
    tensor_case.push_back(dst);

    benchmarker_relayout.exec(tensor_case);
    benchmarker_fbk_relayout.exec(tensor_case);
    benchmarker_relayout.set_times(RUNS);
    benchmarker_fbk_relayout.set_times(RUNS);

    auto used = benchmarker_relayout.exec(tensor_case) / RUNS;
    auto fbk_used = benchmarker_fbk_relayout.exec(tensor_case) / RUNS;
    float bw = 2.f * m * n * 1e-6 / used * dtype.size();
    float fbk_bw = 2.f * m * n * 1e-6 / fbk_used * dtype.size();
    printf("run: %s -> %s , %f GB/s, fbk %f GB/s, speedup %f\n",
           src.to_string().c_str(), dst.to_string().c_str(), bw, fbk_bw, bw / fbk_bw);
}

TEST_F(AARCH64, BENCHMARK_Relayout_2) {
    constexpr size_t WARM_RUNS = 100;
    constexpr size_t RUNS = 600;
    auto dtype = dtype::Float32();
    Benchmarker<Relayout> benchmarker_relayout(handle());
    Benchmarker<Relayout> benchmarker_fbk_relayout(fallback_handle());
    benchmarker_relayout.set_times(WARM_RUNS);
    benchmarker_fbk_relayout.set_times(WARM_RUNS);
    int m = 54;
    int n = 28762;
    TensorLayout src({1, 54, 112, 256}, {54, 1, 16384, 64}, dtype);
    TensorLayout dst({1, 54, 112, 256}, {1548288, 28672, 256, 1}, dtype);
    TensorLayoutArray tensor_case;
    tensor_case.push_back(src);
    tensor_case.push_back(dst);

    benchmarker_relayout.exec(tensor_case);
    benchmarker_fbk_relayout.exec(tensor_case);
    benchmarker_relayout.set_times(RUNS);
    benchmarker_fbk_relayout.set_times(RUNS);

    auto used = benchmarker_relayout.exec(tensor_case) / RUNS;
    auto fbk_used = benchmarker_fbk_relayout.exec(tensor_case) / RUNS;
    float bw = 2.f * m * n * 1e-6 / used * dtype.size();
    float fbk_bw = 2.f * m * n * 1e-6 / fbk_used * dtype.size();
    printf("run: %s -> %s , %f GB/s, fbk %f GB/s, speedup %f\n",
           src.to_string().c_str(), dst.to_string().c_str(), bw, fbk_bw, bw / fbk_bw);
}
#endif

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
