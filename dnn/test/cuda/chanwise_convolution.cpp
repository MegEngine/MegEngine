/**
 * \file dnn/test/cuda/chanwise_convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/nn.h"

#include "test/cuda/fixture.h"
#include "test/cuda/benchmark.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/common/checker.h"
#include "test/common/convolution.h"
#include "test/common/benchmarker.h"
#include "megcore_cuda.h"
#include "cuda.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

using namespace megdnn;
using namespace test;

namespace {

#if MEGDNN_WITH_BENCHMARK
bool check_need_full_bench() {
    if (getenv("MEGDNN_CHANWISE_CONV_FULLBENCH"))
        return true;
    printf("set MEGDNN_CHANWISE_CONV_FULLBENCH to run full benchmark\n");
    return false;
}
#endif

Convolution::Param gconv_param(Convolution::Param p) {
    p.sparse = Convolution::Param::Sparse::GROUP;
    return p;
}

template<int P0, int P1, int P2>
class BenchmarkEnv {
    Handle *handle, *handle_cpu;
    std::unique_ptr<GaussianRNG> rng;
    TensorLayout lsrc, lflt0, lflt1, ldst;
    std::unique_ptr<Tensor<>> src0, src1,
        flt0, flt0_cpu, flt1, flt1_cpu, dst0, dst1;
    cudaEvent_t cuda_ev[3];
    cudaStream_t cuda_stream;
    size_t pad_h, pad_w;

    template<typename T>
    static std::tuple<T, T, T> shuffle(std::tuple<T, T, T> data) {
        return std::make_tuple(
                std::get<P0>(data), std::get<P1>(data), std::get<P2>(data));
    }

public:
    BenchmarkEnv(Handle *handle, Handle *handle_cpu) {
        this->handle = handle;
        this->handle_cpu = handle_cpu;
        rng = handle->create_operator<GaussianRNG>();
        // make cpu handle used
        handle_cpu->create_operator<Sleep>()->exec();

        for (int i = 0; i < 3; ++ i)
            cudaEventCreate(&cuda_ev[i]);
        megcoreGetCUDAStream(handle->megcore_computing_handle(), &cuda_stream);
    }

    ~BenchmarkEnv() {
        for (int i = 0; i < 3; ++ i)
            cudaEventDestroy(cuda_ev[i]);
    }

    void alloc(size_t N, size_t IC, size_t IH, size_t IW,
            size_t CHL_MUL, size_t FH, size_t FW, size_t PH, size_t PW) {
        pad_h = PH;
        pad_w = PW;
        auto mkly = [](const TensorShape &s) {
            return TensorLayout{s, dtype::Float32()};
        };
        lsrc = mkly({N, IC, IH, IW});
        lflt0 = mkly({CHL_MUL*IC, IC, FH, FW});
        lflt1 = mkly({IC, CHL_MUL, 1, FH, FW});
        ldst = mkly({N, IC*CHL_MUL, IH-FH+1+PH*2, IW-FW+1+PW*2});
        src0.reset(new Tensor<>(handle, lsrc));
        src1.reset(new Tensor<>(handle, lsrc));
        flt0.reset(new Tensor<>(handle, lflt0));
        flt0_cpu.reset(new Tensor<>(handle_cpu, lflt0));
        flt1.reset(new Tensor<>(handle, lflt1));
        flt1_cpu.reset(new Tensor<>(handle_cpu, lflt1));
        dst0.reset(new Tensor<>(handle, ldst));
        dst1.reset(new Tensor<>(handle, ldst));
    }

    void fill_src() {
        rng->exec(src0->tensornd(), {});
        megdnn_memcpy_D2D(handle, src1->ptr(), src0->ptr(),
                lsrc.span().dist_byte());
    }

    void fill_flt() {
        rng->exec(flt1->tensornd(), {});
        megdnn_memcpy_D2H(handle,
                flt1_cpu->ptr(), flt1->ptr(), lflt1.span().dist_byte());

        const size_t IC = lflt1[0], CHL_MUL = lflt1[1],
               FSIZE = lflt1[3] * lflt1[4];

        // fill flt0 from flt1
        float* src = flt1_cpu->ptr();
        float* dst = flt0_cpu->ptr();
        memset(dst, 0, lflt0.span().dist_byte());
        for (size_t i = 0; i < IC; ++ i) {
            for (size_t j = 0; j < CHL_MUL; ++ j) {
                memcpy(dst + ((i * CHL_MUL + j) * IC + i) * FSIZE,
                        src + (i * CHL_MUL + j) * FSIZE,
                        FSIZE * sizeof(float));
            }
        }

        megdnn_memcpy_H2D(handle,
                flt0->ptr(), dst, lflt0.span().dist_byte());
    }

    void fill_dst() {
        rng->exec(dst0->tensornd(), {});
        megdnn_memcpy_D2D(handle, dst1->ptr(), dst0->ptr(),
                ldst.span().dist_byte());
    }

    template<class Opr>
    void exec(Opr *opr0, Opr *opr1) {
        opr0->param().pad_h = pad_h;
        opr0->param().pad_w = pad_w;
        opr1->param() = opr0->param();
        opr1->param().sparse = param::Convolution::Sparse::GROUP;

        TensorND a0, b0, c0, a1, b1, c1;
        std::tie(a0, b0, c0) = shuffle(std::make_tuple(
                    src0->tensornd(), flt0->tensornd(), dst0->tensornd()));
        std::tie(a1, b1, c1) = shuffle(std::make_tuple(
                    src1->tensornd(), flt1->tensornd(), dst1->tensornd()));
        WorkspaceWrapper wk(handle,
                std::max(
                    opr0->get_workspace_in_bytes(
                        a0.layout, b0.layout, c0.layout),
                    opr1->get_workspace_in_bytes(
                        a1.layout, b1.layout, c1.layout)
                    ));
        cudaProfilerStart();
        cudaEventRecord(cuda_ev[0], cuda_stream);
        opr0->exec(a0, b0, c0, wk.workspace());
        cudaEventRecord(cuda_ev[1], cuda_stream);
        opr1->exec(a1, b1, c1, wk.workspace());
        cudaEventRecord(cuda_ev[2], cuda_stream);
        cudaProfilerStop();

        if (getenv("MEGDNN_CHANWISE_CONV_VERBOSE") ||
                getenv("MEGDNN_CHANWISE_CONV_FULLBENCH")) {
            cudaStreamSynchronize(cuda_stream);
            float t0 = -1, t1 = -1;
            cudaEventElapsedTime(&t0, cuda_ev[0], cuda_ev[1]);
            cudaEventElapsedTime(&t1, cuda_ev[1], cuda_ev[2]);
            printf("%s;%s;%s: cudnn/megdnn: %.3fms/%.3fms=%.3f\n",
                    lsrc.TensorShape::to_string().c_str(),
                    lflt1.TensorShape::to_string().c_str(),
                    ldst.TensorShape::to_string().c_str(),
                    t0, t1, t0 / t1);
        }
    }

    //! special for weight preprocess
    void exec_convolution(ConvolutionForward* opr0, ConvolutionForward* opr1) {
        opr0->param().pad_h = pad_h;
        opr0->param().pad_w = pad_w;
        opr1->param() = opr0->param();
        opr1->param().sparse = param::Convolution::Sparse::GROUP;

        TensorND a0, b0, c0, a1, b1, c1;
        std::tie(a0, b0, c0) = shuffle(std::make_tuple(
                    src0->tensornd(), flt0->tensornd(), dst0->tensornd()));
        std::tie(a1, b1, c1) = shuffle(std::make_tuple(
                    src1->tensornd(), flt1->tensornd(), dst1->tensornd()));
        WorkspaceWrapper wk(
                handle,
                std::max(opr0->get_workspace_in_bytes(a0.layout, b0.layout,
                                                      c0.layout, nullptr),
                         opr1->get_workspace_in_bytes(a1.layout, b1.layout,
                                                      c1.layout, nullptr)));
        cudaProfilerStart();
        cudaEventRecord(cuda_ev[0], cuda_stream);
        opr0->exec(a0, b0, c0, nullptr, wk.workspace());
        cudaEventRecord(cuda_ev[1], cuda_stream);
        opr1->exec(a1, b1, c1, nullptr, wk.workspace());
        cudaEventRecord(cuda_ev[2], cuda_stream);
        cudaProfilerStop();

        if (getenv("MEGDNN_CHANWISE_CONV_VERBOSE") ||
                getenv("MEGDNN_CHANWISE_CONV_FULLBENCH")) {
            cudaStreamSynchronize(cuda_stream);
            float t0 = -1, t1 = -1;
            cudaEventElapsedTime(&t0, cuda_ev[0], cuda_ev[1]);
            cudaEventElapsedTime(&t1, cuda_ev[1], cuda_ev[2]);
            printf("%s;%s;%s: cudnn/megdnn: %.3fms/%.3fms=%.3f\n",
                    lsrc.TensorShape::to_string().c_str(),
                    lflt1.TensorShape::to_string().c_str(),
                    ldst.TensorShape::to_string().c_str(),
                    t0, t1, t0 / t1);
        }
    }

    void cmp_dst() {
        Tensor<> dst0_cpu(handle_cpu, ldst), dst1_cpu(handle_cpu, ldst);
        megdnn_memcpy_D2H(handle,
                dst0_cpu.ptr(), dst0->ptr(), ldst.span().dist_byte());
        megdnn_memcpy_D2H(handle,
                dst1_cpu.ptr(), dst1->ptr(), ldst.span().dist_byte());
        dst0_cpu.check_with(dst1_cpu);
    }

    void cmp_src() {
        Tensor<> src0_cpu(handle_cpu, lsrc), src1_cpu(handle_cpu, lsrc);
        megdnn_memcpy_D2H(handle,
                src0_cpu.ptr(), src0->ptr(), lsrc.span().dist_byte());
        megdnn_memcpy_D2H(handle,
                src1_cpu.ptr(), src1->ptr(), lsrc.span().dist_byte());
        src0_cpu.check_with(src1_cpu);
    }

    void cmp_flt() {
        Tensor<> flt0_cpu(handle_cpu, lflt0), flt1_cpu(handle_cpu, lflt1);
        float *p0 = flt0_cpu.ptr();
        float *p1 = flt1_cpu.ptr();
        megdnn_memcpy_D2H(handle, p0, flt0->ptr(), lflt0.span().dist_byte());
        megdnn_memcpy_D2H(handle, p1, flt1->ptr(), lflt1.span().dist_byte());

        size_t IC = lflt1[0], CHL_MUL = lflt1[1],
               FSIZE = lflt1[3] * lflt1[4];

        double tot_err = 0, tot_err_num = 0;
        for (size_t i = 0; i < IC; ++ i) {
            for (size_t j = 0; j < CHL_MUL; ++ j) {
                auto t0 = p0 + ((i * CHL_MUL + j) * IC + i) * FSIZE,
                     t1 = p1 + (i * CHL_MUL + j) * FSIZE;
                for (size_t k = 0; k < FSIZE; ++ k) {
                    auto err = std::abs(diff(t0[k], t1[k]));
                    tot_err += err;
                    tot_err_num += 1;
                    ASSERT_LT(err, 1e-2) << "failed at " <<
                        i << " " << j << " " << k <<
                        " vals=" << t0[k] << "," << t1[k];
                }
            }
        }
        auto avg_err = tot_err /  tot_err_num;
        ASSERT_LT(avg_err, 1e-4);

    }
};

} // anonymous namespace

constexpr auto M = Convolution::Mode::CROSS_CORRELATION;

TEST_F(CUDA, CHANWISE_CONVOLUTION_FORWARD) {
    Checker<Convolution> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            ConvBiasForward::algo_name<ConvBiasForward::DirectParam>(
                    "CHANNEL_WISE", {})
                    .c_str(),
            &require_algo));
    for (auto dtype : std::vector<DType>{dtype::Float32(), dtype::Float16()}) {
        checker.set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(2, dtype);
        if (dtype.enumv() == DTypeEnum::Float16)
            checker.set_epsilon(2e-2);

        // simple case
        // clang-format off
        for (uint32_t s : {1, 2})
        for (uint32_t p : {0, 1, 2, 3})
        for (size_t f : {2, 3, 5, 7})
        for (size_t ocpg : {1, 3}) {
            checker.set_param(gconv_param({M, p, p, s, s}))
                    .execs({{2, 3, 16, 16}, {3, ocpg, 1, f, f}, {}});
        }
        // clang-format on

        checker.set_param(gconv_param({M, 2, 3, 2, 1}))
                .execs({{32, 12, 20, 10}, {12, 2, 1, 4, 5}, {}});

        // padding larger than kern
        checker.set_param(gconv_param({M, 20, 30, 4, 5}))
                .execs({{32, 12, 20, 10}, {12, 2, 1, 4, 5}, {}});
    }
}

TEST_F(CUDA, CHANWISE_CONVOLUTION_FORWARD_SMALL) {
    Checker<Convolution> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            ConvBiasForward::algo_name<ConvBiasForward::DirectParam>(
                    "CHANNEL_WISE_SMALL", {}).c_str(),
            &require_algo));
    for (auto dtype : std::vector<DType> {
             dtype::Float32(),
#if CUDA_VERSION >= 9000
                     dtype::Float16()
#endif
         }) {
        checker.set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(2, dtype);
        if (dtype.enumv() == DTypeEnum::Float16)
            checker.set_epsilon(2e-2);

        // clang-format off
        for (uint32_t s : {1})
        for (uint32_t f : {1, 3, 5, 7}) {
            checker.set_param(gconv_param({M, f / 2, f / 2, s, s}))
                    .execs({{2, 3, 16, 16}, {3, 1, 1, f, f}, {}});
        }
        // clang-format on
        checker.set_param(gconv_param({M, 1, 1, 1, 1}))
                .execs({{2, 3, 3, 16}, {3, 1, 1, 3, 3}, {}})
                .execs({{2, 3, 8, 3}, {3, 1, 1, 3, 3}, {}});

    }
}

TEST_F(CUDA, CHANWISE_CONVOLUTION_BACKWARD_DATA) {
    Checker<ConvolutionBackwardData> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<ConvolutionBackwardData>(
            "CHANNEL_WISE", &require_algo));
    for (auto dtype : std::vector<DType>{dtype::Float32(), dtype::Float16()}) {
        checker.set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(2, dtype);
        if (dtype.enumv() == DTypeEnum::Float16)
            checker.set_epsilon(1e-1);
        // simple case
        // clang-format off
        for (uint32_t s : {1, 2})
        for (uint32_t p : {0, 1, 2, 3})
        for (size_t f : {1, 2, 3, 5, 7})
        for (size_t ocpg : {1, 3}) {
            size_t ii = infer_conv_shape(16, f, s, p, true);
            checker.set_param(gconv_param({M, p, p, s, s}))
                    .execs({{3, ocpg, 1, f, f},
                            {2, 3 * ocpg, ii, ii},
                            {2, 3, 16, 16}});
        }
        // clang-format on

        checker.set_param(gconv_param({M, 2, 3, 2, 1}))
                .execs({{12, 3, 1, 4, 5}, {32, 36, 20, 10}, {32, 12, 39, 8}});
        checker.set_param(gconv_param({M, 30, 20, 5, 4}))
                .execs({{6, 2, 1, 5, 4}, {32, 12, 12, 10}, {32, 6, 3, 2}});
        checker.set_param(gconv_param({M, 20, 30, 4, 5}))
                .execs({{6, 2, 1, 4, 5}, {32, 12, 10, 12}, {32, 6, 2, 3}});
    }
}

TEST_F(CUDA, CHANWISE_CONVOLUTION_BACKWARD_DATA_SMALL) {
    Checker<ConvolutionBackwardData> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardData>(
                "CHANNEL_WISE_SMALL", &require_algo));
    for (auto dtype : std::vector<DType> {
            dtype::Float32(),
#if CUDA_VERSION >= 9000
            dtype::Float16()
#endif
         }) {
        checker.set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(2, dtype);

        if (dtype.enumv() == DTypeEnum::Float16)
            checker.set_epsilon(2e-2);

        for (uint32_t f : {1, 3, 5, 7}) {
            checker.set_param(gconv_param({M, f/2, f/2, 1, 1}))
                .execs({{3, 1, 1, f, f}, {2, 3, 16, 16}, {2, 3, 16, 16}});
        }
        checker.set_param(gconv_param({M, 1, 1, 1, 1}))
                .execs({{3, 1, 1, 3, 3}, {2, 3, 3, 16}, {2, 3, 3, 16}})
                .execs({{3, 1, 1, 3, 3}, {2, 3, 8, 3}, {2, 3, 8, 3}});
    }
}

TEST_F(CUDA, CHANWISE_CONVOLUTION_BACKWARD_FILTER) {
    Checker<ConvolutionBackwardFilter> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(AlgoChecker<ConvolutionBackwardFilter>(
                "CHANNEL_WISE", &require_algo));
    UniformFloatRNG rng(-0.1, 0.1);
    for (auto dtype : std::vector<DType>{dtype::Float32(), dtype::Float16()}) {
        checker.set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(2, dtype).set_rng(0, &rng).set_rng(1, &rng);
        if (dtype.enumv() == DTypeEnum::Float16)
            checker.set_epsilon(2e-1);
        // simple case
        // clang-format off
        for (uint32_t s : {1, 2})
        for (uint32_t p : {0, 1, 2, 3})
        for (uint32_t f : {1, 2, 3, 5, 7})
        for (uint32_t ocpg : {1, 3})
        for (uint32_t i : {8, 16, 32, 64}){
            size_t ii = infer_conv_shape(i, f, s, p, true);
            checker.set_param(gconv_param({M, p, p, s, s}))
                    .execs({{2, 3, i, i},
                            {2, 3 * ocpg, ii, ii},
                            {3, ocpg, 1, f, f}});
        }
        // clang-format on

    // padding larger than kern
        checker.set_param(gconv_param({M, 20, 30, 4, 5})).
            execs({{32, 6, 2, 3}, {32, 12, 10, 12}, {6, 2, 1, 4, 5}});
    // unused filter items
        checker.set_param(gconv_param({M, 2, 3, 2, 3})).
            execs({{32, 6, 1, 1}, {32, 12, 1, 1}, {6, 2, 1, 5, 7}});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, CHANWISE_CONVOLUTION_FORWARD_BENCH_CHECK) {
    auto handle = handle_cuda();
    auto handle_cpu = handle_naive();
    auto conv0 = handle->create_operator<ConvolutionForward>();
    auto conv1 = handle->create_operator<ConvolutionForward>();
    BenchmarkEnv<0, 1, 2> benv(handle, handle_cpu);

    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t CHL_MUL, size_t FH, size_t FW, size_t PH, size_t PW) {
        benv.alloc(N, IC, IH, IW, CHL_MUL, FH, FW, PH, PW);
        benv.fill_src();
        benv.fill_flt();
        benv.exec_convolution(conv0.get(), conv1.get());
        benv.cmp_dst();
    };

    run(64, 60, 50, 50, 1, 3, 3, 1, 1);
    if (check_need_full_bench()) {
        run(64, 728, 18, 18, 2, 5, 5, 2, 2);
        run(64, 64, 150, 150, 2, 3, 3, 1, 1);
        run(1, 2048, 4, 4, 2, 3, 3, 1, 1);
    }
}

TEST_F(CUDA, CHANWISE_CONVOLUTION_BWD_DATA_BENCH_CHECK) {
    auto handle = handle_cuda();
    auto handle_cpu = handle_naive();
    auto conv0 = handle->create_operator<ConvolutionBackwardData>();
    auto conv1 = handle->create_operator<ConvolutionBackwardData>();
    BenchmarkEnv<1, 2, 0> benv(handle, handle_cpu);

    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t CHL_MUL, size_t FH, size_t FW, size_t PH, size_t PW) {
        benv.alloc(N, IC, IH, IW, CHL_MUL, FH, FW, PH, PW);
        benv.fill_dst();
        benv.fill_flt();
        benv.exec(conv0.get(), conv1.get());
        benv.cmp_src();
    };

    run(64, 60, 50, 50, 1, 3, 3, 1, 1);
    if (check_need_full_bench()) {
        run(64, 728, 18, 18, 2, 5, 5, 2, 2);
        run(64, 64, 150, 150, 2, 3, 3, 1, 1);
        run(1, 2048, 4, 4, 2, 3, 3, 1, 1);
    }
}

TEST_F(CUDA, CHANWISE_CONVOLUTION_BWD_FILTER_BENCH_CHECK) {
    auto handle = handle_cuda();
    auto handle_cpu = handle_naive();
    auto conv0 = handle->create_operator<ConvolutionBackwardFilter>();
    auto conv1 = handle->create_operator<ConvolutionBackwardFilter>();
    BenchmarkEnv<0, 2, 1> benv(handle, handle_cpu);

    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t CHL_MUL, size_t FH, size_t FW, size_t PH, size_t PW) {
        benv.alloc(N, IC, IH, IW, CHL_MUL, FH, FW, PH, PW);
        benv.fill_src();
        benv.fill_dst();
        benv.exec(conv0.get(), conv1.get());
        benv.cmp_flt();
    };

    run(64, 60, 50, 50, 1, 3, 3, 1, 1);
    if (check_need_full_bench()){
        run(64, 728, 18, 18, 2, 5, 5, 2, 2);
        run(64, 64, 150, 150, 2, 3, 3, 1, 1);
        run(1, 2048, 4, 4, 2, 3, 3, 1, 1);
    }
}

TEST_F(CUDA, CHANWISE_CONVOLUTION_BENCH_ALL_ALGO_FWD) {
    // enable profiling
    std::unique_ptr<OprProxy<ConvolutionForward>> proxy{
            new OprProxy<ConvolutionForward>{true}};
    proxy->warmup_times = 1;
    proxy->exec_times = 10;
    Benchmarker<ConvolutionForward> checker(handle_cuda());
    checker.set_times(1);
    ConvolutionForward::Param param;
    param.sparse = ConvolutionForward::Param::Sparse::GROUP;
    checker.set_param(param);
    checker.set_proxy(proxy);

    auto run = [&](size_t N, size_t C, size_t IH, size_t IW, size_t FH,
                   size_t FW) {
        checker.proxy()->target_algo_info.reset();
        checker.execs({{N, C, IH, IW}, {C, 1, 1, FH, FW}, {}});
    };

    run(128, 64, 90, 80, 3, 3);
    run(128, 90, 100, 100, 3, 5);
    run(128, 32, 62, 62, 5, 5);
}

TEST_F(CUDA, CHANWISE_CONVOLUTION_BENCH_ALL_ALGO_BWD_DATA) {
    // enable profiling
    std::unique_ptr<OprProxy<ConvolutionBackwardData>> proxy{
            new OprProxy<ConvolutionBackwardData>{true}};
    proxy->warmup_times = 1;
    proxy->exec_times = 10;
    Benchmarker<ConvolutionBackwardData> checker(handle_cuda());
    checker.set_times(1);
    ConvolutionBackwardData::Param param;
    param.sparse = ConvolutionForward::Param::Sparse::GROUP;
    checker.set_param(param);
    checker.set_proxy(proxy);

    auto run = [&](size_t N, size_t C, size_t IH, size_t IW, size_t FH,
                   size_t FW) {
        checker.proxy()->target_algo_info.reset();
        checker.execs({{C, 1, 1, FH, FW},
                       {N, C, IH - FH + 1, IW - FW + 1},
                       {N, C, IH, IW}});
    };

    run(128, 64, 90, 80, 3, 3);
    run(128, 90, 100, 100, 3, 5);
    run(128, 32, 62, 62, 5, 5);
}

TEST_F(CUDA, CHANWISE_CONVOLUTION_BENCH_ALL_ALGO_BWD_FILTER) {
    // enable profiling
    std::unique_ptr<OprProxy<ConvolutionBackwardFilter>> proxy{
            new OprProxy<ConvolutionBackwardFilter>{true}};
    proxy->warmup_times = 1;
    proxy->exec_times = 10;
    Benchmarker<ConvolutionBackwardFilter> checker(handle_cuda());
    checker.set_times(1);
    ConvolutionBackwardFilter::Param param;
    param.sparse = ConvolutionForward::Param::Sparse::GROUP;
    checker.set_param(param);
    checker.set_proxy(proxy);

    auto run = [&](size_t N, size_t C, size_t IH, size_t IW, size_t FH,
                   size_t FW) {
        checker.proxy()->target_algo_info.reset();
        checker.execs({{N, C, IH, IW},
                       {N, C, IH - FH + 1, IW - FW + 1},
                       {C, 1, 1, FH, FW}});
    };

    run(128, 64, 90, 80, 3, 3);
    run(128, 90, 100, 100, 3, 5);
    run(128, 32, 62, 62, 5, 5);
}

TEST_F(CUDA, BENCHMARK_CHANWISE_CONV_ALL_ALGO_FORWARD) {
    CUBenchmarker<ConvolutionForward> bencher(handle_cuda());
    size_t RUNS = 10;
    bencher.set_display(false).set_times(RUNS);
    std::unique_ptr<OprProxy<ConvolutionForward>> proxy{
            new OprProxy<ConvolutionForward>{true}};
    bencher.set_proxy(proxy);

    Convolution::Param param;
    param.format = ConvBias::Param::Format::NCHW;
    param.sparse = Convolution::Param::Sparse::GROUP;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t c, size_t ih, size_t iw, size_t f,
                   size_t s) {
        param.pad_h = f / 2;
        param.pad_w = f / 2;
        param.stride_h = s;
        param.stride_w = s;
        param.compute_mode = param::Convolution::ComputeMode::DEFAULT;

        TensorShape src = {batch, c, ih, iw}, filter = {c, 1, 1, f, f};

        TensorLayout dst_layout;
        auto opr = handle_cuda()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({src, dtype::Float32()}, {filter, dtype::Float32()},
                           dst_layout);
        float bandwith = static_cast<float>(src.total_nr_elems() +
                                            filter.total_nr_elems() +
                                            dst_layout.total_nr_elems()) /
                         (1024 * 1024 * 1024) * 1e3;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        bencher.proxy()->target_algo_info.reset();
        auto time_in_ms_fp32 = bencher.execs({src, filter, {}}) / RUNS;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        bencher.proxy()->target_algo_info.reset();
        auto time_in_ms_fp16 = bencher.execs({src, filter, {}}) / RUNS;

        bencher.proxy()->target_algo_info.reset();
        param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
        bencher.set_param(param);
        auto time_in_ms_pseudo_fp16 = bencher.execs({src, filter, {}}) / RUNS;

        printf("stride=%zu src=%s, filter=%s, float32: %.2fms %.2fGB/s "
               "float16: %.2fms %.2fGB/s "
               "pseudo float16: %.2fms %.2fGB/s "
               "speedup: "
               "%0.2f (fp16/fp32) %.2f (fp16/pseudo fp16)\n",
               s, src.to_string().c_str(), filter.to_string().c_str(),
               time_in_ms_fp32, bandwith * 4 / time_in_ms_fp32, time_in_ms_fp16,
               bandwith * 2 / time_in_ms_fp16, time_in_ms_pseudo_fp16,
               bandwith * 2 / time_in_ms_pseudo_fp16,
               time_in_ms_fp32 / time_in_ms_fp16,
               time_in_ms_pseudo_fp16 / time_in_ms_fp16);

    };


    // clang-format off
    for (size_t s : {1, 2})
    for (size_t f : {3, 5, 7})
    for (size_t batch : {64})
    for (size_t c : {16, 32, 64, 128})
    for (size_t ih: {128, 256})
    for (size_t iw : {128, 256})
        run(batch, c, ih, iw, f, s);
    // clang-format on

    run(128, 192, 28, 28, 3, 1);
    run(128, 192, 28, 28, 3, 2);
    run(128, 576, 14, 14, 3, 1);
    run(128, 384, 14, 14, 3, 1);
    run(128, 32, 112, 112, 3, 1);
    run(128, 960, 7, 7, 3, 1);
    run(128, 384, 14, 14, 3, 1);
    run(128, 144, 56, 56, 3, 2);
    run(128, 384, 14, 14, 3, 1);
    run(128, 144, 56, 56, 3, 1);
    run(128, 96, 112, 112, 3, 2);
    run(128, 384, 14, 14, 3, 1);
    run(128, 192, 28, 28, 3, 1);
    run(128, 576, 14, 14, 3, 1);
    run(128, 576, 14, 14, 3, 2);
}

TEST_F(CUDA, BENCHMARK_CHANWISE_CONV_FORWARD_FLOAT) {
    CUBenchmarker<ConvolutionForward> bencher(handle_cuda());
    size_t RUNS = 1;
    bencher.set_display(false).set_times(RUNS);
    bencher.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            ConvBiasForward::algo_name<ConvBiasForward::DirectParam>(
                    "CHANNEL_WISE", {})
                    .c_str()));

    Convolution::Param param;
    param.format = ConvBias::Param::Format::NCHW;
    param.sparse = Convolution::Param::Sparse::GROUP;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t c, size_t ih, size_t iw, size_t f,
                   size_t s) {
        param.pad_h = f / 2;
        param.pad_w = f / 2;
        param.stride_h = s;
        param.stride_w = s;
        param.compute_mode = param::Convolution::ComputeMode::DEFAULT;

        TensorShape src = {batch, c, ih, iw}, filter = {c, 1, 1, f, f};

        TensorLayout dst_layout;
        auto opr = handle_cuda()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({src, dtype::Float32()}, {filter, dtype::Float32()},
                           dst_layout);
        float bandwith = static_cast<float>(src.total_nr_elems() +
                                            filter.total_nr_elems() +
                                            dst_layout.total_nr_elems()) /
                         (1024 * 1024 * 1024) * 1e3;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        auto time_in_ms_fp32 = bencher.execs({src, filter, {}}) / RUNS;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        auto time_in_ms_fp16 = bencher.execs({src, filter, {}}) / RUNS;

        printf("stride=%zu src=%s, filter=%s, float32: %.2fms %.2fGB/s "
               "float16: %.2fms %.2fGB/s "
               "speedup: "
               "%0.2f (fp16/fp32)\n",
               s, src.to_string().c_str(), filter.to_string().c_str(),
               time_in_ms_fp32, bandwith * 4 / time_in_ms_fp32, time_in_ms_fp16,
               bandwith * 2 / time_in_ms_fp16,
               time_in_ms_fp32 / time_in_ms_fp16);

    };


    // clang-format off
    for (size_t s : {1})
    for (size_t f : {3, 5, 7})
    for (size_t batch : {64})
    for (size_t c : {16, 32, 64, 128})
    for (size_t ih: {8, 16, 32, 128, 256})
    for (size_t iw : {8, 16, 32, 128, 256})
        run(batch, c, ih, iw, f, s);
    // clang-format on

}

TEST_F(CUDA, BENCHMARK_CHANWISE_CONV_FORWARD_FLOAT_SMALL) {
    CUBenchmarker<ConvolutionForward> bencher(handle_cuda());
    size_t RUNS = 1;
    bencher.set_display(false).set_times(RUNS);

    Convolution::Param param;
    param.format = ConvBias::Param::Format::NCHW;
    param.sparse = Convolution::Param::Sparse::GROUP;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t c, size_t ih, size_t iw, size_t f,
                   size_t s) {
        param.pad_h = f / 2;
        param.pad_w = f / 2;
        param.stride_h = s;
        param.stride_w = s;
        param.compute_mode = param::Convolution::ComputeMode::DEFAULT;

        TensorShape src = {batch, c, ih, iw}, filter = {c, 1, 1, f, f};

        TensorLayout dst_layout;
        auto opr = handle_cuda()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({src, dtype::Float32()}, {filter, dtype::Float32()},
                           dst_layout);
        float bandwith = static_cast<float>(src.total_nr_elems() +
                                            filter.total_nr_elems() +
                                            dst_layout.total_nr_elems()) /
                         (1024 * 1024 * 1024) * 1e3;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_before_exec_callback(AlgoChecker<ConvolutionForward>(
                        ConvBiasForward::algo_name<
                                ConvBiasForward::DirectParam>("CHANNEL_WISE",
                                                              {})
                                .c_str()));
        auto time_in_ms_fp32_normal = bencher.execs({src, filter, {}}) / RUNS;

        bencher.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
                ConvBiasForward::algo_name<ConvBiasForward::DirectParam>(
                        "CHANNEL_WISE", {})
                        .c_str()));
        auto time_in_ms_fp32_small = bencher.execs({src, filter, {}}) / RUNS;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        auto time_in_ms_fp16_small = bencher.execs({src, filter, {}}) / RUNS;

        printf("stride=%zu src=%s, filter=%s, fp32 normal: %.2fms %.2fGB/s "
               "small: %.2fms %.2fGB/s, fp16 small: %.2fms %.2fGB/s, "
               "speedup: "
               "%0.2f (fp32 small/normal) %0.2f (small fp16/fp32)\n",
               s, src.to_string().c_str(), filter.to_string().c_str(),
               time_in_ms_fp32_normal, bandwith * 4 / time_in_ms_fp32_normal,
               time_in_ms_fp32_small, bandwith * 4 / time_in_ms_fp32_small,
               time_in_ms_fp16_small, bandwith * 2 / time_in_ms_fp16_small,
               time_in_ms_fp32_normal / time_in_ms_fp32_small,
               time_in_ms_fp32_small / time_in_ms_fp16_small);
    };


    // clang-format off
    for (size_t s : {1})
    for (size_t f : {3, 5})
    for (size_t batch : {64})
    for (size_t c : {16, 32, 64, 128})
    for (size_t ih: {8, 16, 32})
    for (size_t iw : {8, 16, 32})
        run(batch, c, ih, iw, f, s);
    // clang-format on

    run(128, 192, 28, 28, 3, 1);
    run(128, 576, 14, 14, 3, 1);
    run(128, 384, 14, 14, 3, 1);
    run(128, 960, 7, 7, 3, 1);
    run(128, 384, 14, 14, 3, 1);
    run(128, 384, 14, 14, 3, 1);
    run(128, 384, 14, 14, 3, 1);
    run(128, 192, 28, 28, 3, 1);
    run(128, 576, 14, 14, 3, 1);

}

TEST_F(CUDA, BENCHMARK_CHANWISE_CONV_FORWARD_CUDNN_DNN) {
    CUBenchmarker<ConvBiasForward> bencher(handle_cuda());
    size_t RUNS = 1;
    bencher.set_display(false).set_times(RUNS);

    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW;
    param.sparse = ConvBias::Param::Sparse::GROUP;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t c, size_t ih, size_t iw, size_t f,
                   size_t s) {
        param.pad_h = f / 2;
        param.pad_w = f / 2;
        param.stride_h = s;
        param.stride_w = s;
        param.compute_mode = param::ConvBias::ComputeMode::DEFAULT;

        TensorShape src = {batch, c, ih, iw}, filter = {c, 1, 1, f, f},
                    bias = {1, c, 1, 1};

        TensorLayout dst_layout;
        auto opr = handle_cuda()->create_operator<ConvBias>();
        opr->param() = param;
        opr->deduce_layout({src, dtype::Float32()}, {filter, dtype::Float32()},
                           {bias, dtype::Float32()}, {}, dst_layout);
        float computation_mops =
                static_cast<float>(dst_layout.total_nr_elems() * f * f * 2) *
                1e-6;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        bencher.set_before_exec_callback(
                AlgoChecker<ConvBiasForward>(".+CHANNEL_WISE.+"));
        auto time_in_ms_dnn = bencher.execs({src, filter, bias, {}, {}}) / RUNS;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        bencher.set_before_exec_callback(AlgoChecker<ConvBiasForward>(
                ".+CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM.+"));
        auto time_in_ms_cudnn =
                bencher.execs({src, filter, bias, {}, {}}) / RUNS;

        printf("stride=%zu src=%s, filter=%s, dst=%s, dnn: %.2fms %.2fGB/s "
               "cudnn: %.2fms %.2fGB/s "
               "speedup: "
               "%0.2f (dnn/cudnn)\n",
               s, src.to_string().c_str(), filter.to_string().c_str(),
               dst_layout.to_string().c_str(), time_in_ms_dnn,
               computation_mops / time_in_ms_dnn, time_in_ms_cudnn,
               computation_mops / time_in_ms_cudnn,
               time_in_ms_cudnn / time_in_ms_dnn);
    };

    // clang-format off
    for(size_t batch:{1, 16, 32, 64, 128}){
        run(batch, 32, 112, 112, 3, 1);
        run(batch, 96, 112, 112, 3, 2);
        run(batch, 96, 112, 112, 3, 1);
        run(batch, 144, 56, 56, 3, 2);
        run(batch, 144, 56, 56, 3, 1);
        run(batch, 192, 28, 28, 3, 1);
        run(batch, 384, 14, 14, 3, 1);
        run(batch, 576, 14, 14, 3, 1);
        run(batch, 960, 7, 7, 3, 1);
        //! calibrate heu algo policy hw_size param
        run(batch, 144, 24, 24, 3, 1);
        run(batch, 144, 22, 22, 3, 1);
        run(batch, 144, 20, 20, 3, 1);
        run(batch, 144, 18, 18, 3, 1);
    }
    // clang-format on
}

TEST_F(CUDA, BENCHMARK_CHANWISE_CONV_BACKWARD_DATA_FLOAT_SMALL) {
    CUBenchmarker<ConvolutionBackwardData> bencher(handle_cuda());
    size_t RUNS = 1;
    bencher.set_display(false).set_times(RUNS);

    ConvolutionBackwardData::Param param;
    param.format = Convolution::Param::Format::NCHW;
    param.sparse = Convolution::Param::Sparse::GROUP;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t c, size_t ih, size_t iw, size_t f,
                   size_t s) {
        param.pad_h = f / 2;
        param.pad_w = f / 2;
        param.stride_h = s;
        param.stride_w = s;
        param.compute_mode = param::Convolution::ComputeMode::DEFAULT;

        TensorShape src = {batch, c, ih, iw}, filter = {c, 1, 1, f, f};
        float bandwith = static_cast<float>(src.total_nr_elems() +
                                            filter.total_nr_elems() +
                                            src.total_nr_elems()) /
                         (1024 * 1024 * 1024) * 1e3;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_before_exec_callback(
                   AlgoChecker<ConvolutionBackwardData>("CHANNEL_WISE"));
        auto time_in_ms_fp32_normal = bencher.execs({filter, src, src}) / RUNS;

        bencher.set_before_exec_callback(
                   AlgoChecker<ConvolutionBackwardData>("CHANNEL_WISE_SMALL"));
        auto time_in_ms_fp32_small = bencher.execs({filter, src, src}) / RUNS;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        auto time_in_ms_fp16_small = bencher.execs({filter, src, src}) / RUNS;

        printf("stride=%zu src=%s, filter=%s, fp32 normal: %.2fms %.2fGB/s "
               "small: %.2fms %.2fGB/s, fp16 small: %.2fms %.2fGB/s, "
               "speedup: "
               "%0.2f (fp32 small/normal) %0.2f (small fp16/fp32)\n",
               s, src.to_string().c_str(), filter.to_string().c_str(),
               time_in_ms_fp32_normal, bandwith * 4 / time_in_ms_fp32_normal,
               time_in_ms_fp32_small, bandwith * 4 / time_in_ms_fp32_small,
               time_in_ms_fp16_small, bandwith * 2 / time_in_ms_fp16_small,
               time_in_ms_fp32_normal / time_in_ms_fp32_small,
               time_in_ms_fp32_small / time_in_ms_fp16_small);
    };


    // clang-format off
    for (size_t s : {1})
    for (size_t f : {3, 5})
    for (size_t batch : {64})
    for (size_t c : {16, 32, 64, 128})
    for (size_t ih: {8, 16, 32})
    for (size_t iw : {8, 16, 32})
        run(batch, c, ih, iw, f, s);
    // clang-format on

    run(128, 192, 28, 28, 3, 1);
    run(128, 576, 14, 14, 3, 1);
    run(128, 384, 14, 14, 3, 1);
    run(128, 960, 7, 7, 3, 1);
    run(128, 384, 14, 14, 3, 1);
    run(128, 384, 14, 14, 3, 1);
    run(128, 384, 14, 14, 3, 1);
    run(128, 192, 28, 28, 3, 1);
    run(128, 576, 14, 14, 3, 1);

}

TEST_F(CUDA, BENCHMARK_CHANWISE_CONV_BWD_DATA) {
    CUBenchmarker<ConvolutionBackwardData> bencher(handle_cuda());
    size_t RUNS = 1;
    bencher.set_display(false).set_times(RUNS);
    bencher.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardData>("CHANNEL_WISE"));

    Convolution::Param param;
    param.format = ConvBias::Param::Format::NCHW;
    param.sparse = Convolution::Param::Sparse::GROUP;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t ocpg, size_t group, size_t ih,
                   size_t iw, size_t f, size_t p, size_t s) {
        param.pad_h = p;
        param.pad_w = p;
        param.stride_h = s;
        param.stride_w = s;
        size_t oh, ow;
        infer_conv_shape2d(ih, iw, f, f, s, s, p, p, oh, ow, true);
        param.compute_mode = param::Convolution::ComputeMode::DEFAULT;

        TensorShape src_grad = {batch, group, ih, iw},
                    dst_grad = {batch, group * ocpg, oh, ow},
                    flt = {group, ocpg, 1, f, f};

        auto opr = handle_cuda()->create_operator<Convolution>();
        opr->param() = param;
        float bandwith = static_cast<float>(flt.total_nr_elems() +
                                            dst_grad.total_nr_elems() +
                                            src_grad.total_nr_elems()) /
                         (1024 * 1024 * 1024) * 1e3;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        auto time_in_ms_fp32 = bencher.execs({flt, dst_grad, src_grad}) / RUNS;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        auto time_in_ms_fp16 = bencher.execs({flt, dst_grad, src_grad}) / RUNS;

        printf("stride=%zu, src_grad=%s, flt=%s, "
               "float32: %.2fms %.2fGB/s "
               "float16: %.2fms %.2fGB/s "
               "speedup: "
               "%0.2f (fp16/fp32)\n",
               s, src_grad.to_string().c_str(), flt.to_string().c_str(),
               time_in_ms_fp32, bandwith * 4 / time_in_ms_fp32, time_in_ms_fp16,
               bandwith * 2 / time_in_ms_fp16,
               time_in_ms_fp32 / time_in_ms_fp16);
    };

    // clang-format off
    for (size_t s : {1, 2})
    for (size_t f : {3, 5, 7})
    for (size_t p : {f / 2})
    for (size_t batch : {64})
    for (size_t ocpg : {1})
    for (size_t group : {16, 32, 64, 128})
    for (size_t ih : {8, 16, 32, 128, 256})
    for (size_t iw : {8, 16, 32, 128, 256})
        run(batch, ocpg, group, ih, iw, f, p, s);
    // clang-format on
}

TEST_F(CUDA, BENCHMARK_CHANWISE_CONV_BWD_FILTER) {
    CUBenchmarker<ConvolutionBackwardFilter> bencher(handle_cuda());
    size_t RUNS = 1;
    bencher.set_display(false).set_times(RUNS);
    bencher.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardFilter>("CHANNEL_WISE"));

    Convolution::Param param;
    param.format = ConvBias::Param::Format::NCHW;
    param.sparse = Convolution::Param::Sparse::GROUP;
    NormalRNG rng;

    auto run = [&](size_t batch, size_t ocpg, size_t group, size_t i,
                   size_t f, size_t p, size_t s) {
        param.pad_h = p;
        param.pad_w = p;
        param.stride_h = s;
        param.stride_w = s;
        size_t d = infer_conv_shape(i, f, s, p, true);
        param.compute_mode = param::Convolution::ComputeMode::DEFAULT;

        TensorShape src = {batch, group, i, i},
                    dst_grad = {batch, group * ocpg, d, d},
                    flt_grad = {group, ocpg, 1, f, f};

        auto opr = handle_cuda()->create_operator<Convolution>();
        opr->param() = param;
        float bandwith = static_cast<float>(flt_grad.total_nr_elems() +
                                            dst_grad.total_nr_elems() +
                                            src.total_nr_elems()) /
                         (1024 * 1024 * 1024) * 1e3;
        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        auto time_in_ms_fp32 = bencher.execs({src, dst_grad, flt_grad}) / RUNS;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        auto time_in_ms_fp16 = bencher.execs({src, dst_grad, flt_grad}) / RUNS;

        printf("stride=%zu, src=%s, flt_grad=%s, "
               "float32: %.2fms %.2fGB/s "
               "float16: %.2fms %.2fGB/s "
               "speedup: "
               "%.2f (fp16/fp32)\n",
               s, src.to_string().c_str(), flt_grad.to_string().c_str(),
               time_in_ms_fp32, bandwith * 4 / time_in_ms_fp32, time_in_ms_fp16,
               bandwith * 2 / time_in_ms_fp16,
               time_in_ms_fp32 / time_in_ms_fp16);
    };

    // clang-format off
    for (size_t s : {1, 2})
    for (size_t f : {3, 5, 7})
    for (size_t p : {f / 2})
    for (size_t batch : {64})
    for (size_t ocpg : {1})
    for (size_t group : {16, 32, 64, 128})
    for (size_t i : {8, 16, 32, 64, 128})
        run(batch, ocpg, group, i, f, p, s);
    // clang-format on
}

#endif

// vim: syntax=cpp.doxygen
