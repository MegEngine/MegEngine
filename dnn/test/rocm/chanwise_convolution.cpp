/**
 * \file dnn/test/rocm/chanwise_convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "include/hcc_detail/hcc_defs_prologue.h"
#include "megdnn/oprs/nn.h"

#include "megcore_rocm.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/convolution.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/rocm/fixture.h"

#include "hip_header.h"

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

template <int P0, int P1, int P2>
class BenchmarkEnv {
    Handle *handle, *handle_cpu;
    std::unique_ptr<GaussianRNG> rng;
    TensorLayout lsrc, lflt0, lflt1, ldst;
    std::unique_ptr<Tensor<>> src0, src1, flt0, flt0_cpu, flt1, flt1_cpu, dst0,
            dst1;
    hipEvent_t hip_ev[3];
    hipStream_t hip_stream;
    size_t pad_h, pad_w;

    template <typename T>
    static std::tuple<T, T, T> shuffle(std::tuple<T, T, T> data) {
        return std::make_tuple(std::get<P0>(data), std::get<P1>(data),
                               std::get<P2>(data));
    }

public:
    BenchmarkEnv(Handle* handle, Handle* handle_cpu) {
        this->handle = handle;
        this->handle_cpu = handle_cpu;
        rng = handle->create_operator<GaussianRNG>();
        // make cpu handle used
        handle_cpu->create_operator<Sleep>()->exec();

        for (int i = 0; i < 3; ++i)
            hipEventCreate(&hip_ev[i]);
        megcoreGetROCMStream(handle->megcore_computing_handle(), &hip_stream);
    }

    ~BenchmarkEnv() {
        for (int i = 0; i < 3; ++i)
            hipEventDestroy(hip_ev[i]);
    }

    void alloc(size_t N, size_t IC, size_t IH, size_t IW, size_t CHL_MUL,
               size_t FH, size_t FW, size_t PH, size_t PW) {
        pad_h = PH;
        pad_w = PW;
        auto mkly = [](const TensorShape& s) {
            return TensorLayout{s, dtype::Float32()};
        };
        lsrc = mkly({N, IC, IH, IW});
        lflt0 = mkly({CHL_MUL * IC, IC, FH, FW});
        lflt1 = mkly({IC, CHL_MUL, 1, FH, FW});
        ldst = mkly(
                {N, IC * CHL_MUL, IH - FH + 1 + PH * 2, IW - FW + 1 + PW * 2});
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
        megdnn_memcpy_D2H(handle, flt1_cpu->ptr(), flt1->ptr(),
                          lflt1.span().dist_byte());

        const size_t IC = lflt1[0], CHL_MUL = lflt1[1],
                     FSIZE = lflt1[3] * lflt1[4];

        // fill flt0 from flt1
        float* src = flt1_cpu->ptr();
        float* dst = flt0_cpu->ptr();
        memset(dst, 0, lflt0.span().dist_byte());
        for (size_t i = 0; i < IC; ++i) {
            for (size_t j = 0; j < CHL_MUL; ++j) {
                memcpy(dst + ((i * CHL_MUL + j) * IC + i) * FSIZE,
                       src + (i * CHL_MUL + j) * FSIZE, FSIZE * sizeof(float));
            }
        }

        megdnn_memcpy_H2D(handle, flt0->ptr(), dst, lflt0.span().dist_byte());
    }

    void fill_dst() {
        rng->exec(dst0->tensornd(), {});
        megdnn_memcpy_D2D(handle, dst1->ptr(), dst0->ptr(),
                          ldst.span().dist_byte());
    }

    template <class Opr>
    void exec(Opr* opr0, Opr* opr1) {
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
                            std::max(opr0->get_workspace_in_bytes(
                                             a0.layout, b0.layout, c0.layout),
                                     opr1->get_workspace_in_bytes(
                                             a1.layout, b1.layout, c1.layout)));
        hipProfilerStart();
        hipEventRecord(hip_ev[0], hip_stream);
        opr0->exec(a0, b0, c0, wk.workspace());
        hipEventRecord(hip_ev[1], hip_stream);
        opr1->exec(a1, b1, c1, wk.workspace());
        hipEventRecord(hip_ev[2], hip_stream);
        hipProfilerStop();

        if (getenv("MEGDNN_CHANWISE_CONV_VERBOSE") ||
            getenv("MEGDNN_CHANWISE_CONV_FULLBENCH")) {
            hipStreamSynchronize(hip_stream);
            float t0 = -1, t1 = -1;
            hipEventElapsedTime(&t0, hip_ev[0], hip_ev[1]);
            hipEventElapsedTime(&t1, hip_ev[1], hip_ev[2]);
            printf("%s;%s;%s: miopen/megdnn: %.3fms/%.3fms=%.3f\n",
                   lsrc.TensorShape::to_string().c_str(),
                   lflt1.TensorShape::to_string().c_str(),
                   ldst.TensorShape::to_string().c_str(), t0, t1, t0 / t1);
        }
    }

    void cmp_dst() {
        Tensor<> dst0_cpu(handle_cpu, ldst), dst1_cpu(handle_cpu, ldst);
        megdnn_memcpy_D2H(handle, dst0_cpu.ptr(), dst0->ptr(),
                          ldst.span().dist_byte());
        megdnn_memcpy_D2H(handle, dst1_cpu.ptr(), dst1->ptr(),
                          ldst.span().dist_byte());
        dst0_cpu.check_with(dst1_cpu);
    }

    void cmp_src() {
        Tensor<> src0_cpu(handle_cpu, lsrc), src1_cpu(handle_cpu, lsrc);
        megdnn_memcpy_D2H(handle, src0_cpu.ptr(), src0->ptr(),
                          lsrc.span().dist_byte());
        megdnn_memcpy_D2H(handle, src1_cpu.ptr(), src1->ptr(),
                          lsrc.span().dist_byte());
        src0_cpu.check_with(src1_cpu);
    }

    void cmp_flt() {
        Tensor<> flt0_cpu(handle_cpu, lflt0), flt1_cpu(handle_cpu, lflt1);
        float* p0 = flt0_cpu.ptr();
        float* p1 = flt1_cpu.ptr();
        megdnn_memcpy_D2H(handle, p0, flt0->ptr(), lflt0.span().dist_byte());
        megdnn_memcpy_D2H(handle, p1, flt1->ptr(), lflt1.span().dist_byte());

        size_t IC = lflt1[0], CHL_MUL = lflt1[1], FSIZE = lflt1[3] * lflt1[4];

        double tot_err = 0, tot_err_num = 0;
        for (size_t i = 0; i < IC; ++i) {
            for (size_t j = 0; j < CHL_MUL; ++j) {
                auto t0 = p0 + ((i * CHL_MUL + j) * IC + i) * FSIZE,
                     t1 = p1 + (i * CHL_MUL + j) * FSIZE;
                for (size_t k = 0; k < FSIZE; ++k) {
                    auto err = std::abs(diff(t0[k], t1[k]));
                    tot_err += err;
                    tot_err_num += 1;
                    ASSERT_LT(err, 1e-2)
                            << "failed at " << i << " " << j << " " << k
                            << " vals=" << t0[k] << "," << t1[k];
                }
            }
        }
        auto avg_err = tot_err / tot_err_num;
        ASSERT_LT(avg_err, 1e-4);
    }
};

}  // anonymous namespace

constexpr auto M = Convolution::Mode::CROSS_CORRELATION;

TEST_F(ROCM, CHANWISE_CONVOLUTION_FORWARD) {
    Checker<Convolution> checker(handle_rocm());
    bool require_algo = false;
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CHANNEL_WISE", &require_algo));

    // simple case
    checker.set_param(gconv_param({M, 0, 0, 1, 1}))
            .execs({{1, 1, 2, 2}, {1, 1, 1, 2, 2}, {}})
            .execs({{1, 1, 5, 5}, {1, 1, 1, 2, 2}, {}});

    checker.execs({{2, 2, 5, 5}, {2, 3, 1, 2, 2}, {2, 6, 4, 4}});

    checker.set_param(gconv_param({M, 1, 1, 1, 1}))
            .execs({{2, 2, 5, 5}, {2, 1, 1, 2, 2}, {}});

    checker.set_param(gconv_param({M, 2, 3, 2, 1}))
            .execs({{32, 12, 20, 10}, {12, 2, 1, 4, 5}, {}});

    // padding larger than kern
    checker.set_param(gconv_param({M, 20, 30, 4, 5}))
            .execs({{32, 12, 20, 10}, {12, 2, 1, 4, 5}, {}});
}

TEST_F(ROCM, CHANWISE_CONVOLUTION_BACKWARD_DATA) {
    Checker<ConvolutionBackwardData> checker(handle_rocm());

    checker.set_param(gconv_param({M, 0, 0, 1, 1}))
            .execs({{1, 1, 1, 2, 2}, {1, 1, 1, 1}, {1, 1, 2, 2}})
            .execs({{1, 1, 1, 2, 2}, {1, 1, 5, 5}, {1, 1, 6, 6}});

    checker.execs({{2, 1, 1, 2, 2}, {1, 2, 1, 1}, {1, 2, 2, 2}})
            .execs({{2, 1, 1, 2, 2}, {1, 2, 5, 5}, {1, 2, 6, 6}})
            .execs({{2, 3, 1, 2, 2}, {2, 6, 5, 5}, {2, 2, 6, 6}});

    checker.set_param(gconv_param({M, 1, 1, 1, 1}))
            .execs({{2, 1, 1, 2, 2}, {2, 2, 5, 5}, {2, 2, 4, 4}});

    checker.set_param(gconv_param({M, 2, 3, 2, 1}))
            .execs({{12, 3, 1, 4, 5}, {32, 36, 20, 10}, {32, 12, 39, 8}});

    // padding larger than kern
    checker.set_param(gconv_param({M, 20, 30, 4, 5}))
            .execs({{6, 2, 1, 4, 5}, {32, 12, 10, 12}, {32, 6, 2, 3}});
}

TEST_F(ROCM, CHANWISE_CONVOLUTION_BACKWARD_FILTER) {
    Checker<ConvolutionBackwardFilter> checker(handle_rocm());

    checker.set_param(gconv_param({M, 0, 0, 1, 1}))
            .execs({{1, 1, 2, 2}, {1, 1, 1, 1}, {1, 1, 1, 2, 2}})
            .execs({{1, 1, 6, 6}, {1, 1, 5, 5}, {1, 1, 1, 2, 2}})
            .execs({{256, 1, 2, 2}, {256, 1, 1, 1}, {1, 1, 1, 2, 2}});
    checker.execs({{1, 2, 2, 2}, {1, 2, 1, 1}, {2, 1, 1, 2, 2}})
            .execs({{1, 2, 6, 6}, {1, 2, 5, 5}, {2, 1, 1, 2, 2}})
            .execs({{2, 2, 6, 6}, {2, 6, 5, 5}, {2, 3, 1, 2, 2}});

    checker.set_param(gconv_param({M, 1, 1, 1, 1}))
            .execs({{2, 2, 4, 4}, {2, 2, 5, 5}, {2, 1, 1, 2, 2}});

    checker.set_param(gconv_param({M, 0, 0, 1, 1}))
            .execs({{40960, 1, 1, 1}, {40960, 1, 1, 1}, {1, 1, 1, 1, 1}});

    checker.set_param(gconv_param({M, 2, 3, 2, 1}))
            .execs({{32, 12, 39, 8}, {32, 36, 20, 10}, {12, 3, 1, 4, 5}});

    // padding larger than kern
    checker.set_param(gconv_param({M, 20, 30, 4, 5}))
            .execs({{32, 6, 2, 3}, {32, 12, 10, 12}, {6, 2, 1, 4, 5}});

    // unused filter items
    checker.set_param(gconv_param({M, 2, 3, 2, 3}))
            .execs({{32, 6, 1, 1}, {32, 12, 1, 1}, {6, 2, 1, 5, 7}});
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(ROCM, CHANWISE_CONVOLUTION_FORWARD_BENCH_CHECK) {
    auto handle = handle_rocm();
    auto handle_cpu = handle_naive();
    auto conv0 = handle->create_operator<ConvolutionForward>();
    auto conv1 = handle->create_operator<ConvolutionForward>();
    BenchmarkEnv<0, 1, 2> benv(handle, handle_cpu);

    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW, size_t CHL_MUL,
                   size_t FH, size_t FW, size_t PH, size_t PW) {
        benv.alloc(N, IC, IH, IW, CHL_MUL, FH, FW, PH, PW);
        benv.fill_src();
        benv.fill_flt();
        benv.exec(conv0.get(), conv1.get());
        benv.cmp_dst();
    };

    run(64, 60, 50, 50, 1, 3, 3, 1, 1);
    if (check_need_full_bench()) {
        run(64, 728, 18, 18, 2, 5, 5, 2, 2);
        run(64, 64, 150, 150, 2, 3, 3, 1, 1);
        run(1, 2048, 4, 4, 2, 3, 3, 1, 1);
    }
}

TEST_F(ROCM, CHANWISE_CONVOLUTION_BWD_DATA_BENCH_CHECK) {
    auto handle = handle_rocm();
    auto handle_cpu = handle_naive();
    auto conv0 = handle->create_operator<ConvolutionBackwardData>();
    auto conv1 = handle->create_operator<ConvolutionBackwardData>();
    BenchmarkEnv<1, 2, 0> benv(handle, handle_cpu);

    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW, size_t CHL_MUL,
                   size_t FH, size_t FW, size_t PH, size_t PW) {
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

TEST_F(ROCM, CHANWISE_CONVOLUTION_BWD_FILTER_BENCH_CHECK) {
    auto handle = handle_rocm();
    auto handle_cpu = handle_naive();
    auto conv0 = handle->create_operator<ConvolutionBackwardFilter>();
    auto conv1 = handle->create_operator<ConvolutionBackwardFilter>();
    BenchmarkEnv<0, 2, 1> benv(handle, handle_cpu);

    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW, size_t CHL_MUL,
                   size_t FH, size_t FW, size_t PH, size_t PW) {
        benv.alloc(N, IC, IH, IW, CHL_MUL, FH, FW, PH, PW);
        benv.fill_src();
        benv.fill_dst();
        benv.exec(conv0.get(), conv1.get());
        benv.cmp_flt();
    };

    run(64, 60, 50, 50, 1, 3, 3, 1, 1);
    if (check_need_full_bench()) {
        run(64, 728, 18, 18, 2, 5, 5, 2, 2);
        run(64, 64, 150, 150, 2, 3, 3, 1, 1);
        run(1, 2048, 4, 4, 2, 3, 3, 1, 1);
    }
}

TEST_F(ROCM, CHANWISE_CONVOLUTION_BENCH_ALL_ALGO_FWD) {
    // enable profiling
    OprProxy<ConvolutionForward> proxy(true);
    proxy.warmup_times = 1;
    proxy.exec_times = 10;
    Benchmarker<ConvolutionForward> checker(handle_rocm());
    checker.set_times(1);
    ConvolutionForward::Param param;
    param.sparse = ConvolutionForward::Param::Sparse::GROUP;
    checker.set_param(param);

    auto run = [&](size_t N, size_t C, size_t IH, size_t IW, size_t FH,
                   size_t FW) {
        checker.set_proxy(proxy);
        checker.execs({{N, C, IH, IW}, {C, 1, 1, FH, FW}, {}});
    };

    run(128, 64, 90, 80, 3, 3);
    run(128, 90, 100, 100, 3, 5);
    run(128, 32, 62, 62, 5, 5);
}

TEST_F(ROCM, CHANWISE_CONVOLUTION_BENCH_ALL_ALGO_BWD_DATA) {
    // enable profiling
    OprProxy<ConvolutionBackwardData> proxy(true);
    proxy.warmup_times = 1;
    proxy.exec_times = 10;
    Benchmarker<ConvolutionBackwardData> checker(handle_rocm());
    checker.set_times(1);
    ConvolutionBackwardData::Param param;
    param.sparse = ConvolutionForward::Param::Sparse::GROUP;
    checker.set_param(param);

    auto run = [&](size_t N, size_t C, size_t IH, size_t IW, size_t FH,
                   size_t FW) {
        checker.set_proxy(proxy);
        checker.execs({{C, 1, 1, FH, FW},
                       {N, C, IH - FH + 1, IW - FW + 1},
                       {N, C, IH, IW}});
    };

    run(128, 64, 90, 80, 3, 3);
    run(128, 90, 100, 100, 3, 5);
    run(128, 32, 62, 62, 5, 5);
}

TEST_F(ROCM, CHANWISE_CONVOLUTION_BENCH_ALL_ALGO_BWD_FILTER) {
    // enable profiling
    OprProxy<ConvolutionBackwardFilter> proxy(true);
    proxy.warmup_times = 1;
    proxy.exec_times = 10;
    Benchmarker<ConvolutionBackwardFilter> checker(handle_rocm());
    checker.set_times(1);
    ConvolutionBackwardFilter::Param param;
    param.sparse = ConvolutionForward::Param::Sparse::GROUP;
    checker.set_param(param);

    auto run = [&](size_t N, size_t C, size_t IH, size_t IW, size_t FH,
                   size_t FW) {
        checker.set_proxy(proxy);
        checker.execs({{N, C, IH, IW},
                       {N, C, IH - FH + 1, IW - FW + 1},
                       {C, 1, 1, FH, FW}});
    };

    run(128, 64, 90, 80, 3, 3);
    run(128, 90, 100, 100, 3, 5);
    run(128, 32, 62, 62, 5, 5);
}

#endif

// vim: syntax=cpp.doxygen
