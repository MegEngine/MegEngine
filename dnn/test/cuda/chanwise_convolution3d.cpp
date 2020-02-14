/**
 * \file dnn/test/cuda/chanwise_convolution3d.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/nn.h"

#include "megcore_cuda.h"
#include "test/common/checker.h"
#include "test/common/convolution3d.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/cuda/fixture.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

using namespace megdnn;
using namespace test;

namespace {

#if MEGDNN_WITH_BENCHMARK
bool check_need_full_bench() {
    if (getenv("MEGDNN_CHANWISE_CONV3D_FULLBENCH"))
        return true;
    printf("set MEGDNN_CHANWISE_CONV3D_FULLBENCH to run full benchmark\n");
    return false;
}
#endif
Convolution3D::Param gconv_param(Convolution3D::Param p) {
    p.sparse = Convolution3D::Param::Sparse::GROUP;
    return p;
}

template <int P0, int P1, int P2>
class BenchmarkEnv {
    Handle *handle, *handle_cpu;
    std::unique_ptr<GaussianRNG> rng;
    TensorLayout lsrc, lflt0, lflt1, ldst;
    std::unique_ptr<Tensor<>> src0, src1, flt0, flt0_cpu, flt1, flt1_cpu, dst0,
            dst1;
    cudaEvent_t cuda_ev[3];
    cudaStream_t cuda_stream;
    size_t pad_d, pad_h, pad_w;

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
            cudaEventCreate(&cuda_ev[i]);
        megcoreGetCUDAStream(handle->megcore_computing_handle(), &cuda_stream);
    }

    ~BenchmarkEnv() {
        for (int i = 0; i < 3; ++i)
            cudaEventDestroy(cuda_ev[i]);
    }

    void alloc(size_t N, size_t IC, size_t ID, size_t IH, size_t IW,
               size_t CHL_MUL, size_t FD, size_t FH, size_t FW, size_t PD,
               size_t PH, size_t PW) {
        pad_d = PD;
        pad_h = PH;
        pad_w = PW;
        auto mkly = [](const TensorShape& s) {
            return TensorLayout{s, dtype::Float32()};
        };
        lsrc = mkly({N, IC, ID, IH, IW});
        lflt0 = mkly({CHL_MUL * IC, IC, FD, FH, FW});
        lflt1 = mkly({IC, CHL_MUL, 1, FD, FH, FW});
        ldst = mkly({N, IC * CHL_MUL, ID - FD + 1 + PD * 2,
                     IH - FH + 1 + PH * 2, IW - FW + 1 + PW * 2});
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
                     FSIZE = lflt1[3] * lflt1[4] * lflt1[5];

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
        opr0->param().pad_d = pad_d;
        opr0->param().pad_h = pad_h;
        opr0->param().pad_w = pad_w;
        opr1->param() = opr0->param();
        opr1->param().sparse = param::Convolution3D::Sparse::GROUP;

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
        cudaProfilerStart();
        cudaEventRecord(cuda_ev[0], cuda_stream);
        opr0->exec(a0, b0, c0, wk.workspace());
        cudaEventRecord(cuda_ev[1], cuda_stream);
        opr1->exec(a1, b1, c1, wk.workspace());
        cudaEventRecord(cuda_ev[2], cuda_stream);
        cudaProfilerStop();

        if (getenv("MEGDNN_CHANWISE_CONV3D_VERBOSE") ||
            getenv("MEGDNN_CHANWISE_CONV3D_FULLBENCH")) {
            cudaStreamSynchronize(cuda_stream);
            float t0 = -1, t1 = -1;
            cudaEventElapsedTime(&t0, cuda_ev[0], cuda_ev[1]);
            cudaEventElapsedTime(&t1, cuda_ev[1], cuda_ev[2]);
            printf("%s;%s;%s: cudnn/megdnn: %.3fms/%.3fms=%.3f\n",
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

        size_t IC = lflt1[0], CHL_MUL = lflt1[1],
               FSIZE = lflt1[3] * lflt1[4] * lflt1[5];

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

constexpr auto M = Convolution3D::Mode::CROSS_CORRELATION;

TEST_F(CUDA, CHANWISE_CONVOLUTION3D_FORWARD) {
    constexpr auto M = Convolution3D::Mode::CROSS_CORRELATION;
    Checker<Convolution3D> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(
            AlgoChecker<Convolution3DForward>(
                    "CHANNEL_WISE", &require_algo));
    checker.set_param(gconv_param({M, 0, 0, 0, 1, 1, 1}))
            .execs({{1, 1, 2, 2, 2}, {1, 1, 1, 2, 2, 2}, {}})
            .execs({{1, 1, 5, 5, 5}, {1, 1, 1, 2, 2, 2}, {}});
    checker.set_param(gconv_param({M, 0, 0, 0, 1, 1, 1}))
            .execs({{1, 2, 2, 2, 2}, {2, 1, 1, 2, 2, 2}, {}})
            .execs({{1, 2, 5, 5, 5}, {2, 1, 1, 2, 2, 2}, {}})
            .execs({{2, 2, 5, 5, 5}, {2, 3, 1, 2, 2, 2}, {2, 6, 4, 4, 4}});

    checker.set_param(gconv_param({M, 1, 1, 1, 1, 1, 1}))
            .execs({{2, 2, 5, 5, 5}, {2, 1, 1, 2, 2, 2}, {}});

    checker.set_param(gconv_param({M, 2, 3, 3, 2, 1, 1}))
            .execs({{4, 12, 10, 5, 10}, {12, 2, 1, 4, 5, 5}, {}});

    // padding larger than kern
    checker.set_param(gconv_param({M, 10, 15, 15, 4, 5, 5}))
            .execs({{4, 12, 10, 5, 10}, {12, 2, 1, 4, 5, 5}, {}});

    for (uint32_t n : {8, 12})
        for (uint32_t id : {12})
            for (uint32_t ih : {12})
                for (uint32_t iw : {16})
                    for (uint32_t ic : {4})
                        for (uint32_t oc : {4})
                            for (uint32_t fd : {2, 5})
                                for (uint32_t pd : {2})
                                    for (uint32_t sd : {1})
                                        for (uint32_t dd : {1}) {
                                            checker
                                                    .set_param(gconv_param(
                                                            {M, pd, pd, pd, sd,
                                                             sd, sd, dd, dd,
                                                             dd}))
                                                    .execs({{n, ic, id, ih, iw},
                                                            {ic, oc, 1, fd, fd,
                                                             fd},
                                                            {}});
                                        }
}

TEST_F(CUDA, CHANWISE_CONVOLUTION3D_BACKWARD_DATA) {
    Checker<Convolution3DBackwardData> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(
            AlgoChecker<Convolution3DBackwardData>(
                    "CHANNEL_WISE", &require_algo));

    checker.set_param(gconv_param({M, 0, 0, 0, 1, 1, 1}))
            .execs({{1, 1, 1, 2, 2, 2}, {1, 1, 1, 1, 1}, {1, 1, 2, 2, 2}})
            .execs({{1, 1, 1, 2, 2, 2}, {1, 1, 5, 5, 5}, {1, 1, 6, 6, 6}});

    require_algo = true;
    checker.execs({{2, 1, 1, 2, 2, 2}, {1, 2, 1, 1, 1}, {1, 2, 2, 2, 2}})
            .execs({{2, 1, 1, 2, 2, 2}, {1, 2, 5, 5, 5}, {1, 2, 6, 6, 6}})
            .execs({{2, 3, 1, 2, 2, 2}, {2, 6, 5, 5, 5}, {2, 2, 6, 6, 6}});

    checker.set_param(gconv_param({M, 1, 1, 1, 1, 1, 1}))
            .execs({{2, 1, 1, 2, 2, 2}, {2, 2, 5, 5, 5}, {2, 2, 4, 4, 4}});

    checker.set_param(gconv_param({M, 2, 3, 3, 2, 1, 1}))
            .execs({{12, 2, 1, 4, 5, 5},
                    {32, 24, 20, 10, 10},
                    {32, 12, 39, 8, 8}});

    // padding larger than kern
    checker.set_param(gconv_param({M, 20, 30, 20, 4, 5, 4}))
            .execs({{6, 2, 1, 4, 5, 4},
                    {32, 12, 10, 12, 10},
                    {32, 6, 2, 3, 2}});
}

TEST_F(CUDA, CHANWISE_CONVOLUTION3D_BACKWARD_FILTER) {
    Checker<Convolution3DBackwardFilter> checker(handle_cuda());
    bool require_algo = false;
    checker.set_before_exec_callback(
            AlgoChecker<Convolution3DBackwardFilter>(
                    "CHANNEL_WISE", &require_algo));

    checker.set_param(gconv_param({M, 0, 0, 0, 1, 1, 1}))
            .execs({{1, 1, 2, 2, 2}, {1, 1, 1, 1, 1}, {1, 1, 1, 2, 2, 2}})
            .execs({{1, 1, 6, 6, 6}, {1, 1, 5, 5, 5}, {1, 1, 1, 2, 2, 2}})
            .execs({{256, 1, 2, 2, 2}, {256, 1, 1, 1, 1}, {1, 1, 1, 2, 2, 2}});
    require_algo = true;
    checker.execs({{1, 2, 2, 2, 2}, {1, 2, 1, 1, 1}, {2, 1, 1, 2, 2, 2}})
            .execs({{1, 2, 6, 6, 6}, {1, 2, 5, 5, 5}, {2, 1, 1, 2, 2, 2}})
            .execs({{2, 2, 6, 6, 6}, {2, 6, 5, 5, 5}, {2, 3, 1, 2, 2, 2}});

    checker.set_param(gconv_param({M, 1, 1, 1, 1, 1, 1}))
            .execs({{2, 2, 4, 4, 4}, {2, 2, 5, 5, 5}, {2, 1, 1, 2, 2, 2}});

    require_algo = false;
    checker.set_param(gconv_param({M, 0, 0, 0, 1, 1, 1}))
            .execs({{40960, 1, 1, 1, 1},
                    {40960, 1, 1, 1, 1},
                    {1, 1, 1, 1, 1, 1}});
    require_algo = true;

    checker.set_param(gconv_param({M, 2, 3, 2, 2, 1, 2}))
            .execs({{32, 12, 39, 8, 20},
                    {32, 36, 20, 10, 10},
                    {12, 3, 1, 4, 5, 6}});

    // padding larger than kern
    checker.set_param(gconv_param({M, 20, 30, 30, 4, 5, 5}))
            .execs({{32, 6, 2, 3, 3},
                    {32, 12, 10, 12, 12},
                    {6, 2, 1, 4, 5, 5}});

    // unused filter items
    checker.set_param(gconv_param({M, 2, 3, 3, 2, 3, 3}))
            .execs({{32, 6, 1, 1, 1}, {32, 12, 1, 1, 1}, {6, 2, 1, 5, 7, 7}});
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, CHANWISE_CONVOLUTION3D_FORWARD_BENCH_CHECK) {
    auto handle = handle_cuda();
    auto handle_cpu = handle_naive();
    auto conv0 = handle->create_operator<Convolution3DForward>();
    auto conv1 = handle->create_operator<Convolution3DForward>();
    BenchmarkEnv<0, 1, 2> benv(handle, handle_cpu);

    auto run = [&](size_t N, size_t IC, size_t ID, size_t IH, size_t IW,
                   size_t CHL_MUL, size_t FD, size_t FH, size_t FW, size_t PD,
                   size_t PH, size_t PW) {
        benv.alloc(N, IC, ID, IH, IW, CHL_MUL, FD, FH, FW, PD, PH, PW);
        benv.fill_src();
        benv.fill_flt();
        benv.exec(conv0.get(), conv1.get());
        benv.cmp_dst();
    };

    run(64, 30, 10, 10, 10, 1, 3, 3, 3, 1, 1, 1);
    if (check_need_full_bench()) {
        run(64, 728, 9, 9, 9, 2, 5, 5, 5, 2, 2, 2);
        run(64, 64, 30, 30, 30, 2, 3, 3, 3, 1, 1, 1);
        run(1, 2048, 4, 4, 4, 2, 3, 3, 3, 1, 1, 1);
    }
}

TEST_F(CUDA, CHANWISE_CONVOLUTION3D_BWD_DATA_BENCH_CHECK) {
    auto handle = handle_cuda();
    auto handle_cpu = handle_naive();
    auto conv0 = handle->create_operator<Convolution3DBackwardData>();
    auto conv1 = handle->create_operator<Convolution3DBackwardData>();
    BenchmarkEnv<1, 2, 0> benv(handle, handle_cpu);

    auto run = [&](size_t N, size_t IC, size_t ID, size_t IH, size_t IW,
                   size_t CHL_MUL, size_t FD, size_t FH, size_t FW, size_t PD,
                   size_t PH, size_t PW) {
        benv.alloc(N, ID, IC, IH, IW, CHL_MUL, FD, FH, FW, PD, PH, PW);
        benv.fill_dst();
        benv.fill_flt();
        benv.exec(conv0.get(), conv1.get());
        benv.cmp_src();
    };

    run(64, 60, 50, 50, 50, 1, 3, 3, 3, 1, 1, 1);
    if (check_need_full_bench()) {
        run(64, 728, 18, 18, 18, 2, 5, 5, 5, 2, 2, 2);
        run(64, 64, 32, 32, 32, 2, 3, 3, 3, 1, 1, 1);
        run(1, 2048, 4, 4, 4, 2, 3, 3, 3, 1, 1, 1);
    }
}

TEST_F(CUDA, CHANWISE_CONVOLUTION3D_BWD_FILTER_BENCH_CHECK) {
    auto handle = handle_cuda();
    auto handle_cpu = handle_naive();
    auto conv0 = handle->create_operator<Convolution3DBackwardFilter>();
    auto conv1 = handle->create_operator<Convolution3DBackwardFilter>();
    BenchmarkEnv<0, 2, 1> benv(handle, handle_cpu);

    auto run = [&](size_t N, size_t IC, size_t ID, size_t IH, size_t IW,
                   size_t CHL_MUL, size_t FD, size_t FH, size_t FW, size_t PD,
                   size_t PH, size_t PW) {
        benv.alloc(N, IC, ID, IH, IW, CHL_MUL, FD, FH, FW, PD, PH, PW);
        benv.fill_src();
        benv.fill_dst();
        benv.exec(conv0.get(), conv1.get());
        benv.cmp_flt();
    };
    run(67, 729, 20, 20, 20, 1, 3, 3, 3, 1, 1, 1);
    if (check_need_full_bench()) {
        run(64, 728, 18, 18, 18, 2, 5, 5, 5, 2, 2, 2);
        // the case below is an sample that select unexpected algo_1
        run(64, 64, 32, 32, 32, 2, 3, 3, 3, 1, 1, 1);
        run(1, 2048, 4, 4, 4, 2, 3, 3, 3, 1, 1, 1);
    }
}
#endif

// vim: syntax=cpp.doxygen
