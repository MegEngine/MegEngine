/**
 * \file src/opr/test/blas.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/blas.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/megdnn_helper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/tensor_gen.h"
#include <random>

using namespace mgb;

namespace {
template <typename dt_src, typename dt_dst>
void brute_force_gemm(size_t M, size_t N, size_t K, bool transa, bool transb,
                      const dt_src* x, const dt_src* y, dt_dst* z) {
    for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n) {
            dt_dst cur = dt_dst(0);
            for (size_t k = 0; k < K; ++k) {
                cur += x[transa ? (k * M + m) : (m * K + k)] *
                       y[transb ? (n * K + k) : (k * N + n)];
            }
            z[m * N + n] = cur;
        }
}

float brute_force_dot(const HostTensorND& a, const HostTensorND& b) {
    auto sz = std::max(a.shape(0), b.shape(0));
    size_t ap = 0, bp = 0;
    float ret = 0;
    auto pa = a.ptr<float>(), pb = b.ptr<float>();
    auto as = a.layout().stride[0], bs = b.layout().stride[0];
    if (a.shape(0) != sz)
        as = 0;
    if (b.shape(0) != sz)
        bs = 0;
    for (size_t i = 0; i < sz; ++i) {
        ret += pa[ap] * pb[bp];
        ap += as;
        bp += bs;
    }
    return ret;
}

// (m,k) * (k,n) = (m,n)
void run_sgemm_test(bool transa, bool transb) {
    using Checker = AutoOprChecker<2, 1>;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto param = opr::MatrixMul::Param{transa, transb};
        return {opr::MatrixMul::make(inputs[0], inputs[1], param)};
    };
    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        size_t M, N, K;
        M = inp[0]->shape().shape[0];
        K = inp[0]->shape().shape[1];
        if (transa)
            std::swap(M, K);
        N = inp[1]->shape().shape[transb ? 0 : 1];

        auto z = dest[0].comp_node(inp[0]->comp_node())
                         .resize({M, N})
                         .ptr<float>();
        // brute-force gemm
        brute_force_gemm(M, N, K, transa, transb, inp[0]->ptr<float>(),
                         inp[1]->ptr<float>(), z);
    };

    auto mkshp = [](bool trans, size_t m, size_t k) {
        TensorShape rst{m, k};
        if (trans)
            std::swap(rst.shape[0], rst.shape[1]);
        return rst;
    };
    using namespace std::placeholders;
    auto mkx = std::bind(mkshp, transa, _1, _2);
    auto mky = std::bind(mkshp, transb, _1, _2);

    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    Checker(make_graph, fwd)
            .run({mkx(4, 6), mky(6, 2)}, opt)
            .run({mkx(2, 3), mky(3, 100)}, opt)
            .run({mkx(20, 3), mky(3, 20)}, opt);
}

#define FWD_BATCH_GEMM(dt_src, dt_dst)                                       \
    [transa, transb](Checker::NumOutArray& dest, Checker::NumInpArray inp) { \
        bool ta(transa), tb(transb);                                         \
        HostTensorND a, b;                                                   \
        size_t B, M, N, K;                                                   \
        a.copy_from(*(inp[0]));                                              \
        b.copy_from(*(inp[1]));                                              \
        B = a.shape().shape[0];                                              \
        M = a.shape().shape[1];                                              \
        K = a.shape().shape[2];                                              \
        N = b.shape().shape[tb ? 1 : 2];                                     \
        if (ta)                                                              \
            std::swap(M, K);                                                 \
        auto x = a.ptr<dt_src>(), y = b.ptr<dt_src>();                       \
        auto z = dest[0].resize({B, M, N}).ptr<dt_dst>();                    \
        for (size_t b = 0; b < B; ++b) {                                     \
            brute_force_gemm(M, N, K, ta, tb, x + b * M * K, y + b * K * N,  \
                             z + b * M * N);                                 \
        }                                                                    \
    }

void run_batched_sgemm_test(bool transa, bool transb) {
    using Checker = AutoOprChecker<2, 1>;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::BatchedMatrixMul::make(inputs[0], inputs[1],
                                            {transa, transb})};
    };

    auto fwd = FWD_BATCH_GEMM(float, float);

    auto mkshp = [](bool trans, size_t b, size_t m, size_t k) {
        TensorShape rst{b, m, k};
        if (trans)
            std::swap(rst.shape[1], rst.shape[2]);
        return rst;
    };
    using namespace std::placeholders;
    auto mkx = std::bind(mkshp, transa, _1, _2, _3);
    auto mky = std::bind(mkshp, transb, _1, _2, _3);

    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    Checker(make_graph, fwd)
            .run({mkx(3, 5, 7), mky(3, 7, 2)}, opt)
            .run({mkx(64, 1, 2), mky(64, 2, 1)}, opt)
            .run({mkx(1, 2, 3), mky(1, 3, 4)}, opt);
}

auto gen_fp16 = [](HostTensorND& dest) {
    RNGxorshf rng{next_rand_seed()};
    auto rand_real = [&rng]() {
        std::uniform_real_distribution<float> dist(-1, 1);
        return dist(rng);
    };
    auto ptr = dest.ptr<dt_float16>();
    size_t elems = dest.shape().total_nr_elems();
    for (size_t i = 0; i < elems; i++) {
        ptr[i] = dt_float16(rand_real());
    }
};

auto gen_int8 = [](HostTensorND& dest) {
    HostTensorGenerator<dtype::Int8, RandomDistribution::UNIFORM>
            int8_generator{-128, 127};
    dest = *int8_generator(dest.shape(), dest.comp_node());
};

void run_batched_hgemm_test(bool transa, bool transb) {
    using Checker = AutoOprChecker<2, 1>;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::BatchedMatrixMul::make(inputs[0], inputs[1],
                                            {transa, transb})};
    };
    auto fwd = FWD_BATCH_GEMM(dt_float16, dt_float16);
    auto mkshp = [](bool trans, size_t b, size_t m, size_t k) {
        TensorShape rst{b, m, k};
        if (trans)
            std::swap(rst.shape[1], rst.shape[2]);
        return rst;
    };

    using namespace std::placeholders;
    auto mkx = std::bind(mkshp, transa, _1, _2, _3);
    auto mky = std::bind(mkshp, transb, _1, _2, _3);

    Checker checker(make_graph, fwd);
    Checker::RunOptions opt;
    opt.outputs_max_err = 1e-2;

    checker.set_input_dtype(0, dtype::Float16())
            .set_input_dtype(1, dtype::Float16())
            .set_input_generator(0, gen_fp16)
            .set_input_generator(1, gen_fp16)
            .set_input_allow_grad(0, false)
            .set_input_allow_grad(1, false)
            .set_output_allow_grad(0, false);

    checker.run({mkx(3, 5, 7), mky(3, 7, 2)}, opt)
            .run({mkx(64, 1, 2), mky(64, 2, 1)}, opt)
            .run({mkx(1, 2, 3), mky(1, 3, 4)}, opt);
}

void run_batched_igemm_test(bool transa, bool transb) {
    using Checker = AutoOprChecker<2, 1>;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::BatchedMatrixMul::make(inputs[0], inputs[1],
                                            {transa, transb})};
    };

    auto fwd = FWD_BATCH_GEMM(int8_t, int32_t);

    auto mkshp = [](bool trans, size_t b, size_t m, size_t k) {
        TensorShape rst{b, m, k};
        if (trans)
            std::swap(rst.shape[1], rst.shape[2]);
        return rst;
    };

    using namespace std::placeholders;
    auto mkx = std::bind(mkshp, transa, _1, _2, _3);
    auto mky = std::bind(mkshp, transb, _1, _2, _3);

    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    Checker checker(make_graph, fwd);

    checker.set_input_dtype(0, dtype::Int8())
            .set_input_dtype(1, dtype::Int8())
            .set_input_generator(0, gen_int8)
            .set_input_generator(1, gen_int8)
            .set_input_allow_grad(0, false)
            .set_input_allow_grad(1, false)
            .set_output_allow_grad(0, false);

    checker.run({mkx(3, 5, 7), mky(3, 7, 2)}, opt)
            .run({mkx(64, 1, 2), mky(64, 2, 1)}, opt)
            .run({mkx(1, 2, 3), mky(1, 3, 4)}, opt);
}

template <typename ctype>
float getter(ctype val) {
    return val;
}

template <>
float getter<dt_qint32>(dt_qint32 val) {
    return (float)val.as_int32();
}

template <typename dt_src, typename dt_dst>
void run_trans_inp_test_case(bool trans_a, bool trans_b) {
    HostTensorGenerator<typename DTypeTrait<dt_src>::dtype> gen;
    std::shared_ptr<HostTensorND> host_x = gen({1, 1}), host_y = gen({1, 1});
    auto graph = ComputingGraph::make();
    auto do_trans = [](SymbolVar x) {
        return opr::Dimshuffle::make(x, {1, 0});
    };
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y);
    if (trans_a) {
        x = do_trans(x);
    }
    if (trans_b) {
        y = do_trans(y);
    }
    OperatorNodeConfig config;
    if (DTypeTrait<dt_dst>::enumv == DTypeEnum::Int16) {
        config.output_dtype(dtype::Int16());
    }
    auto z = opr::MatrixMul::make(x, y, {}, config);
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});

    auto run = [&](size_t M, size_t K, size_t N) {
        *host_x = *(trans_a ? gen({K, M}) : gen({M, K}));
        *host_y = *(trans_b ? gen({N, K}) : gen({K, N}));
        func->execute();
        ASSERT_EQ(TensorShape({M, N}), host_z.shape());
        ASSERT_EQ(!trans_a, x.node()->dev_tensor().layout().is_contiguous());
        ASSERT_EQ(!trans_b, y.node()->dev_tensor().layout().is_contiguous());

        auto px = host_x->ptr<dt_src>(), py = host_y->ptr<dt_src>();
        auto pz = host_z.ptr<dt_dst>();
        auto make_strd = [](bool trans, int h, int w, int* dst) {
            if (trans) {
                dst[0] = 1;
                dst[1] = h;
            } else {
                dst[0] = w;
                dst[1] = 1;
            }
        };
        int strd_x[2], strd_y[2];
        make_strd(trans_a, M, K, strd_x);
        make_strd(trans_b, K, N, strd_y);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                dt_dst sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    dt_dst xv = px[i * strd_x[0] + k * strd_x[1]],
                           yv = py[k * strd_y[0] + j * strd_y[1]];
                    sum += xv * yv;
                }
                MGB_ASSERT_FLOAT_EQ(getter(sum), getter(pz[i * N + j]))
                        << trans_a << ' ' << trans_b;
            }
        }
    };
    run(4, 8, 12);
    run(8, 12, 16);
}

template <typename dt_src, typename dt_dst>
void run_trans_inp_test() {
    for (bool ta : {false, true}) {
        for (bool tb : {false, true}) {
            run_trans_inp_test_case<dt_src, dt_dst>(ta, tb);
        }
    }
}

template <typename dt_src, typename dt_dst>
void inline mul_add(dt_src& a, dt_src& b, dt_dst& c) {
    c += dt_dst(a) * dt_dst(b);
}

template <>
void inline mul_add(dt_qint8& a, dt_qint8& b, dt_qint32& c) {
    c += dt_qint32(a.as_int8()) * dt_qint32(b.as_int8());
}

template <typename dt_gen>
std::shared_ptr<HostTensorND> bgemm_gen(const TensorShape& shp) {
    HostTensorGenerator<typename DTypeTrait<dt_gen>::dtype> gen;
    return gen(shp);
}

template <>
std::shared_ptr<HostTensorND> bgemm_gen<dt_float16>(const TensorShape& shp) {
    CompNode cn = CompNode::load("xpu0");
    std::shared_ptr<HostTensorND> ret =
            std::make_shared<HostTensorND>(cn, dtype::Float16{});
    (*ret).resize(shp);
    gen_fp16(*ret);
    return ret;
}

template <typename dt_src, typename dt_dst>
void run_bgemm_trans_inp_test_case(bool trans_a, bool trans_b) {
    std::shared_ptr<HostTensorND> host_x = bgemm_gen<dt_src>({1, 1, 1}),
                                  host_y = bgemm_gen<dt_src>({1, 1, 1});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y);

    trans_a ? (x = opr::Dimshuffle::make(x, {0, 2, 1})) : 0;
    trans_b ? (y = opr::Dimshuffle::make(y, {0, 2, 1})) : 0;

    auto z = opr::BatchedMatrixMul::make(x, y, {}, OperatorNodeConfig{});
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    auto run = [&](size_t B, size_t M, size_t K, size_t N) {
        *host_x = *(trans_a ? bgemm_gen<dt_src>({B, K, M})
                            : bgemm_gen<dt_src>({B, M, K}));
        *host_y = *(trans_b ? bgemm_gen<dt_src>({B, N, K})
                            : bgemm_gen<dt_src>({B, K, N}));
        func->execute();
        ASSERT_EQ(TensorShape({B, M, N}), host_z.shape());
        ASSERT_EQ(!trans_a, x.node()->dev_tensor().layout().is_contiguous());
        ASSERT_EQ(!trans_b, y.node()->dev_tensor().layout().is_contiguous());

        int strd_x[3], strd_y[3];
        auto px = host_x->ptr<dt_src>(), py = host_y->ptr<dt_src>();
        auto pz = host_z.ptr<dt_dst>();
        auto make_strd = [](bool trans, int h, int w, int* dst) {
            dst[0] = h * w;
            dst[1] = trans ? 1 : w;
            dst[2] = trans ? h : 1;
        };
        make_strd(trans_a, M, K, strd_x);
        make_strd(trans_b, K, N, strd_y);
        for (size_t b = 0; b < B; ++b)
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j) {
                    dt_dst sum = dt_dst(0);
                    for (size_t k = 0; k < K; ++k) {
                        dt_src xv = px[b * strd_x[0] + i * strd_x[1] +
                                       k * strd_x[2]],
                               yv = py[b * strd_y[0] + k * strd_y[1] +
                                       j * strd_y[2]];
                        mul_add(xv, yv, sum);
                    }
                    MGB_ASSERT_FLOAT_NEAR(getter(sum),
                                          getter(pz[(b * M + i) * N + j]), 5e-3)
                            << trans_a << ' ' << trans_b;
                }
    };
    run(2, 4, 8, 12);
    run(2, 8, 12, 16);
}

}  // anonymous namespace

TEST(TestOprBlas, MatrixMul_NN) {
    run_sgemm_test(false, false);
}

TEST(TestOprBlas, MatrixMul_NT) {
    run_sgemm_test(false, true);
}

TEST(TestOprBlas, MatrixMul_TN) {
    run_sgemm_test(true, false);
}

TEST(TestOprBlas, MatrixMul_TT) {
    run_sgemm_test(true, true);
}

TEST(TestOprBlas, BatchedMatrixMulFp32_NN) {
    run_batched_sgemm_test(false, false);
}

TEST(TestOprBlas, BatchedMatrixMulFp32_NT) {
    run_batched_sgemm_test(false, true);
}

TEST(TestOprBlas, BatchedMatrixMulFp32_TN) {
    run_batched_sgemm_test(true, false);
}

TEST(TestOprBlas, BatchedMatrixMulFp32_TT) {
    run_batched_sgemm_test(true, true);
}

TEST(TestOprBlas, BatchedMatrixMulFp16_NN) {
    run_batched_hgemm_test(false, false);
}

TEST(TestOprBlas, BatchedMatrixMulFp16_NT) {
    run_batched_hgemm_test(false, true);
}

TEST(TestOprBlas, BatchedMatrixMulFp16_TN) {
    run_batched_hgemm_test(true, false);
}

TEST(TestOprBlas, BatchedMatrixMulFp16_TT) {
    run_batched_hgemm_test(true, true);
}

TEST(TestOprBlas, BatchedMatrixMulInt8_NN) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_batched_igemm_test(false, false);
}

TEST(TestOprBlas, BatchedMatrixMulInt8_NT) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_batched_igemm_test(false, true);
}

TEST(TestOprBlas, BatchedMatrixMulInt8_TN) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_batched_igemm_test(true, false);
}

TEST(TestOprBlas, BatchedMatrixMulInt8_TT) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_batched_igemm_test(true, true);
}

TEST(TestOprBlas, TransBatchedMatrixMulFp32_NN) {
    run_bgemm_trans_inp_test_case<float, float>(false, false);
}

TEST(TestOprBlas, TransBatchedMatrixMulFp32_NT) {
    run_bgemm_trans_inp_test_case<float, float>(false, true);
}

TEST(TestOprBlas, TransBatchedMatrixMulFp32_TN) {
    run_bgemm_trans_inp_test_case<float, float>(true, false);
}

TEST(TestOprBlas, TransBatchedMatrixMulFp32_TT) {
    run_bgemm_trans_inp_test_case<float, float>(true, true);
}

TEST(TestOprBlas, TransBatchedMatrixMulInt8_NN) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_bgemm_trans_inp_test_case<int8_t, int32_t>(false, false);
}

TEST(TestOprBlas, TransBatchedMatrixMulInt8_NT) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_bgemm_trans_inp_test_case<int8_t, int32_t>(false, true);
}

TEST(TestOprBlas, TransBatchedMatrixMulInt8_TN) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_bgemm_trans_inp_test_case<int8_t, int32_t>(true, false);
}

TEST(TestOprBlas, TransBatchedMatrixMulInt8_TT) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_bgemm_trans_inp_test_case<int8_t, int32_t>(true, true);
}

TEST(TestOprBlas, TransBatchedMatrixMulFp16_NN) {
    run_bgemm_trans_inp_test_case<dt_float16, dt_float16>(false, false);
}

TEST(TestOprBlas, TransBatchedMatrixMulFp16_NT) {
    run_bgemm_trans_inp_test_case<dt_float16, dt_float16>(false, true);
}

TEST(TestOprBlas, TransBatchedMatrixMulFp16_TN) {
    run_bgemm_trans_inp_test_case<dt_float16, dt_float16>(true, false);
}

TEST(TestOprBlas, TransBatchedMatrixMulFp16_TT) {
    run_bgemm_trans_inp_test_case<dt_float16, dt_float16>(true, true);
}

TEST(TestOprBlas, TransBatchedMatrixMulQS8_NN) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_bgemm_trans_inp_test_case<dt_qint8, dt_qint32>(false, false);
}

TEST(TestOprBlas, TransBatchedMatrixMulQS8_NT) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_bgemm_trans_inp_test_case<dt_qint8, dt_qint32>(false, true);
}

TEST(TestOprBlas, TransBatchedMatrixMulQS8_TN) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_bgemm_trans_inp_test_case<dt_qint8, dt_qint32>(true, false);
}

TEST(TestOprBlas, TransBatchedMatrixMulQS8_TT) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_bgemm_trans_inp_test_case<dt_qint8, dt_qint32>(true, true);
}

TEST(TestOprBlas, DotBasic) {
    HostTensorGenerator<> gen;
    auto host_x = gen({123}), host_y = gen({123});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = opr::Dot::make(x, y);
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
    MGB_ASSERT_FLOAT_EQ(brute_force_dot(*host_x, *host_y),
                        *host_z.ptr<float>());
}

TEST(TestOprBlas, Dot) {
    using Checker = AutoOprChecker<2, 1>;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::Dot::make(inputs[0], inputs[1])};
    };

    auto fwd = [](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&i0 = *inp[0], &&i1 = *inp[1];
        auto&& out = dest[0].resize({1});
        *out.ptr<float>() = brute_force_dot(i0, i1);
    };

    Checker(make_graph, fwd)
            .run({TensorShape{15}, TensorShape{1}})
            .run({TensorShape{1}, TensorShape{16}})
            .run({TensorShape{23}, TensorShape{23}})
            .run({TensorShape{1000}, TensorShape{1000}});
}

TEST(TestOprBlas, TransMatMul) {
    run_trans_inp_test<float, float>();
}

TEST(TestOprBlas, TransMatMul8x8x16) {
    if (CompNode::load("xpux").device_type() != CompNode::DeviceType::CUDA) {
        run_trans_inp_test<dt_int8, dt_int16>();
    } else {
        printf("testcase skipped on unsupported arch\n");
    }
}

TEST(TestOprBlas, TransMatMul8x8x32) {
    if (CompNode::load("xpux").device_type() == CompNode::DeviceType::CUDA &&
        !check_compute_capability(6, 1)) {
        return;
    }
    run_trans_inp_test<dt_int8, dt_int32>();
}

TEST(TestOprBlas, NonContigMatmul) {
    using Checker = AutoOprChecker<2, 1>;
    auto make_graph =
            [](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {

        using Ad = opr::Subtensor::AxisIndexer;
        auto x = inputs[0],
             xsub = opr::Subtensor::make(
                     x, {Ad::make_interval(0, None, None, x.make_scalar(2))}),
             y = inputs[1],
             ysub = opr::Subtensor::make(
                     y, {Ad::make_interval(1, None, None, x.make_scalar(3))});
        return {opr::MatrixMul::make(xsub, ysub)};
    };
    auto fwd = [](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&shp0 = inp[0]->shape(), &&shp1 = inp[1]->shape();
        size_t m = (shp0.shape[0] + 1) / 2, k = shp0.shape[1],
               n = (shp1.shape[1] + 2) / 3;
        auto dptr = dest[0].resize({m, n}).ptr<float>();
        memset(dptr, 0, sizeof(float) * m * n);
        for (size_t i = 0; i < m; ++i) {
            auto ptr_a = inp[0]->ptr<float>({i * 2}),
                 ptr_c = dest[0].ptr<float>({i});
            for (size_t kk = 0; kk < k; ++kk) {
                auto va = ptr_a[kk];
                auto ptr_b = inp[1]->ptr<float>({kk});
                for (size_t j = 0; j < n; ++j) {
                    ptr_c[j] += va * ptr_b[j * 3];
                }
            }
        }
    };

    Checker(make_graph, fwd)
            .run({TensorShape{2, 1}, TensorShape{1, 3}})
            .run({TensorShape{5, 2}, TensorShape{2, 6}})
            .run({TensorShape{6, 3}, TensorShape{3, 8}});
}

TEST(TestOprBlas, MatrixInverse) {
    using Checker = AutoOprChecker<1, 1>;
    auto make_graph =
            [=](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::MatrixInverse::make(inputs[0])};
    };
    auto fwd = [=](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr =
                megdnn_naive_handle()->create_operator<megdnn::MatrixInverse>();

        auto wk_size =
                opr->get_workspace_in_bytes(inp[0]->layout(), inp[0]->layout());
        std::unique_ptr<dt_byte[]> wk{new dt_byte[wk_size]};
        opr->exec(inp[0]->as_megdnn(),
                  dest[0].resize(inp[0]->shape()).as_megdnn(),
                  {wk.get(), wk_size});
    };
    // ensure low condition number for generated matrices
    auto input_coord = [](const Checker::NumInpArray& inp) {
        auto shp = inp[0]->shape();
        size_t n = shp[shp.ndim - 1];
        size_t batch = 1;
        for (size_t i = 0; i < shp.ndim - 2; ++i) {
            batch *= shp[i];
        }
        std::vector<int> perm(n);
        for (size_t i = 0; i < n; ++i) {
            perm[i] = i;
        }
        auto ptr = inp[0]->ptr<float>();
        for (size_t i = 0; i < batch; ++i, ptr += n * n) {
#if __cplusplus >= 201703L
            std::default_random_engine rng_engine;
            std::shuffle(perm.begin(), perm.end(), rng_engine);
#else
            std::random_shuffle(perm.begin(), perm.end());
#endif
            for (size_t j = 0; j < n; ++j) {
                ptr[j * n + perm[j]] += 5;
            }
        }
    };

    Checker{make_graph, fwd}
            .set_input_coordinator(input_coord)
            .run({TensorShape{5, 5}})
            .run({TensorShape{2, 5, 5}})
            .run({TensorShape{2, 6, 3, 3}});
}

namespace {

void gen_svd_input(HostTensorND& dest) {
    auto ptr = dest.ptr<float>();
    auto dim = dest.layout().ndim;
    size_t n = dest.layout().shape[dim - 2], m = dest.layout().shape[dim - 1];
    size_t j = 0, k = 0;
    float batch_off = 0;
    float max_val = std::min(m, n) * std::min(m, n) + 0.99;
    for (size_t i = 0, it = dest.layout().total_nr_elems(); i < it; ++i) {
        if (i % (n * m) == 0) {
            batch_off += 0.32;
            j = k = 0;
        }
        if (!((i % (n * m)) % (m + 1)))
            ptr[i] = (j++) + ((++k / 10.0));
        else
            ptr[i] = (j++);
        ptr[i] += batch_off;
        ptr[i] = std::fmod(ptr[i], max_val);
    }
}

template <int have_u, int have_s, int have_v>
void run_svd_empty_grad_test() {
    using Checker = AutoOprChecker<1, have_u + have_s + have_v>;
    auto make_graph = [=](const typename Checker::SymInpArray& inputs) {
        auto out = opr::SVD::make(inputs[0], opr::SVD::Param{false, true});
        typename Checker::SymOutArray ret;
        int idx = 0;
        if (have_u) {
            ret[idx++] = out[0];
        }
        if (have_s) {
            ret[idx++] = out[1];
        }
        if (have_v) {
            ret[idx++] = out[2];
        }
        return ret;
    };
    auto fwd = [=](typename Checker::NumOutArray& dest,
                   typename Checker::NumInpArray inp) {
        auto opr = megdnn_naive_handle()->create_operator<megdnn::SVDForward>();
        opr->param().compute_uv = true;
        TensorLayout ul, sl, vtl;
        opr->deduce_layout(inp[0]->layout(), ul, sl, vtl);
        HostTensorND tmp_u{dest[0].comp_node(), ul},
                tmp_s{dest[0].comp_node(), sl}, tmp_v{dest[0].comp_node(), vtl};
        auto wk_size =
                opr->get_workspace_in_bytes(inp[0]->layout(), ul, sl, vtl);
        auto wk = std::make_unique<dt_byte[]>(wk_size);
        auto out0 = tmp_u.as_megdnn(), out1 = tmp_s.as_megdnn(),
             out2 = tmp_v.as_megdnn();
        int idx = 0;
        if (have_u) {
            out0 = dest[idx++].resize(ul).as_megdnn();
        }
        if (have_s) {
            out1 = dest[idx++].resize(sl).as_megdnn();
        }
        if (have_v) {
            out2 = dest[idx++].resize(vtl).as_megdnn();
        }
        opr->exec(inp[0]->as_megdnn(), out0, out1, out2, {wk.get(), wk_size});
    };
    Checker checker{make_graph, fwd};
    checker.set_input_generator(0, gen_svd_input);
    if (have_u) {
        checker.set_output_allow_check(0, false);
    }
    if (have_v) {
        checker.set_output_allow_check(have_u + have_s, false);
    }
    checker.run({TensorShape{3, 3}})
            .run({TensorShape{2, 3, 3}})
            .run({TensorShape{2, 4, 2}})
            .run({TensorShape{3, 1, 2, 4}})
            .run({TensorShape{2, 3, 2, 3}});
}

}  // anonymous namespace

TEST(TestOprBlas, SingularValueDecomposition) {
    using Checker = AutoOprChecker<1, 3>;
    auto make_graph =
            [=](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto out = opr::SVD::make(inputs[0], opr::SVD::Param{false, true});
        return {out[0], out[1], out[2]};
    };
    auto fwd = [=](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr = megdnn_naive_handle()->create_operator<megdnn::SVDForward>();
        opr->param().compute_uv = true;
        TensorLayout ul, sl, vtl;
        opr->deduce_layout(inp[0]->layout(), ul, sl, vtl);
        auto wk_size =
                opr->get_workspace_in_bytes(inp[0]->layout(), ul, sl, vtl);
        auto wk = std::make_unique<dt_byte[]>(wk_size);
        opr->exec(inp[0]->as_megdnn(), dest[0].resize(ul).as_megdnn(),
                  dest[1].resize(sl).as_megdnn(),
                  dest[2].resize(vtl).as_megdnn(), {wk.get(), wk_size});
    };
    Checker{make_graph, fwd}
            .set_input_generator(0, gen_svd_input)
            .set_output_allow_check(0, false)
            .set_output_allow_check(2, false)
            .run({TensorShape{3, 3}})
            .run({TensorShape{2, 3, 3}})
            .run({TensorShape{2, 4, 2}})
            .run({TensorShape{3, 1, 2, 4}})
            .run({TensorShape{2, 3, 2, 3}});
}

TEST(TestOprBlas, SingularValueDecompositionZeroGrad) {
    run_svd_empty_grad_test<0, 0, 1>();
    run_svd_empty_grad_test<0, 1, 0>();
    run_svd_empty_grad_test<0, 1, 1>();
    run_svd_empty_grad_test<1, 0, 0>();
    run_svd_empty_grad_test<1, 0, 1>();
    run_svd_empty_grad_test<1, 1, 0>();
    run_svd_empty_grad_test<1, 1, 1>();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
//
