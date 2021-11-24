/**
 * \file test/naive/record1.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "test/naive/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/task_record_check.h"

#include "test/common/adaptive_pooling.h"
#include "test/common/cond_take.h"
#include "test/common/convolution3d.h"
#include "test/common/local.h"
#include "test/common/matrix_mul.h"
#include "test/common/rng.h"
#include "test/common/separable_conv.h"
#include "test/common/warp_affine.h"
#include "test/common/warp_perspective.h"

namespace {
using namespace megdnn;
using namespace test;

class ArgmxxRNG final : public RNG {
public:
    void gen(const TensorND& tensor) override {
        auto offset = tensor.layout.span().low_elem;
        auto nr_elems = tensor.layout.span().dist_elem();

#define cb(DType)                                             \
    if (tensor.layout.dtype == DType()) {                     \
        using ctype = typename DTypeTrait<DType>::ctype;      \
        auto ptr = tensor.ptr<ctype>();                       \
        for (size_t i = 0; i < nr_elems; ++i) {               \
            ptr[offset + i] = i;                              \
        }                                                     \
        COMPAT_RANDOM(ptr + offset, ptr + offset + nr_elems); \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
    }
};

template <typename Argmxx>
void test_argmxx() {
    TaskRecordChecker<Argmxx> checker(2);
    checker.set_dtype(1, dtype::Int32());
    using Param = typename Argmxx::Param;
    ArgmxxRNG rng;
    checker.set_rng(0, &rng);
    for (size_t axis = 0; axis < 4; ++axis) {
        Param param;
        param.axis = axis;
        checker.set_param(param)
                .set_dtype(0, dtype::Float32())
                .execs({{2, 3, 4, 5}, {}});
        checker.set_param(param)
                .set_dtype(0, dtype::Float16())
                .execs({{2, 3, 4, 5}, {}});
        checker.set_param(param).set_dtype(0, dtype::Int32()).execs({{2, 3, 4, 5}, {}});
        checker.set_param(param).set_dtype(0, dtype::Int16()).execs({{2, 3, 4, 5}, {}});
        checker.set_param(param).set_dtype(0, dtype::Int8()).execs({{2, 3, 4, 5}, {}});
        checker.set_param(param).set_dtype(0, dtype::Uint8()).execs({{2, 3, 4, 5}, {}});
    }
    checker.set_dtype(0, dtype::Float32());
    Param param;
    param.axis = 1;
    checker.set_param(param);
    // 1-step
    checker.execs({{2, 64, 32}, {}});
    // 2-step
    checker.execs({{2, 192, 32}, {}});
    // 3-step
    checker.execs({{2, 4333, 32}, {}});
    // single reduce
    checker.execs({{2, 1, 1}, {}});
    checker.execs({{2, 1 + 1, 1}, {}});
    checker.execs({{2, 2048 + 1, 1}, {}});
    checker.execs({{2, 2048 * 2048 + 1, 1}, {}});
    checker.execs({{2, 1 + 1, 31}, {}});
    checker.execs({{2, 16 + 1, 31}, {}});
    checker.execs({{2, 16 * 16 + 1, 31}, {}});
    checker.execs({{2, 16 * 16 * 16 + 1, 31}, {}});
    checker.execs({{2, 16 * 16 * 16 * 16 + 1, 31}, {}});
    checker.execs({{3, 256 * 256 + 1, 2}, {}});
    checker.execs({{3, 128 * 128 + 1, 3}, {}});
    checker.execs({{3, 64 * 64 + 1, 7}, {}});
    checker.execs({{3, 32 * 32 + 1, 15}, {}});
    checker.execs({{3, 512, 500}, {}});
    // very large reduce
    checker.execs({{1, 4194304, 1}, {}});
}

class ArgsortRNG final : public RNG {
    bool m_rev_order = false;
    DType m_dtype;

    template <typename T>
    void fill(T* ptr, int n) {
        if (m_rev_order) {
            for (int i = 0; i < n; ++i)
                ptr[i] = static_cast<T>(n / 2 - i);
        } else {
            for (int i = 0; i < n; ++i)
                ptr[i] = static_cast<T>(i - n / 2);
            COMPAT_RANDOM(ptr, ptr + n);
        }
    }

    void gen(const TensorND& tensor) override {
        auto n = tensor.layout.total_nr_elems();
        if (m_dtype == dtype::Float32{}) {
            fill(tensor.ptr<dt_float32>(), n);
        } else {
            megdnn_assert(m_dtype == dtype::Int32{});
            fill(tensor.ptr<dt_int32>(), n);
        }
    }

public:
    ArgsortRNG(DType dt) : m_dtype{dt} {}

    void set_rev_order(bool flag) { m_rev_order = flag; }
};

void run_forward_test(DType dtype) {
    TaskRecordChecker<ArgsortForward> checker(2);
    using Param = Argsort::Param;
    using Order = Param::Order;
    ArgsortRNG rng{dtype};
    checker.set_dtype(2, dtype::Int32());
    checker.set_dtype(0, dtype).set_rng(0, &rng);
    for (size_t i = 3; i < 10240; i *= 2) {
        Param param;

        param.order = Order::ASCENDING;
        checker.set_param(param).execs({{3, i + 1}, {}, {}});
        param.order = Order::DESCENDING;
        checker.set_param(param).execs({{3, i - 1}, {}, {}});
        checker.set_param(param).execs({{13, i + 3}, {}, {}});
    }
    {
        // reverse sort large array
        constexpr size_t N = 200003;
        rng.set_rev_order(true);
        Param param;
        param.order = Order::ASCENDING;
        checker.set_param(param).execs({{1, N}, {}, {}});
    }
}

class IdxRng final : public RNG {
    void gen(const TensorND& tensor) override {
        auto ptr = tensor.ptr<dt_int32>();
        auto m = tensor.layout[0], n = tensor.layout[1];
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                ptr[j] = j;
            }
            COMPAT_RANDOM(ptr, ptr + n);
            ptr += n;
        }
    }
};

void run_backward_test(DType dtype) {
    IdxRng rng;
    TaskRecordChecker<ArgsortBackward> checker(2);
    checker.set_dtype(1, dtype::Int32()).set_rng(1, &rng);
    checker.set_dtype(0, dtype);
    checker.set_dtype(2, dtype);
    for (size_t i = 16; i < 4096; i *= 2) {
        checker.execs({{3, i}, {3, i}, {3, i}});
        checker.execs({{3, i + 3}, {3, i + 3}, {3, i + 3}});
        checker.execs({{3, i + 3}, {3, i + 3}, {3, i + 7}});
    }
}

}  // anonymous namespace

namespace megdnn {
namespace test {

//! adaptive pooling
TEST_F(NAIVE, ADAPTIVE_POOLING_FORWARD_RECORD) {
    TaskRecordChecker<AdaptivePooling> checker(2);
    auto args = adaptive_pooling::get_args();
    using Format = param::AdaptivePooling::Format;
    DType dtype = dtype::Float32();
    for (auto&& arg : args) {
        auto param = arg.param;
        auto src = arg.ishape;
        auto dst = arg.oshape;
        param.format = Format::NCHW;
        checker.set_epsilon(1e-2);
        checker.set_param(param).set_dtype(0, dtype).set_dtype(1, dtype).exec(
                TensorShapeArray{src, dst, {}});
        break;
    }
}

TEST_F(NAIVE, ADAPTIVE_POOLING_BACKWARD_RECORD) {
    TaskRecordChecker<AdaptivePooling> checker(2);
    auto args = adaptive_pooling::get_args();
    for (auto&& arg : args) {
        TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
        TensorLayout olayout = TensorLayout(arg.oshape, dtype::Float32());
        DType dtype = dtype::Float32();
        checker.set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype)
                .set_param(arg.param)
                .exec(TensorShapeArray{ilayout, olayout, olayout, ilayout});
        break;
    }
}

//! add update
TEST_F(NAIVE, ADD_UPDATE_RECORD) {
    TaskRecordChecker<AddUpdate> checker(2);
    param::AddUpdate p{2, -1, 3};
    checker.set_param(p)
            .set_dtype(0, dtype::BFloat16())
            .set_dtype(1, dtype::BFloat16())
            .execs({{2, 2, 3}, {2, 2, 3}});
}

//! argxx
TEST_F(NAIVE, ARGXX_RECORD) {
    test_argmxx<Argmax>();
    test_argmxx<Argmin>();
}

//! argsort
TEST_F(NAIVE, ARGSORT_FORWARD_RECORD) {
    run_forward_test(dtype::Float32{});
    run_forward_test(dtype::Int32{});
}

TEST_F(NAIVE, ARGSORT_BACKWARD_RECORD) {
    run_backward_test(dtype::Float32{});
    run_backward_test(dtype::Int32{});
}

TEST_F(NAIVE, BATCH_CONV_BIAS_QS8_RECORD) {
    TaskRecordChecker<BatchConvBiasForward> checker(2);
    UniformIntRNG const_rng{1, 1};
    UniformIntRNG rng{-5, 5};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.1f})
            .set_epsilon(1 + 1e-3);
    param::BatchConvBias param;
    param.pad_h = 2, param.pad_w = 1;
    param.stride_h = 1, param.stride_w = 2;
    param.format = param::BatchConvBias::Format::NCHW4;
    checker.set_param(param).execs(
            {{32, 4, 24, 24, 4}, {32, 32, 4, 1, 1, 4}, {1, 8, 1, 1, 4}, {}, {}});
}

//! batched_matmul
TEST_F(NAIVE, BATCH_MAT_MUL_RECORD) {
    TaskRecordChecker<BatchedMatrixMulForward> checker(2);
    using TestArg = matrix_mul::TestArg;
    //! return expect if stride == -1, stride otherwise
    auto stride_val = [](size_t stride, size_t expect) -> size_t {
        if (stride == TestArg::UNSET_STRIDE_VAL) {
            return expect;
        } else {
            return stride;
        }
    };

    using Param = MatrixMul::Param;
    std::vector<TestArg> args;
    args = matrix_mul::get_batched_matmul_args();

    for (auto& arg : args) {
        if (arg.b == 1) {
            continue;
        }
        size_t m = arg.m, n = arg.n, k = arg.k;

        Param param;
        param.transposeA = arg.mask & 0x1;
        param.transposeB = arg.mask & 0x2;
        size_t A0 = m, A1 = k, B0 = k, B1 = n;
        TensorShape A, B;
        if (param.transposeA) {
            std::swap(A0, A1);
        }
        if (param.transposeB) {
            std::swap(B0, B1);
        }
        ptrdiff_t A_stride = arg.A_stride, B_stride = arg.B_stride,
                  C_stride = arg.C_stride, A_batch_stride = arg.A_batch_stride,
                  B_batch_stride = arg.B_batch_stride,
                  C_batch_stride = arg.C_batch_stride;
        A_stride = stride_val(A_stride, A1);
        B_stride = stride_val(B_stride, B1);
        C_stride = stride_val(C_stride, n);
        A_batch_stride = stride_val(A_batch_stride, A0 * A_stride);
        B_batch_stride = stride_val(B_batch_stride, B0 * B_stride);
        C_batch_stride = stride_val(C_batch_stride, m * C_stride);

        checker.set_param(param);
        checker.execl(
                {TensorLayout{
                         {arg.b, A0, A1},
                         {A_batch_stride, A_stride, 1},
                         dtype::Float32()},
                 TensorLayout{
                         {arg.b, B0, B1},
                         {B_batch_stride, B_stride, 1},
                         dtype::Float32()},
                 TensorLayout{
                         {arg.b, m, n},
                         {C_batch_stride, C_stride, 1},
                         dtype::Float32()}});
        break;
    }
}

//! BN
TEST_F(NAIVE, BN_FORWARD_RECORD) {
    TaskRecordChecker<BNForward> checker(2);
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_epsilon(1e-3);

    param::BN param;
    param.fwd_mode = param::BN::FwdMode::TRAINING;
    param.param_dim = param::BN::ParamDim::DIM_1C11;
    param.epsilon = 1e-3;

    for (size_t n : {1, 2}) {
        for (size_t c : {1, 128}) {
            for (size_t i : {2, 14}) {
                for (float f : {0.5, 1.0}) {
                    param.avg_factor = f;
                    checker.set_param(param);
                    TensorShape src{n, c, i, i};
                    TensorShape inp{1, c, 1, 1};
                    checker.execs(
                            {src,  //! src                 -> input
                             inp,  //! bn_scale            -> input
                             inp,  //! bn_bias             -> input
                             inp,  //! mean                -> output
                             inp,  //! variance            -> output
                             inp,  //! batch_mean          -> output
                             inp,  //! batch_inv_variance  -> output
                             {},   //! reserve             -> output
                             {}});
                }
            }
        }
    }

    UniformFloatRNG rng(1.0f, 2.0f);
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Float32())
            .set_dtype(4, dtype::Float32())
            .set_rng(3, &rng)
            .set_rng(4, &rng)
            .set_epsilon(1e-3);

    param.fwd_mode = param::BN::FwdMode::INFERENCE;
    param.param_dim = param::BN::ParamDim::DIM_1C11;
    param.epsilon = 1e-3;
    checker.set_param(param);

    for (size_t n : {1, 2}) {
        for (size_t c : {1, 128}) {
            for (size_t i : {2, 14}) {
                TensorShape src{n, c, i, i};
                TensorShape inp{1, c, 1, 1};
                checker.exec({
                        src,  //! src                 -> input
                        inp,  //! bn_scale            -> input
                        inp,  //! bn_bias             -> input
                        inp,  //! mean                -> input
                        inp,  //! variance            -> input
                        {},   //! batch_mean          -> output[unused]
                        {},   //! batch_inv_variance  -> output[unused]
                        {},   //! reserve             -> output
                        {}    //! dst                 -> output[shape got by
                              //! deduced]
                });
            }
        }
    }
}

TEST_F(NAIVE, BN_BACKWARD_RECORD) {
    TaskRecordChecker<BNBackward> checker(2);
    UniformFloatRNG rng(1.0f, 2.0f);
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Float32())
            .set_dtype(4, dtype::Float32())
            .set_rng(3, &rng);

    param::BN param;
    param.fwd_mode = param::BN::FwdMode::TRAINING;
    param.epsilon = 0.0f;
    checker.set_param(param);

    for (size_t n : {1, 2}) {
        for (size_t c : {3, 128}) {
            for (size_t i : {2, 14}) {
                TensorShape src{n, c, i, i};
                TensorShape inp{1, c, 1, 1};
                checker.exec({
                        src,  //! x                 -> input
                        src,  //! dy                -> input
                        inp,  //! bn_mean           -> input
                        inp,  //! bn_ivar           -> input
                        inp,  //! bn_scale          -> input
                        {},   //! reserve           -> input
                        inp,  //! d_bn_scale        -> output
                        inp,  //! d_bn_bias         -> output
                        src   //! dx                -> output
                });
            }
        }
    }
}

//! concat
TEST_F(NAIVE, CONCAT_RECORD) {
    TaskRecordChecker<Concat> checker(2);
    using Param = Concat::Param;
    for (auto dtype : std::vector<DType>{dtype::Float32(), dtype::Float16()})
        for (size_t axis = 0; axis < 4; ++axis) {
            Param param;
            param.axis = axis;
            TensorShapeArray shapes(4, TensorShape({2, 3, 4, 5}));
            for (size_t i = 0; i < 4; ++i) {
                shapes[i].shape[axis] = i + 1;
            }
            shapes.emplace_back();
            for (size_t i = 0; i < shapes.size(); ++i)
                checker.set_dtype(i, dtype);
            checker.set_param(param).execs(shapes);
        }
}

//! ConvBias
TEST_F(NAIVE, CONV_BIAS_RECORD) {
    TaskRecordChecker<ConvBias> checker(2);
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW;
    checker.set_dtype(0, dtype::QuantizedS8(0.1f))
            .set_dtype(1, dtype::QuantizedS8(0.2f))
            .set_dtype(2, dtype::QuantizedS32(0.02f))
            .set_dtype(3, dtype::QuantizedS32(0.3f))
            .set_dtype(4, dtype::QuantizedS32(0.02f));
    checker.set_param(param).execs(
            {{1, 1, 4, 4}, {3, 1, 3, 3}, {1, 3, 1, 1}, {1, 3, 2, 2}, {}});
}

//! Convolution
TEST_F(NAIVE, CONV_RECORD) {
    TaskRecordChecker<Convolution> checker(2);
    Convolution::Param param;
    param.format = Convolution::Param::Format::NCHW;
    checker.set_param(param).execs({{1, 1, 4, 4}, {3, 1, 3, 3}, {}});
}

//! Conv3D
TEST_F(NAIVE, CONV3D_RECORD) {
    using TestArg = convolution3d::TestArg;
    std::vector<TestArg> args = convolution3d::get_args();
    TaskRecordChecker<Convolution3DForward> checker(2);
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] * arg.filter[3] *
                                  arg.filter[4]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

//! cumsum
TEST_F(NAIVE, CUMSUM_RECORD) {
    TaskRecordChecker<Cumsum> checker(2);
    struct TestArg {
        param::Cumsum param;
        TensorShape shape;
        TestArg(param::Cumsum param, TensorShape shape) : param(param), shape(shape) {}
    };
    std::vector<TestArg> args, args_int32;
    for (auto shape : TensorShapeArray{{1000}, {330, 33}, {10, 10, 10}, {5, 5, 5, 5}}) {
        for (size_t axis = 0; axis < shape.ndim; ++axis) {
            args.emplace_back(param::Cumsum(axis, true, true), shape);
            args.emplace_back(param::Cumsum(axis, true, false), shape);
            args.emplace_back(param::Cumsum(axis, false, true), shape);
            args.emplace_back(param::Cumsum(axis, false, false), shape);
        }
    }
    for (auto shape : TensorShapeArray{{1}, {10}, {100}, {1000}, {10000}}) {
        args.emplace_back(param::Cumsum(0, true, true), shape);
        args.emplace_back(param::Cumsum(0, true, false), shape);
        args.emplace_back(param::Cumsum(0, false, true), shape);
        args.emplace_back(param::Cumsum(0, false, false), shape);
    }
    for (auto shape : TensorShapeArray{{1}, {10}, {100}, {1000}, {10000}}) {
        args_int32.emplace_back(param::Cumsum(0, true, true), shape);
        args_int32.emplace_back(param::Cumsum(0, true, false), shape);
        args_int32.emplace_back(param::Cumsum(0, false, true), shape);
        args_int32.emplace_back(param::Cumsum(0, false, false), shape);
    }
    for (auto arg : args) {
        checker.set_param(arg.param);
        checker.set_epsilon(1e-2);
        checker.set_dtype(0, dtype::Float32()).execs({{arg.shape}, {}});
        checker.set_dtype(0, dtype::Int16()).execs({{arg.shape}, {}});
        checker.set_dtype(0, dtype::Int32()).execs({{arg.shape}, {}});
    }
    for (auto arg : args_int32) {
        checker.set_param(arg.param);
        checker.set_epsilon(1e-2);
        checker.set_dtype(0, dtype::Int32()).execs({{arg.shape}, {}});
    }
}

//! dct
TEST_F(NAIVE, DCT_RECORD) {
    TaskRecordChecker<DctChannelSelectForward> checker(2);
    DctChannelSelectForward::Param param;
    param.format = DctChannelSelectForward::Param::Format::NCHW4;
    checker.set_dtype(0, dtype::Uint8()).set_dtype(3, dtype::QuantizedS8(10.f));
    checker.set_param(param).execs({{1, 1, 16, 16}, {}, {}, {}});
}

//! deformable_conv
TEST_F(NAIVE, DEFORMABLE_CONV_FWD_RECORD) {
    TaskRecordChecker<DeformableConv> checker(2);
    DeformableConv::Param param;

    UniformIntRNG im_rng{0, 4};
    UniformIntRNG filter_rng{0, 4};
    UniformIntRNG offset_rng{-2, 2};
    UniformIntRNG mask_rng{0, 1};

    checker.set_rng(0, &im_rng)
            .set_rng(1, &filter_rng)
            .set_rng(2, &offset_rng)
            .set_rng(3, &mask_rng);

    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilate_h = 1;
    param.dilate_w = 1;
    param.format = DeformableConv::Param::Format::NCHW;
    param.sparse = DeformableConv::Param::Sparse::GROUP;

    checker.set_param(param).execs(
            {{1, 2, 5, 5},
             {2, 1, 1, 3, 3},
             {1, 2 * 2 * 3 * 3, 5, 5},
             {1, 2 * 3 * 3, 5, 5},
             {}});

    checker.set_param(param).execs(
            {{1, 2, 5, 5},
             {2, 1, 1, 3, 3},
             {1, 2 * 2 * 3 * 3, 5, 5},
             {1, 2 * 3 * 3, 5, 5},
             {}});

    param.sparse = DeformableConv::Param::Sparse::DENSE;
    checker.set_param(param).execs(
            {{1, 2, 5, 5},
             {2, 2, 3, 3},
             {1, 2 * 2 * 3 * 3, 5, 5},
             {1, 2 * 3 * 3, 5, 5},
             {}});
}

TEST_F(NAIVE, DEFORMABLE_CONV_BWD_FILTER_RECORD) {
    TaskRecordChecker<DeformableConvBackwardFilter> checker(2);
    DeformableConv::Param param;

    UniformIntRNG im_rng{0, 4};
    UniformIntRNG offset_rng{-2, 2};
    UniformIntRNG mask_rng{0, 1};
    UniformIntRNG out_grad_rng{0, 1};

    checker.set_rng(0, &im_rng)
            .set_rng(1, &offset_rng)
            .set_rng(2, &mask_rng)
            .set_rng(3, &out_grad_rng);
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilate_h = 1;
    param.dilate_w = 1;
    param.format = DeformableConv::Param::Format::NCHW;
    param.sparse = DeformableConv::Param::Sparse::GROUP;

    checker.set_param(param).execs(
            {{1, 2, 5, 5},
             {1, 2 * 2 * 3 * 3, 5, 5},
             {1, 2 * 3 * 3, 5, 5},
             {1, 2, 5, 5},
             {2, 1, 1, 3, 3}});
}

TEST_F(NAIVE, DEFORMABLE_CONV_BWD_DATA_RECORD) {
    TaskRecordChecker<DeformableConvBackwardData> checker(2);
    DeformableConv::Param param;

    ConstValue im_rng{1};
    ConstValue filter_rng{0.99};
    ConstValue offset_rng{1.1};
    ConstValue mask_rng{1};
    ConstValue out_grad_rng{1};

    checker.set_rng(0, &im_rng)
            .set_rng(1, &filter_rng)
            .set_rng(2, &offset_rng)
            .set_rng(3, &mask_rng)
            .set_rng(4, &out_grad_rng);

    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilate_h = 1;
    param.dilate_w = 1;
    param.format = DeformableConv::Param::Format::NCHW;
    param.sparse = DeformableConv::Param::Sparse::GROUP;

    checker.set_param(param).execs(
            {{1, 2, 5, 5},
             {2, 1, 1, 3, 3},
             {1, 1 * 2 * 3 * 3, 5, 5},
             {1, 1 * 3 * 3, 5, 5},
             {1, 2, 5, 5},
             {1, 2, 5, 5},
             {1, 1 * 2 * 3 * 3, 5, 5},
             {1, 1 * 3 * 3, 5, 5}});
}

//! elemwise
TEST_F(NAIVE, ELEMWISE_COMMON_RECORD) {
    TaskRecordChecker<ElemwiseForward> checker(2);
    using Mode = ElemwiseForward::Param::Mode;
    auto run_activate = [&](size_t N, size_t C, size_t H, size_t W, Mode mode,
                            DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype);
        checker.execs({{N, C, H, W}, {}});
    };
    auto run_binary = [&](size_t N, size_t C, size_t H, size_t W, Mode mode,
                          DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(
                2, dtype);
        checker.execs({{N, C, H, W}, {N, C, H, W}, {}});
    };
    auto run_unary = [&](size_t N, size_t C, size_t H, size_t W, Mode mode,
                         DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype);
        checker.execs({{N, C, H, W}, {}});
    };

#define RUN_ACTIVATE(_dt)                         \
    run_activate(4, 32, 10, 10, Mode::RELU, _dt); \
    run_activate(4, 32, 10, 10, Mode::SIGMOID, _dt);
    RUN_ACTIVATE(dtype::Float32());
    RUN_ACTIVATE(dtype::Float16());
    checker.set_epsilon(1e-2);
    RUN_ACTIVATE(dtype::BFloat16());
#undef RUN_ACTIVATE

    checker.set_epsilon(1e-3);

#define RUN_BINARY(_dt)                        \
    run_binary(4, 32, 10, 10, Mode::ADD, _dt); \
    run_binary(4, 32, 10, 10, Mode::SUB, _dt); \
    run_binary(4, 32, 10, 10, Mode::MUL, _dt); \
    run_binary(4, 32, 10, 10, Mode::MIN, _dt); \
    run_binary(4, 32, 10, 10, Mode::MAX, _dt);
    RUN_BINARY(dtype::Float32());
    RUN_BINARY(dtype::Float16());
    RUN_BINARY(dtype::BFloat16());
    RUN_BINARY(dtype::Int32());
    RUN_BINARY(dtype::Int16());

    //! true_div
    run_binary(4, 32, 10, 10, Mode::TRUE_DIV, dtype::Float32());
    RUN_BINARY(dtype::Float16());
    checker.set_epsilon(1e-2);
    run_binary(4, 32, 10, 10, Mode::TRUE_DIV, dtype::Float16());
    RUN_BINARY(dtype::BFloat16());
    //! FIXME: precision is especially low
    checker.set_epsilon(1e-1);
    run_binary(4, 32, 10, 10, Mode::TRUE_DIV, dtype::BFloat16());
#undef RUN_BINARY

#define RUN_UNARY(_dt)                         \
    run_unary(4, 32, 10, 10, Mode::ABS, _dt);  \
    run_unary(4, 32, 10, 10, Mode::SIN, _dt);  \
    run_unary(4, 32, 10, 10, Mode::COS, _dt);  \
    run_unary(4, 32, 10, 10, Mode::EXP, _dt);  \
    run_unary(4, 32, 10, 10, Mode::CEIL, _dt); \
    run_unary(4, 32, 10, 10, Mode::TANH, _dt);
    RUN_UNARY(dtype::Float32());
    RUN_UNARY(dtype::BFloat16());
    checker.set_epsilon(1e-2);
    RUN_UNARY(dtype::Float16());

    //! FLOOR
    run_unary(4, 32, 10, 10, Mode::FLOOR, dtype::Float32());
    run_unary(4, 32, 10, 10, Mode::FLOOR, dtype::Float16());

    //! INT TEST
    run_unary(4, 32, 10, 10, Mode::ABS, dtype::Int16());
    run_unary(4, 32, 10, 10, Mode::ABS, dtype::Int32());
#undef RUN_UNARY

    //! naive impl
    run_binary(4, 32, 10, 10, Mode::LT, dtype::Float32());
    run_binary(4, 32, 10, 10, Mode::LT, dtype::Int32());

    run_binary(4, 32, 10, 10, Mode::LEQ, dtype::Float32());
    run_binary(4, 32, 10, 10, Mode::LEQ, dtype::Int32());

    run_binary(4, 32, 10, 10, Mode::EQ, dtype::Float32());
    run_binary(4, 32, 10, 10, Mode::EQ, dtype::Int32());

    auto rng = UniformFloatRNG(0.01, 2.0);
    checker.set_rng(0, &rng);

    run_unary(4, 32, 10, 10, Mode::LOG, dtype::Float32());
    run_unary(4, 32, 10, 10, Mode::LOG, dtype::BFloat16());
    checker.set_epsilon(1e-2);
    run_unary(4, 32, 10, 10, Mode::LOG, dtype::Float16());

    run_unary(4, 32, 10, 10, Mode::NEGATE, dtype::Float32());
    run_unary(4, 32, 10, 10, Mode::NEGATE, dtype::BFloat16());
    run_unary(4, 32, 10, 10, Mode::NEGATE, dtype::Float16());

    auto rng_int = UniformIntNonZeroRNG(1, 65535);
    checker.set_rng(0, &rng_int);
    run_unary(4, 32, 10, 10, Mode::NEGATE, dtype::Int32());
    run_unary(4, 32, 10, 10, Mode::NEGATE, dtype::Int16());
}

TEST_F(NAIVE, ELEMWISE_BROADCAST_RECORD) {
    TaskRecordChecker<ElemwiseForward> checker(2);
    using Mode = ElemwiseForward::Param::Mode;
    //! do broadcast test
    auto run_binary_broadcast = [&](size_t N, size_t C, size_t H, size_t W, Mode mode,
                                    DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype);
        checker.execs({{N, C, H, W}, {N, C, 1, 1}, {}});
        checker.execs({{N, C, 1, 1}, {N, C, H, W}, {}});
        checker.execs({{N, C, H, W}, {1}, {}});
        checker.execs({{1}, {N, C, H, W}, {}});
        checker.execs({{N, C, H, W}, {1, C, H, W}, {}});
        checker.execs({{1, C, H, W}, {N, C, H, W}, {}});
    };
#define RUN_BINARY(_dt)                                  \
    run_binary_broadcast(4, 32, 10, 10, Mode::ADD, _dt); \
    run_binary_broadcast(4, 32, 10, 10, Mode::SUB, _dt); \
    run_binary_broadcast(4, 32, 10, 10, Mode::MUL, _dt); \
    run_binary_broadcast(4, 32, 10, 10, Mode::MIN, _dt); \
    run_binary_broadcast(4, 32, 10, 10, Mode::MAX, _dt);
    RUN_BINARY(dtype::Float32());
    run_binary_broadcast(4, 32, 10, 10, Mode::TRUE_DIV, dtype::Float32());
    RUN_BINARY(dtype::Float16());
    checker.set_epsilon(1e-2);
    run_binary_broadcast(4, 32, 10, 10, Mode::TRUE_DIV, dtype::Float16());
    RUN_BINARY(dtype::BFloat16());
    //! FIXME: precision is especially low
    checker.set_epsilon(1e-1);
    run_binary_broadcast(4, 32, 10, 10, Mode::TRUE_DIV, dtype::BFloat16());
    RUN_BINARY(dtype::Int16());
    RUN_BINARY(dtype::Int32());
#undef RUN_BINARY
}

TEST_F(NAIVE, ELEMWISE_FUSE_MUL_ADD3_RECORD) {
    TaskRecordChecker<ElemwiseForward> checker(2);
    using Mode = ElemwiseForward::Param::Mode;
    auto run_mul_add = [&](size_t N, size_t C, size_t H, size_t W, DType dtype) {
        checker.set_param(Mode::FUSE_MUL_ADD3)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype);
        checker.execs({{1}, {N, C, H, W}, {1}, {}});
        checker.execs({{N, C, 1, 1}, {N, C, H, W}, {1}, {}});
        checker.execs({{N, C, H, W}, {N, C, H, W}, {1}, {}});
        checker.execs({{N, C, 1, 1}, {N, C, H, W}, {N, C, 1, 1}, {}});
    };
    run_mul_add(4, 32, 10, 10, dtype::Float32());
    checker.set_epsilon(1e-2);
    run_mul_add(4, 32, 10, 10, dtype::Float16());
    //! FIXME: precision is especially low
    checker.set_epsilon(1e-1);
    run_mul_add(4, 32, 10, 10, dtype::BFloat16());
    run_mul_add(4, 32, 10, 10, dtype::Int16());
    run_mul_add(4, 32, 10, 10, dtype::Int32());
}

TEST_F(NAIVE, ELEMWISE_FUSE_MUL_ADD4_RECORD) {
    TaskRecordChecker<ElemwiseForward> checker(2);
    using Mode = ElemwiseForward::Param::Mode;
    auto run_mul_add = [&](size_t N, size_t C, size_t H, size_t W, DType dtype) {
        checker.set_param(Mode::FUSE_MUL_ADD4)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype)
                .set_dtype(4, dtype);
        checker.execs({{1}, {N, C, H, W}, {1}, {N, C, H, W}, {}});
        checker.execs({{1}, {N, C, H, W}, {N, C, H, W}, {1}, {}});
        checker.execs({{N, C, 1, 1}, {N, C, H, W}, {N, C, 1, 1}, {N, C, H, W}, {}});
        checker.execs({{N, C, H, W}, {N, C, H, W}, {N, C, H, W}, {N, C, H, W}, {}});
    };
    run_mul_add(4, 32, 10, 10, dtype::Float32());
    checker.set_epsilon(1e-2);
    run_mul_add(4, 32, 10, 10, dtype::Float16());
    //! FIXME: precision is especially low
    checker.set_epsilon(1e-1);
    run_mul_add(4, 32, 10, 10, dtype::BFloat16());
    run_mul_add(4, 32, 10, 10, dtype::Int16());
    run_mul_add(4, 32, 10, 10, dtype::Int32());
}

TEST_F(NAIVE, ELEMWISE_FUSE_ADD_RELU_RECORD) {
    TaskRecordChecker<ElemwiseForward> checker(2);
    using Mode = ElemwiseForward::Param::Mode;
    auto run_mul_add = [&](size_t N, size_t C, size_t H, size_t W, DType dtype) {
        checker.set_param(Mode::FUSE_ADD_RELU)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype);
        checker.execs({{N, C, H, W}, {N, C, H, W}, {}});
    };
    run_mul_add(4, 32, 10, 10, dtype::Float32());
    checker.set_epsilon(1e-2);
    run_mul_add(4, 32, 10, 10, dtype::Float16());
    //! FIXME: precision is especially low
    checker.set_epsilon(1e-1);
    run_mul_add(4, 32, 10, 10, dtype::BFloat16());
}

TEST_F(NAIVE, ELEMWISE_FUSE_ADD_SIGMOID_RECORD) {
    TaskRecordChecker<ElemwiseForward> checker(2);
    using Mode = ElemwiseForward::Param::Mode;
    auto run_mul_add = [&](size_t N, size_t C, size_t H, size_t W, DType dtype) {
        checker.set_param(Mode::FUSE_ADD_SIGMOID)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype);
        checker.execs({{N, C, H, W}, {N, C, H, W}, {}});
    };
    run_mul_add(4, 32, 10, 10, dtype::Float32());
    checker.set_epsilon(1e-2);
    run_mul_add(4, 32, 10, 10, dtype::Float16());
    //! FIXME: precision is especially low
    checker.set_epsilon(1e-1);
    run_mul_add(4, 32, 10, 10, dtype::BFloat16());
}

TEST_F(NAIVE, ELEMWISE_FUSE_ADD_TANH_RECORD) {
    TaskRecordChecker<ElemwiseForward> checker(2);
    using Mode = ElemwiseForward::Param::Mode;
    auto run_mul_add = [&](size_t N, size_t C, size_t H, size_t W, DType dtype) {
        checker.set_param(Mode::FUSE_ADD_TANH)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype);
        checker.execs({{N, C, H, W}, {N, C, H, W}, {}});
    };
    run_mul_add(4, 32, 10, 10, dtype::Float32());
    checker.set_epsilon(1e-2);
    run_mul_add(4, 32, 10, 10, dtype::Float16());
    //! FIXME: precision is especially low
    checker.set_epsilon(1e-1);
    run_mul_add(4, 32, 10, 10, dtype::BFloat16());
}

TEST_F(NAIVE, ELEMWISE_VECTOR_RECORD) {
    TaskRecordChecker<ElemwiseForward> checker(2);
    using Mode = ElemwiseForward::Param::Mode;
    auto run_vector = [&](size_t N, DType dtype, Mode mode) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(
                2, dtype);
        checker.execs({{N}, {1, N}, {}});
        checker.execs({{1, N}, {N}, {}});
        checker.execs({{N}, {1}, {}});
        checker.execs({{1}, {N}, {}});
        checker.execs({{1}, {1, 1}, {}});
        checker.execs({{1, 1, 1}, {1}, {}});
    };
    run_vector(1000, dtype::Float32(), Mode::ADD);
    run_vector(1000, dtype::Float32(), Mode::MUL);
    checker.set_epsilon(1e-2);
    run_vector(1000, dtype::Float16(), Mode::ADD);
    run_vector(1000, dtype::Float16(), Mode::MUL);
    //! FIXME: precision is especially low
    checker.set_epsilon(1e-1);
    run_vector(1000, dtype::BFloat16(), Mode::ADD);
    run_vector(1000, dtype::BFloat16(), Mode::MUL);
}

//! EYE
TEST_F(NAIVE, EYE_RECORD) {
    TaskRecordChecker<Eye> checker(2);
    for (DType dtype :
         std::vector<DType>{dtype::Float16(), dtype::Int32(), dtype::Float32()})
        for (int k = -20; k < 20; ++k) {
            checker.set_param({k, dtype.enumv()});
            checker.set_dtype(0, dtype);
            checker.execs(TensorShapeArray{{3, 4}});
            checker.execs(TensorShapeArray{{4, 3}});
        }
}

//! FILL
TEST_F(NAIVE, FILL_RECORD) {
    TaskRecordChecker<Fill> checker(2);
    for (DType dtype :
         std::vector<DType>{dtype::Float16(), dtype::Int32(), dtype::Float32()})
        for (float value : {-1.23, 0.0, 0.001, 234.0, 2021.072}) {
            checker.set_param({value});
            checker.set_dtype(0, dtype);
            checker.exec(TensorShapeArray{{1, 1}});
            checker.exec(TensorShapeArray{{2, 3, 4}});
        }
}

//! LINSPACE
TEST_F(NAIVE, LINSPACE_RECORD) {
    TaskRecordChecker<Linspace> checker(2);
    Linspace::Param param;
    param.start = 0.5;
    param.stop = 1.5;
    param.endpoint = true;
    for (DType dtype :
         std::vector<DType>{dtype::Float16(), dtype::Int32(), dtype::Float32()}) {
        checker.set_dtype(0, dtype).set_param(param).exec(TensorShapeArray{{11}});
    }
    param.endpoint = false;
    for (DType dtype :
         std::vector<DType>{dtype::Float16(), dtype::Int32(), dtype::Float32()}) {
        checker.set_dtype(0, dtype).set_param(param).exec(TensorShapeArray{{11}});
    }
}

//! LOCAL
TEST_F(NAIVE, LOCAL_FORWARD_RECORD) {
    auto args = local::get_args_for_cuda();
    for (size_t i = 0; i < 2; ++i) {
        auto&& arg = args[i];
        TaskRecordChecker<LocalForward> checker(2);
        checker.set_param(arg.param).exec(
                TensorShapeArray{arg.sshape(), arg.fshape(), arg.dshape()});
    }
}

TEST_F(NAIVE, LOCAL_BACKWARD_DATA_RECORD) {
    using namespace local;
    auto args = local::get_args_bwd_data_for_cuda();
    for (size_t i = 0; i < 2; ++i) {
        auto&& arg = args[i];
        TaskRecordChecker<LocalBackwardData> checker(2);
        checker.set_param(arg.param).exec(
                TensorShapeArray{arg.fshape(), arg.dshape(), arg.sshape()});
    }
}

TEST_F(NAIVE, LOCAL_BACKWARD_FILTER_RECORD) {
    using namespace local;
    auto args = local::get_args_bwd_filter_for_cuda();
    for (size_t i = 0; i < 2; ++i) {
        auto&& arg = args[i];
        TaskRecordChecker<LocalBackwardFilter> checker(2);
        checker.set_param(arg.param).exec(
                TensorShapeArray{arg.sshape(), arg.dshape(), arg.fshape()});
    }
}

//! matrix inverse
TEST_F(NAIVE, MATRIX_INVERSE_RECORD) {
    TaskRecordChecker<MatrixInverse> checker(2);
    checker.exec({{10, 20, 20}, {}});
}

//! matmul
TEST_F(NAIVE, MATRIX_MUL_RECORD) {
    TaskRecordChecker<MatrixMul> checker(2);
    MatrixMul::Param param;
    param.transposeA = false;
    param.transposeB = false;

    checker.set_dtype(0, dtype::Quantized8Asymm(0.1f, (uint8_t)128))
            .set_dtype(1, dtype::Quantized8Asymm(0.2f, (uint8_t)233))
            .set_dtype(2, dtype::QuantizedS32(0.1f * 0.2f));
    checker.set_param(param).exec({{4, 7}, {7, 5}, {}});

    param.transposeA = true;
    checker.set_dtype(0, dtype::Quantized8Asymm(0.7f, (uint8_t)128))
            .set_dtype(1, dtype::Quantized8Asymm(0.4f, (uint8_t)128))
            .set_dtype(2, dtype::QuantizedS32(0.7f * 0.4f));
    checker.set_param(param).exec({{2, 1}, {2, 1}, {}});
}

//! pooling
TEST_F(NAIVE, POOLING_QUANTIZED_RECORD) {
    using Mode = Pooling::Param::Mode;

    TaskRecordChecker<Pooling> checker(2);
    Pooling::Param param{Mode::MAX, 1, 1, 2, 2, 2, 2};
    auto dt = dtype::Quantized8Asymm(0.1f, (uint8_t)128);
    checker.set_dtype(0, dt).set_dtype(1, dt);
    checker.set_param(param).exec({{1, 1, 3, 3}, {}});

    param = {Mode::AVERAGE, 1, 1, 2, 2, 2, 2};
    checker.set_param(param).exec({{1, 1, 3, 3}, {}});

    param = {Mode::AVERAGE_COUNT_EXCLUDE_PADDING, 1, 1, 2, 2, 2, 2};
    checker.set_param(param).exec({{1, 1, 3, 3}, {}});

    auto dt32 = dtype::QuantizedS32(0.233f);
    checker.set_dtype(0, dt32).set_dtype(1, dt32);
    param = {Mode::MAX, 1, 1, 2, 2, 2, 2};
    checker.set_param(param).exec({{1, 1, 3, 3}, {}});
}

TEST_F(NAIVE, REDUCE_QUANTIZED_RECORD) {
    using Mode = Reduce::Param::Mode;

    TaskRecordChecker<Reduce> checker(2);

    Reduce::Param param;
    param.mode = Mode::SUM;
    param.data_type = param::Reduce::DataType::QUINT_I8xO32;
    param.axis = 0;

    checker.set_dtype(0, dtype::Quantized8Asymm(0.1f, (uint8_t)128))
            .set_dtype(1, dtype::QuantizedS32(0.1f));
    checker.set_param(param).exec({{3, 4}, {}});

    param.data_type = param::Reduce::DataType::DEFAULT;
    param.mode = Mode::MEAN;
    checker.set_dtype(0, dtype::Quantized8Asymm(1.f, (uint8_t)128))
            .set_dtype(1, dtype::Quantized8Asymm(1.f, (uint8_t)128));
    checker.set_param(param).exec({{3, 4}, {}});

    checker.set_dtype(0, dtype::Quantized8Asymm(0.00233f, (uint8_t)128))
            .set_dtype(1, dtype::Quantized8Asymm(0.00233f, (uint8_t)128));
    checker.set_param(param).exec({{3, 4}, {}});

    checker.set_dtype(0, dtype::Quantized8Asymm(7e-10f, (uint8_t)45))
            .set_dtype(1, dtype::Quantized8Asymm(7e-10f, (uint8_t)45));
    checker.set_param(param).exec({{3, 4}, {}});
}

//! relayout format
TEST_F(NAIVE, RELAYOUT_FORMAT_NCHW4_NCHW_RECORD) {
    TaskRecordChecker<RelayoutFormat> checker(2);
    RelayoutFormat::Param param{RelayoutFormat::Param::Mode::NCHW4_NCHW};
    checker.set_param(param).exec({{1, 2, 1, 2, 4}, {}});

    param.oc = 7;
    checker.set_param(param).exec({{1, 2, 1, 2, 4}, {}});

    param.oc = 6;
    param.group = 2;
    checker.set_param(param).exec({{1, 2, 1, 2, 4}, {}});
}

TEST_F(NAIVE, RELAYOUT_FORMAT_NCHW_NCHW4_WEIGHT_RECORD) {
    TaskRecordChecker<RelayoutFormat> checker(2);
    RelayoutFormat::Param param{RelayoutFormat::Param::Mode::NCHW_NCHW4_WEIGHT};
    checker.set_param(param);
    checker.exec({{2, 2, 2, 2}, {}});
    checker.exec({{2, 2, 1, 2, 2}, {}});
}

TEST_F(NAIVE, RELAYOUT_FORMAT_NCHW_NCHW4_RECORD) {
    TaskRecordChecker<RelayoutFormat> checker(2);
    RelayoutFormat::Param param{RelayoutFormat::Param::Mode::NCHW_NCHW4};
    checker.set_param(param).exec({{1, 8, 1, 2}, {}});

    param.group = 4;
    checker.set_param(param).exec({{1, 8, 1, 2}, {}});

    param.group = 2;
    checker.set_param(param).exec({{1, 6, 1, 2}, {}});
}

TEST_F(NAIVE, RELAYOUT_FORMAT_NCHW88_RECORD) {
    TaskRecordChecker<RelayoutFormat> checker(2);
    {
        RelayoutFormat::Param param{RelayoutFormat::Param::Mode::NCHW_NCHW88};
        checker.set_param(param);
        checker.exec({{1, 8, 1, 2}, {}});
        checker.exec({{2, 8, 1, 2}, {}});
        checker.exec({{2, 4, 1, 2}, {}});
        checker.exec({{1, 3, 64, 64}, {}});
    }
    {
        RelayoutFormat::Param param{RelayoutFormat::Param::Mode::NCHW88_NCHW};
        checker.set_param(param).exec({{1, 1, 1, 2, 8}, {}});
    }
}

TEST_F(NAIVE, RELAYOUT_FORMAT_NCHW88_DENSE_RECORD) {
    TaskRecordChecker<RelayoutFormat> checker(2);
    RelayoutFormat::Param param{
            RelayoutFormat::Param::Mode::NCHW_NCHW88_CONV_DENSE_WEIGHT};
    checker.set_param(param);
    checker.exec({{8, 8, 1, 1}, {}});
    checker.exec({{8, 2, 1, 1}, {}});
}

TEST_F(NAIVE, RELAYOUT_FORMAT_NCHW88_CHAIN_RECORD) {
    TaskRecordChecker<RelayoutFormat> checker(2);
    RelayoutFormat::Param param{
            RelayoutFormat::Param::Mode::NCHW_NCHW88_CONV_CHAN_WEIGHT};
    checker.set_param(param);
    checker.exec({{8, 1, 1, 1, 2}, {}});
    checker.exec({{2, 1, 1, 1, 2}, {}});
}

TEST_F(NAIVE, RELAYOUT_FORMAT_NCHW88_GROUP_RECORD) {
    TaskRecordChecker<RelayoutFormat> checker(2);
    {
        RelayoutFormat::Param param{
                RelayoutFormat::Param::Mode::NCHW_NCHW88_CONV_GROUP_WEIGHT};
        checker.set_param(param);
        checker.exec({{1, 8, 8, 1, 1}, {}});
        checker.exec({{1, 8, 2, 1, 1}, {}});
    }
    {
        RelayoutFormat::Param param{RelayoutFormat::Param::Mode::NCHW88_NCHW};
        checker.set_param(param).exec({TensorShape{1, 8, 64, 64, 8}, {}});
    }
}

//! separable conv
TEST_F(NAIVE, SEPARABLE_CONV_RECORD) {
    using TestArg = megdnn::test::separable_conv::TestArg;
    std::vector<TestArg> args = separable_conv::get_args();
    TaskRecordChecker<SeparableConvForward> checker(2);
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter_x, arg.filter_y, {}});
    }
}

//！ warp affine
TEST_F(NAIVE, WARP_AFFINE_RECORD) {
    TaskRecordChecker<WarpAffine> checker(2);
    WarpAffine::Param param;
    param.border_mode = WarpAffine::Param::BorderMode::BORDER_REFLECT;
    param.imode = WarpAffine::Param::InterpolationMode::LINEAR;
    param.format = WarpAffine::Param::Format::NCHW;

    checker.set_dtype(0, dtype::Uint8{})
            .set_dtype(1, dtype::Float32{})
            .set_dtype(2, dtype::Uint8{});
    checker.set_param(param).exec({{1, 1, 3, 3}, {1, 2, 3}, {1, 1, 2, 2}});

    checker.set_dtype(0, dtype::Quantized8Asymm{1.4f, static_cast<uint8_t>(127)})
            .set_dtype(1, dtype::Float32{})
            .set_dtype(2, dtype::Quantized8Asymm{1.4f, static_cast<uint8_t>(127)});
    checker.set_param(param).exec({{1, 1, 3, 3}, {1, 2, 3}, {1, 1, 2, 2}});
}

TEST_F(NAIVE, WARP_AFFINE_CV_RECORD) {
    using TestArg = warp_affine::TestArg;
    std::vector<TestArg> args = warp_affine::get_cv_args();
    TaskRecordChecker<WarpAffine> checker(2);

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Uint8())
                .execs({arg.src, arg.trans, arg.dst});
    }

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .execs({arg.src, arg.trans, arg.dst});
    }
}

//! warp perspective
TEST_F(NAIVE, WARP_PERSPECTIVE_RECORD) {
    TaskRecordChecker<WarpPerspective> checker(2);
    WarpPerspective::Param param;
    param.bmode = WarpPerspective::Param::BorderMode::BORDER_REFLECT;
    param.imode = WarpPerspective::Param::InterpolationMode::LINEAR;
    param.format = WarpPerspective::Param::Format::NCHW;

    checker.set_dtype(0, dtype::Uint8{})
            .set_dtype(1, dtype::Float32{})
            .set_dtype(2, dtype::Uint8{});
    checker.set_param(param).exec({{1, 1, 3, 3}, {1, 3, 3}, {1, 1, 2, 2}});

    checker.set_dtype(0, dtype::Quantized8Asymm{1.4f, static_cast<uint8_t>(127)})
            .set_dtype(1, dtype::Float32{})
            .set_dtype(2, dtype::Quantized8Asymm{1.4f, static_cast<uint8_t>(127)});
    checker.set_param(param).exec({{1, 1, 3, 3}, {1, 3, 3}, {1, 1, 2, 2}});
}

TEST_F(NAIVE, WARP_PERSPECTIVE_NCHW4_RECORD) {
    using Param = WarpPerspective::Param;
    WarpPerspective::Param param;
    TaskRecordChecker<WarpPerspectiveForward> checker(2);
    WarpPerspectiveMatRNG rng;
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::QuantizedS8(0.1f));
    checker.set_dtype(2, dtype::QuantizedS8(0.1f));
    for (auto bmode :
         {WarpPerspective::BorderMode::WRAP, WarpPerspective::BorderMode::REFLECT,
          WarpPerspective::BorderMode::REPLICATE,
          WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        param.format = Param::Format::NCHW4;
        checker.set_param(param);
        checker.execs({{2, 1, 10, 11, 4}, {2, 3, 3}, {2, 1, 11, 12, 4}});
        checker.execs({{1, 25, 25, 25, 4}, {1, 3, 3}, {1, 25, 25, 510, 4}});
        checker.execs({{1, 25, 25, 25, 4}, {1, 3, 3}, {1, 25, 51, 51, 4}});
        checker.execs({{1, 25, 51, 51, 4}, {1, 3, 3}, {1, 25, 25, 25, 4}});
    }
}

TEST_F(NAIVE_MULTI_THREADS, WARP_PERSPECTIVE_RECORD) {
    TaskRecordChecker<WarpPerspective> checker(2);
    WarpPerspective::Param param;
    param.bmode = WarpPerspective::Param::BorderMode::BORDER_REFLECT;
    param.imode = WarpPerspective::Param::InterpolationMode::LINEAR;
    param.format = WarpPerspective::Param::Format::NCHW;

    checker.set_dtype(0, dtype::Uint8{})
            .set_dtype(1, dtype::Float32{})
            .set_dtype(2, dtype::Uint8{});
    checker.set_param(param).exec({{1, 1, 3, 3}, {1, 3, 3}, {1, 1, 2, 2}});

    checker.set_dtype(0, dtype::Quantized8Asymm{1.4f, static_cast<uint8_t>(127)})
            .set_dtype(1, dtype::Float32{})
            .set_dtype(2, dtype::Quantized8Asymm{1.4f, static_cast<uint8_t>(127)});
    checker.set_param(param).exec({{1, 1, 3, 3}, {1, 3, 3}, {1, 1, 2, 2}});
}

TEST_F(NAIVE_MULTI_THREADS, WARP_PERSPECTIVE_NCHW4_RECORD) {
    using Param = WarpPerspective::Param;
    WarpPerspective::Param param;
    TaskRecordChecker<WarpPerspectiveForward> checker(2);
    WarpPerspectiveMatRNG rng;
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::QuantizedS8(0.1f));
    checker.set_dtype(2, dtype::QuantizedS8(0.1f));
    for (auto bmode :
         {WarpPerspective::BorderMode::WRAP, WarpPerspective::BorderMode::REFLECT,
          WarpPerspective::BorderMode::REPLICATE,
          WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        param.format = Param::Format::NCHW4;
        checker.set_param(param);
        checker.execs({{2, 1, 10, 11, 4}, {2, 3, 3}, {2, 1, 11, 12, 4}});
        checker.execs({{1, 25, 25, 25, 4}, {1, 3, 3}, {1, 25, 25, 510, 4}});
        checker.execs({{1, 25, 25, 25, 4}, {1, 3, 3}, {1, 25, 51, 51, 4}});
        checker.execs({{1, 25, 51, 51, 4}, {1, 3, 3}, {1, 25, 25, 25, 4}});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
