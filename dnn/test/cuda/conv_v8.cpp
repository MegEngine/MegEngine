#include "megdnn/dtype.h"
#include "test/cuda/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "src/cuda/handle.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/cuda/utils.h"

using namespace megdnn;
using namespace test;
using namespace conv_bias;

#if CUDNN_VERSION >= 8020
TEST_F(CUDA, CONV_V8_FLOAT) {
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(ExecutionPolicyAlgoName{
                    ConvBiasForward::algo_name<ConvBiasForward::DefaultParam>(
                            "CUDNN:ConvolutionV8", {})
                            .c_str()}));

    UniformFloatRNG rng(0.f, 1.f);
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Float32());
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::NCHW;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {64, 64, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {64, 64, 3, 3}, {1, 64, 1, 1}, {1, 64, 7, 7}, {}});

    // group
    param.sparse = param::ConvBias::Sparse::GROUP;
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {8, 8, 8, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {8, 8, 8, 3, 3}, {1, 64, 1, 1}, {1, 64, 7, 7}, {}});

    // NHWC
    param.format = param::ConvBias::Format::NHWC;
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {8, 8, 3, 3, 8}, {1, 1, 1, 64}, {}, {}});
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {8, 8, 3, 3, 8}, {1, 1, 1, 64}, {1, 7, 7, 64}, {}});
}

TEST_F(CUDA, CONV_V8_HALF) {
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(ExecutionPolicyAlgoName{
                    ConvBiasForward::algo_name<ConvBiasForward::DefaultParam>(
                            "CUDNN:ConvolutionV8", {})
                            .c_str()}));

    UniformFloatRNG rng(0.f, 1.f);
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16())
            .set_dtype(3, dtype::Float16())
            .set_dtype(4, dtype::Float16())
            .set_epsilon(5e-2);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::NCHW;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.compute_mode = param::ConvBias::ComputeMode::FLOAT32;
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {64, 64, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {64, 64, 3, 3}, {1, 64, 1, 1}, {1, 64, 7, 7}, {}});

    // group
    param.sparse = param::ConvBias::Sparse::GROUP;
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {8, 8, 8, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {8, 8, 8, 3, 3}, {1, 64, 1, 1}, {1, 64, 7, 7}, {}});

    // NHWC
    param.format = param::ConvBias::Format::NHWC;
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {8, 8, 3, 3, 8}, {1, 1, 1, 64}, {}, {}});
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {8, 8, 3, 3, 8}, {1, 1, 1, 64}, {1, 7, 7, 64}, {}});
}

TEST_F(CUDA, CONV_BIAS_V8_FLOAT) {
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(ExecutionPolicyAlgoName{
                    ConvBiasForward::algo_name<ConvBiasForward::DefaultParam>(
                            "CUDNN:ConvBiasActivationV8", {})
                            .c_str()}));

    UniformFloatRNG rng(0.f, 1.f);
    UniformFloatRNG crng(0.f, 0.f);
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Float32());
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::NCHW;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {64, 64, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {64, 64, 3, 3}, {1, 64, 1, 1}, {1, 64, 7, 7}, {}});

    // group
    param.sparse = param::ConvBias::Sparse::GROUP;
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {8, 8, 8, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {8, 8, 8, 3, 3}, {1, 64, 1, 1}, {1, 64, 7, 7}, {}});

    // NHWC
    param.format = param::ConvBias::Format::NHWC;
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {8, 8, 3, 3, 8}, {1, 1, 1, 64}, {}, {}});
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {8, 8, 3, 3, 8}, {1, 1, 1, 64}, {1, 7, 7, 64}, {}});
}

TEST_F(CUDA, CONV_BIAS_V8_HALF) {
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(ExecutionPolicyAlgoName{
                    ConvBiasForward::algo_name<ConvBiasForward::DefaultParam>(
                            "CUDNN:ConvBiasActivationV8", {})
                            .c_str()}));

    UniformFloatRNG rng(0.f, 1.f);
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16())
            .set_dtype(3, dtype::Float16())
            .set_dtype(4, dtype::Float16())
            .set_epsilon(5e-2);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::NCHW;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.compute_mode = param::ConvBias::ComputeMode::FLOAT32;
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {64, 64, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {64, 64, 3, 3}, {1, 64, 1, 1}, {1, 64, 7, 7}, {}});

    // group
    param.sparse = param::ConvBias::Sparse::GROUP;
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {8, 8, 8, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.set_param(param).execs(
            {{1, 64, 7, 7}, {8, 8, 8, 3, 3}, {1, 64, 1, 1}, {1, 64, 7, 7}, {}});

    // NHWC
    param.format = param::ConvBias::Format::NHWC;
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {8, 8, 3, 3, 8}, {1, 1, 1, 64}, {}, {}});
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {8, 8, 3, 3, 8}, {1, 1, 1, 64}, {1, 7, 7, 64}, {}});
}

TEST_F(CUDA, CONV_BIAS_V8_DP4A) {
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(ExecutionPolicyAlgoName{
                    ConvBiasForward::algo_name<ConvBiasForward::DefaultParam>(
                            "CUDNN:ConvBiasActivationV8", {})
                            .c_str()}));

    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.0f})
            .set_epsilon(1 + 1e-3);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::NCHW4;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    checker.set_param(param).execs(
            {{1, 16, 7, 7, 4}, {64, 16, 3, 3, 4}, {1, 16, 1, 1, 4}, {}, {}});
    checker.set_param(param).execs(
            {{1, 16, 7, 7, 4},
             {64, 16, 3, 3, 4},
             {1, 16, 1, 1, 4},
             {1, 16, 7, 7, 4},
             {}});

    param.nonlineMode = param::ConvBias::NonlineMode::IDENTITY;
    checker.set_param(param).execs(
            {{1, 16, 7, 7, 4}, {64, 16, 3, 3, 4}, {1, 16, 1, 1, 4}, {}, {}});
    checker.set_param(param).execs(
            {{1, 16, 7, 7, 4},
             {64, 16, 3, 3, 4},
             {1, 16, 1, 1, 4},
             {1, 16, 7, 7, 4},
             {}});

    param.format = param::ConvBias::Format::NHWC;
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {64, 3, 3, 64}, {1, 1, 1, 64}, {}, {}});
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {64, 3, 3, 64}, {1, 1, 1, 64}, {1, 7, 7, 64}, {}});
    param.sparse = param::ConvBias::Sparse::GROUP;
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {8, 8, 3, 3, 8}, {1, 1, 1, 64}, {}, {}});
    checker.set_param(param).execs(
            {{1, 7, 7, 64}, {8, 8, 3, 3, 8}, {1, 1, 1, 64}, {1, 7, 7, 64}, {}});
}

TEST_F(CUDA, CONV_BIAS_V8_IMMA) {
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(ExecutionPolicyAlgoName{
                    ConvBiasForward::algo_name<ConvBiasForward::DefaultParam>(
                            "CUDNN:ConvBiasActivationV8", {})
                            .c_str()}));

    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.0f})
            .set_epsilon(1 + 1e-3);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::NCHW32;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    checker.set_param(param).execs(
            {{1, 2, 7, 7, 32}, {64, 2, 3, 3, 32}, {1, 2, 1, 1, 32}, {}, {}});
    checker.set_param(param).execs(
            {{1, 2, 7, 7, 32},
             {64, 2, 3, 3, 32},
             {1, 2, 1, 1, 32},
             {1, 2, 7, 7, 32},
             {}});

    param.nonlineMode = NonlineMode::RELU;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 0;

    checker.set_param(param).execs(
            {{2, 8, 12, 12, 32}, {512, 8, 1, 1, 32}, {1, 16, 1, 1, 32}, {}, {}});
}

#endif
// vim: syntax=cpp.doxygen
