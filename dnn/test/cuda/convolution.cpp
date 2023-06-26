#include "test/common/convolution.h"
#include "megdnn/dtype.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "test/common/accuracy_shake_checker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"

#include <cudnn.h>

#define V1(x) #x
#define V(x)  V1(x)
#define CUDNN_VERSION_STRING \
    "v" V(CUDNN_MAJOR) "." V(CUDNN_MINOR) "." V(CUDNN_PATCHLEVEL)

namespace megdnn {
namespace test {

#if CUDNN_MAJOR > 9
TEST_F(CUDA, CONVOLUTION_8X8X32) {
    require_compute_capability(6, 1);

    using namespace convolution;
    std::vector<TestArg> args;
    {
        auto v = get_args();
        for (auto&& a : v) {
            args.push_back(std::move(a));
        }
    }
    {
        auto v = get_dilated_args();
        for (auto&& a : v) {
            args.push_back(std::move(a));
        }
    }
    {
        auto v = get_chanwise_args();
        for (auto&& a : v) {
            args.push_back(std::move(a));
        }
    }
    Checker<ConvolutionForward> checker(handle_cuda());
    UniformIntRNG rng(-4, 4);
    for (auto arg : args) {
        arg.param.format = param::Convolution::Format::NHWC;
        arg.src = cvt_src_or_dst_nchw2nhwc(arg.src);
        arg.filter = cvt_filter_nchw2nhwc(arg.filter);
        checker.set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8())
                .set_dtype(2, dtype::Int32())
                .set_param(arg.param)
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .execs({arg.src, arg.filter, {}});
    }
}
#endif

TEST_F(CUDA, CONVOLUTION_FORWARD) {
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    Checker<ConvolutionForward> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
        arg.param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
        checker.set_dtype(0, dtype::BFloat16())
                .set_dtype(1, dtype::BFloat16())
                .set_dtype(2, dtype::BFloat16())
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

TEST_F(CUDA, CONV_FORWARD_MATMUL_NCHW4) {
    require_compute_capability(6, 1);
    using namespace convolution;
    Checker<Convolution> checker(handle_cuda());
    UniformIntRNG int_rng{-127, 127};
    Convolution::Param param;
    param.format = Convolution::Param::Format::NCHW4;

    checker.set_dtype(0, dtype::QuantizedS8(0.132f))
            .set_dtype(1, dtype::QuantizedS8(0.0239f))
            .set_dtype(2, dtype::QuantizedS32(0.132f * 0.0239f))
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_param(param);

    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>(ExecutionPolicyAlgoName{
                    "DEFAULT",
                    {{ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>(
                              "MATMUL8X8X32", {})
                              .c_str(),
                      {}}}}));

    param.sparse = Convolution::Param::Sparse::DENSE;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    checker.set_param(param);
    checker.exec({{8, 4, 10, 10, 4}, {16, 4, 3, 3, 4}, {}});
    checker.exec({{1, 4, 2, 2, 4}, {16, 4, 3, 3, 4}, {}});
    checker.exec({{8, 64, 12, 12, 4}, {256, 64, 3, 3, 4}, {}});
}

TEST_F(CUDA, CONVOLUTION_1X1_FORWARD) {
    using namespace convolution;
    std::vector<TestArg> args = get_1x1_args();
    Checker<ConvolutionForward> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_DATA) {
    using namespace convolution;
    std::vector<TestArg> args = get_args_cuda_conv_bwd_data();
    Checker<ConvolutionBackwardData> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 64.f / sqrt(arg.filter[0] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
        if (!check_compute_capability(6, 0)) {
            src.dtype = dst.dtype = filter.dtype = dtype::Float16();
            checker.set_rng(0, &rng)
                    .set_rng(1, &rng)
                    .set_epsilon(1e-1)
                    .set_param(arg.param)
                    .exec(TensorLayoutArray{filter, dst, src});
            arg.param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
            checker.set_rng(0, &rng)
                    .set_rng(1, &rng)
                    .set_epsilon(1e-1)
                    .set_param(arg.param)
                    .exec(TensorLayoutArray{filter, dst, src});
        }
        checker.set_before_exec_callback(
                AlgoChecker<ConvolutionBackwardData>(ExecutionPolicyAlgoName{
                        "CONVOLUTION_BACKWARD_DATD_BFLOAT16",
                        {{"MATMUL", {{"CUBLAS", {}}}}}}));
        src.dtype = dst.dtype = filter.dtype = dtype::BFloat16();
        arg.param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
        checker.reset_before_exec_callback();
        checker.opr()->execution_policy() = {};
    }
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_DATA_FP16_CUDNN7_5) {
    // algo CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 with
    // TensorCore operations produces incorrect result.
    // Maybe nvidia has fixed this issue
    // There is a test using incorrect case:
    // inp={2x8x18x18}, kern={8x8x2x2}, pad_h=pad_w=2, stride_h=stride_w=2,
    // dtype=float16
    using namespace convolution;
    std::vector<TestArg> args = get_args_cudnn_5_1_backward();
    Checker<ConvolutionBackwardData> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 128.f / sqrt(arg.filter[0] * arg.filter[2] * arg.filter[3]);
        scale = std::max(scale, 1.f);
        UniformFloatRNG rng(scale, 2 * scale);
        arg.param.format = param::Convolution::Format::NHWC;
        arg.src = cvt_src_or_dst_nchw2nhwc(arg.src);
        arg.filter = cvt_filter_nchw2nhwc(arg.filter);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        src.dtype = dst.dtype = filter.dtype = dtype::Float16();
        arg.param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-2)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        arg.param.compute_mode = param::Convolution::ComputeMode::DEFAULT;
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-2)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
    }
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_DATA_NHWC) {
    using namespace convolution;
    std::vector<TestArg> args = get_args_cuda_conv_bwd_data();
    Checker<ConvolutionBackwardData> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 64.f / sqrt(arg.filter[0] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        arg.param.format = param::Convolution::Format::NHWC;
        arg.src = cvt_src_or_dst_nchw2nhwc(arg.src);
        arg.filter = cvt_filter_nchw2nhwc(arg.filter);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        src.dtype = dst.dtype = filter.dtype = dtype::Float16();
        arg.param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-2)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        arg.param.compute_mode = param::Convolution::ComputeMode::DEFAULT;
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-2)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
    }
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_DATA_CUDNN) {
    require_compute_capability(7, 0);
    using namespace convolution;
    Checker<ConvolutionBackwardData> checker(handle_cuda());
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardData>("CUDNN_CONVOLUTION"));
    //! noncontiguous case
    {
        param::Convolution param;
        param.pad_h = param.pad_w = 1;
        checker.set_param(param).execl(TensorLayoutArray{
                {{16, 16, 3, 3}, {144, 9, 3, 1}, dtype::Float32()},
                {{2, 16, 7, 7}, {1568, 49, 7, 1}, dtype::Float32()},
                {{2, 16, 7, 7}, {1568, 49, 7, 1}, dtype::Float32()},
        });
    }
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_DATA_MATMUL) {
    using namespace convolution;
    std::vector<TestArg> args = get_args_cuda_conv_bwd_data();
    Checker<ConvolutionBackwardData> checker(handle_cuda());

    checker.set_before_exec_callback(AlgoChecker<ConvolutionBackwardData>(
            ExecutionPolicyAlgoName{"MATMUL", {{"CUBLAS", {}}}}));
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 64.f / sqrt(arg.filter[0] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
    }
    //! noncontiguous case
    {
        param::Convolution param;
        param.pad_h = param.pad_w = 1;
        checker.set_param(param).execl(TensorLayoutArray{
                {{16, 16, 3, 3}, {144, 9, 3, 1}, dtype::Float32()},
                {{2, 16, 7, 7}, {1568, 49, 7, 1}, dtype::Float32()},
                {{2, 16, 7, 7}, {1568, 49, 7, 1}, dtype::Float32()},
        });
    }
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_DATA_INT8_NCHW4_DP4A) {
    require_compute_capability(6, 1);

    using namespace convolution;
    std::vector<TestArg> args = get_args_int8_nchw4_conv_bwd_data();

    struct AlgoParam {
        int threadblock_m;
        int threadblock_n;
        int threadblock_k;
        int warp_m;
        int warp_n;
        int warp_k;
        int stage;
        std::string to_string() {
            return ssprintf(
                    "_%dX%dX%d_%dX%dX%d_%dstage", threadblock_m, threadblock_n,
                    threadblock_k, warp_m, warp_n, warp_k, stage);
        }
    };

    std::vector<AlgoParam> all_params;

    all_params.emplace_back(AlgoParam{16, 64, 8, 16, 64, 8, 2});
    all_params.emplace_back(AlgoParam{16, 128, 16, 16, 64, 16, 2});
    all_params.emplace_back(AlgoParam{16, 128, 16, 16, 128, 16, 1});
    all_params.emplace_back(AlgoParam{32, 128, 32, 32, 64, 32, 2});

    for (auto algo_param : all_params) {
        Checker<ConvolutionBackwardData> checker(handle_cuda());
        std::string algo_name(ssprintf(
                "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM%s", algo_param.to_string().c_str()));
        checker.set_before_exec_callback(
                AlgoChecker<ConvolutionBackwardData>(algo_name.c_str()));

        checker.set_epsilon(1 + 1e-3).set_max_avg_error(1e-1);

        for (auto&& arg : args) {
            UniformIntRNG rng(-3, 3);
            auto src = TensorLayout(arg.src, dtype::QuantizedS8{1.2f});
            auto filter = TensorLayout(arg.filter, dtype::QuantizedS8{1.3f});
            TensorLayout dst;
            dst.dtype = dtype::QuantizedS8{1.2f};
            {
                auto opr = handle_cuda()->create_operator<Convolution>();
                opr->param() = arg.param;
                opr->deduce_layout(src, filter, dst);
            }
            checker.set_rng(0, &rng).set_rng(1, &rng).set_param(arg.param).exec(
                    TensorLayoutArray{filter, dst, src});
        }
    }
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_DATA_INT8_NCHW_DP4A) {
    require_compute_capability(6, 1);
    using namespace convolution;
    std::vector<TestArg> args = get_args_int8_nchw_conv_bwd_data();
    Checker<ConvolutionBackwardData> checker(handle_cuda());

    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardData>("INT8_NCHW_DOTPROD_IMPLICIT_GEMM"));

    checker.set_epsilon(1 + 1e-3).set_max_avg_error(1e-1);

    for (auto&& arg : args) {
        UniformIntRNG rng(-3, 3);
        auto src = TensorLayout(arg.src, dtype::QuantizedS8{1.2f});
        auto filter = TensorLayout(arg.filter, dtype::QuantizedS8{1.3f});
        TensorLayout dst;
        dst.dtype = dtype::QuantizedS8{1.2f};
        {
            auto opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        checker.set_rng(0, &rng).set_rng(1, &rng).set_param(arg.param).exec(
                TensorLayoutArray{filter, dst, src});
    }
}

#if CUDA_VERSION >= 10020
TEST_F(CUDA, CONVOLUTION_BACKWARD_DATA_INT8_NHWC_IMMA) {
    require_compute_capability(7, 5);

    using namespace convolution;
    std::vector<TestArg> args = get_args_int8_nhwc_conv_bwd_data();

    struct AlgoParam {
        int threadblock_m;
        int threadblock_n;
        int threadblock_k;
        int warp_m;
        int warp_n;
        int warp_k;
        int stage;
        int access_size;
        std::string to_string() {
            return ssprintf(
                    "_%dX%dX%d_%dX%dX%d_%dstage_%d", threadblock_m, threadblock_n,
                    threadblock_k, warp_m, warp_n, warp_k, stage, access_size);
        }
    };

    std::vector<AlgoParam> all_params;

    all_params.emplace_back(AlgoParam{64, 16, 32, 64, 16, 32, 2, 4});
    all_params.emplace_back(AlgoParam{64, 16, 32, 64, 16, 32, 2, 8});
    all_params.emplace_back(AlgoParam{64, 16, 32, 64, 16, 32, 2, 16});
    all_params.emplace_back(AlgoParam{128, 32, 32, 64, 32, 32, 1, 4});
    all_params.emplace_back(AlgoParam{128, 32, 32, 64, 32, 32, 1, 8});
    all_params.emplace_back(AlgoParam{128, 32, 32, 64, 32, 32, 1, 16});

    for (auto algo_param : all_params) {
        Checker<ConvolutionBackwardData> checker(handle_cuda());
        std::string algo_name(ssprintf(
                "INT8_NHWC_IMMA_IMPLICIT_GEMM%s", algo_param.to_string().c_str()));
        checker.set_before_exec_callback(
                AlgoChecker<ConvolutionBackwardData>(algo_name.c_str()));

        checker.set_epsilon(1 + 1e-3).set_max_avg_error(1e-1);

        for (auto&& arg : args) {
            UniformIntRNG rng(-3, 3);
            auto src = TensorLayout(arg.src, dtype::QuantizedS8{1.2f});
            auto filter = TensorLayout(arg.filter, dtype::QuantizedS8{1.3f});
            TensorLayout dst;
            dst.dtype = dtype::QuantizedS8{1.2f};
            {
                auto opr = handle_cuda()->create_operator<Convolution>();
                opr->param() = arg.param;
                opr->deduce_layout(src, filter, dst);
            }
            checker.set_rng(0, &rng).set_rng(1, &rng).set_param(arg.param).exec(
                    TensorLayoutArray{filter, dst, src});
        }
    }
}
#endif

TEST_F(CUDA, CONVOLUTION_BACKWARD_DATA_FAILED_CUDNN7_5) {
    // BRAIN-481 failed on architectures 7.0, remove the following if statement,
    // when cudnn fixed the problem.
    require_compute_capability(7, 0);
    using namespace convolution;
    std::vector<TestArg> args = get_args_cudnn_7_5_failures();
    Checker<ConvolutionBackwardData> checker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 128.f / sqrt(arg.filter[0] * arg.filter[2] * arg.filter[3]);
        scale = std::max(scale, 1.f);
        UniformFloatRNG rng(scale, 2 * scale);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
        src.dtype = dst.dtype = filter.dtype = dtype::Float16();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
        arg.param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
    }
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_FILTER) {
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    Checker<ConvolutionBackwardFilter> checker(handle_cuda());
    bool f16_checked = false;
    for (auto&& arg : args) {
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        float scale = 1.0f / sqrt(dst[2] * dst[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});

        // reduce on large f16 array may introduce significant error
        if (dst.total_nr_elems() >= 1000 && f16_checked)
            continue;

        f16_checked = true;
        src.dtype = dst.dtype = filter.dtype = dtype::Float16();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});
        arg.param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});

        checker.set_before_exec_callback(
                AlgoChecker<ConvolutionBackwardFilter>(ExecutionPolicyAlgoName{
                        "CONVOLUTION_BACKWARD_FILTER_BFLOAT16",
                        {{"MATMUL", {{"CUBLAS", {}}}}}}));
        src.dtype = dst.dtype = filter.dtype = dtype::BFloat16();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});
        checker.reset_before_exec_callback();
        checker.opr()->execution_policy() = {};
    }
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_FILTER_MATMUL) {
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    Checker<ConvolutionBackwardFilter> checker(handle_cuda());
    checker.set_before_exec_callback(AlgoChecker<ConvolutionBackwardFilter>(
            ExecutionPolicyAlgoName{"MATMUL", {{"CUBLAS", {}}}}));
    for (auto&& arg : args) {
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        float scale = 1.0f / sqrt(dst[2] * dst[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});
    }
    //! noncontiguous case
    {
        NormalRNG default_rng;
        param::Convolution param;
        param.pad_h = param.pad_w = 1;
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_param(param)
                .execl(TensorLayoutArray{
                        {{2, 16, 7, 7}, {1568, 49, 7, 1}, dtype::Float32()},
                        {{2, 16, 7, 7}, {1568, 49, 7, 1}, dtype::Float32()},
                        {{16, 16, 3, 3}, {144, 9, 3, 1}, dtype::Float32()}});
    }
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_FILTER_CUDNN) {
    require_compute_capability(7, 0);
    using namespace convolution;
    Checker<ConvolutionBackwardFilter> checker(handle_cuda());
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardFilter>("CUDNN_CONVOLUTION"));
    //! noncontiguous case
    {
        param::Convolution param;
        param.pad_h = param.pad_w = 1;
        checker.set_param(param).execl(TensorLayoutArray{
                {{2, 16, 7, 7}, {1568, 49, 7, 1}, dtype::Float32()},
                {{2, 16, 7, 7}, {1568, 49, 7, 1}, dtype::Float32()},
                {{16, 16, 3, 3}, {144, 9, 3, 1}, dtype::Float32()}});
    }
}

TEST_F(CUDA, CONV_CONFIG_COMBINATIONS) {
    auto eps_getter = [](bool f16, int stage, const char* name) -> float {
        if (f16) {
            return stage == 2 ? 0.5 : 0.2;
        }
        if (strstr(name, "WINOGRAD_NONFUSED"))
            return 0.3;
        return 1e-3;
    };
    convolution::test_conv_config_combinations(
            2, handle_cuda(), false, true, true, eps_getter, true);
    convolution::test_conv_config_combinations(
            3, handle_cuda(), false, true, true, eps_getter, true);
    convolution::test_conv_config_combinations(
            5, handle_cuda(), false, true, true, eps_getter, true);
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_DATA_1) {
    require_compute_capability(7, 0);
    using namespace convolution;
    Checker<ConvolutionBackwardData> checker(handle_cuda());
    checker.set_before_exec_callback(AlgoChecker<ConvolutionBackwardData>(
            "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1" CUDNN_VERSION_STRING));
    NormalRNG default_rng;
    TensorShape s_filter = TensorShape{8, 8, 2, 2}, s_src = TensorShape{2, 8, 18, 18};
    float scale = 1.0f / sqrt(s_filter[0] * s_filter[2] * s_filter[3]);
    UniformFloatRNG rng(scale, 2 * scale);
    auto src = TensorLayout(s_src, dtype::Float16());
    auto filter = TensorLayout(s_filter, dtype::Float16());
    TensorLayout dst;
    param::Convolution param;
    param.pad_h = param.pad_w = 2;
    param.stride_h = param.stride_w = 2;
    {
        auto opr = handle_cuda()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout(src, filter, dst);
    }
    src.dtype = dst.dtype = filter.dtype = dtype::Float16();
    param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
    checker.set_rng(0, &rng).set_rng(1, &rng).set_epsilon(0.2).set_param(param).exec(
            TensorLayoutArray{filter, dst, src});
}

TEST_F(CUDA, CONVOLUTION_BACKWARD_DEPTHWISE_LARGE_FILTER) {
    Checker<ConvolutionBackwardData> checker(handle_cuda());
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardData>("DEPTHWISE_LARGE_FILTER"));
    for (auto dtype : std::vector<DType> {
             dtype::Float32(),
#if CUDA_VERSION >= 9000
                     dtype::Float16()
#endif
         }) {
        auto run = [&checker, &dtype](
                           size_t n, size_t g, size_t h, size_t fh, size_t padding,
                           size_t stride) {
            param::Convolution param;
            param.stride_h = param.stride_w = stride;
            param.pad_h = param.pad_w = padding;
            param.mode = Convolution::Mode::CROSS_CORRELATION;
            param.sparse = param::Convolution::Sparse::GROUP;
            checker.set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(2, dtype);
            float scale = 64.f / sqrt(fh * fh);
            UniformFloatRNG rng(scale, scale * 2);
            checker.set_rng(0, &rng).set_rng(1, &rng).set_rng(2, &rng);
            if (dtype.enumv() == DTypeEnum::Float16)
                checker.set_epsilon(1e-1);

            checker.set_param(param).execs(
                    {{g, 1, 1, fh, fh},
                     {n, g, (h + 2 * padding - fh + 1) / stride,
                      (h + 2 * padding - fh + 1) / stride},
                     {n, g, h, h}});
        };
        run(4, 8, 32, 5, 5 / 2, 1);
        run(4, 8, 32, 7, 7 / 2, 1);
        run(4, 8, 32, 9, 9 / 2, 1);
        run(4, 8, 32, 11, 11 / 2, 1);
        run(4, 8, 32, 13, 13 / 2, 1);
        run(4, 8, 32, 15, 15 / 2, 1);
        run(4, 8, 32, 17, 17 / 2, 1);
        run(4, 8, 32, 19, 19 / 2, 1);
        run(4, 8, 32, 21, 21 / 2, 1);
        run(4, 8, 32, 23, 23 / 2, 1);
        run(4, 8, 32, 25, 25 / 2, 1);
        run(4, 8, 32, 27, 27 / 2, 1);
        run(4, 8, 32, 29, 29 / 2, 1);
        run(4, 8, 32, 31, 31 / 2, 1);
        run(4, 8, 64, 5, 5 / 2, 2);
        run(4, 8, 64, 7, 7 / 3, 2);
        run(4, 8, 64, 9, 9 / 3, 2);
        run(4, 8, 64, 11, 11 / 3, 2);
        run(4, 8, 64, 13, 13 / 3, 2);
        run(4, 8, 64, 15, 15 / 3, 2);
        run(4, 8, 64, 17, 17 / 3, 2);
        run(4, 8, 64, 19, 19 / 3, 2);
        run(4, 8, 64, 21, 21 / 3, 2);
        run(4, 8, 64, 23, 23 / 3, 2);
        run(4, 8, 64, 25, 25 / 3, 2);
        run(4, 8, 64, 27, 27 / 3, 2);
        run(4, 8, 64, 29, 29 / 3, 2);
        run(4, 8, 64, 31, 31 / 3, 2);
        run(1, 2, 128, 31, 31 / 3, 2);
        run(1, 2, 256, 31, 31 / 3, 2);
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_CONVOLUTION_1X1_FORWARD) {
    using namespace convolution;
    std::vector<TestArg> args = get_1x1_args();
    Benchmarker<ConvolutionForward> marker(handle_cuda());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        marker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

TEST_F(CUDA, CONV_FWD_BENCHMARK) {
    auto run = [&](size_t N, size_t OC, size_t IC, size_t IH, size_t IW, size_t SH = 1,
                   size_t SW = 1, size_t FH = 1, size_t FW = 1, size_t PH = 0,
                   size_t PW = 0, bool fp16io_c32 = false) {
        auto benchmarker = Benchmarker<ConvolutionForward>(handle_cuda());
        benchmarker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16());
        ConvolutionForward::Param param;
        param.stride_h = SH;
        param.stride_w = SW;
        param.pad_h = PH;
        param.pad_w = PW;
        if (fp16io_c32) {
            param.compute_mode = ConvolutionForward::Param::ComputeMode::FLOAT32;
        }
        benchmarker.set_param(param);
        std::unique_ptr<OprProxy<ConvolutionForward>> proxy{
                new OprProxy<ConvolutionForward>{true}};
        benchmarker.set_proxy(proxy);
        size_t OH = (IH - FH + 2 * PH) / SH + 1;
        size_t OW = (IW - FW + 2 * PW) / SW + 1;
        auto time =
                benchmarker.execs({{N, IC, IH, IW}, {OC, IC, FH, FW}, {N, OC, OH, OW}});
        time /= 1000.0 * 10.0;
        auto flo = (double)N * OC * IC * OH * OW * FH * FW * 2;
        auto flops = flo / time / 1e12;
        printf("comp_type %s: ", fp16io_c32 ? "32" : "16");
        printf("%.3fG FLO, flops %.3fTFLOPS\n", flo / 1e9, flops);
    };
    run(32, 512, 256, 56, 56, 1, 1, 1, 1, 0, 0, false);
    run(32, 512, 256, 56, 56, 1, 1, 1, 1, 0, 0, true);
}

TEST_F(CUDA, CONVOLUTION_FWD_BENCHMARK) {
    CUBenchmarker<ConvolutionForward> bench{handle_cuda()};
    std::unique_ptr<OprProxy<ConvolutionForward>> proxy{
            new OprProxy<ConvolutionForward>{true}};
    size_t RUNS = 10;
    bench.set_proxy(proxy).set_times(RUNS);

    auto run = [&](size_t N, size_t OC, size_t IC, size_t IH, size_t IW, size_t FH,
                   size_t SH, size_t PH) {
        bench.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32());
        param::Convolution param;
        param.stride_h = param.stride_w = SH;
        param.pad_h = param.pad_w = PH;
        param.compute_mode = param::Convolution::ComputeMode::DEFAULT;
        bench.set_param(param);
        bench.proxy()->target_execution_policy.algo.reset();
        TensorLayout src{{N, IC, IH, IW}, dtype::Float32()},
                filter{{OC, IC, FH, FH}, dtype::Float32()};
        TensorLayout dst;
        {
            auto&& opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = param;
            opr->deduce_layout(src, filter, dst);
        }
        auto time_ms_fp32 = bench.execl({src, filter, dst}) / RUNS;
        src.dtype = filter.dtype = dst.dtype = dtype::Float16();
        bench.proxy()->target_execution_policy.algo.reset();
        bench.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16());
        auto time_ms_true_fp16 = bench.execl({src, filter, dst}) / RUNS;
        param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
        bench.proxy()->target_execution_policy.algo.reset();
        bench.set_param(param);
        auto time_ms_pseudo_fp16 = bench.execl({src, filter, dst}) / RUNS;
        float flo = 2.0 * N * OC * IC * dst[2] * dst[3] * FH * FH;
        printf("inp=%s, kern=%s, dst=%s ", src.to_string().c_str(),
               filter.to_string().c_str(), dst.to_string().c_str());
        printf("time_fp32=%.2fms, flops=%.3fTFLOPS\ntime_true_fp16=%.2fms, "
               "flops=%.3fTFLOPS\ntime_pseudo_fp16=%.2fms, flops=%.3fFLOPS\n",
               time_ms_fp32, (flo / (time_ms_fp32 * 1e9)), time_ms_true_fp16,
               (flo / (time_ms_true_fp16 * 1e9)), time_ms_pseudo_fp16,
               (flo / (time_ms_pseudo_fp16 * 1e9)));
        printf("speedup (true_fp16/fp32)=%.2f, (true_fp16/pseudo_fp16)=%.2f\n",
               time_ms_fp32 / time_ms_true_fp16,
               time_ms_pseudo_fp16 / time_ms_true_fp16);
    };
    run(32, 64, 3, 224, 224, 7, 2, 3);
    run(32, 128, 128, 28, 28, 3, 1, 1);
    run(32, 256, 256, 14, 14, 3, 1, 1);
    run(32, 512, 512, 7, 7, 3, 1, 1);
    run(32, 64, 64, 56, 56, 3, 1, 1);
    run(32, 512, 256, 56, 56, 1, 2, 0);
    run(32, 1024, 512, 28, 28, 1, 2, 0);
    run(32, 2048, 1024, 14, 14, 1, 2, 0);
    run(32, 512, 128, 28, 28, 1, 1, 0);
    run(32, 128, 512, 28, 28, 1, 1, 0);
    run(32, 1024, 256, 14, 14, 1, 1, 0);
    run(32, 256, 1024, 14, 14, 1, 1, 0);
    run(32, 2048, 512, 7, 7, 1, 1, 0);
    run(32, 512, 2048, 7, 7, 1, 1, 0);
    run(32, 256, 64, 56, 56, 1, 1, 0);
    run(32, 64, 256, 56, 56, 1, 1, 0);
    run(32, 128, 256, 56, 56, 1, 2, 0);
    run(32, 256, 512, 28, 28, 1, 2, 0);
    run(32, 512, 1024, 14, 14, 1, 2, 0);
    run(32, 64, 64, 56, 56, 1, 1, 0);
}

TEST_F(CUDA, CONVOLUTION_BWD_DATA_BENCHMARK) {
    CUBenchmarker<ConvolutionBackwardData> bench{handle_cuda()};
    std::unique_ptr<OprProxy<ConvolutionBackwardData>> proxy{
            new OprProxy<ConvolutionBackwardData>{true}};
    size_t RUNS = 10;
    bench.set_proxy(proxy).set_times(RUNS);

    auto run = [&](size_t N, size_t OC, size_t IC, size_t IH, size_t IW, size_t FH,
                   size_t SH, size_t PH) {
        bench.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32());
        param::Convolution param;
        param.stride_h = param.stride_w = SH;
        param.pad_h = param.pad_w = PH;
        param.compute_mode = param::Convolution::ComputeMode::DEFAULT;
        bench.set_param(param);
        bench.proxy()->target_execution_policy.algo.reset();
        TensorLayout src{{N, IC, IH, IW}, dtype::Float32()},
                filter{{OC, IC, FH, FH}, dtype::Float32()};
        TensorLayout dst;
        {
            auto&& opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = param;
            opr->deduce_layout(src, filter, dst);
        }
        auto time_ms_fp32 = bench.execl({filter, dst, src}) / RUNS;
        src.dtype = filter.dtype = dst.dtype = dtype::Float16();
        bench.proxy()->target_execution_policy.algo.reset();
        bench.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16());
        auto time_ms_true_fp16 = bench.execl({filter, dst, src}) / RUNS;
        param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
        bench.proxy()->target_execution_policy.algo.reset();
        bench.set_param(param);
        auto time_ms_pseudo_fp16 = bench.execl({filter, dst, src}) / RUNS;
        float flo = 2.0 * N * OC * IC * dst[2] * dst[3] * FH * FH;
        printf("inp=%s, kern=%s, dst=%s ", src.to_string().c_str(),
               filter.to_string().c_str(), dst.to_string().c_str());
        printf("time_fp32=%.2fms, flops=%.3fTFLOPS\ntime_true_fp16=%.2fms, "
               "flops=%.3fTFLOPS\ntime_pseudo_fp16=%.2fms, flops=%.3fFLOPS\n",
               time_ms_fp32, (flo / (time_ms_fp32 * 1e9)), time_ms_true_fp16,
               (flo / (time_ms_true_fp16 * 1e9)), time_ms_pseudo_fp16,
               (flo / (time_ms_pseudo_fp16 * 1e9)));
        printf("speedup (true_fp16/fp32)=%.2f, (true_fp16/pseudo_fp16)=%.2f\n",
               time_ms_fp32 / time_ms_true_fp16,
               time_ms_pseudo_fp16 / time_ms_true_fp16);
    };
    run(32, 64, 3, 224, 224, 7, 2, 3);
    run(32, 128, 128, 28, 28, 3, 1, 1);
    run(32, 256, 256, 14, 14, 3, 1, 1);
    run(32, 512, 512, 7, 7, 3, 1, 1);
    run(32, 64, 64, 56, 56, 3, 1, 1);
    run(32, 512, 256, 56, 56, 1, 2, 0);
    run(32, 1024, 512, 28, 28, 1, 2, 0);
    run(32, 2048, 1024, 14, 14, 1, 2, 0);
    run(32, 512, 128, 28, 28, 1, 1, 0);
    run(32, 128, 512, 28, 28, 1, 1, 0);
    run(32, 1024, 256, 14, 14, 1, 1, 0);
    run(32, 256, 1024, 14, 14, 1, 1, 0);
    run(32, 2048, 512, 7, 7, 1, 1, 0);
    run(32, 512, 2048, 7, 7, 1, 1, 0);
    run(32, 256, 64, 56, 56, 1, 1, 0);
    run(32, 64, 256, 56, 56, 1, 1, 0);
    run(32, 128, 256, 56, 56, 1, 2, 0);
    run(32, 256, 512, 28, 28, 1, 2, 0);
    run(32, 512, 1024, 14, 14, 1, 2, 0);
    run(32, 64, 64, 56, 56, 1, 1, 0);
}

TEST_F(CUDA, BENCHMARK_CONVOLUTION_BWD_DATA_DEPTHWISE_LARGE_FILTER_FP32) {
    CUBenchmarker<ConvolutionBackwardData> bencher{handle_cuda()};
    bencher.set_display(false);
    bencher.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardData>("DEPTHWISE_LARGE_FILTER"));

    auto run = [&](size_t N, size_t OC, size_t g, size_t IH, size_t IW, size_t FH,
                   size_t SH, size_t nr_times) {
        bencher.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32());
        param::Convolution param;
        param.stride_h = param.stride_w = SH;
        param.pad_h = param.pad_w = FH / 2;
        param.sparse = param::Convolution::Sparse::GROUP;
        bencher.set_param(param);
        bencher.set_times(nr_times);
        TensorLayout src{{N, g, IH, IW}, dtype::Float32()},
                filter{{g, 1, 1, FH, FH}, dtype::Float32()};
        TensorLayout dst;
        {
            auto&& opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = param;
            opr->deduce_layout(src, filter, dst);
        }
        auto time_ms_fp32 = bencher.execl({filter, dst, src}) / nr_times;
        float flo = 2.0 * N * g * dst[2] * dst[3] * FH * FH;
        printf("inp=%s, kern=%s, dst=%s ", src.to_string().c_str(),
               filter.to_string().c_str(), dst.to_string().c_str());
        printf("time_fp32=%.2fms, flops=%.3fTFLOPS\n", time_ms_fp32,
               (flo / (time_ms_fp32 * 1e9)));
    };
    run(64, 384, 384, 32, 32, 3, 1, 10);
    run(64, 384, 384, 32, 32, 5, 1, 10);
    run(64, 384, 384, 32, 32, 7, 1, 10);
    run(64, 384, 384, 32, 32, 9, 1, 10);
    run(64, 384, 384, 32, 32, 11, 1, 10);
    run(64, 384, 384, 32, 32, 13, 1, 10);
    run(64, 384, 384, 32, 32, 15, 1, 10);
    run(64, 384, 384, 32, 32, 17, 1, 10);
    run(64, 384, 384, 32, 32, 19, 1, 10);
    run(64, 384, 384, 32, 32, 21, 1, 10);
    run(64, 384, 384, 32, 32, 23, 1, 10);
    run(64, 384, 384, 32, 32, 25, 1, 10);
    run(64, 384, 384, 32, 32, 27, 1, 10);
    run(64, 384, 384, 32, 32, 29, 1, 10);
    run(64, 384, 384, 32, 32, 31, 1, 10);
}

TEST_F(CUDA, BENCHMARK_CONVOLUTION_BWD_DATA_DEPTHWISE_LARGE_FILTER_FP16) {
    CUBenchmarker<ConvolutionBackwardData> bencher{handle_cuda()};
    bencher.set_display(false);
    bencher.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardData>("DEPTHWISE_LARGE_FILTER"));

    auto run = [&](size_t N, size_t OC, size_t g, size_t IH, size_t IW, size_t FH,
                   size_t SH, size_t nr_times) {
        bencher.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16());
        param::Convolution param;
        param.stride_h = param.stride_w = SH;
        param.pad_h = param.pad_w = FH / 2;
        param.sparse = param::Convolution::Sparse::GROUP;
        bencher.set_param(param);
        bencher.set_times(nr_times);
        TensorLayout src{{N, g, IH, IW}, dtype::Float16()},
                filter{{g, 1, 1, FH, FH}, dtype::Float16()};
        TensorLayout dst;
        {
            auto&& opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = param;
            opr->deduce_layout(src, filter, dst);
        }
        auto time_ms_fp16 = bencher.execl({filter, dst, src}) / nr_times;
        float flo = 2.0 * N * g * dst[2] * dst[3] * FH * FH;
        printf("inp=%s, kern=%s, dst=%s ", src.to_string().c_str(),
               filter.to_string().c_str(), dst.to_string().c_str());
        printf("time_fp16=%.2fms, flops=%.3fTFLOPS\n", time_ms_fp16,
               (flo / (time_ms_fp16 * 1e9)));
    };
    run(64, 384, 384, 32, 32, 3, 1, 10);
    run(64, 384, 384, 32, 32, 5, 1, 10);
    run(64, 384, 384, 32, 32, 7, 1, 10);
    run(64, 384, 384, 32, 32, 9, 1, 10);
    run(64, 384, 384, 32, 32, 11, 1, 10);
    run(64, 384, 384, 32, 32, 13, 1, 10);
    run(64, 384, 384, 32, 32, 15, 1, 10);
    run(64, 384, 384, 32, 32, 17, 1, 10);
    run(64, 384, 384, 32, 32, 19, 1, 10);
    run(64, 384, 384, 32, 32, 21, 1, 10);
    run(64, 384, 384, 32, 32, 23, 1, 10);
    run(64, 384, 384, 32, 32, 25, 1, 10);
    run(64, 384, 384, 32, 32, 27, 1, 10);
    run(64, 384, 384, 32, 32, 29, 1, 10);
    run(64, 384, 384, 32, 32, 31, 1, 10);
}

TEST_F(CUDA, BENCHMARK_CONVOLUTION_BWD_DATA_BF16) {
    CUBenchmarker<ConvolutionBackwardData> bench{handle_cuda()};
    std::unique_ptr<OprProxy<ConvolutionBackwardData>> proxy{
            new OprProxy<ConvolutionBackwardData>{true}};
    size_t RUNS = 10;
    bench.set_proxy(proxy).set_times(RUNS);

    auto run = [&](size_t N, size_t OC, size_t IC, size_t IH, size_t IW, size_t FH,
                   size_t SH, size_t PH) {
        bench.set_dtype(0, dtype::BFloat16())
                .set_dtype(1, dtype::BFloat16())
                .set_dtype(2, dtype::BFloat16());
        param::Convolution param;
        param.stride_h = param.stride_w = SH;
        param.pad_h = param.pad_w = PH;
        param.compute_mode = param::Convolution::ComputeMode::DEFAULT;
        bench.set_param(param);
        bench.proxy()->target_execution_policy = {};
        TensorLayout src{{N, IC, IH, IW}, dtype::BFloat16()},
                filter{{OC, IC, FH, FH}, dtype::BFloat16()};
        TensorLayout dst;
        {
            auto&& opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = param;
            opr->deduce_layout(src, filter, dst);
        }
        auto used = bench.execl({filter, dst, src}) / RUNS;
        float flo = 2.0 * N * OC * IC * dst[2] * dst[3] * FH * FH;
        printf("inp=%s, kern=%s, dst=%s ", src.to_string().c_str(),
               filter.to_string().c_str(), dst.to_string().c_str());
        printf("time_fp32=%.2fms, flops=%.3fTFLOPS\n", used, (flo / (used * 1e9)));
    };
    run(32, 64, 3, 224, 224, 7, 2, 3);
    run(32, 128, 128, 28, 28, 3, 1, 1);
    run(32, 256, 256, 14, 14, 3, 1, 1);
    run(32, 512, 512, 7, 7, 3, 1, 1);
    run(32, 64, 64, 56, 56, 3, 1, 1);
    run(32, 512, 256, 56, 56, 1, 2, 0);
    run(32, 1024, 512, 28, 28, 1, 2, 0);
    run(32, 2048, 1024, 14, 14, 1, 2, 0);
    run(32, 512, 128, 28, 28, 1, 1, 0);
    run(32, 128, 512, 28, 28, 1, 1, 0);
    run(32, 1024, 256, 14, 14, 1, 1, 0);
    run(32, 256, 1024, 14, 14, 1, 1, 0);
    run(32, 2048, 512, 7, 7, 1, 1, 0);
    run(32, 512, 2048, 7, 7, 1, 1, 0);
    run(32, 256, 64, 56, 56, 1, 1, 0);
    run(32, 64, 256, 56, 56, 1, 1, 0);
    run(32, 128, 256, 56, 56, 1, 2, 0);
    run(32, 256, 512, 28, 28, 1, 2, 0);
    run(32, 512, 1024, 14, 14, 1, 2, 0);
    run(32, 64, 64, 56, 56, 1, 1, 0);
}

TEST_F(CUDA, BENCHMARK_CONVOLUTION_BWD_DATA_INT8_DP4A) {
    CUBenchmarker<ConvolutionBackwardData> bench{handle_cuda()};
    std::unique_ptr<OprProxy<ConvolutionBackwardData>> proxy{
            new OprProxy<ConvolutionBackwardData>{true}};
    size_t RUNS = 10;
    bench.set_proxy(proxy).set_times(RUNS);

    auto run = [&](size_t N, size_t OC, size_t IC, size_t IH, size_t IW, size_t FH,
                   size_t SH, size_t PH) {
        bench.set_dtype(0, dtype::QuantizedS8{1.0f})
                .set_dtype(1, dtype::QuantizedS8{1.0f})
                .set_dtype(2, dtype::QuantizedS8{1.0f});
        param::Convolution param;
        param.format = param::Convolution::Format::NCHW4;
        param.stride_h = param.stride_w = SH;
        param.pad_h = param.pad_w = PH;
        param.compute_mode = param::Convolution::ComputeMode::DEFAULT;
        bench.set_param(param);
        bench.proxy()->target_execution_policy = {};
        TensorLayout src{{N, IC / 4, IH, IW, 4}, dtype::QuantizedS8{1.0f}},
                filter{{OC, IC / 4, FH, FH, 4}, dtype::QuantizedS8{1.0f}};
        TensorLayout dst;
        dst.dtype = dtype::QuantizedS8{1.0f};
        {
            auto&& opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = param;
            opr->deduce_layout(src, filter, dst);
        }
        auto used = bench.execl({filter, dst, src}) / RUNS;
        float flo = 2.0 * N * OC * IC * dst[2] * dst[3] * FH * FH;
        printf("inp=%s, kern=%s, dst=%s ", src.to_string().c_str(),
               filter.to_string().c_str(), dst.to_string().c_str());
        printf("time_fp32=%.2fms, flops=%.3fTFLOPS\n", used, (flo / (used * 1e9)));
    };
    run(64, 32, 32, 92, 180, 4, 2, 2);
    run(64, 32, 32, 46, 80, 4, 2, 2);
    run(16, 16, 16, 92, 180, 4, 2, 2);
    run(16, 16, 16, 46, 80, 4, 2, 2);
}

TEST_F(CUDA, CONVOLUTION_BWD_FILTER_BENCHMARK) {
    CUBenchmarker<ConvolutionBackwardFilter> bench{handle_cuda()};
    std::unique_ptr<OprProxy<ConvolutionBackwardFilter>> proxy{
            new OprProxy<ConvolutionBackwardFilter>{true}};
    size_t RUNS = 10;
    bench.set_proxy(proxy).set_times(RUNS);

    auto run = [&](size_t N, size_t OC, size_t IC, size_t IH, size_t IW, size_t FH,
                   size_t SH, size_t PH) {
        bench.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32());
        param::Convolution param;
        param.stride_h = param.stride_w = SH;
        param.pad_h = param.pad_w = PH;
        param.compute_mode = param::Convolution::ComputeMode::DEFAULT;
        bench.set_param(param);
        bench.proxy()->target_execution_policy.algo.reset();
        TensorLayout src{{N, IC, IH, IW}, dtype::Float32()},
                filter{{OC, IC, FH, FH}, dtype::Float32()};
        TensorLayout dst;
        {
            auto&& opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = param;
            opr->deduce_layout(src, filter, dst);
        }
        auto time_ms_fp32 = bench.execl({src, dst, filter}) / RUNS;
        src.dtype = filter.dtype = dst.dtype = dtype::Float16();
        bench.proxy()->target_execution_policy.algo.reset();
        bench.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16());
        auto time_ms_true_fp16 = bench.execl({src, dst, filter}) / RUNS;
        param.compute_mode = param::Convolution::ComputeMode::FLOAT32;
        bench.proxy()->target_execution_policy.algo.reset();
        bench.set_param(param);
        auto time_ms_pseudo_fp16 = bench.execl({src, dst, filter}) / RUNS;
        float flo = 2.0 * N * OC * IC * dst[2] * dst[3] * FH * FH;
        printf("inp=%s, kern=%s, dst=%s ", src.to_string().c_str(),
               filter.to_string().c_str(), dst.to_string().c_str());
        printf("time_fp32=%.2fms, flops=%.3fTFLOPS\ntime_true_fp16=%.2fms, "
               "flops=%.3fTFLOPS\ntime_pseudo_fp16=%.2fms, flops=%.3fFLOPS\n",
               time_ms_fp32, (flo / (time_ms_fp32 * 1e9)), time_ms_true_fp16,
               (flo / (time_ms_true_fp16 * 1e9)), time_ms_pseudo_fp16,
               (flo / (time_ms_pseudo_fp16 * 1e9)));
        printf("speedup (true_fp16/fp32)=%.2f, (true_fp16/pseudo_fp16)=%.2f\n",
               time_ms_fp32 / time_ms_true_fp16,
               time_ms_pseudo_fp16 / time_ms_true_fp16);
    };
    run(32, 64, 3, 224, 224, 7, 2, 3);
    run(32, 128, 128, 28, 28, 3, 1, 1);
    run(32, 256, 256, 14, 14, 3, 1, 1);
    run(32, 512, 512, 7, 7, 3, 1, 1);
    run(32, 64, 64, 56, 56, 3, 1, 1);
    run(32, 512, 256, 56, 56, 1, 2, 0);
    run(32, 1024, 512, 28, 28, 1, 2, 0);
    run(32, 2048, 1024, 14, 14, 1, 2, 0);
    run(32, 512, 128, 28, 28, 1, 1, 0);
    run(32, 128, 512, 28, 28, 1, 1, 0);
    run(32, 1024, 256, 14, 14, 1, 1, 0);
    run(32, 256, 1024, 14, 14, 1, 1, 0);
    run(32, 2048, 512, 7, 7, 1, 1, 0);
    run(32, 512, 2048, 7, 7, 1, 1, 0);
    run(32, 256, 64, 56, 56, 1, 1, 0);
    run(32, 64, 256, 56, 56, 1, 1, 0);
    run(32, 128, 256, 56, 56, 1, 2, 0);
    run(32, 256, 512, 28, 28, 1, 2, 0);
    run(32, 512, 1024, 14, 14, 1, 2, 0);
    run(32, 64, 64, 56, 56, 1, 1, 0);
}

TEST_F(CUDA, BENCHMARK_CONVOLUTION_BWD_FILTER_DEPTHWISE_LARGE_FILTER) {
    CUBenchmarker<ConvolutionBackwardFilter> bench{handle_cuda()};
    std::unique_ptr<OprProxy<ConvolutionBackwardFilter>> proxy{
            new OprProxy<ConvolutionBackwardFilter>{true}};
    size_t RUNS = 10;
    bench.set_proxy(proxy).set_times(RUNS);

    bench.set_before_exec_callback(AlgoChecker<ConvolutionBackwardFilter>(
            "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFTv7.6.3"));

    auto run = [&](size_t N, size_t OC, size_t g, size_t IH, size_t IW, size_t FH,
                   size_t SH, size_t PH) {
        bench.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32());
        param::Convolution param;
        param.stride_h = param.stride_w = SH;
        param.pad_h = param.pad_w = FH / 2;
        param.sparse = param::Convolution::Sparse::GROUP;
        bench.set_param(param);
        bench.proxy()->target_execution_policy.algo.reset();
        TensorLayout src{{N, g, IH, IW}, dtype::Float32()},
                filter{{g, 1, 1, FH, FH}, dtype::Float32()};
        TensorLayout dst;
        {
            auto&& opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = param;
            opr->deduce_layout(src, filter, dst);
        }
        auto time_ms_fp32 = bench.execl({src, dst, filter}) / RUNS;
        float flo = 2.0 * N * g * dst[2] * dst[3] * FH * FH;
        printf("inp=%s, kern=%s, dst=%s ", src.to_string().c_str(),
               filter.to_string().c_str(), dst.to_string().c_str());
        printf("time_fp32=%.2fms, flops=%.3fTFLOPS\n", time_ms_fp32,
               (flo / (time_ms_fp32 * 1e9)));
    };
    run(64, 384, 384, 32, 32, 31, 1, 15);
}

#endif

#undef CUDNN_VERSION_STRING
#undef V
#undef V1

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
