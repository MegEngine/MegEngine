#include "test/common/convolution.h"
#include "megdnn/dtype.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "test/atlas/fixture.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

namespace megdnn {
namespace test {

TEST_F(ATLAS, CONVOLUTION_FORWARD) {
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    Checker<ConvolutionForward> checker(handle_atlas());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        if (arg.param.format == param::Convolution::Format::NHWC) {
            arg.param.format = param::Convolution::Format::NCHW;
            arg.src = {arg.src[0], arg.src[3], arg.src[1], arg.src[2]};
            arg.filter = {arg.filter[0], arg.filter[3], arg.filter[1], arg.filter[2]};
        }
        if (arg.param.format != param::Convolution::Format::NCHW) {
            continue;
        }
        arg.param.compute_mode = param::Convolution::ComputeMode::DEFAULT;
        arg.param.mode = param::Convolution::Mode::CROSS_CORRELATION;
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
                .set_epsilon(5e-2)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

TEST_F(ATLAS, CONVOLUTION_1X1_FORWARD) {
    using namespace convolution;
    std::vector<TestArg> args = get_1x1_args();
    Checker<ConvolutionForward> checker(handle_atlas());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        if (arg.param.format == param::Convolution::Format::NHWC) {
            arg.param.format = param::Convolution::Format::NCHW;
            arg.src = {arg.src[0], arg.src[3], arg.src[1], arg.src[2]};
            arg.filter = {arg.filter[0], arg.filter[3], arg.filter[1], arg.filter[2]};
        }
        if (arg.param.format != param::Convolution::Format::NCHW) {
            continue;
        }
        arg.param.compute_mode = param::Convolution::ComputeMode::DEFAULT;
        arg.param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

TEST_F(ATLAS, CONVOLUTION_BACKWARD_DATA) {
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    Checker<ConvolutionBackwardData> checker(handle_atlas());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        if (arg.param.format == param::Convolution::Format::NHWC) {
            arg.param.format = param::Convolution::Format::NCHW;
            arg.src = {arg.src[0], arg.src[3], arg.src[1], arg.src[2]};
            arg.filter = {arg.filter[0], arg.filter[3], arg.filter[1], arg.filter[2]};
        }
        if (arg.param.format != param::Convolution::Format::NCHW) {
            continue;
        }
        arg.param.compute_mode = param::Convolution::ComputeMode::DEFAULT;
        arg.param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        float scale = 64.f / sqrt(arg.filter[0] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_atlas()->create_operator<Convolution>();
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
                .set_epsilon(5e-2)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
    }
}

TEST_F(ATLAS, CONVOLUTION_BACKWARD_FILTER) {
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    Checker<ConvolutionBackwardFilter> checker(handle_atlas());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        if (arg.param.format == param::Convolution::Format::NHWC) {
            arg.param.format = param::Convolution::Format::NCHW;
            arg.src = {arg.src[0], arg.src[3], arg.src[1], arg.src[2]};
            arg.filter = {arg.filter[0], arg.filter[3], arg.filter[1], arg.filter[2]};
        }
        if (arg.param.format != param::Convolution::Format::NCHW) {
            continue;
        }
        arg.param.compute_mode = param::Convolution::ComputeMode::DEFAULT;
        arg.param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_atlas()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        float scale = 1.0f / sqrt(dst[2] * dst[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});
        // similar to cuda convolution_backward_filter, too large f16 array may
        // introduce significant error
        if (dst.total_nr_elems() >= 1000)
            continue;
        src.dtype = dst.dtype = filter.dtype = dtype::Float16();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});
    }
}
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
