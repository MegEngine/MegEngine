#include "megdnn/dtype.h"
#include "test/atlas/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "src/atlas/handle.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

using namespace megdnn;
using namespace test;
using namespace conv_bias;

// TODO: precision for float16 is a little low
TEST_F(ATLAS, CONV_BIAS_FORWARD_FLOAT) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_args();
    Checker<ConvBiasForward> checker(handle_atlas());

    NormalRNG default_rng;
    for (auto&& arg : args) {
        // TODO: support other mode.
        arg.param.nonlineMode = NonlineMode::IDENTITY;
        arg.param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_dtype(3, dtype::Float32())
                .set_dtype(4, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_rng(2, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, arg.bias, {}, {}});
        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_dtype(3, dtype::Float16())
                .set_dtype(4, dtype::Float16())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_rng(2, &default_rng)
                .set_epsilon(5e-2)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, arg.bias, {}, {}});

        size_t src_h = arg.src[2], src_w = arg.src[3], filter_h = arg.filter[2],
               filter_w = arg.filter[3];
        size_t n = arg.src[0], oc = arg.filter[0];
        size_t pad_h = arg.param.pad_h, pad_w = arg.param.pad_w;
        size_t stride_h = arg.param.stride_h, stride_w = arg.param.stride_w;
        size_t dst_h = (src_h - filter_h + 2 * pad_h) / stride_h + 1,
               dst_w = (src_w - filter_w + 2 * pad_w) / stride_w + 1;
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_dtype(3, dtype::Float32())
                .set_dtype(4, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_rng(2, &default_rng)
                .set_rng(3, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src,
                        arg.filter,
                        arg.bias,
                        TensorShape{n, oc, dst_h, dst_w},
                        {}});
        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_dtype(3, dtype::Float16())
                .set_dtype(4, dtype::Float16())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_rng(2, &default_rng)
                .set_rng(3, &default_rng)
                .set_epsilon(6e-2)
                .set_param(arg.param)
                .execs({arg.src,
                        arg.filter,
                        arg.bias,
                        TensorShape{n, oc, dst_h, dst_w},
                        {}});
    }
}

TEST_F(ATLAS, CONV_BIAS_FORWARD_GROUP) {
    using NLMode = ConvBias::Param::NonlineMode;

    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW, size_t FH, size_t FW,
                   size_t OC, size_t PH, size_t PW, size_t SH, size_t SW, size_t DH,
                   size_t DW, size_t group, NLMode mode) {
        {
            // TODO: support other mode.
            mode = NonlineMode::IDENTITY;
            // float case
            Checker<ConvBiasForward> checker(handle_atlas());
            ConvBias::Param param;
            param.sparse = ConvBias::Param::Sparse::GROUP;
            param.nonlineMode = mode;
            param.pad_h = PH;
            param.pad_w = PW;
            param.stride_h = SH;
            param.stride_w = SW;
            param.dilate_h = DH;
            param.dilate_w = DW;
            auto ICg = IC / group;
            auto OCg = OC / group;
            auto OH = (IH - FH - (FH - 1) * (DH - 1) + 2 * PH) / SH + 1;
            auto OW = (IW - FW - (FW - 1) * (DW - 1) + 2 * PW) / SW + 1;
            checker.set_param(param).exec(
                    {{N, IC, IH, IW},
                     {group, OCg, ICg, FH, FW},
                     {1, OCg * group, 1, 1},
                     {},
                     {}});
            checker.set_param(param).exec(
                    {{N, IC, IH, IW},
                     {group, OCg, ICg, FH, FW},
                     {1, OCg * group, 1, 1},
                     {N, OCg * group, OH, OW},
                     {}});
        }
    };

    for (NLMode nlmode :
         {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID, NLMode::H_SWISH}) {
        // normal case
        run(2, 64, 7, 7, 3, 3, 32, 0, 0, 1, 1, 1, 1, 2, nlmode);
        // padded case
        run(2, 32, 7, 7, 3, 3, 64, 1, 1, 1, 1, 1, 1, 4, nlmode);
        // strided case
        run(2, 32, 7, 7, 3, 3, 64, 0, 0, 2, 2, 1, 1, 8, nlmode);
        // dilated case
        run(2, 32, 7, 7, 3, 3, 64, 0, 0, 1, 1, 2, 2, 8, nlmode);
    }
}

TEST_F(ATLAS, CONV_BIAS_FORWARD_DILATED) {
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW, size_t FH, size_t FW,
                   size_t OC, size_t PH, size_t PW, size_t SH, size_t SW, size_t DH,
                   size_t DW) {
        {
            // float case
            Checker<ConvBiasForward> checker(handle_atlas());
            ConvBias::Param param;
            param.sparse = ConvBias::Param::Sparse::DENSE;
            param.pad_h = PH;
            param.pad_w = PW;
            param.stride_h = SH;
            param.stride_w = SW;
            param.dilate_h = DH;
            param.dilate_w = DW;
            param.nonlineMode = ConvBias::Param::NonlineMode::IDENTITY;
            auto OH = (IH - FH - (FH - 1) * (DH - 1) + 2 * PH) / SH + 1;
            auto OW = (IW - FW - (FW - 1) * (DW - 1) + 2 * PW) / SW + 1;
            checker.set_param(param).exec(
                    {{N, IC, IH, IW}, {OC, IC, FH, FW}, {1, OC, 1, 1}, {}, {}});
            checker.set_param(param).exec(
                    {{N, IC, IH, IW},
                     {OC, IC, FH, FW},
                     {1, OC, 1, 1},
                     {N, OC, OH, OW},
                     {}});
        }
    };

    // dilated case
    run(2, 8, 7, 7, 3, 3, 4, 0, 0, 1, 1, 2, 2);
}

TEST_F(ATLAS, CONV_BIAS_FORWARD_1X1) {
    using namespace conv_bias;
    Checker<ConvBiasForward> checker(handle_atlas());
    using NLMode = param::ConvBias::NonlineMode;
    param::ConvBias cur_param;
    cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
    cur_param.format = param::ConvBias::Format::NCHW;
    // TODO: support other mode.
    for (auto nonlineMode : {NLMode::IDENTITY}) {
        for (size_t n : {1, 2}) {
            for (size_t ic : {8, 16, 64, 128}) {
                for (size_t oc : {32, 256}) {
                    cur_param.nonlineMode = nonlineMode;
                    checker.set_dtype(0, dtype::Float32())
                            .set_dtype(1, dtype::Float32())
                            .set_dtype(2, dtype::Float32())
                            .set_dtype(3, dtype::Float32())
                            .set_dtype(4, dtype::Float32());
                    checker.set_param(cur_param);
                    //! bias
                    checker.execs(
                            {{n, ic, 28, 28},
                             {oc, ic, 1, 1},
                             {1, oc, 1, 1},
                             {},
                             {n, oc, 28, 28}});
                    //! z
                    checker.execs(
                            {{n, ic, 28, 28},
                             {oc, ic, 1, 1},
                             {1, oc, 1, 1},
                             {n, oc, 28, 28},
                             {n, oc, 28, 28}});
                }
            }
        }
    }
}

// vim: syntax=cpp.doxygen
