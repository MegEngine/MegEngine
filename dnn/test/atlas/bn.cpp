#include "test/cuda/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"

#include "megdnn/oprs.h"
#include "megdnn/oprs/nn.h"
#include "test/atlas/fixture.h"

#include "test/common/bn.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm.h"
#include "aclnnop/aclnn_batch_norm_backward.h"

namespace megdnn {
namespace test {

using namespace batch_normalization;
using namespace megdnn;

std::vector<TestArg> get_atlas_args() {
    std::vector<TestArg> args;

    for (size_t i = 4; i < 10; i *= 4) {
        param::BN param;
        param.fwd_mode = param::BN::FwdMode::TRAINING;
        param.param_dim = param::BN::ParamDim::DIM_1C11;
        args.emplace_back(
                param, TensorShape{3, 3, i, i}, TensorShape{1, 3, 1, 1},
                dtype::Float32());
        args.emplace_back(
                param, TensorShape{3, 3, i, i}, TensorShape{1, 3, 1, 1},
                dtype::Float16());
    }
    param::BN param;
    param.fwd_mode = param::BN::FwdMode::TRAINING;
    param.param_dim = param::BN::ParamDim::DIM_1C11;
    int C = 6;
    args.emplace_back(
            param, TensorShape{3, C, 3, 3}, TensorShape{1, C, 1, 1}, dtype::Float32());
    args.emplace_back(
            param, TensorShape{3, C, 3, 3}, TensorShape{1, C, 1, 1}, dtype::Float16());

    return args;
}

TEST_F(ATLAS, BN_FORWARD) {
    std::vector<TestArg> args = get_atlas_args();
    Checker<BNForward> checker(handle_atlas());
    for (auto&& arg : args) {
        for (int i = 0; i < 8; ++i) {
            checker.set_dtype(i, dtype::Float32());
        }
        checker.set_dtype(0, arg.dtype);
        checker.set_dtype(8, arg.dtype);
        checker.set_epsilon(1e-3).set_param(arg.param);
        for (bool need_statistic : {true})
            checker.exec({
                    arg.src,
                    arg.param_shape,                                      // bn_scale
                    arg.param_shape,                                      // bn_bias
                    need_statistic ? arg.param_shape : TensorShape({0}),  // mean
                    need_statistic ? arg.param_shape : TensorShape({0}),  // variance
                    arg.param_shape,                                      // batch_mean
                    arg.param_shape,  // batch_inv_variance
                    {1},              // reserve
                    arg.src           // dst
            });
    }
}

TEST_F(ATLAS, BN_BACKWARD) {
    std::vector<TestArg> args = get_atlas_args();
    Checker<BNBackward> checker(handle_atlas());
    UniformFloatRNG ui_rng{.1f, 1000.f};
    checker.set_rng(3, &ui_rng);

    for (auto&& arg : args) {
        for (int i = 0; i < 9; ++i) {
            checker.set_dtype(i, dtype::Float32());
        }
        checker.set_dtype(0, arg.dtype)    // x
                .set_dtype(1, arg.dtype)   // dy
                .set_dtype(8, arg.dtype);  // dx
        checker.set_epsilon(1e-3).set_param(arg.param).exec(
                {arg.src,
                 arg.src,
                 arg.param_shape,
                 arg.param_shape,
                 arg.param_shape,
                 {1},
                 arg.param_shape,
                 arg.param_shape,
                 arg.src});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
