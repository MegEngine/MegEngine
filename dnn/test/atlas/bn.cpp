#include "megdnn/oprs.h"
#include "megdnn/oprs/nn.h"
#include "test/atlas/fixture.h"

#include "test/common/bn.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

namespace megdnn {
namespace test {

TEST_F(ATLAS, BN_FORWARD_BACKWARD) {
    using namespace batch_normalization;
    std::vector<TestArg> args = get_args();
    Checker<BNForward> checker_fwd(handle_atlas());
    Checker<BNBackward> checker_bwd(handle_atlas());
    for (auto&& arg : args) {
        // atlas only support (1,C,1,1)
        if (arg.param.param_dim != param::BN::ParamDim::DIM_1C11) {
            continue;
        }

        arg.param.epsilon = 1e-3;
        size_t reserve = 0;
        // Forward
        for (int i = 0; i < 9; ++i) {
            checker_fwd.set_dtype(i, dtype::Float32());
        }
        checker_fwd.set_dtype(0, arg.dtype);
        checker_fwd.set_dtype(7, dtype::Byte());
        checker_fwd.set_dtype(8, arg.dtype);
        checker_fwd.set_bypass(7);
        checker_fwd.set_epsilon(1e-3);
        UniformFloatRNG rng(1, 100);
        checker_fwd.set_rng(4, &rng);
        using FMode = param::BN::FwdMode;
        for (auto mode : {FMode::TRAINING, FMode::INFERENCE}) {
            arg.param.fwd_mode = mode;
            checker_fwd.set_param(arg.param);
            checker_fwd.exec({
                    arg.src,
                    arg.param_shape,  // bn_scale
                    arg.param_shape,  // bn_bias
                    arg.param_shape,  // mean
                    arg.param_shape,  // variance
                    arg.param_shape,  // batch_mean
                    arg.param_shape,  // batch_inv_variance
                    {reserve},        // reserve
                    arg.src           // dst
            });
        }

        // Backward
        arg.param.fwd_mode = FMode::TRAINING;
        for (int i = 0; i < 9; ++i) {
            checker_bwd.set_dtype(i, dtype::Float32());
        }
        checker_bwd
                .set_dtype(0, arg.dtype)      // x
                .set_dtype(1, arg.dtype)      // dy
                .set_dtype(5, dtype::Byte())  // reserve
                .set_dtype(8, arg.dtype)      // dx
                .set_bypass(5);
        checker_bwd.set_epsilon(1e-2).set_param(arg.param).exec(
                {arg.src,
                 arg.src,
                 arg.param_shape,
                 arg.param_shape,
                 arg.param_shape,
                 {reserve},
                 arg.param_shape,
                 arg.param_shape,
                 arg.src});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
