#include "test/cambricon/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "src/cambricon/batch_normalization/opr_impl.h"
#include "src/cambricon/utils.h"
#include "test/common/bn.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

namespace megdnn {
namespace test {

TEST_F(CAMBRICON, BN_FORWARD_BACKWARD) {
    using namespace batch_normalization;
    std::vector<TestArg> args = batch_normalization::get_nhwc_args();
    Checker<BNForward> checker(handle_cambricon());
    Checker<BNBackward> checker_bwd(handle_cambricon());
    for (auto&& arg : args) {
        size_t reserve = 0;
        // Forward
        for (int i = 0; i < 9; ++i) {
            checker.set_dtype(i, dtype::Float32());
        }
        checker.set_dtype(0, arg.dtype);
        checker.set_dtype(7, dtype::Byte());
        checker.set_dtype(8, arg.dtype);
        checker.set_bypass(7);
        checker.set_epsilon(1e-3).set_param(arg.param);
        for (bool need_statistic : {false, true})
            checker.exec({
                    arg.src,
                    arg.param_shape,                                      // bn_scale
                    arg.param_shape,                                      // bn_bias
                    need_statistic ? arg.param_shape : TensorShape({0}),  // mean
                    need_statistic ? arg.param_shape : TensorShape({0}),  // variance
                    arg.param_shape,                                      // batch_mean
                    arg.param_shape,  // batch_inv_variance
                    {reserve},        // reserve
                    arg.src           // dst
            });

        // Backward
        for (int i = 0; i < 9; ++i) {
            checker_bwd.set_dtype(i, dtype::Float32());
        }
        checker_bwd
                .set_dtype(0, arg.dtype)      // x
                .set_dtype(1, arg.dtype)      // dy
                .set_dtype(5, dtype::Byte())  // reserve
                .set_dtype(8, arg.dtype)      // dx
                .set_bypass(5);
        checker_bwd.set_epsilon(1e-3).set_param(arg.param).exec(
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
