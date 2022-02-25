#include "test/cuda/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "src/cuda/batch_normalization/opr_impl.h"
#include "src/cuda/utils.h"
#include "test/common/bn.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, BN_FORWARD_BACKWARD) {
    using namespace batch_normalization;
    using cuda::cudnn_handle;
    using cuda::batch_normalization::BNTensorDescHolder;
    using cuda::batch_normalization::get_reserve_size;
    std::vector<TestArg> args = batch_normalization::get_args();
    Checker<BNForward> checker(handle_cuda());
    Checker<BNBackward> checker_bwd(handle_cuda());
    for (auto&& arg : args) {
        auto tensor_desc = BNTensorDescHolder(
                {arg.src, arg.dtype}, arg.param.param_dim, arg.param.fwd_mode);
        auto reserve = get_reserve_size(cudnn_handle(handle_cuda()), tensor_desc);
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
