#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/cambricon/fixture.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CAMBRICON, GROUPNORM_FORWARD) {
    using Param = GroupNormForward::Param;
    Param param;
    param.affine = true;
    param.eps = 1e-6;
    Checker<GroupNormForward> checker(handle_cambricon());
    checker.set_epsilon(1e-2);

    // cambricon_inv_std = 1 / (naive_std * naive_std)
    auto reset_batch_inv_variancep = [](const TensorNDArray& tensors) {
        auto batch_inv_variance = tensors.back();
        auto size_in_bytes = batch_inv_variance.layout.span().dist_byte();
        memset(batch_inv_variance.raw_ptr(), 0, size_in_bytes);
    };
    checker.set_output_canonizer(reset_batch_inv_variancep);

    auto run = [&](DType d) {
        for (size_t group : {1})
            for (size_t C : {6}) {
                param.group = group;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, d)
                        .set_dtype(4, dtype::Float32())
                        .set_dtype(5, dtype::Float32())
                        .execs({{2, C, 2, 1},
                                {C},
                                {C},
                                {2, C, 2, 1},
                                {2, group},
                                {2, group}});
            }
    };

    run(dtype::Float32());
    // the mean and std must be float
    // run(dtype::Float16());
    // run(dtype::BFloat16());
}

}  // namespace test
}  // namespace megdnn