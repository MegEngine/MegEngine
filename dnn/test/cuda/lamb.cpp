#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/common/rng.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, LAMBUpdate) {
    LAMBUpdate::Param param;
    param.beta_1 = 0.9;
    param.beta_2 = 0.999;
    param.eps = 1e-5;
    param.weight_decay = 0.4;
    param.lr = 1e-3;
    param.step = 1;
    param.bias_correction = true;
    param.always_adapt = false;

    Checker<LAMBUpdate> checker(handle_cuda());
    checker.set_epsilon(1e-3);
    UniformFloatRNG rng0(0, 1);

    auto run = [&](DType d) {
        checker.set_param(param)
                .set_rng(0, &rng0)
                .set_rng(1, &rng0)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_dtype(3, d)
                .set_dtype(4, dtype::Float32())
                .set_dtype(5, dtype::Float32())
                .set_dtype(6, dtype::Float32())
                .execs({{2}, {2}, {2}, {2}, {}, {}, {}});
    };

    run(dtype::Float32());
    run(dtype::Float16());
    run(dtype::BFloat16());
}

}  // namespace test
}  // namespace megdnn
