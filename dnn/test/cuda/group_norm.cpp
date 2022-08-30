#include "test/cuda/fixture.h"

#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, GROUPNORM_FORWARD) {
    using Param = GroupNormForward::Param;
    Param param;
    param.affine = true;
    param.eps = 1e-6;
    Checker<GroupNormForward> checker(handle_cuda());
    checker.set_epsilon(1e-2);

    auto run = [&](DType d) {
        for (size_t group : {1, 3})
            for (size_t C : {6, 9}) {
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
    run(dtype::Float16());
    run(dtype::BFloat16());
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
