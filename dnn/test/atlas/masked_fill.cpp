#include "test/atlas/fixture.h"

#include "test/common/checker.h"
#include "test/common/rng.h"

namespace megdnn {
namespace test {

TEST_F(ATLAS, MASKEDFILL) {
    Checker<MaskedFill> checker(handle_atlas(), true);
    using Param = MaskedFill::Param;
    Param param;
    param.value = 1.0;
    checker.set_epsilon(1e-2);

    UniformFloatRNG ui_rng{-1000.0f, 1000.0f};
    BoolRNG bool_rng(0);
    checker.set_rng(0, &ui_rng);
    checker.set_rng(1, &bool_rng);
    checker.set_rng(2, &ui_rng);

    auto run = [&](DType d) {
        for (size_t A : {2, 3})
            for (size_t B : {6, 9}) {
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, dtype::Bool())
                        .set_dtype(2, d)
                        .execs({{A, B, 2, 1}, {A, B}, {A, B, 2, 1}});
            }
        for (size_t A : {2, 3})
            for (size_t B : {6, 9}) {
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, dtype::Bool())
                        .set_dtype(2, d)
                        .execs({{A, B, 2, 1}, {A, B, 2, 1}, {A, B, 2, 1}});
            }
    };

    run(dtype::Float32());
    run(dtype::Int8());
}

}  // namespace test
}  // namespace megdnn