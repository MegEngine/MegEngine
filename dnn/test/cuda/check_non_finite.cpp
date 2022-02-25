#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, CHECK_NON_FINITE_BASIC) {
    Checker<CheckNonFinite> checker(handle_cuda());
    checker.set_allow_invalid_check(true);
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    UniformFloatWithValueRNG rng(-1.0f, 1.0f, 0.1f, inf);
    checker.set_rng(0, &rng);
    //! while deduce layout, dst tensor dtype will be set to Int32
    checker.execs({{512 * 4}, {4}, {}});
    rng = UniformFloatWithValueRNG(-1.0f, 1.0f, 1.f, inf);
    checker.set_rng(0, &rng);
    checker.execs({{4}, {512 * 4}, {}});
    rng = UniformFloatWithValueRNG(-1.0f, 1.0f, 1.f, nan);
    checker.set_rng(0, &rng);
    checker.execs({{32}, {256}, {}});
    rng = UniformFloatWithValueRNG(-1.0f, 1.0f, 0.f, nan);
    checker.set_rng(0, &rng);
    checker.execs({{16}, {16}, {2}, {}});
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
