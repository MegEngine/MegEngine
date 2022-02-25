#include "./helper.h"
#include "megbrain/imperative/ops/autogen.h"

using namespace mgb;
using namespace imperative;

TEST(TestImperative, CondTake) {
    auto op = imperative::CondTake::make();
    auto msk = HostTensorGenerator<dtype::Bool>()({42});
    OprChecker(op).run({TensorShape{42}, *msk});
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
