/**
 * \file imperative/src/test/imperative.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#include "./helper.h"
#include "megbrain/imperative/ops/cond_take.h"

using namespace mgb;
using namespace imperative;

TEST(TestImperative, CondTake) {
    auto op = imperative::CondTake::make();
    auto msk = HostTensorGenerator<dtype::Bool>()({42});
    OprChecker(op).run({TensorShape{42}, *msk});
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
