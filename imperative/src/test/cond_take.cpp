/**
 * \file imperative/src/test/cond_take.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

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
