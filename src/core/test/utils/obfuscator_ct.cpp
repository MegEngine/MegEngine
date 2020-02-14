/**
 * \file src/core/test/utils/obfuscator_ct.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/obfuscator_ct.h"
#include "megbrain/test/helper.h"

using namespace mgb;

TEST(TestObFucatorCT, Normal) {
    auto run = []() {
        std::string obfuscator_str = MGB_OBFUSCATE_STR("mgb0001");
        ASSERT_STREQ(obfuscator_str.c_str(), "mgb0001");
    };

    //! invoke twice
    run();
    run();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
