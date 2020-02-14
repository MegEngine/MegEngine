/**
 * \file test/src/rng_seed.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./rng_seed.h"

#include "megbrain/common.h"
#include "megbrain/utils/hash.h"

#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace mgb;

RNGSeedManager& RNGSeedManager::inst() {
    static RNGSeedManager inst;
    return inst;
}

RNGSeedManager::RNGSeedManager() {
    if (getenv("MGB_STABLE_RNG")) {
        mgb_log_warn("use stable rand seed");
        m_stable = true;
        m_next_seed = 0;
    } else {
        m_stable = false;
        m_next_seed = time(nullptr);
    }
}

void RNGSeedManager::OnTestStart(const ::testing::TestInfo& test_info) {
    if (m_stable) {
        auto cname = test_info.test_case_name(), tname = test_info.name();
        m_next_seed = mgb::XXHash{}
                              .update(cname, strlen(cname))
                              .update(".", 1)
                              .update(tname, strlen(tname))
                              .digest();
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
