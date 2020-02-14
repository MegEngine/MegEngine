/**
 * \file test/src/rng_seed.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include <gtest/gtest.h>

#include <atomic>

namespace mgb {

/*!
 * \brief manage RNG seed
 *
 * The seed is reset on every test case if MGB_STABLE_RNG is set
 */
class RNGSeedManager : public ::testing::EmptyTestEventListener {
    bool m_stable;
    std::atomic_uint_fast64_t m_next_seed;

    RNGSeedManager();
    RNGSeedManager(const RNGSeedManager&) = delete;

    void OnTestStart(const ::testing::TestInfo& test_info) override;

public:
    static RNGSeedManager& inst();

    uint64_t next_seed() { return m_next_seed++; }

    void set_seed(uint64_t seed) {
        if (m_stable) {
            m_next_seed = seed;
        }
    }
};

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
