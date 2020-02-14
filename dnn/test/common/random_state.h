/**
 * \file dnn/test/common/random_state.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <random>

namespace megdnn {
namespace test {

class RandomState {
public:
    static std::mt19937& generator() { return instance()->m_generator; }

    static void reset() { instance()->m_generator.seed(m_seed); }

private:
    RandomState() : m_generator(m_seed) {}
    std::mt19937 m_generator;
    const static int m_seed = 42;
    static RandomState* instance() { return &m_instance; }
    static RandomState m_instance;
};

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
