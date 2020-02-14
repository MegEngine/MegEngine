/**
 * \file src/core/impl/utils/hash.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./xxhash/xxhash.h"

#include "megbrain/utils/hash.h"
#include <algorithm>

using namespace mgb;

XXHash::XXHash() {
    reset();
}

void XXHash::reset() {
    static_assert(sizeof(m_state) == sizeof(XXH64_state_t),
            "bad state size");
    XXH64_reset(reinterpret_cast<XXH64_state_t*>(m_state),
            0x4b4e74b36b5d11);
}

XXHash& XXHash::update(const void *addr, size_t len) {
    XXH64_update(reinterpret_cast<XXH64_state_t*>(m_state),
            addr, len);
    return *this;
}

uint64_t XXHash::digest() const {
    return std::max<uint64_t>(
            XXH64_digest(reinterpret_cast<const XXH64_state_t*>(m_state)),
            1);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

