#include "./xxhash/xxhash.h"

#include <algorithm>
#include "megbrain/utils/hash.h"

using namespace mgb;

XXHash::XXHash() {
    reset();
}

void XXHash::reset() {
    static_assert(sizeof(m_state) == sizeof(XXH64_state_t), "bad state size");
    XXH64_reset(reinterpret_cast<XXH64_state_t*>(m_state), 0x4b4e74b36b5d11);
}

XXHash& XXHash::update(const void* addr, size_t len) {
    XXH64_update(reinterpret_cast<XXH64_state_t*>(m_state), addr, len);
    return *this;
}

uint64_t XXHash::digest() const {
    return std::max<uint64_t>(
            XXH64_digest(reinterpret_cast<const XXH64_state_t*>(m_state)), 1);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
