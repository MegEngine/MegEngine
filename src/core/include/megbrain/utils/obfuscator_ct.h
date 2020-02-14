/**
 * \file src/core/include/megbrain/utils/obfuscator_ct.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace mgb {

#if defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

template <char KEY>
struct XORObfuscator {
    static_assert(KEY != '\0', "KEY must not be the \0 character.");

    static constexpr char encrypt(char ch) { return ch ^ KEY; }
    static constexpr char decrypt(char ch) { return ch ^ KEY; }
};

/*!
 * \brief Obfuscates the string 'data' at compile-time and returns a reference
 * to a object with global lifetime that implicitly convertable to a char*
 *
 * \param Indexes string indexes
 * \param Obfucator the obfuscator used to encrypt/decrypt data
 */
template <typename Indexes, typename Obfucator>
class ObfucatorCT;

template <size_t... I, typename Obfucator>
class ObfucatorCT<std::index_sequence<I...>, Obfucator> {
public:
    constexpr ALWAYS_INLINE ObfucatorCT(const char* data)
            : m_buffer{Obfucator::encrypt(data[I])...} {}

    ALWAYS_INLINE std::string decrypt() {
        std::string ret;
        for (size_t i = 0; i < sizeof...(I); ++i) {
            ret.push_back(Obfucator::decrypt(m_buffer[i]));
        }
        return ret;
    }

private:
    //! "volatile" is important to avoid uncontrolled over-optimization by the
    //! compiler
    volatile char m_buffer[sizeof...(I) + 1]{};
};

#undef ALWAYS_INLINE

}  // namespace mgb

#define MGB_OBFUSCATE_STR(data) \
    MGB_OBFUSCATE_STR_KEY(data, mgb::XORObfuscator<'.'>)

#define MGB_OBFUSCATE_STR_KEY(data, ob)                                    \
    mgb::ObfucatorCT<std::make_index_sequence<sizeof(data) - 1>, ob>(data) \
            .decrypt()

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
