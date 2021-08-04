/**
 * \file src/decryption/rc4/rc4_cryption_impl.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */
#pragma once
#include "rc4_cryption_base.h"

#include <memory>
#include <vector>

namespace lite {

class RC4Impl {
    struct RC4State {
        rc4::RC4RandStream enc_stream;
        rc4::RC4RandStream hash_stream;
    } m_state;

public:
    RC4Impl(const void* model_mem, size_t size, const std::vector<uint8_t>& key)
            : m_model_mem(model_mem), m_model_length(size) {
        const uint8_t* data = key.data();
        m_hash_key = *reinterpret_cast<const uint64_t*>(data);
        m_enc_key = *reinterpret_cast<const uint64_t*>(data + 8);
    }

    std::vector<uint8_t> encrypt_model();
    std::vector<uint8_t> decrypt_model();

    /*! \brief Read the input stream once in order to initialize the decryption
     *         state.
     */
    void init_rc4_state();

private:
    const void* m_model_mem;
    size_t m_model_length;

    uint64_t m_hash_key;
    uint64_t m_enc_key;
};

class SimpleFastRC4Impl {
    struct SFRC4State {
        rc4::RC4RandStream enc_stream;
        rc4::RC4RandStream hash_stream;
    } m_state;

public:
    SimpleFastRC4Impl(const void* model_mem, size_t size,
                      const std::vector<uint8_t>& key)
            : m_model_mem(model_mem), m_model_length(size) {
        const uint8_t* data = key.data();
        m_hash_key = *reinterpret_cast<const uint64_t*>(data);
        m_enc_key = *reinterpret_cast<const uint64_t*>(data + 8);
    }
    std::vector<uint8_t> encrypt_model();
    std::vector<uint8_t> decrypt_model();

    /*! \brief Read the input stream once in order to initialize the decryption
     *         state.
     */
    void init_sfrc4_state();

private:
    const void* m_model_mem;
    size_t m_model_length;

    uint64_t m_hash_key;
    uint64_t m_enc_key;
};

}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
