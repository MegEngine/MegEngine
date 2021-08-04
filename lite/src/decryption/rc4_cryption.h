/**
 * \file src/decryption/rc4_cryption.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */
#pragma once

#include "rc4/rc4_cryption_base.h"

#include <vector>

namespace lite {

class RC4 {
public:
    static std::vector<uint8_t> decrypt_model(const void* model_mem,
                                              size_t size,
                                              const std::vector<uint8_t>& key);

    static std::vector<uint8_t> encrypt_model(const void* model_mem,
                                              size_t size,
                                              const std::vector<uint8_t>& key);

    static std::vector<uint8_t> get_decrypt_key();
};

class SimpleFastRC4 {
public:
    static std::vector<uint8_t> decrypt_model(const void* model_mem,
                                              size_t size,
                                              const std::vector<uint8_t>& key);
    static std::vector<uint8_t> encrypt_model(const void* model_mem,
                                              size_t size,
                                              const std::vector<uint8_t>& key);

    static std::vector<uint8_t> get_decrypt_key();
};

}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
