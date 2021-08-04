/**
 * \file src/decryption/rc4_cryption.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
