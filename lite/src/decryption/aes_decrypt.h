/**
 * \file src/decryption/aes_decrypt.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#include "./mbedtls/aes.h"
#include "decrypt_base.h"

namespace lite {

class AESDcryption {
public:
    static std::vector<uint8_t> decrypt_model(const void* model_mem,
                                              size_t size,
                                              const std::vector<uint8_t>& key) {
        mbedtls_aes_context ctx;
        mbedtls_aes_init(&ctx);
        mbedtls_aes_setkey_dec(&ctx, key.data(), 256);

        auto data = static_cast<const uint8_t*>(model_mem);
        //! first 16 bytes is IV
        uint8_t iv[16];
        //! last 8 bytes is file size(length)
        auto length_ptr = data + size - 8;
        size_t length = 0;
        for (int i = 0; i < 8; i++) {
            length |= length_ptr[i] << (8 * (7 - i));
        }
        std::copy(data, data + 16, iv);
        auto output = std::vector<uint8_t>(size - 24);
        mbedtls_aes_crypt_cbc(&ctx, MBEDTLS_AES_DECRYPT, size - 24, iv,
                              data + 16, output.data());
        mbedtls_aes_free(&ctx);
        output.erase(output.begin() + length, output.end());
        return output;
    }

    static std::vector<uint8_t> get_decrypt_key() {
        std::vector<uint8_t> key = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                    0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14,
                                    0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B,
                                    0x1C, 0x1D, 0x1E, 0x1F};
        return key;
    }
};
}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
