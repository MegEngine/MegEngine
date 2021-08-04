/**
 * \file src/decryption/rc4/rc4_cryption_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "rc4_cryption_impl.h"
#include "../../misc.h"

#include <cstring>

using namespace lite;

/*!
 * \brief Read the input stream once in order to initialize the decryption
 *        state.
 */
void RC4Impl::init_rc4_state() {
    rc4::RC4RandStream enc_stream(m_enc_key);
    rc4::FastHash64 dechash(m_hash_key);

    size_t offset = 0;

    std::vector<uint64_t> buffer(128);
    size_t remaining = m_model_length - sizeof(uint64_t);
    while (remaining > 0) {
        size_t toread = std::min(remaining, buffer.size() * sizeof(uint64_t));
        memcpy(buffer.data(), static_cast<const uint8_t*>(m_model_mem) + offset,
               toread);
        offset += toread;
        remaining -= toread;

        for (size_t i = 0; i < toread / sizeof(uint64_t); ++i) {
            uint64_t value = buffer[i];
            value ^= enc_stream.next64();
            dechash.feed(value);
        }
    }

    uint64_t hashvalue;
    memcpy(&hashvalue, static_cast<const uint8_t*>(m_model_mem) + offset,
           sizeof(hashvalue));
    offset += sizeof(hashvalue);

    hashvalue ^= dechash.get() ^ enc_stream.next64();
    m_state.hash_stream.reset(hashvalue);
    m_state.enc_stream.reset(m_enc_key);
}

std::vector<uint8_t> RC4Impl::decrypt_model() {
    std::vector<uint8_t> result(m_model_length, 0);

    uint8_t* ptr = result.data();
    for (size_t i = 0; i < m_model_length; ++i) {
        ptr[i] = static_cast<const uint8_t*>(m_model_mem)[i];
        ptr[i] ^= m_state.hash_stream.next8() ^ m_state.enc_stream.next8();
    }
    return result;
}

/*! \brief Encrypt the data in m_buffer.
 *
 * The basic idea is to calculate a 64-bit hash from the buffer and append
 * it to the end of the buffer. The basic requirement is that the change of
 * every byte including the hash value will destroy the whole model in every
 * byte.
 *
 * Encryption:
 *
 * 1. First calculate a 64-bit hash, called plain hash value, from the
 * buffer.
 * 2. Initialize a RC4 stream with the plain hash value.
 * 3. Obfuscate the model body with the RC4 stream defined in step 2.
 * 4. Calculate the hash value of the obfuscated model, called hash value
 *    after hashing.
 * 5. Encrypt the model body with a RC4 stream made from the encryption key.
 * 6. Bit-xor the hash value after hashing with the plain hash value, called
 *    mixed hash.
 * 7. Encrypt the mixed hash with the RC4 stream defined in step 5, called
 * the protected hash.
 * 8. Append the protected hash to the buffer.
 *
 * Decryption:
 * 1. Decrypt the model body with a RC4 stream made from the encryption key,
 *    which is the reverse of step 5 and 7 of encryption and get the mixed
 *    hash.
 * 2. Calculate the hash value of the decrypted model, which equals to the
 *    hash value after hashing in step 4 of encryption.
 * 3. Bit-xor the hash value after hashing and the mixed hash to get the
 * plain hash value, which is the reverse of step 6 of encryption.
 * 4. Un-obfuscate the model body with the plain hash value, which is the
 *    reverse of step 3 of encryption.
 *
 * Think:
 * 1. If any byte in the model body is broken, the hash value after hashing
 *    will be broken in step 2, and hence the plain hash value in step 3
 * will be also broken, and finally, the model body will be broken in
 * step 4.
 * 2. If the protected hash is broken, the plain hash value in step 3 will
 * be broken, and finally the model body will be broken.
 */
std::vector<uint8_t> RC4Impl::encrypt_model() {
    size_t total_length = (m_model_length + (sizeof(size_t) - 1)) /
                          sizeof(size_t) * sizeof(size_t);
    std::vector<uint8_t> pad_model(total_length, 0);
    memcpy(pad_model.data(), m_model_mem, m_model_length);

    // Calculate the hash of the model.
    rc4::FastHash64 plainhash(m_hash_key);
    uint64_t* ptr = reinterpret_cast<uint64_t*>(pad_model.data());
    size_t len = pad_model.size() / sizeof(uint64_t);

    for (size_t i = 0; i < len; ++i)
        plainhash.feed(ptr[i]);
    uint64_t plainhash_value = plainhash.get();

    // Encrypt the model.
    rc4::RC4RandStream hash_enc(plainhash_value);
    rc4::RC4RandStream outmost_enc(m_enc_key);
    rc4::FastHash64 afterhashenc_hash(m_hash_key);

    for (size_t i = 0; i < len; ++i) {
        uint64_t value = ptr[i] ^ hash_enc.next64();
        afterhashenc_hash.feed(value);
        ptr[i] = value ^ outmost_enc.next64();
    }

    uint64_t protected_hash =
            plainhash_value ^ afterhashenc_hash.get() ^ outmost_enc.next64();

    size_t end = pad_model.size();
    pad_model.resize(pad_model.size() + sizeof(uint64_t));
    ptr = reinterpret_cast<uint64_t*>(&pad_model[end]);
    *ptr = protected_hash;
    return pad_model;
}

/*!
 * \brief Read the input stream once in order to initialize the decryption
 *        state.
 */
void SimpleFastRC4Impl::init_sfrc4_state() {
    rc4::RC4RandStream enc_stream(m_enc_key);
    rc4::FastHash64 dechash(m_hash_key);

    size_t offset = 0;
    std::vector<uint64_t> buffer(128);
    size_t remaining = m_model_length - sizeof(uint64_t);
    while (remaining > 0) {
        size_t toread = std::min(remaining, buffer.size() * sizeof(uint64_t));
        memcpy(buffer.data(), static_cast<const uint8_t*>(m_model_mem) + offset,
               toread);
        offset += toread;
        remaining -= toread;

        for (size_t i = 0; i < toread / sizeof(uint64_t); ++i) {
            uint64_t value = buffer[i];
            dechash.feed(value);
        }
    }

    uint64_t hashvalue;
    memcpy(&hashvalue, static_cast<const uint8_t*>(m_model_mem) + offset,
           sizeof(hashvalue));

    offset += sizeof(hashvalue);

    /*! \brief test the hash_val. */
    if (hashvalue != dechash.get())
        LITE_THROW(
                "The checksum of the file cannot be verified. The file may "
                "be encrypted in the wrong algorithm or different keys.");

    m_state.hash_stream.reset(m_hash_key);
    m_state.enc_stream.reset(m_enc_key);
}

std::vector<uint8_t> SimpleFastRC4Impl::decrypt_model() {
    std::vector<uint8_t> result(m_model_length, 0);
    uint8_t* ptr = result.data();
    for (size_t i = 0; i < m_model_length; ++i) {
        ptr[i] = static_cast<const uint8_t*>(m_model_mem)[i];
        ptr[i] ^= m_state.enc_stream.next8();
    }
    return result;
}

std::vector<uint8_t> SimpleFastRC4Impl::encrypt_model() {
    size_t total_length = (m_model_length + (sizeof(size_t) - 1)) /
                          sizeof(size_t) * sizeof(size_t);
    std::vector<uint8_t> pad_model(total_length, 0);
    memcpy(pad_model.data(), m_model_mem, m_model_length);

    // Calculate the hash of the model.
    rc4::FastHash64 enchash(m_hash_key);
    uint64_t* ptr = reinterpret_cast<uint64_t*>(pad_model.data());
    size_t len = pad_model.size() / sizeof(uint64_t);

    // Encrypt the model.
    rc4::RC4RandStream out_enc(m_enc_key);
    for (size_t i = 0; i < len; ++i) {
        ptr[i] = ptr[i] ^ out_enc.next64();
        enchash.feed(ptr[i]);
    }

    uint64_t hash_value = enchash.get();

    size_t end = pad_model.size();
    pad_model.resize(pad_model.size() + sizeof(uint64_t));
    ptr = reinterpret_cast<uint64_t*>(&pad_model[end]);
    *ptr = hash_value;

    return pad_model;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
