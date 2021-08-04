/**
 * \file src/decryption/rc4/rc4_cryption_base.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */
#pragma once

#include <algorithm>
#include <cstdint>

namespace lite {
namespace rc4 {

#define m256(x) static_cast<uint8_t>(x)

/*! \brief Pseudo-random byte stream for RC4.
 */
class RC4RandStream {
public:
    RC4RandStream() = default;

    RC4RandStream(uint64_t key) { reset(key); }

    void reset(uint64_t init_key) {
        i_ = j_ = 0;
        for (int i = 0; i < 256; i++)
            s_[i] = i;
        uint8_t j = 0;
        for (int i = 0; i < 256; i++) {
            j = j + s_[i] + m256(init_key >> ((i % 8) * 8));
            std::swap(s_[i], s_[j]);
        }
        // drop
        for (int i = 0; i < 768; i++) {
            next8();
        }
        for (int i = 0, t = next8(); i < t; i++) {
            next8();
        }
    }

    uint8_t next8() {
        i_++;
        uint8_t a = s_[i_];
        j_ += a;
        uint8_t b = s_[j_];
        s_[i_] = b;
        s_[j_] = a;
        uint8_t c = s_[m256((i_ << 5) ^ (j_ >> 3))] +
                    s_[m256((j_ << 5) ^ (i_ >> 3))];
        return (s_[m256(a + b)] + s_[c ^ 0xAA]) ^ s_[m256(j_ + b)];
    }

    uint64_t next64() {
        uint64_t rst;
        uint8_t* buf = reinterpret_cast<uint8_t*>(&rst);
        for (int i = 0; i < 8; i++) {
            buf[i] = next8();
        }
        return rst;
    }

private:
    uint8_t s_[256], i_ = 0, j_ = 0;
};
#undef m256

/*!
 * \brief fast and secure 64-bit hash
 * see https://code.google.com/p/fast-hash/
 */
class FastHash64 {
public:
    FastHash64(uint64_t seed)
            : hash_{seed},
              mul0_{key_gen_hash_mul0()},
              mul1_{key_gen_hash_mul1()} {}

    void feed(uint64_t val) {
        val ^= val >> 23;
        val *= mul0_;
        val ^= val >> 47;
        hash_ ^= val;
        hash_ *= mul1_;
    }

    uint64_t get() { return hash_; }

private:
    uint64_t hash_;
    const uint64_t mul0_, mul1_;

    static uint64_t key_gen_hash_mul0() {
        uint64_t rst;
        uint8_t volatile* buf = reinterpret_cast<uint8_t*>(&rst);
        buf[2] = 50;
        buf[3] = 244;
        buf[6] = 39;
        buf[1] = 92;
        buf[5] = 89;
        buf[4] = 155;
        buf[0] = 55;
        buf[7] = 33;
        return rst;
    }

    static uint64_t key_gen_hash_mul1() {
        uint64_t rst;
        uint8_t volatile* buf = reinterpret_cast<uint8_t*>(&rst);
        buf[6] = 3;
        buf[2] = 109;
        buf[7] = 136;
        buf[1] = 25;
        buf[5] = 85;
        buf[0] = 101;
        buf[4] = 242;
        buf[3] = 30;
        return rst;
    }
};

// The encryption keys are always inlined.
static inline uint64_t key_gen_enc_key() {
    uint64_t rst;
    uint8_t volatile* buf = reinterpret_cast<uint8_t*>(&rst);
    buf[4] = 120;
    buf[3] = 121;
    buf[7] = 122;
    buf[6] = 123;
    buf[0] = 124;
    buf[5] = 125;
    buf[2] = 126;
    buf[1] = 127;
    return rst;
}

static inline uint64_t key_gen_hash_key() {
    uint64_t rst;
    uint8_t volatile* buf = reinterpret_cast<uint8_t*>(&rst);
    buf[2] = 101;
    buf[5] = 102;
    buf[4] = 103;
    buf[7] = 104;
    buf[1] = 105;
    buf[3] = 106;
    buf[6] = 107;
    buf[0] = 108;
    return rst;
}
}  // namespace rc4
}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
