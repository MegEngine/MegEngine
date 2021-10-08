/**
 * \file dnn/test/common/rng.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/rng.h"

#include <gtest/gtest.h>
#include "test/common/random_state.h"
#include "test/common/tensor.h"

using namespace megdnn;
using namespace test;

/*!
 * \brief xorshift+ RNG, which is very fast
 *
 * see https://en.wikipedia.org/wiki/Xorshift#xorshift.2B
 */
class RNG::RNGxorshf {
    uint64_t s[2];

public:
    using result_type = uint64_t;

#ifdef WIN32
    static uint64_t min() { return 0; }
    static uint64_t max() { return std::numeric_limits<uint64_t>::max(); }
#else
    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return std::numeric_limits<uint64_t>::max(); }
#endif

    template <typename T>
    explicit RNGxorshf(T&& gen) {
        s[0] = gen();
        s[1] = gen();
    }

    uint64_t operator()() {
        uint64_t x = s[0];
        uint64_t const y = s[1];
        s[0] = y;
        x ^= x << 23;                          // a
        s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);  // b, c
        return s[1] + y;
    }
};

Float16PeriodicalRNG::Float16PeriodicalRNG() : m_offset(0) {
    for (size_t x = 0; x < (1u << 16); ++x) {
        size_t exponent = (x >> 10) & 0x1F;
        if (exponent == 0x1F) {
            // +inf, -inf, NaN
            continue;
        }
        union U {
            U() {}
            uint16_t i;
            dt_float16 f;
        } i2f;
        i2f.i = static_cast<uint16_t>(x);
        m_sequence.push_back(i2f.f);
    }
    COMPAT_RANDOM(m_sequence.begin(), m_sequence.end());
}

Float16PeriodicalRNG::Float16PeriodicalRNG(size_t range) : m_offset(0) {
    union U {
        U() {}
        uint16_t i;
        dt_float16 f;
    } i2f;
    size_t x = 0;
    i2f.i = static_cast<uint16_t>(x);
    for (size_t i = 0; i < range; i++) {
        x += 1;
        i2f.i = static_cast<uint16_t>(x);
        m_sequence.push_back(i2f.f);
    }
    x = 1u << 15;
    i2f.i = static_cast<uint16_t>(x);
    for (size_t i = 0; i < range; i++) {
        x += 1;
        i2f.i = static_cast<uint16_t>(x);
        m_sequence.push_back(i2f.f);
    }

    COMPAT_RANDOM(m_sequence.begin(), m_sequence.end());
}

void Float16PeriodicalRNG::gen(const TensorND& tensor) {
    megdnn_assert(tensor.layout.dtype == dtype::Float16());
    size_t nr_elems = tensor.layout.span().dist_elem();
    auto offset = tensor.layout.span().low_elem;
    for (size_t i = 0; i < nr_elems; ++i) {
        tensor.ptr<dt_float16>()[offset + i] = get_single_val();
    }
}

dt_float16 Float16PeriodicalRNG::get_single_val() {
    if (m_offset >= m_sequence.size()) {
        m_offset = 0;
    }
    return m_sequence[m_offset++];
}

void IIDRNG::gen(const TensorND& tensor) {
    if (tensor.layout.dtype == dtype::Float32() && has_fast_float32() &&
        tensor.layout.is_physical_contiguous()) {
        fill_fast_float32(tensor.ptr<dt_float32>(), tensor.layout.total_nr_elems());
        return;
    }

    auto offset = tensor.layout.span().low_elem;
    auto nr_elems = tensor.layout.span().dist_elem();
#define cb(DType)                                                   \
    if (tensor.layout.dtype == DType()) {                           \
        using ctype = typename DTypeTrait<DType>::ctype;            \
        auto ptr = tensor.ptr<ctype>();                             \
        for (size_t i = 0; i < nr_elems; ++i) {                     \
            ptr[offset + i] = static_cast<ctype>(gen_single_val()); \
        }                                                           \
        return;                                                     \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
#define cb(DType)                                                              \
    if (tensor.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {             \
        using ctype = typename DTypeTrait<DType>::ctype;                       \
        auto ptr = tensor.ptr<ctype>();                                        \
        if (output_is_float()) {                                               \
            for (size_t i = 0; i < nr_elems; ++i) {                            \
                ptr[offset + i] = tensor.layout.dtype.param<DType>().quantize( \
                        static_cast<float>(gen_single_val()));                 \
            }                                                                  \
        } else {                                                               \
            for (size_t i = 0; i < nr_elems; ++i) {                            \
                ptr[offset + i] = static_cast<ctype>(gen_single_val());        \
            }                                                                  \
        }                                                                      \
        return;                                                                \
    }
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
    //! In order to avoid an unnecessary increase in binary size, we just
    //! use QuantizedS16 dtype in winograd_filter_preprocess now.
    cb(::megdnn::dtype::QuantizedS16)
#undef cb
            if (tensor.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        auto ptr = static_cast<uint8_t*>(tensor.raw_ptr);
        if (output_is_float()) {
            for (size_t i = 0; i < nr_elems; i += 2) {
                uint8_t val0 = tensor.layout.dtype.param<dt_quint4>()
                                       .quantize(static_cast<float>(gen_single_val()))
                                       .as_uint8();
                uint8_t val1 = tensor.layout.dtype.param<dt_quint4>()
                                       .quantize(static_cast<float>(gen_single_val()))
                                       .as_uint8();
                ptr[(offset + i) / 2] = (val1 << 4) | val0;
            }
        } else {
            for (size_t i = 0; i < nr_elems; i += 2) {
                uint8_t val0 = static_cast<uint8_t>(gen_single_val());
                uint8_t val1 = static_cast<uint8_t>(gen_single_val());
                ptr[(offset + i) / 2] = (val1 << 4) | val0;
            }
        }
        return;
    }
    if (tensor.layout.dtype.enumv() == DTypeEnum::QuantizedS4) {
        auto ptr = static_cast<int8_t*>(tensor.raw_ptr);
        if (output_is_float()) {
            for (size_t i = 0; i < nr_elems; i += 2) {
                int8_t val0 = tensor.layout.dtype.param<dt_qint4>()
                                      .quantize(static_cast<float>(gen_single_val()))
                                      .as_int8();
                int8_t val1 = tensor.layout.dtype.param<dt_qint4>()
                                      .quantize(static_cast<float>(gen_single_val()))
                                      .as_int8();
                ptr[(offset + i) / 2] = (val0 & 0xF) | (val1 << 4);
            }
        } else {
            for (size_t i = 0; i < nr_elems; i += 2) {
                int8_t val0 = static_cast<int8_t>(gen_single_val());
                int8_t val1 = static_cast<int8_t>(gen_single_val());

                val0 = std::min(val0, DTypeTrait<dtype::QuantizedS4>::max());
                val0 = std::max(val0, DTypeTrait<dtype::QuantizedS4>::min());
                val1 = std::min(val1, DTypeTrait<dtype::QuantizedS4>::max());
                val1 = std::max(val1, DTypeTrait<dtype::QuantizedS4>::min());
                ptr[(offset + i) / 2] = (val0 & 0xF) | (val1 << 4);
            }
        }
        return;
    }
    if (tensor.layout.dtype.enumv() == DTypeEnum::Byte) {
        memset(tensor.raw_ptr, 0, tensor.layout.access_bytes());
        return;
    }
    if (tensor.layout.dtype.enumv() == DTypeEnum::Uint16) {
        return;
    }
    megdnn_assert(
            0, "IIDRNG does not know how to generate value for DType %s",
            tensor.layout.dtype.name());
}

bool IIDRNG::has_fast_float32() {
    return false;
}

void IIDRNG::fill_fast_float32(dt_float32*, size_t) {
    megdnn_assert(0);
}

dt_float32 NormalRNG::gen_single_val() {
    auto&& gen = RandomState::generator();
    return m_dist(gen);
}

bool NormalRNG::has_fast_float32() {
    return true;
}

void NormalRNG::fill_fast_float32(dt_float32* dest, size_t size) {
    RNGxorshf gen{RandomState::generator()};
    for (size_t i = 0; i < size; ++i) {
        dest[i] = m_dist(gen);
    }
}

void ConstValue::fill_fast_float32(dt_float32* dest, size_t size) {
    for (size_t i = 0; i < size; ++i)
        dest[i] = value_;
}

dt_float32 UniformIntRNG::gen_single_val() {
    auto&& gen = RandomState::generator();
    return static_cast<dt_float32>(m_dist(gen));
}

dt_float32 UniformIntNonZeroRNG::gen_single_val() {
    auto&& gen = RandomState::generator();
    auto ret = UniformIntRNG::gen_single_val();
    if (m_dist_flip(gen)) {
        ret = -ret;
    }
    megdnn_assert(ret != 0);
    return ret;
}

dt_float32 UniformFloatRNG::gen_single_val() {
    auto&& gen = RandomState::generator();
    return m_dist(gen);
}

bool UniformFloatRNG::has_fast_float32() {
    return true;
}

void UniformFloatRNG::fill_fast_float32(dt_float32* dest, size_t size) {
    RNGxorshf gen{RandomState::generator()};
    auto k = double(m_dist.b() - m_dist.a()) /
             double(RNGxorshf::max() - RNGxorshf::min() + 1.0);
    auto b = m_dist.a() - RNGxorshf::min() * k;
    for (size_t i = 0; i < size; ++i) {
        dest[i] = gen() * k + b;
    }
}

dt_float32 UniformFloatNonZeroRNG::gen_single_val() {
    auto&& gen = RandomState::generator();
    auto ret = UniformFloatRNG::gen_single_val();
    if (m_dist_flip(gen)) {
        ret = -ret;
    }
    megdnn_assert(ret != 0);
    return ret;
}

void UniformFloatNonZeroRNG::fill_fast_float32(dt_float32* dest, size_t size) {
    RNGxorshf gen{RandomState::generator()};
    UniformFloatRNG::fill_fast_float32(dest, size);
    for (size_t i = 0; i < size; ++i) {
        if (m_dist_flip(gen)) {
            dest[i] = -dest[i];
        }
    }
}

void UniformFloatWithValueRNG::fill_fast_float32(dt_float32* dest, size_t size) {
    RNGxorshf gen{RandomState::generator()};
    auto k = double(m_dist.b() - m_dist.a()) /
             double(RNGxorshf::max() - RNGxorshf::min() + 1.0);
    auto b = m_dist.a() - RNGxorshf::min() * k;

    auto p = 1.0 / double(RNGxorshf::max() - RNGxorshf::min() + 1.0);
    auto pb = 0.f - RNGxorshf::min() * p;
    for (size_t i = 0; i < size; ++i) {
        float rnd = gen() * p + pb;
        if (rnd < val_proportion_) {
            dest[i] = val_;
        } else {
            dest[i] = gen() * k + b;
        }
    }
}

BernoulliRNG::BernoulliRNG(float probability_) : m_dist(0, 1) {
    megdnn_assert(0.0f <= probability_ && probability_ < 1.0f);
    m_probability = probability_;
}

dt_float32 BernoulliRNG::gen_single_val() {
    auto&& gen = RandomState::generator();
    return m_dist(gen) < m_probability ? 1.0 : 0.0;
}

void NoReplacementRNG::gen(const TensorND& tensor) {
    auto offset = tensor.layout.span().low_elem;
    auto nr_elems = tensor.layout.span().dist_elem();
#define cb(DType)                                                      \
    if (tensor.layout.dtype == DType()) {                              \
        using ctype = typename DTypeTrait<DType>::ctype;               \
        std::set<ctype> values;                                        \
        auto ptr = tensor.ptr<ctype>();                                \
        for (size_t i = 0; i < nr_elems; ++i) {                        \
            ctype val;                                                 \
            do {                                                       \
                val = static_cast<ctype>(m_iid_rng->gen_single_val()); \
            } while (!values.insert(val).second);                      \
            ptr[offset + i] = val;                                     \
        }                                                              \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
}

InvertibleMatrixRNG::InvertibleMatrixRNG()
        : m_rng{new RNGxorshf{RandomState::generator()}} {}

InvertibleMatrixRNG::~InvertibleMatrixRNG() noexcept = default;

template <typename ctype>
void InvertibleMatrixRNG::do_gen(ctype* ptr, size_t batch, size_t n) {
    auto&& gen = *m_rng;
    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) {
        perm[i] = i;
    }
    for (size_t i = 0; i < batch; ++i, ptr += n * n) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                ptr[j * n + k] =
                        static_cast<ctype>(gen() / (RNGxorshf::max() + 1.0) * 2 - 0.5);
            }
        }
        for (size_t i = 0; i < n; ++i) {
            auto idx = gen() % (n - i) + i;
            ptr[i * n + perm[idx]] +=
                    static_cast<ctype>(gen() / (RNGxorshf::max() + 1.0) + 3);
            std::swap(perm[idx], perm[i]);
        }
    }
}

void InvertibleMatrixRNG::gen(const TensorND& tensor) {
#define cb(DType)                                                                   \
    if (tensor.layout.dtype == DType()) {                                           \
        using ctype = typename DTypeTrait<DType>::ctype;                            \
        auto ptr = tensor.ptr<ctype>();                                             \
        megdnn_assert(                                                              \
                tensor.layout.ndim >= 2 && tensor.layout.is_physical_contiguous()); \
        size_t batch = 1;                                                           \
        for (size_t i = 0; i < tensor.layout.ndim - 2; ++i) {                       \
            batch *= tensor.layout[i];                                              \
        }                                                                           \
        size_t n = tensor.layout[tensor.layout.ndim - 1];                           \
        megdnn_assert(n == tensor.layout[tensor.layout.ndim - 2]);                  \
        do_gen<ctype>(ptr, batch, n);                                               \
        return;                                                                     \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
}
void ConsecutiveRNG::fill_fast_float32(dt_float32* dest, size_t size) {
    for (size_t i = 0; i < size; ++i)
        dest[i] = value_ + i * delta_;
}

TEST(RNG, NO_REPLACEMENT_RNG) {
    static const size_t N = 10, TIMES = 100;
    UniformIntRNG base_rng(0, N - 1);
    NoReplacementRNG rng(&base_rng);
    auto handle = create_cpu_handle(2, false);
    for (size_t t = 0; t < TIMES; ++t) {
        TensorLayout layout({N}, dtype::Float32());
        Tensor<> tensor(handle.get(), layout);
        rng.gen(tensor.tensornd());
        std::vector<float> vals;
        for (size_t i = 0; i < N; ++i)
            vals.push_back(tensor.ptr()[i]);
        std::sort(vals.begin(), vals.end());
        for (size_t i = 0; i < N; ++i)
            ASSERT_EQ(static_cast<float>(i), vals[i]);
    }
}
// vim: syntax=cpp.doxygen
