#pragma once
#include "megdnn/dtype.h"

#include <random>
#include <set>
#include "test/common/random_state.h"
#include "test/common/utils.h"

namespace megdnn {
namespace test {

#if __cplusplus >= 201703L
#define COMPAT_RANDOM(begin, end)              \
    {                                          \
        std::default_random_engine rng_engine; \
        std::shuffle(begin, end, rng_engine);  \
    }
#else
#define COMPAT_RANDOM(begin, end) std::random_shuffle(begin, end);
#endif

class RNG {
protected:
    class RNGxorshf;

public:
    virtual void gen(const TensorND& tensor) = 0;
    virtual ~RNG() = default;
};

class Float16PeriodicalRNG : public RNG {
public:
    Float16PeriodicalRNG();
    Float16PeriodicalRNG(size_t range);

    void gen(const TensorND& tensor) override;
    dt_float16 get_single_val();

private:
    void gen_all_valid_float16();
    size_t m_offset;
    std::vector<dt_float16> m_sequence;
};

class BFloat16PeriodicalRNG : public RNG {
public:
    BFloat16PeriodicalRNG() {
        size_t bits = sizeof(dt_bfloat16) * 8;
        size_t mantissa_bits = std::numeric_limits<dt_bfloat16>::digits - 1;
        size_t exponent_bits = bits - mantissa_bits - 1;
        for (size_t exp = 1u << (exponent_bits - 2);
             exp < (1u << exponent_bits) - (1u << (exponent_bits - 2)); ++exp) {
            for (size_t x = 0; x < 1u << mantissa_bits; ++x) {
                size_t pos_num = (exp << mantissa_bits) + x;
                size_t neg_num = (1u << (bits - 1)) + (exp << mantissa_bits) + x;
                union U {
                    U() {}
                    uint16_t i;
                    dt_bfloat16 f;
                } i2f;
                i2f.i = static_cast<uint16_t>(pos_num);
                m_sequence.push_back(i2f.f);
                i2f.i = static_cast<uint16_t>(neg_num);
                m_sequence.push_back(i2f.f);
            }
        }
        std::shuffle(m_sequence.begin(), m_sequence.end(), RandomState::generator());
    }

    void gen(const TensorND& tensor) override {
        megdnn_assert(tensor.layout.dtype.enumv() == DTypeTrait<dt_bfloat16>::enumv);
        size_t nr_elems = tensor.layout.span().dist_elem();
        auto offset = tensor.layout.span().low_elem;
        for (size_t i = 0; i < nr_elems; ++i) {
            tensor.ptr<dt_bfloat16>()[offset + i] = get_single_val();
        }
    }

    dt_bfloat16 get_single_val() {
        if (m_offset >= m_sequence.size()) {
            m_offset = 0;
        }
        return m_sequence[m_offset++];
    }

private:
    size_t m_offset = 0;
    std::vector<dt_bfloat16> m_sequence;
};

class IIDRNG : public RNG {
public:
    void gen(const TensorND& tensor) override;
    virtual dt_float32 gen_single_val() = 0;
    virtual bool output_is_float() { return true; }

protected:
    virtual bool has_fast_float32();
    virtual void fill_fast_float32(dt_float32* dest, size_t size);
};

class NormalRNG final : public IIDRNG {
public:
    NormalRNG(dt_float32 mean = 0.0f, dt_float32 stddev = 1.0f)
            : m_dist(mean, stddev) {}

    void fill_fast_float32(dt_float32* dest, size_t size) override;

protected:
    dt_float32 gen_single_val() override;

private:
    std::normal_distribution<dt_float32> m_dist;
    bool has_fast_float32() override;
};

class ConstValue final : public IIDRNG {
public:
    ConstValue(dt_float32 value = 0.0f) : value_(value) {}
    void fill_fast_float32(dt_float32* dest, size_t size) override;

protected:
    dt_float32 gen_single_val() override { return value_; }

private:
    dt_float32 value_;
    bool has_fast_float32() override { return true; }
};

class UniformIntRNG : public IIDRNG {
public:
    UniformIntRNG(dt_int32 a, dt_int32 b) : m_dist(a, b) {}
    dt_float32 gen_single_val() override;
    bool output_is_float() override { return false; }

protected:
    std::uniform_int_distribution<dt_int32> m_dist;
};

//! range must be positive; each value would be negated with prob 0.5
class UniformIntNonZeroRNG : public UniformIntRNG {
    std::uniform_int_distribution<dt_int32> m_dist_flip{0, 1};

public:
    UniformIntNonZeroRNG(int a, int b) : UniformIntRNG(a, b) {
        megdnn_assert(a > 0 && b > a);
    }

    dt_float32 gen_single_val() override;
};

class UniformFloatRNG : public IIDRNG {
public:
    UniformFloatRNG(dt_float32 a, dt_float32 b) : m_dist(a, b) {}
    dt_float32 gen_single_val() override;

protected:
    std::uniform_real_distribution<dt_float32> m_dist;
    bool has_fast_float32() override;
    void fill_fast_float32(dt_float32* dest, size_t size) override;
};

//! range must be positive; each value would be negated with prob 0.5
class UniformFloatNonZeroRNG : public UniformFloatRNG {
    std::uniform_int_distribution<dt_int32> m_dist_flip{0, 1};

public:
    UniformFloatNonZeroRNG(float a, float b) : UniformFloatRNG(a, b) {
        megdnn_assert(a > 0 && b > a);
    }

    dt_float32 gen_single_val() override;
    void fill_fast_float32(dt_float32* dest, size_t size) override;
};

class UniformFloatWithValueRNG : public UniformFloatRNG {
public:
    UniformFloatWithValueRNG(
            dt_float32 a, dt_float32 b, float val_proportion, float val)
            : UniformFloatRNG(a, b), val_(val) {
        if (val_proportion < 0.f)
            val_proportion_ = 0.f;
        else if (val_proportion > 1.f)
            val_proportion_ = 1.f;
        else
            val_proportion_ = val_proportion;
    }

private:
    float val_proportion_, val_;
    void fill_fast_float32(dt_float32* dest, size_t size) override;
};

class UniformFloatWithZeroRNG final : public UniformFloatWithValueRNG {
public:
    UniformFloatWithZeroRNG(dt_float32 a, dt_float32 b, float zero_val_proportion)
            : UniformFloatWithValueRNG(a, b, zero_val_proportion, 0.f) {}
};

class BernoulliRNG final : public IIDRNG {
public:
    BernoulliRNG(dt_float32 probability_);
    dt_float32 gen_single_val() override;

private:
    dt_float32 m_probability;
    std::uniform_real_distribution<dt_float32> m_dist;
};

/**
 * \brief RNG without replacement, so that no two values in the tensor are
 * equal.
 *
 * Each value is generated repeatedly by IIDRNG, until the newly-generated value
 * differs from any previous value.
 */
class NoReplacementRNG final : public RNG {
private:
    IIDRNG* m_iid_rng;

public:
    NoReplacementRNG(IIDRNG* iid_rng) : m_iid_rng(iid_rng) {}
    void gen(const TensorND& tensor) override;
};

//! generate a batch of matrices that are likely to have a small condition num
class InvertibleMatrixRNG final : public RNG {
    std::unique_ptr<RNGxorshf> m_rng;

public:
    InvertibleMatrixRNG();
    ~InvertibleMatrixRNG() noexcept;

    void gen(const TensorND& tensor) override;

private:
    template <typename ctype>
    void do_gen(ctype* ptr, size_t batch, size_t n);
};

//! generate a continuous number of delta, start from value
class ConsecutiveRNG final : public IIDRNG {
public:
    ConsecutiveRNG(dt_float32 value = 0.0f, dt_float32 delta = 1.0f)
            : value_(value), delta_(delta) {}
    void fill_fast_float32(dt_float32* dest, size_t size) override;

protected:
    dt_float32 gen_single_val() override {
        auto res = value_;
        value_ += delta_;
        return res;
    }

private:
    dt_float32 value_, delta_;
    bool has_fast_float32() override { return true; }
};

class BoolRNG final : public RNG {
    std::mt19937_64 m_rng;

public:
    BoolRNG(size_t seed) : m_rng(seed) {}

    void gen(const TensorND& tensor) override {
        std::uniform_int_distribution<int> dist(0, 1);
        auto ptr = tensor.ptr<bool>() + tensor.layout.span().low_elem;
        for (size_t i = 0; i < tensor.layout.span().dist_elem(); ++i)
            ptr[i] = (dist(m_rng) == 1);
    }
};

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
