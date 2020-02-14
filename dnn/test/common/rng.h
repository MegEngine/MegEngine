/**
 * \file dnn/test/common/rng.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/dtype.h"

#include "test/common/utils.h"
#include <random>
#include <set>

namespace megdnn {
namespace test {

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

class UniformFloatWithZeroRNG final : public UniformFloatRNG {
public:
    UniformFloatWithZeroRNG(dt_float32 a, dt_float32 b,
                            float zero_val_proportion)
            : UniformFloatRNG(a, b) {
        if (zero_val_proportion < 0.f)
            zero_val_proportion_ = 0.f;
        else if (zero_val_proportion > 1.f)
            zero_val_proportion_ = 1.f;
        else
            zero_val_proportion_ = zero_val_proportion;
    }

private:
    float zero_val_proportion_;
    void fill_fast_float32(dt_float32* dest, size_t size) override;
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

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
