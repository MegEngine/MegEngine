/**
 * \file test/src/include/megbrain/test/helper.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief helper functions for testing
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/graph.h"

#include "gtest/gtest.h"

#include <iostream>

#if !MGB_ENABLE_EXCEPTION
#pragma GCC diagnostic ignored  "-Wunused-variable"
#endif

namespace megdnn {
    static inline bool operator == (
            const TensorShape &a, const TensorShape &b) {
        return a.eq_shape(b);
    }

    static inline std::ostream& operator << (
            std::ostream &ostr, const TensorShape &s) {
        return ostr << s.to_string();
    }

    bool operator == (const TensorLayout &a, const TensorLayout &b);

    static inline std::ostream& operator << (
            std::ostream &ostr, const TensorLayout &l) {
        return ostr << l.to_string();
    }

    static inline std::ostream& operator << (
            std::ostream &ostr, const DType &dt) {
        return ostr << dt.name();
    }
} // namespace megdnn

namespace mgb {

static inline std::ostream& operator << (
        std::ostream &ostr, const CompNode &cn) {
    return ostr << cn.to_string();
}

namespace cg {
    static inline bool operator == (SymbolVar a, SymbolVar b) {
        return a.node() == b.node();
    }

    static inline std::ostream& operator << (
            std::ostream &ostr, const VarNode* var) {
        return ostr << "var@" << static_cast<const void*>(var) << "("
            << (var ? var->cname() : "") << ")";
    }

    static inline std::ostream& operator << (
            std::ostream &ostr, SymbolVar var) {
        return ostr << var.node();
    }

    static inline std::ostream& operator << (
            std::ostream &ostr, OperatorNodeBase *opr) {
        return ostr << ssprintf("opr@%p{id=%zu,type=%s,name=%s}",
                opr, opr->id(), opr->dyn_typeinfo()->name, opr->cname());
    }

} // namespace cg

/*!
 * \brief set the random seed for current test case
 *
 * This is only effective when MGB_STABLE_RNG is set
 */
void set_rand_seed(uint64_t seed);

/*!
 * \brief get random seed to be used for this test case
 *
 * If MGB_STABLE_RNG is set, the seed would be based on current test name;
 * otherwise it is based on time
 */
uint64_t next_rand_seed();

/*!
 * \brief get items in a container which has begin() and end() methods as vector
 */
template<typename Container>
decltype(auto) container_to_vector(Container &&ct) {
    std::vector<decltype(*ct.begin())> rst;
    for (auto i = ct.begin(); i != ct.end(); ++ i)
        rst.push_back(*i);
    return rst;
}

::testing::AssertionResult __assert_float_equal(
        const char *expr0, const char *expr1, const char *expr_maxerr,
        float v0, float v1, float maxerr);

#define MGB_ASSERT_FLOAT_NEAR(v0, v1, maxerr) \
    ASSERT_PRED_FORMAT3(::mgb::__assert_float_equal, v0, v1, maxerr)

#define MGB_ASSERT_FLOAT_EQ(v0, v1) \
    MGB_ASSERT_FLOAT_NEAR(v0, v1, 1e-6)

#define MK(name) {dev_ ## name, [&](DeviceTensorND &var) {\
    host_ ## name.copy_from(var).sync();\
}}

::testing::AssertionResult __assert_tensor_equal(
        const char *expr0, const char *expr1, const char *expr_maxerr,
        const HostTensorND &v0, const HostTensorND &v1, float maxerr);

#define MGB_ASSERT_TENSOR_NEAR(v0, v1, maxerr) \
    ASSERT_PRED_FORMAT3(::mgb::__assert_tensor_equal, v0, v1, maxerr)

#define MGB_ASSERT_TENSOR_EQ(v0, v1) \
    MGB_ASSERT_TENSOR_NEAR(v0, v1, 1e-6)

::testing::AssertionResult __assert_shape_equal(const TensorShape& v0,
                                                const TensorShape& v1);

#define MGB_ASSERT_SHAPE_EQ(v0, v1) \
    ASSERT_TRUE(::mgb::__assert_shape_equal(v0, v1))

/*!
 * \brief xorshift+ RNG, which is very fast
 *
 * see https://en.wikipedia.org/wiki/Xorshift#xorshift.2B
 */
class RNGxorshf {
    uint64_t s[2];

    public:
        using result_type = uint64_t;

        static constexpr uint64_t min() {
            return 0;
        }

        static constexpr uint64_t max() {
#if WIN32
            return ~static_cast<uint64_t>(0);
#else
            return std::numeric_limits<uint64_t>::max();
#endif
        }

        explicit RNGxorshf(uint64_t seed);

        uint64_t operator() () {
            uint64_t x = s[0];
            uint64_t const y = s[1];
            s[0] = y;
            x ^= x << 23; // a
            s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
            return s[1] + y;
        }
};

enum class RandomDistribution {
    GAUSSIAN, UNIFORM, CONSTANT, CONSECUTIVE
};

template<class dtype>
struct RandomDistributionDTypeDefault;

template<>
struct RandomDistributionDTypeDefault<dtype::Float32> {
    static constexpr auto dist = RandomDistribution::GAUSSIAN;
};
template<>
struct RandomDistributionDTypeDefault<dtype::Int8> {
    static constexpr auto dist = RandomDistribution::UNIFORM;
};
template<>
struct RandomDistributionDTypeDefault<dtype::Uint8> {
    static constexpr auto dist = RandomDistribution::UNIFORM;
};
template<>
struct RandomDistributionDTypeDefault<dtype::Int16> {
    static constexpr auto dist = RandomDistribution::UNIFORM;
};
template<>
struct RandomDistributionDTypeDefault<dtype::Int32> {
    static constexpr auto dist = RandomDistribution::UNIFORM;
};
template<>
struct RandomDistributionDTypeDefault<dtype::Bool> {
    static constexpr auto dist = RandomDistribution::UNIFORM;
};
template<>
struct RandomDistributionDTypeDefault<dtype::QuantizedS8> {
    static constexpr auto dist = RandomDistribution::UNIFORM;
};


//! base class for host tensor generator
class HostTensorGeneratorBase {
    public:
        HostTensorGeneratorBase(uint64_t seed):
            m_rng{seed}
        {}

        virtual ~HostTensorGeneratorBase() = default;

        virtual std::shared_ptr<HostTensorND> operator ()(
                const TensorShape &shape, CompNode cn = {}) = 0;

        std::shared_ptr<HostTensorND> operator ()(
                const TensorShape &shape, const char *cn_name) {
            return operator()(shape, CompNode::load(cn_name));
        }
    protected:
        RNGxorshf m_rng;
};

/*!
 * \brief generate random tensor with given distribution and dtype
 */
template<class dtype = dtype::Float32,
         RandomDistribution dist = RandomDistributionDTypeDefault<dtype>::dist>
class HostTensorGenerator;

template<class dtype>
struct UniformRNGDefaultRange;

template<>
struct UniformRNGDefaultRange<dtype::Float32> {
    static constexpr dt_float32 LO = -1.732, HI = 1.732;
};

template<>
struct UniformRNGDefaultRange<dtype::Int8> {
    static constexpr dt_int8 LO = -127, HI = 127;
};
template<>
struct UniformRNGDefaultRange<dtype::Uint8> {
    static constexpr dt_uint8 LO = 0, HI = 255;
};
template<>
struct UniformRNGDefaultRange<dtype::Bool> {
    static constexpr dt_bool LO = false, HI = true;
};
template<>
struct UniformRNGDefaultRange<dtype::Int16> {
    static constexpr dt_int16 LO = -32767, HI = 32767;
};
template<>
struct UniformRNGDefaultRange<dtype::Int32> {
    static constexpr dt_int32 LO = -32768, HI = 32768;
};
template<>
struct UniformRNGDefaultRange<dtype::QuantizedS8> {
    static const dt_qint8 LO, HI;
};

template<>
struct UniformRNGDefaultRange<dtype::Quantized8Asymm> {
    static const dt_quint8 LO, HI;
};
//! gaussian
template<class dtype>
class HostTensorGenerator<dtype, RandomDistribution::GAUSSIAN> final:
        public HostTensorGeneratorBase {

    public:
        using ctype = typename DTypeTrait<dtype>::ctype;

        HostTensorGenerator(ctype mean = 0, ctype std = 1,
                uint64_t seed = next_rand_seed()):
            HostTensorGeneratorBase{seed}, m_mean{mean}, m_std{std}
        {
        }

        std::shared_ptr<HostTensorND> operator ()(
                const TensorShape &shape, CompNode cn = {}) override;
        using HostTensorGeneratorBase::operator();

        //! set mean
        auto&& mean(ctype val) {
            m_mean = val;
            return *this;
        }

        //! set std
        auto&& std(ctype val) {
            m_std = val;
            return *this;
        }

    private:
        ctype m_mean, m_std;

};

//! uniform
template<class dtype>
class HostTensorGenerator<dtype, RandomDistribution::UNIFORM> final:
        public HostTensorGeneratorBase {

    public:
        using ctype = typename DTypeTrait<dtype>::ctype;

        HostTensorGenerator(
                ctype lo = UniformRNGDefaultRange<dtype>::LO,
                ctype hi = UniformRNGDefaultRange<dtype>::HI,
                uint64_t seed = next_rand_seed()):
            HostTensorGeneratorBase{seed}, m_lo{lo}, m_hi{hi}
        {
        }

        std::shared_ptr<HostTensorND> operator ()(
                const TensorShape &shape, CompNode cn = {}) override;
        using HostTensorGeneratorBase::operator();

    private:
        ctype m_lo, m_hi;
};

//! const value
template<class dtype>
class HostTensorGenerator<dtype, RandomDistribution::CONSTANT> final:
        public HostTensorGeneratorBase {

    public:
        using ctype = typename DTypeTrait<dtype>::ctype;

        HostTensorGenerator(ctype default_val)
                : HostTensorGeneratorBase{next_rand_seed()},
                  m_default_val{default_val} {}

        std::shared_ptr<HostTensorND> operator ()(
                const TensorShape &shape, CompNode cn = {}) override;
        using HostTensorGeneratorBase::operator();

    private:
        ctype m_default_val;
};

//! consecutive value
template<class dtype>
class HostTensorGenerator<dtype, RandomDistribution::CONSECUTIVE> final:
        public HostTensorGeneratorBase {

    public:
        using ctype = typename DTypeTrait<dtype>::ctype;

        HostTensorGenerator(ctype val, ctype delta)
                : HostTensorGeneratorBase{next_rand_seed()},
                  m_val{val}, m_delta{delta} {}

        std::shared_ptr<HostTensorND> operator ()(
                const TensorShape &shape, CompNode cn = {}) override;
        using HostTensorGeneratorBase::operator();

    private:
        ctype m_val;
        ctype m_delta;
};


template <>
class HostTensorGenerator<dtype::Bool, RandomDistribution::UNIFORM> final
        : public HostTensorGeneratorBase {
public:
    using ctype = typename DTypeTrait<dtype::Bool>::ctype;

    HostTensorGenerator(uint64_t seed = next_rand_seed())
            : HostTensorGeneratorBase{seed} {}

    std::shared_ptr<HostTensorND> operator()(const TensorShape& shape,
                                             CompNode cn = {}) override;
    using HostTensorGeneratorBase::operator();

};

template <>
class HostTensorGenerator<dtype::QuantizedS8, RandomDistribution::UNIFORM> final
        : public HostTensorGeneratorBase {
    public:
        using ctype = typename DTypeTrait<dtype::QuantizedS8>::ctype;

        HostTensorGenerator(
                ctype lo = UniformRNGDefaultRange<dtype::QuantizedS8>::LO,
                ctype hi = UniformRNGDefaultRange<dtype::QuantizedS8>::HI,
                float scale = 1.f, uint64_t seed = next_rand_seed())
                : HostTensorGeneratorBase{seed},
                  m_scale{scale},
                  m_lo{lo},
                  m_hi{hi} {}

        std::shared_ptr<HostTensorND> operator()(const TensorShape& shape,
                                                 CompNode cn = {}) override;
        using HostTensorGeneratorBase::operator();

    private:
        float m_scale;
        ctype m_lo, m_hi;
};

template <>
class HostTensorGenerator<dtype::Quantized8Asymm, RandomDistribution::UNIFORM>
        final : public HostTensorGeneratorBase {
public:
    using ctype = typename DTypeTrait<dtype::Quantized8Asymm>::ctype;

    HostTensorGenerator(
            ctype lo = UniformRNGDefaultRange<dtype::Quantized8Asymm>::LO,
            ctype hi = UniformRNGDefaultRange<dtype::Quantized8Asymm>::HI,
            float scale = 1.f, uint8_t zero_point = 0,
            uint64_t seed = next_rand_seed())
            : HostTensorGeneratorBase{seed},
              m_scale{scale},
              m_zero_point(zero_point),
              m_lo{lo},
              m_hi{hi} {}

    std::shared_ptr<HostTensorND> operator()(const TensorShape& shape,
                                             CompNode cn = {}) override;
    using HostTensorGeneratorBase::operator();

private:
    float m_scale;
    uint8_t m_zero_point;
    ctype m_lo, m_hi;
};

/*!
 * \brief get output file name in test output dir
 * \param check_writable whether to ensure the file is writable
 * \return absolute output file path
 */
std::string output_file(const std::string &fname, bool check_writable = true);

//! write tensor to binary file
void write_tensor_to_file(const HostTensorND &hv,
        const char *fname, char mode = 'w');

/*!
 * \brief a named temporary file, which would be deleted upon object destruction
 */
class NamedTemporaryFile {
    std::string m_fpath;
    int m_fd;

    public:
        NamedTemporaryFile();
        ~NamedTemporaryFile();

        int fd() const {
            return m_fd;
        }

        const std::string& path() const {
            return m_fpath;
        }
};

cg::ComputingGraph::OutputSpecItem
make_callback_copy(SymbolVar dev, HostTensorND &host, bool sync = true);

static inline const dt_byte* dev_ptr(SymbolVar var) {
    return var.node()->dev_tensor().raw_ptr();
}

static inline const void* prev_dev_ptr(SymbolVar var) {
    return var.node()->prev_dev_ptr();
}

static inline void set_priority(SymbolVar var, int pri) {
    var.node()->owner_opr()->node_prop().attribute().priority = pri;
}

/*!
 * \brief load multipl comp nodes on xpu that belong to the same type
 *
 * If there are not enough devices for xpu, then cpu would be used
 */
std::vector<CompNode> load_multiple_xpus(size_t num);

//! check whether given number of GPUs is available
bool check_gpu_available(size_t num);

//! check whether given number of AMD GPUs is available
bool check_amd_gpu_available(size_t num);

//! check whether given number of cambricon devices is available
bool check_cambricon_device_available(size_t num);

//! check current capability >= major.minor
bool check_compute_capability(int major, int minor);

//! check compnode avaiable
bool check_device_type_avaiable(CompNode::DeviceType device_type);

//! hook persistent cache get calls during the lifetime
class PersistentCacheHook {
    class HookedImpl;

    std::shared_ptr<HookedImpl> m_impl;

public:
    //! if value is not available, \p val and \p val_size would be zero
    using GetHook = thin_function<void(const std::string& category,
                                       const void* key, size_t key_size,
                                       const void* val, size_t val_size)>;
    PersistentCacheHook(GetHook on_get);
    ~PersistentCacheHook();
};

//! skip a testcase if gpu not available
#define REQUIRE_GPU(n) do { \
    if (!check_gpu_available(n)) \
        return; \
} while(0)

#define REQUIRE_CUDA_COMPUTE_CAPABILITY(major, minor) \
    do {                                              \
        if (!check_compute_capability(major, minor))  \
            return;                                   \
    } while (0)

//! skip a testcase if amd gpu not available
#define REQUIRE_AMD_GPU(n) do { \
    if (!check_amd_gpu_available(n)) \
        return; \
} while(0)

//! skip a testcase if cambricon device not available
#define REQUIRE_CAMBRICON_DEVICE(n)               \
    do {                                          \
        if (!check_cambricon_device_available(n)) \
            return;                               \
    } while (0)

#if MGB_HAVE_THREAD
#define REQUIRE_THREAD()
#else
#define REQUIRE_THREAD() do { \
    return; \
} while (0)
#endif  //  MGB_HAVE_THREAD

} // namespace mgb

#if !MGB_ENABLE_EXCEPTION
#undef ASSERT_THROW
#undef EXPECT_THROW
#undef ASSERT_NO_THROW
#undef EXPECT_NO_THROW
#undef ASSERT_ANY_THROW
#undef EXPECT_ANY_THROW
#define ASSERT_THROW(...)
#define EXPECT_THROW(...)
#define ASSERT_ANY_THROW(...)
#define EXPECT_ANY_THROW(...)
#define ASSERT_NO_THROW(stmt) stmt
#define EXPECT_NO_THROW(stmt) stmt
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
