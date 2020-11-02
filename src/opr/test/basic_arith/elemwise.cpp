/**
 * \file src/opr/test/basic_arith/elemwise.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./erfinv.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/io.h"

#include <cmath>
#include <algorithm>

using namespace mgb;

namespace {
    using Mode = opr::Elemwise::Mode;

    using InputGenerator = Maybe<thin_function<void(HostTensorND&)>>;
    // msvc would check for callable of None, so we use this to replace None
    const InputGenerator NONE_INPUT_GEN;

    std::unordered_set<Mode, enumhash> tested_mode;

    /* ======================= opr special impls ======================= */
    float do_mod(float a, float b) {
        return std::fmod(a, b);
    }

    int do_mod(int a, int b) {
        return a % b;
    }

    float do_erfinv(float x) {
        return erfinvf(x);
    }

    float do_erfcinv(float x) {
        return erfcinvf(x);
    }

    float do_h_swish(float x){
        return x * fmaxf(fminf(x + 3.f, 6.f), 0.f) / 6.f;
    }

    float do_h_swish_grad(float x, float y){
        return  x < -3.f ? 0.f : (x > 3.f ? y : (2.f * x + 3.f) / 6.f * y);
    }

    template<typename T>
    T do_log_sum_exp(T a, T b) {
        return std::log(std::exp(a) + std::exp(b));
    }

    float do_fast_tanh(float x) {
        return x * (27.f + x * x) / (27.f + 9.f * x * x);
    }

    float do_fast_tanh_grad(float x, float y) {
        float x_pow2 = x * x;
        float deno = 3.f + x_pow2;
        return ((-48.f * x_pow2) / deno + 27.f + x_pow2) / (deno * 9.f) * y;
    }

    float do_fuse_add_h_swish(float x, float y) {
        float z = x + y;
        return z * fmaxf(fminf(z + 3.f, 6.f), 0.f) / 6.f;
    }

    template<typename T>
    T do_shl(T, T); // undefined
    template<typename T>
    T do_shr(T, T); // undefined
    int do_shl(int x, int y) {
        return x << y;
    }
    int do_shr(int x, int y) {
        return x >> y;
    }

    template <typename T>
    struct MulType {};
    template <>
    struct MulType<int8_t> {
        typedef int16_t type;
    };
    template <>
    struct MulType<int16_t> {
        typedef int32_t type;
    };
    template <>
    struct MulType<int32_t> {
        typedef int64_t type;
    };
    template <>
    struct MulType<uint8_t> {
        typedef uint16_t type;
    };

    template <typename T>
    T rounding_shift_right_upward(T x, int k) {
        T mask = (T(1) << k) - 1;
        T threshold = mask >> 1;
        return (x >> k) + ((x & mask) > threshold);
    }

    template <typename T>
    T do_round_mulh_saturate(T a, T b) {
        MEGDNN_STATIC_ASSERT(std::numeric_limits<T>::digits <= 32,
                             "Portable RMULH is not supported for integer "
                             "types larger than 32 bits.");
        MEGDNN_STATIC_ASSERT(std::numeric_limits<T>::is_integer,
                             "Input types should be integer for RMULH");
        bool overflow = a == b && a == DTypeTrait<T>::min();
        // TODO: This really should be
        // rounding_shift_right_away_from_zero, but we haven't yet found a fast
        // way to implement it on ARM NEON. For now, we just try to align with
        // NEON's VQRDMULH and hope that it does not harm our NN badly.
        return overflow ? DTypeTrait<T>::max()
                        : static_cast<T>(rounding_shift_right_upward(
                                  typename MulType<T>::type(a) *
                                          typename MulType<T>::type(b),
                                  std::numeric_limits<T>::digits));
    }

    /* ======================= basic framework ======================= */

    template<typename ctype, bool stable_sign = false>
    void gen_nozero(HostTensorND &dest)  {
        static RNGxorshf rng{next_rand_seed()};
        auto ptr = dest.template ptr<ctype>();

        if (DTypeTrait<ctype>::category == DTypeCategory::FLOAT) {
            for (size_t i = 0, it = dest.shape().total_nr_elems();
                    i < it; ++ i) {
                auto v = rng() / (rng.max() + 1.0) * 3 - 1.5;
                bool vsign = v > 0;
                if (stable_sign) {
                    vsign = i % 2;
                }
                v = std::abs(v) + 0.1;
                ptr[i] = vsign ? v : -v;
            }
        } else {
            for (size_t i = 0, it = dest.shape().total_nr_elems();
                    i < it; ++ i) {
                ctype v = rng() / (rng.max() + 1.0) * 65536 - 32767,
                      vsat = i % 2 * 2 - 1;
                ptr[i] = v == 0 ? vsat : v;
            }
        }
    }

    template <class Trait>
    struct CheckerConfig {
        static constexpr bool enable_binary_inp_swap() {
            return true;
        }

        static constexpr bool allow_inp_grad(size_t idx) {
            MGB_MARK_USED_VAR(idx);
            return true;
        }

        template<typename ctype>
        static InputGenerator get_inp_gen(size_t idx) {
            MGB_MARK_USED_VAR(idx);
            return NONE_INPUT_GEN;
        }

        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 1e-2;
        }

        template<class Checker>
        static void update_checker(Checker &checker) {
            MGB_MARK_USED_VAR(checker);
        }
    };

    template<typename ctype>
    InputGenerator get_inp_gen_f32_range(float low, float high) {
        mgb_assert(std::is_same<ctype MGB_COMMA dt_float32>::value &&
                high - low >= 0.1);
        auto gen = [low, high](HostTensorND &dest) {
            HostTensorGenerator<
                dtype::Float32, RandomDistribution::UNIFORM>
                gen{low, high};
            dest = *gen(dest.shape());
        };
        return gen;
    }

#define DEF_TRAIT(_mode, _expr) \
    struct _mode { \
        static constexpr size_t ARITY = _CUR_ARITY; \
        static constexpr Mode MODE = Mode::_mode; \
        static constexpr bool ALLOW_INT = _ALLOW_INT; \
        static constexpr bool ALLOW_FLOAT = _ALLOW_FLOAT; \
        static constexpr bool ALLOW_BOOL = _ALLOW_BOOL; \
        static constexpr const char* NAME = #_mode; \
        template<typename ctype> \
        static inline ctype apply( \
                std::array<const ctype*, ARITY> inp, size_t idx) { \
            _EXPAND_PARAMS; \
            return _expr; \
        } \
    };

#include "./elemwise_unary_trait_def.inl"
#include "./elemwise_binary_trait_def.inl"
#include "./elemwise_ternary_trait_def.inl"

#undef DEF_TRAIT

    //! ensure nonzero value on some specific input
    template<size_t nozero_idx, bool large_eps = true>
    struct NoZeroCheckerConfig: public CheckerConfig<void> {
        static constexpr bool enable_binary_inp_swap() {
            return false;
        }

        template<typename ctype>
        static InputGenerator get_inp_gen(size_t idx) {
            if (idx != nozero_idx)
                return NONE_INPUT_GEN;
            return gen_nozero<ctype>;
        }

        template<class Opt>
        static void update_opt(Opt &opt) {
            if (large_eps)
                opt.numdiff_eps_single_inp[nozero_idx] = 0.05;
        }
    };
    struct NoGradCheckerConfig: public CheckerConfig<void> {
        static constexpr bool allow_inp_grad(size_t) {
            return false;
        }
    };

    /* ======================= unary config ======================= */
    template<> struct CheckerConfig<RELU>: public NoZeroCheckerConfig<0> {};
    template<> struct CheckerConfig<ABS>: public NoZeroCheckerConfig<0> {};
    template<> struct CheckerConfig<CEIL>: public NoGradCheckerConfig {};
    template<> struct CheckerConfig<FLOOR>: public NoGradCheckerConfig {};
    template<> struct CheckerConfig<ROUND>: public NoGradCheckerConfig {};
    template<> struct CheckerConfig<LOG>: public CheckerConfig<void> {
        template<typename ctype>
        static InputGenerator get_inp_gen(size_t) {
            return get_inp_gen_f32_range<ctype>(0.1, 4);
        }
        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 1e-2;
            opt.numdiff_max_err = 0.1;
        }
    };
    template<> struct CheckerConfig<LOG1P>: public CheckerConfig<void> {
        template<typename ctype>
        static InputGenerator get_inp_gen(size_t) {
            return get_inp_gen_f32_range<ctype>(-0.2, 0.2);
        }
    };
    template<> struct CheckerConfig<ACOS>: public CheckerConfig<void> {
        template<typename ctype>
        static InputGenerator get_inp_gen(size_t) {
            return get_inp_gen_f32_range<ctype>(-0.95, 0.95);
        }
        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 2e-3;
            opt.numdiff_max_err = 4e-3;
        }
    };
    template<> struct CheckerConfig<ASIN>: public CheckerConfig<ACOS> {};
    template<> struct CheckerConfig<TANH>: public CheckerConfig<void> {
        template<typename ctype>
        static InputGenerator get_inp_gen(size_t) {
            return get_inp_gen_f32_range<ctype>(-5, 5);
        }
        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 2e-2;
        }
    };
    template<> struct CheckerConfig<SIGMOID_GRAD>: public CheckerConfig<void> {
        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 2e-2;
        }
    };
    template<> struct CheckerConfig<ERF>: public CheckerConfig<void> {
        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 2e-2;
        }
    };
    template<> struct CheckerConfig<ERFINV>: public NoGradCheckerConfig {
        template<typename ctype>
        static InputGenerator get_inp_gen(size_t) {
            return get_inp_gen_f32_range<ctype>(-1, 1);
        }
        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 2e-2;
        }
    };
    template<> struct CheckerConfig<ERFC>: public CheckerConfig<void> {
        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 2e-2;
        }
    };
    template<> struct CheckerConfig<ERFCINV>: public NoGradCheckerConfig {
        template<typename ctype>
        static InputGenerator get_inp_gen(size_t) {
            return get_inp_gen_f32_range<ctype>(0, 2);
        }
        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 2e-2;
        }
    };

    template <> struct CheckerConfig<H_SWISH> : public CheckerConfig<void> {};
    template <>
    struct CheckerConfig<H_SWISH_GRAD> : public NoGradCheckerConfig {};

    /* ======================= binary config ======================= */
    template<bool for_mod>
    struct BinaryInputMinGap: public CheckerConfig<void> {
        template<typename ctype, class Checker>
        static void do_update_checker(Checker &checker) {
            auto icoord = [](const typename Checker::NumInpArray &inp) {
                static const ctype GAP{for_mod ? 0.01f : 0.1f};
                if (DTypeTrait<ctype>::category != DTypeCategory::FLOAT)
                    return;
                auto p0 = inp[0]->template ptr<ctype>(),
                     p1 = inp[1]->template ptr<ctype>();
                for (size_t i = 0, it = inp[0]->shape().total_nr_elems();
                        i < it; ++ i) {
                    if (for_mod) {
                        auto p1v = std::abs(p1[i]),
                             mod = std::fmod(p0[i], p1v);
                        mod += mod < 0 ? p1v : 0;
                        if (mod < GAP || mod > p1v - GAP) {
                            mgb_assert(p1v > GAP * 4);
                            ctype m0, m1;
                            do {
                                p0[i] += GAP;
                                m0 = std::fmod(p0[i] - GAP, p1[i]);
                                m1 = std::fmod(p0[i] + GAP, p1[i]);
                            } while (std::abs(m1 - m0) > GAP * 2 + 1e-3);
                        }
                    } else {
                        if (std::abs(p0[i] - p1[i]) < GAP) {
                            p1[i] += p0[i] < p1[i] ? GAP : -GAP;
                        }
                    }
                }
            };
            checker.set_input_coordinator(icoord);
        }


        template<class Checker>
        static void update_checker(Checker &checker) {
            using ctype = typename Checker::ctype;
            if (std::is_integral<ctype>::value)
                return;
            if (std::is_same<ctype, dt_float16>::value)
                return do_update_checker<dt_float16>(checker);
            if (std::is_same<ctype, dt_float32>::value)
                return do_update_checker<dt_float32>(checker);
            mgb_assert(0);
        }
    };

    struct BinaryEQInput: public CheckerConfig<void> {
        static constexpr bool allow_inp_grad(size_t idx) {
            return idx >= 2;
        }

        template<class Checker>
        static void update_checker(Checker &checker) {
            using ctype = typename Checker::ctype;
            auto icoord = [](const typename Checker::NumInpArray &inp) {
                if (DTypeTrait<ctype>::category != DTypeCategory::FLOAT)
                    return;
                auto p0 = inp[0]->template ptr<ctype>(),
                     p1 = inp[1]->template ptr<ctype>();
                RNGxorshf rng{next_rand_seed()};
                for (size_t i = 0, it = inp[0]->shape().total_nr_elems();
                        i < it; ++ i) {
                    p0[i] = rng() % 3 == 0 ? p1[i] : p0[i];
                }
            };
            checker.set_input_coordinator(icoord);
        }
    };

    struct BinaryPlaneNoPiInput : public CheckerConfig<void> {
        template <class Checker>
        static void update_checker(Checker& checker) {
            using ctype = typename Checker::ctype;
            auto icoord = [](const typename Checker::NumInpArray& inp) {
                if (DTypeTrait<ctype>::category != DTypeCategory::FLOAT)
                    return;
                auto p0 = inp[0]->template ptr<ctype>(),
                     p1 = inp[1]->template ptr<ctype>();
                RNGxorshf rng{next_rand_seed()};
                auto maxv = rng.max() + 1.0;
                for (size_t i = 0, it = inp[0]->shape().total_nr_elems();
                     i < it; ++i) {
                    //! To be numerical stable, r cannot be too small
                    auto r = rng() / maxv * 2 + 0.5;  //! radious
                    //! Avoid pi value due to periodicity
                    //! Numerical diff will be wrong there
                    //! Range [-pi+eps, pi-eps]
                    auto t = rng() / maxv * 3.1 * 2 - 3.1;  //! angle
                    //! First input is y in space
                    p0[i] = r * std::sin(t);
                    //! Second input is x in space
                    p1[i] = r * std::cos(t);
                }
            };
            checker.set_input_coordinator(icoord);
        }
        static constexpr bool enable_binary_inp_swap() { return false; }
    };
    template <>
    struct CheckerConfig<ATAN2> : public BinaryPlaneNoPiInput {
        template <class Opt>
        static void update_opt(Opt& opt) {
            opt.numdiff_eps = 1e-3;
            opt.numdiff_max_err = 0.02;
        }
    };

    template<> struct CheckerConfig<ABS_GRAD>: public NoZeroCheckerConfig<0> {};
    template<> struct CheckerConfig<FLOOR_DIV>:
        public NoZeroCheckerConfig<1, false> {
            static constexpr bool allow_inp_grad(size_t) {
                return false;
            }
    };
    template<> struct CheckerConfig<TRUE_DIV>: public
                                               NoZeroCheckerConfig<1, false> {
        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 1e-2;
            opt.numdiff_max_err = 0.1;
        }
    };
    template<> struct CheckerConfig<EQ>: public BinaryEQInput {};
    template<> struct CheckerConfig<LEQ>: public NoGradCheckerConfig {};
    template<> struct CheckerConfig<LT>: public NoGradCheckerConfig {};
    template <>
    struct CheckerConfig<FUSE_ADD_H_SWISH> : public CheckerConfig<void> {};
    template<> struct CheckerConfig<SWITCH_GT0>:
        public NoZeroCheckerConfig<0> { };
    template<> struct CheckerConfig<POW>: public CheckerConfig<void> {
        static constexpr bool enable_binary_inp_swap() {
            return false;
        }
        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 1e-2;
            opt.numdiff_max_err = 0.06;
        }
        template<typename ctype>
        static InputGenerator get_inp_gen(size_t idx) {
            auto func = [](HostTensorND &dest) {
                dest = *HostTensorGenerator<typename DTypeTrait<ctype>::dtype
                    >{}(dest.shape());
                auto ptr = dest.ptr<ctype>();
                for (size_t i = 0, t = dest.shape().total_nr_elems();
                        i < t; ++ i) {
                    ptr[i] = std::abs(ptr[i]) + 0.1;
                }
            };
            if (idx == 0)
                return func;
            return NONE_INPUT_GEN;
        }
    };
    template<> struct CheckerConfig<MAX>: public BinaryInputMinGap<false> {};
    template<> struct CheckerConfig<MIN>: public BinaryInputMinGap<false> {};
    template<> struct CheckerConfig<MOD>:
        public NoZeroCheckerConfig<1, false>,
        public BinaryInputMinGap<true>
    {
        using NoZeroCheckerConfig<1, false>::get_inp_gen;
        using NoZeroCheckerConfig<1, false>::enable_binary_inp_swap;
        using BinaryInputMinGap<true>::update_checker;

        template<class Opt>
        static void update_opt(Opt &opt) {
            opt.numdiff_eps = 0.003;
        }

        static constexpr bool allow_inp_grad(size_t idx) {
            return idx == 0;
        }
    };

    template<> struct CheckerConfig<SHL>: public CheckerConfig<void> {
        static constexpr bool enable_binary_inp_swap() {
            return false;
        }

        static constexpr bool allow_inp_grad(size_t idx) {
            return false;
        }

        template<typename ctype>
        static InputGenerator get_inp_gen(size_t);
    };
    template<> struct CheckerConfig<SHR>: public CheckerConfig<SHL> {};

    template<>
    InputGenerator CheckerConfig<SHL>::get_inp_gen<int>(size_t idx) {
        if (!idx)
            return NONE_INPUT_GEN;
        auto gen = [](HostTensorND &dest) {
            HostTensorGenerator<dtype::Int32, RandomDistribution::UNIFORM>
                gen{0, 32};
            dest = *gen(dest.shape());
        };
        return gen;
    }

    template<> struct CheckerConfig<FUSE_ADD_RELU>: public CheckerConfig<void> {
        template<typename ctype>
        static InputGenerator get_inp_gen(size_t) {
            return gen_nozero<ctype, true>;
        }
    };

    template<>
    struct CheckerConfig<FAST_TANH>: public CheckerConfig<void> {
        template<typename ctype>
        static InputGenerator get_inp_gen(size_t) {
            return get_inp_gen_f32_range<ctype>(0.1, 5);
        }
    };

    template<>
    struct CheckerConfig<FAST_TANH_GRAD>: public CheckerConfig<FAST_TANH> {
        static constexpr bool allow_inp_grad(size_t idx) {
            MGB_MARK_USED_VAR(idx);
            return false;
        }
    };

    /* ======================= ternary config ======================= */
    template<> struct CheckerConfig<COND_LEQ_MOV>:
        public BinaryInputMinGap<false> {};


    /* ======================= test runner ======================= */
    namespace detail {
        template<typename dtype, class Trait>
        struct enable_for_dtype_impl;

        template<class Trait>
        struct enable_for_dtype_impl<dtype::Float32, Trait> {
            static constexpr bool value = Trait::ALLOW_FLOAT;
        };
        template<>
        struct enable_for_dtype_impl<dtype::Float32, void> {
            static constexpr bool value = false;
        };
        template<class Trait>
        struct enable_for_dtype_impl<dtype::Int32, Trait> {
            static constexpr bool value = Trait::ALLOW_INT;
        };
        template<>
        struct enable_for_dtype_impl<dtype::Int32, void> {
            static constexpr bool value = false;
        };
        template<class Trait>
        struct enable_for_dtype_impl<dtype::Bool, Trait> {
            static constexpr bool value = Trait::ALLOW_BOOL;
        };
    }

    //! whether to enable test for specific dtype and Trait
    template<typename dtype, class Trait>
    constexpr bool enable_for_dtype =
    detail::enable_for_dtype_impl<dtype, Trait>::value;

    template<typename Trait, typename dtype,
        bool enable = enable_for_dtype<dtype, Trait>>
    struct TestRunner;

    template<typename Trait, typename dtype>
    struct TestRunner<Trait, dtype, true> {
        static void run();
    };
    template<typename Trait, typename dtype>
    struct TestRunner<Trait, dtype, false> {
        static void run() {
        }
    };
    template<typename dtype>
    struct TestRunner<void, dtype, false> {
        static void run() {
        }
    };

    template<typename Trait>
    class TestOprBasicArithUnaryElemwise: public ::testing::Test {
    };
    template<typename Trait>
    class TestOprBasicArithBinaryElemwise: public ::testing::Test {
    };
    template<typename Trait>
    class TestOprBasicArithTernaryElemwise: public ::testing::Test {
    };


    typedef ::testing::Types<
#define DEF_TRAIT(_mode, _expr) _mode,
#include "./elemwise_unary_trait_def.inl"
#undef DEF_TRAIT
        void // extra void to consume last comma
        > UnaryTraitTypes;
    TYPED_TEST_CASE(TestOprBasicArithUnaryElemwise, UnaryTraitTypes);

    typedef ::testing::Types<
#define DEF_TRAIT(_mode, _expr) _mode,
#include "./elemwise_binary_trait_def.inl"
#undef DEF_TRAIT
        void // extra void to consume last comma
        > BinaryTraitTypes;
    TYPED_TEST_CASE(TestOprBasicArithBinaryElemwise, BinaryTraitTypes);

    typedef ::testing::Types<
#define DEF_TRAIT(_mode, _expr) _mode,
#include "./elemwise_ternary_trait_def.inl"
#undef DEF_TRAIT
        void // extra void to consume last comma
        > TernaryTraitTypes;
    TYPED_TEST_CASE(TestOprBasicArithTernaryElemwise, TernaryTraitTypes);

} // anonymous namespace

template<typename Trait, typename dtype>
void TestRunner<Trait, dtype, true>::run() {
    {
        Mode mode = Trait::MODE;
        // copy to temporary var to avoid undefined reference when linking
        tested_mode.insert(mode);
    }

    using ctype = typename DTypeTrait<dtype>::ctype;

    HostTensorGenerator<> gen;
    using Config = CheckerConfig<Trait>;

    static constexpr bool TEST_REV_INP = Trait::ARITY == 2 &&
        Config::allow_inp_grad(0) == Config::allow_inp_grad(1) &&
        Config::enable_binary_inp_swap();
    using Checker = AutoOprChecker<Trait::ARITY, TEST_REV_INP + 1, dtype>;
    auto make_graph = [&](const typename Checker::SymInpArray &inputs) {
        typename Checker::SymOutArray out;
        SymbolVarArray vinp(inputs.begin(), inputs.end());
        out[0] = opr::Elemwise::make(vinp, Trait::MODE);
        if (TEST_REV_INP) {
            std::swap(vinp[0], vinp[1]);
            out[1] = opr::Elemwise::make(vinp, Trait::MODE);
        }
        return out;
    };

    auto fwd = [&](typename Checker::NumOutArray &dest,
            typename Checker::NumInpArray inp) {
        dest[0].resize(inp[0]->shape());
        if (TEST_REV_INP)
            dest[1].resize(inp[0]->shape());

        std::array<const ctype*, Trait::ARITY> iptr;
        for (size_t i = 0; i < Trait::ARITY; ++ i)
            iptr[i] = inp[i]->template ptr<ctype>();

        size_t sz = dest[0].shape().total_nr_elems();

        ctype* optr = dest[0].template ptr<ctype>();
        for (size_t i = 0; i < sz; ++ i)
            optr[i] = Trait::apply(iptr, i);

        if (TEST_REV_INP) {
            std::swap(iptr[0], iptr[1]);
            ctype* optr = dest[1].template ptr<ctype>();
            for (size_t i = 0; i < sz; ++ i)
                optr[i] = Trait::apply(iptr, i);
        }
    };

    Checker checker{make_graph, fwd};
    checker.set_extra_err_msg(ssprintf("mode=%s", Trait::NAME));
    for (size_t i = 0; i < Trait::ARITY; ++ i) {
        auto func = Config::template get_inp_gen<ctype>(i);
        if (func.valid())
            checker.set_input_generator(i, func.val());

        checker.set_input_allow_grad(i, Config::allow_inp_grad(i));
    }

    TensorShape shapes[] = {{1}, {23, 3}, {666}};
    typename Checker::RunOptions opt;
    Config::update_opt(opt);
    Config::update_checker(checker);
    for (auto &&ishp: shapes) {
        typename Checker::ShapeInpArray inp;
        std::fill(inp.begin(), inp.end(), ishp);
        checker.run(inp, opt);
    }
}

TYPED_TEST(TestOprBasicArithUnaryElemwise, Int32) {
    TestRunner<TypeParam, dtype::Int32>::run();
}
TYPED_TEST(TestOprBasicArithBinaryElemwise, Int32) {
    TestRunner<TypeParam, dtype::Int32>::run();
}
TYPED_TEST(TestOprBasicArithTernaryElemwise, Int32) {
    TestRunner<TypeParam, dtype::Int32>::run();
}

TYPED_TEST(TestOprBasicArithUnaryElemwise, Float32) {
    set_rand_seed(19931102);
    TestRunner<TypeParam, dtype::Float32>::run();
}
TYPED_TEST(TestOprBasicArithBinaryElemwise, Float32) {
    set_rand_seed(19931150);
    TestRunner<TypeParam, dtype::Float32>::run();
}
TYPED_TEST(TestOprBasicArithTernaryElemwise, Float32) {
    set_rand_seed(19931102);
    TestRunner<TypeParam, dtype::Float32>::run();
}

TEST(TestOprBasicArithElemwise, CheckAllModeTested) {
    size_t nr_member = opr::Elemwise::Param::MODE_NR_MEMBER;
    ASSERT_EQ(nr_member, tested_mode.size() + 4);
    // Not using TestRunner: NOT, AND, OR, XOR
}
#define TEST_OPR_BASIC_ARITH_UNARY_BOOL(_mode, _op) \
    TEST(TestOprBasicArithElemwise, _mode) { \
        HostTensorGenerator<dtype::Bool> gen; \
        auto host_x = gen({2, 1}); \
        auto ptr = host_x->ptr<dt_bool>(); \
        for (size_t i = 0; i < 2; ++i) { \
            ptr[i] = (i & 1); \
        } \
        auto graph = ComputingGraph::make(); \
        using Mode = opr::Elemwise::Mode; \
        auto x = opr::Host2DeviceCopy::make(*graph, host_x), \
             y = opr::Elemwise::make({x}, Mode::_mode); \
        HostTensorND host_y; \
        auto func = graph->compile({make_callback_copy(y, host_y)}); \
        func->execute(); \
        ASSERT_EQ(TensorShape({2, 1}), host_y.shape()); \
        auto ptry = host_y.ptr<dt_bool>(); \
        for (int i = 0;i < 2;i ++) { \
            ASSERT_EQ(_op ptr[i], ptry[i]); \
        } \
    } \

TEST_OPR_BASIC_ARITH_UNARY_BOOL(NOT, !)

#define TEST_OPR_BASIC_ARITH_BINARY_BOOL(_mode, _op) \
    TEST(TestOprBasicArithElemwise, _mode) { \
        HostTensorGenerator<dtype::Bool> gen; \
        auto host_x1 = gen({2, 2}), host_x2 = gen({2, 2}); \
        auto ptr1 = host_x1->ptr<dt_bool>(), ptr2 = host_x2->ptr<dt_bool>(); \
        for (size_t i = 0; i < 4; ++i) { \
            ptr1[i] = (i < 2); \
            ptr2[i] = (i & 1); \
        } \
        auto graph = ComputingGraph::make(); \
        using Mode = opr::Elemwise::Mode; \
        auto x1 = opr::Host2DeviceCopy::make(*graph, host_x1), \
             x2 = opr::Host2DeviceCopy::make(*graph, host_x2), \
             y = opr::Elemwise::make({x1, x2}, Mode::_mode); \
        HostTensorND host_y; \
        auto func = graph->compile({make_callback_copy(y, host_y)}); \
        func->execute(); \
        ASSERT_EQ(TensorShape({2, 2}), host_y.shape()); \
        auto ptry = host_y.ptr<dt_bool>(); \
        for (int i = 0;i < 4;i ++) { \
            ASSERT_EQ(ptr1[i] _op ptr2[i], ptry[i]); \
        } \
    } \

TEST_OPR_BASIC_ARITH_BINARY_BOOL(AND, &&)
TEST_OPR_BASIC_ARITH_BINARY_BOOL(OR, ||)
TEST_OPR_BASIC_ARITH_BINARY_BOOL(XOR, ^)
TEST_OPR_BASIC_ARITH_BINARY_BOOL(LT, <)
TEST_OPR_BASIC_ARITH_BINARY_BOOL(LEQ, <=)
TEST_OPR_BASIC_ARITH_BINARY_BOOL(EQ, ==)

TEST(TestOprBasicArithElemwise, FuseMulAdd3Shapes) {
    using Checker = AutoOprChecker<3, 1>;

    opr::Elemwise *opr;
    auto make_graph = [&](const typename Checker::SymInpArray &i) ->
            Checker::SymOutArray {
        i[0].node()->owner_graph()->options().graph_opt_level = 0;
        auto ret = opr::Elemwise::make(i, Mode::FUSE_MUL_ADD3);
        opr = &ret.node()->owner_opr()->cast_final_safe<opr::Elemwise>();
        return {ret};

    };

    auto fwd = [&](typename Checker::NumOutArray &dest,
            typename Checker::NumInpArray inp) {
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = false;
        auto i = [&](size_t idx) {
            return opr::Host2DeviceCopy::make(*graph, inp[idx]);
        };
        auto ans = i(0) * i(1) + i(2);
        graph->compile({make_callback_copy(ans, dest[0])})->execute();
    };

    Checker checker{make_graph, fwd};
    checker.
        run({TensorShape{1, 2}, {2, 1}, {1, 2}}).
        run({TensorShape{1, 2}, {2, 1}, {1}});
    ASSERT_FALSE(opr->fuse_badlayout_warn_printed());
    checker.run({TensorShape{1, 1, 4}, {1, 3, 1}, {2, 1, 1}});
    ASSERT_TRUE(opr->fuse_badlayout_warn_printed());
}

TEST(TestOprBasicArithElemwise, FuseMulAdd4Shapes) {
    using Checker = AutoOprChecker<4, 1>;

    opr::Elemwise *opr;
    auto make_graph = [&](const typename Checker::SymInpArray &i) ->
            Checker::SymOutArray {
        i[0].node()->owner_graph()->options().graph_opt_level = 0;
        auto ret = opr::Elemwise::make(i, Mode::FUSE_MUL_ADD4);
        opr = &ret.node()->owner_opr()->cast_final_safe<opr::Elemwise>();
        return {ret};

    };

    auto fwd = [&](typename Checker::NumOutArray &dest,
            typename Checker::NumInpArray inp) {
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = false;
        auto i = [&](size_t idx) {
            return opr::Host2DeviceCopy::make(*graph, inp[idx]);
        };
        auto ans = i(0) * i(1) + i(2) * i(3);
        graph->compile({make_callback_copy(ans, dest[0])})->execute();
    };

    Checker checker{make_graph, fwd};
    checker.
        run({TensorShape{1, 2}, {2, 1}, {1, 2}, {2, 1}}).
        run({TensorShape{1, 2, 1, 2, 1, 2}, {2, 1, 2, 1, 2, 1},
                {2, 1, 2, 1, 2, 1}, {1, 2, 1, 2, 1, 2}});
    ASSERT_FALSE(opr->fuse_badlayout_warn_printed());
    checker.run({TensorShape{1, 2}, {2, 1}, {2, 2}, {2, 2}});
    ASSERT_TRUE(opr->fuse_badlayout_warn_printed());
}

TEST(TestOprBasicArithElemwise, WritableFwdForSameStorage) {
    HostTensorGenerator<> gen;

    auto run = [&](int idx_val, bool should_overwrite) {
        auto host_x = gen({100});
        auto make_y = [&](ComputingGraph &graph) {
            using S = opr::Subtensor;
            auto x = opr::Host2DeviceCopy::make_no_fwd(graph, host_x),
                 idx = x.make_scalar(idx_val),
                 sub0 = S::make(x,
                         {S::AxisIndexer::make_interval(0, None, idx, None)}),
                 sub1 = S::make(x,
                         {S::AxisIndexer::make_interval(0, -idx, None, None)}),
                 y = sub0 + sub1;
            auto chk_overwrite = [sub0, sub1, y]() {
                auto py = y.node()->prev_dev_ptr();
                return sub0.node()->prev_dev_ptr() == py ||
                    sub1.node()->prev_dev_ptr() == py;
            };
            return std::make_pair(y, chk_overwrite);
        };
        auto g0 = ComputingGraph::make(), g1 = ComputingGraph::make();
        g1->options().seq_opt.enable_mem_plan_opt = false;
        auto y0 = make_y(*g0), y1 = make_y(*g1);
        HostTensorND host_y0, host_y1;
        auto f0 = g0->compile({make_callback_copy(y0.first, host_y0)}),
             f1 = g1->compile({make_callback_copy(y1.first, host_y1)});

        f0->execute();
        f1->execute();
        ASSERT_EQ(host_y1.shape(), TensorShape{static_cast<size_t>(idx_val)});
        MGB_ASSERT_TENSOR_EQ(host_y1, host_y0);
        ASSERT_EQ(should_overwrite, y0.second());
        ASSERT_FALSE(y1.second());

    };

    run(10, true);
    run(90, false);
}

TEST(TestOprBasicArithElemwise, NonContigInput) {
    HostTensorGenerator<> gen;

    auto graph = ComputingGraph::make();
    constexpr size_t SIZE = 100;
    auto host_x = gen({SIZE});
    using S = opr::Subtensor;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         xsub = S::make(x, {S::AxisIndexer::make_interval(0, None, None,
                     x.make_scalar(2))}),
         y = xsub + x.make_scalar(1.f);
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    ASSERT_FALSE(xsub.node()->dev_tensor().layout().is_contiguous());

    ASSERT_EQ(SIZE / 2, host_y.layout().total_nr_elems());
    auto px = host_x->ptr<float>(), py = host_y.ptr<float>();
    for (size_t i = 0; i < SIZE / 2; ++ i) {
        MGB_ASSERT_FLOAT_EQ(px[i * 2] + 1, py[i]);
    }
}

TEST(TestOprBasicArithElemwise, CommutableDedup) {
    auto cn = CompNode::load("xpux");
    auto graph = ComputingGraph::make();
    auto host_x = std::make_shared<HostTensorND>(cn, TensorShape{100}),
         host_y = std::make_shared<HostTensorND>(cn, TensorShape{100});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y);
    auto mk = [](Mode mode, SymbolVar x, SymbolVar y) {
        return opr::Elemwise::make({x, y}, mode);
    };
#define CHK(_a, _b) ASSERT_EQ((_a).node(), (_b).node())
    CHK(x + y, y + x);
    CHK(x * y, y * x);
    CHK(mk(Mode::EQ, x, y), mk(Mode::EQ, y, x));
    CHK(mk(Mode::MIN, x, y), mk(Mode::MIN, y, x));
    CHK(mk(Mode::MAX, x, y), mk(Mode::MAX, y, x));
    CHK(mk(Mode::LOG_SUM_EXP, x, y), mk(Mode::LOG_SUM_EXP, y, x));
    CHK(x < y, y > x);
#undef CHK
    ASSERT_NE((x - y).node(), (y - x).node());
}


TEST(TestLayoutUtil, CollectiveCollapse) {
    using namespace opr;
    auto shp2layout = [](const TensorShapeArray& tshps) {
        TensorLayoutArray tlayouts(tshps.size());
        for (size_t i = 0; i < tshps.size(); i++) {
            tlayouts[i] = TensorLayout(tshps[i], dtype::Float32());
        }
        return tlayouts;
    };
    auto check = [](const TensorLayoutArray& res,
                    const TensorLayoutArray& std) {
        for (size_t i = 0; i < res.size(); i++) {
            ASSERT_EQ(std[i], res[i]);
        }
    };
    TensorShapeArray tshps1 = {{3, 3}, {3, 3},  {3, 3}};
    auto cc_res1 = Elemwise::collective_collapse(shp2layout(tshps1));
    TensorShapeArray std_res1 = {{9}, {9}, {9}};
    check(cc_res1, shp2layout(std_res1));

    TensorShapeArray tshps2 = {{3, 3, 3}, {1, 3, 3}};
    auto cc_res2 = Elemwise::collective_collapse(shp2layout(tshps2));
    TensorShapeArray std_res2 {{3, 9}, {1, 9}};
    check(cc_res2, shp2layout(std_res2));

    TensorShapeArray tshp3 = {{3, 3, 3}, {3, 3, 1}};
    auto cc_res3 = Elemwise::collective_collapse(shp2layout(tshp3));
    TensorShapeArray std_res3 {{9, 3}, {9, 1}};
    check(cc_res3, shp2layout(std_res3));

    TensorShapeArray tshp4 = {{3, 3, 3, 3}, {1, 3, 3, 1}};
    auto cc_res4 = Elemwise::collective_collapse(shp2layout(tshp4));
    TensorShapeArray std_res4 {{3, 9, 3}, {1, 9, 1}};
    check(cc_res4, shp2layout(std_res4));

    TensorLayoutArray inp5 = {
        TensorLayout(TensorShape{3, 3}, {1, 3}, dtype::Float32()),
        TensorLayout(TensorShape{3, 3}, {1, 3}, dtype::Float32())
    };
    auto cc_res5 = Elemwise::collective_collapse(inp5);
    auto std_res5 = inp5;
    check(cc_res5, std_res5);
}

TEST(TestOprBasicArithElemwise, EmptyInputOutputUnary) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({3, 0, 1, 3});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Elemwise::make(
                 {x}, opr::Elemwise::Param(opr::Elemwise::Param::Mode::RELU));
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    ASSERT_NO_THROW(func->execute().wait());
    ASSERT_TRUE(host_y.empty());
    ASSERT_TRUE(host_y.shape().is_empty());
    MGB_ASSERT_SHAPE_EQ(host_y.shape(), TensorShape({3, 0, 1, 3}));
}

TEST(TestOprBasicArithElemwise, EmptyInputOutputBinary) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({0, 8, 1, 7}), host_y = gen({0, 8, 1, 7});

    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = x + y;
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});

    // Invalid broadcast
    host_y->resize({0, 9, 1, 7});
    ASSERT_ANY_THROW(func->execute().wait());

    // Broadcast to 0
    host_y->resize({1, 8, 0, 7});
    ASSERT_NO_THROW(func->execute().wait());
    ASSERT_TRUE(host_z.empty());
    ASSERT_TRUE(host_z.shape().is_empty());
    MGB_ASSERT_SHAPE_EQ(host_z.shape(), TensorShape({0, 8, 0, 7}));

    // Broadcast to 0 (2)
    host_y->resize({2, 8, 1, 7});
    ASSERT_NO_THROW(func->execute().wait());
    ASSERT_TRUE(host_z.empty());
    ASSERT_TRUE(host_z.shape().is_empty());
    MGB_ASSERT_SHAPE_EQ(host_z.shape(), TensorShape({0, 8, 1, 7}));

    // Scalar broadcast
    z = x + x.make_scalar(1.f);
    func = graph->compile({make_callback_copy(z, host_z)});
    ASSERT_NO_THROW(func->execute().wait());
    ASSERT_TRUE(host_z.empty());
    ASSERT_TRUE(host_z.shape().is_empty());
    MGB_ASSERT_SHAPE_EQ(host_z.shape(), TensorShape({0, 8, 1, 7}));
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
