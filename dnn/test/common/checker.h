/**
 * \file dnn/test/common/checker.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megdnn/basic_types.h"
#include "megdnn/tensor_iter.h"
#include "test/common/opr_algo_proxy.h"
#include "test/common/opr_proxy.h"
#include "test/common/rng.h"

#include <gtest/gtest.h>

#include <memory>
#include <regex>
#include <unordered_map>

// clang-format off
#if defined(__has_feature)
    #if __has_feature(address_sanitizer)
        #define MEGDNN_TEST_ASAN 1
    #else
        #define MEGDNN_TEST_ASAN 0
    #endif
#elif defined(__SANITIZE_ADDRESS__)
    #define MEGDNN_TEST_ASAN 1
#else
    #define MEGDNN_TEST_ASAN 0
#endif
// clang-format on

namespace megdnn {
namespace test {

class CheckerHelper {
    // TensorLayoutArray and TensorValueArray should be protected in theory;
    // but g++-4.9 bugs handle access privilege wrongfully, so we change it
    // to public.
public:
    using TensorValueArray = TensorNDArray;
    using TensorsConstriant = std::function<void(TensorValueArray& tensors)>;
    using ExtraOprImpl = std::function<void(const TensorNDArray&)>;
    using OutputCanonizer = std::function<void(const TensorValueArray&)>;
    static std::shared_ptr<TensorValueArray> alloc_tensors(
            Handle* handle, const TensorLayoutArray& layouts, size_t offset);

    Handle* handle() const { return m_handle_cur; }

protected:
    //! whether to use physically contiguous (i.e. default layout) for naive
    //! impl
    bool m_enable_contig_naive = false;

    bool m_prev_succ = true;
    const char* m_input_tensors_fpath = nullptr;
    thin_function<void()> m_expect_exec_fail;
    std::unique_ptr<Handle> m_handle_naive;
    Handle* m_handle_cur;
    std::unique_ptr<RNG> m_default_rng;
    std::unordered_map<size_t, RNG*> m_rng;
    std::unordered_map<size_t, DType> m_dtype;
    std::unordered_map<size_t, TensorFormat> m_fmt;
    std::set<size_t> m_bypass;
    float_t m_epsilon = 1e-3, m_max_avg_error = 1e-3, m_max_avg_biased_error = 1e-3;
    float_t m_perf_check_threshold = -1;
    bool m_perf_check = false;
    ExtraOprImpl m_extra_opr_impl;
    OutputCanonizer m_output_canonizer;
    TensorsConstriant m_tensor_constraint;
    bool m_no_naive_and_check = false;
    bool m_stable_check = false;
    bool m_force_deduce_dst = true;
    bool m_allow_invalid_check = false;
    /**
     * the offset from the start of malloc memory
     *
     * \note alloc \p m_offset more memory when alloc memory for a tensor,
     * the start of tensor just begin at \p m_offset.
     * \warning current only used for opencl
     */
    size_t m_offset = 0;

    CheckerHelper(Handle* handle, bool check_dispatch = true);

    ~CheckerHelper() noexcept;

    using OprExec = std::function<void(const TensorValueArray&)>;

    void do_exec_with_testcases(
            const TensorValueArray& testcase_in, const TensorValueArray& testcase_out,
            const OprExec& exec_opr);

    void do_exec(
            const TensorLayoutArray& user_layouts,
            const TensorLayoutArray& deduced_layouts, const OprExec& exec_naive,
            const OprExec& exec_opr);

    void enable_contig_naive() { m_enable_contig_naive = true; }

    void copy_tensors_to_device(
            const TensorValueArray& dest, const TensorValueArray& src);
    void copy_tensors_from_device(
            const TensorValueArray& dest, const TensorValueArray& src);

private:
    std::shared_ptr<TensorValueArray> m_tensors_naive;

    void init_naive_values();
    void check_tensors(
            const TensorValueArray& expected, const TensorValueArray& computed);
};

template <typename Opr, typename Proxy = OprProxy<Opr>>
class Checker : public CheckerHelper {
public:
    using Param = typename Opr::Param;
    using BeforeExecCallback = std::function<void(Opr*, const TensorValueArray&)>;
    Checker(Handle* handle, bool check_dispatch = true)
            : CheckerHelper(handle, check_dispatch), m_param(Param()) {}

    TensorLayoutArray make_layouts(const TensorShapeArray& shapes) {
        TensorLayoutArray layouts(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            DType dt =
                    (m_dtype.find(i) != m_dtype.end() ? m_dtype[i] : dtype::Float32());
            if (m_fmt.find(i) == m_fmt.end()) {
                layouts[i] = TensorLayout(shapes[i], dt);
            } else
                layouts[i] = TensorLayout(shapes[i], dt, m_fmt[i]);
        }
        return layouts;
    }

    /*!
     * \brief execute opr on current param/dtype/rng config
     * \param shapes input/output shapes, which would be passed as
     *      arguments to Opr::deduce_layout
     *
     * Checker would construct TensorLayout vectors from shapes and dtypes,
     * and call exec(TensorLayoutArray &).
     */
    Checker& exec(const TensorShapeArray& shapes) {
        exec(make_layouts(shapes));
        return *this;
    }

    void exec(TensorLayoutArray layouts);

    //! explicitly require argument to be TensorShape
    Checker& execs(const TensorShapeArray& shapes) { return exec(shapes); }

    //! explicitly require argument to be TensorLayout
    Checker& execl(const TensorLayoutArray& layouts) {
        exec(layouts);
        return *this;
    }

    Checker& exect(
            const TensorValueArray& testcase_in, const TensorValueArray& testcase_out);

    Checker& set_param(Param param) {
        m_param = param;
        opr()->param() = param;
        return *this;
    }
    Checker& set_dtype(size_t idx, DType dtype) {
        m_dtype[idx] = dtype;
        return *this;
    }
    Checker& set_fmt(size_t idx, TensorFormat fmt) {
        m_fmt[idx] = fmt;
        return *this;
    }
    Checker& set_rng(size_t idx, RNG* rng) {
        m_rng[idx] = rng;
        return *this;
    }
    Checker& set_bypass(size_t idx) {
        m_bypass.insert(idx);
        return *this;
    }
    //! max error of a single element
    Checker& set_epsilon(dt_float32 epsilon) {
        m_epsilon = epsilon;
        m_max_avg_error = epsilon;
        m_max_avg_biased_error = epsilon;
        return *this;
    }
    //! max average error; defaults to epsilon
    Checker& set_max_avg_error(dt_float32 error) {
        m_max_avg_error = error;
        return *this;
    }
    //! max average biased error; defaults to epsilon
    Checker& set_max_avg_biased_error(dt_float32 error) {
        m_max_avg_biased_error = error;
        return *this;
    }
    Checker& set_offset(size_t offset) {
        m_offset = offset;
        return *this;
    }

    Checker& set_proxy(const Proxy& proxy) {
        m_naive_proxy = proxy;
        m_cur_proxy = proxy;
        return *this;
    }

    //! set_perf_check and set_perf_check_threshold control the
    //! performance checking behavior.
    //!
    //! If perf_check is on (default to off), the running time of the
    //! current operator and the naive operator would be measured and
    //! checked when calling exec.
    //! The accelerating ratio should be larger than perf_check_threshold,
    //! otherwise errors would be reported.
    //! perf_check_threshold must be set in advance since the default value
    //! (which is negative) is invalid.
    Checker& set_perf_check(bool perf_check) {
        m_perf_check = perf_check;
        return *this;
    }

    Checker& set_perf_check_threshold(float perf_check_threshold) {
        m_perf_check_threshold = perf_check_threshold;
        return *this;
    }

    //! stable check will run many iter and compare result with first iter
    Checker& set_stable_check(bool stable_check) {
        m_stable_check = stable_check;
        return *this;
    }

    //! froce deduce dst
    Checker& set_force_deduce_dst(bool force_deduce_dst) {
        m_force_deduce_dst = force_deduce_dst;
        return *this;
    }

    Checker& set_no_naive_check(bool no_naive_and_check) {
        m_no_naive_and_check = no_naive_and_check;
        return *this;
    }

    Checker& set_allow_invalid_check(bool allow_invalid_check) {
        m_allow_invalid_check = allow_invalid_check;
        return *this;
    }

    //! load input tensors from file for next run
    Checker& load_input_tensors(const char* fpath) {
        m_input_tensors_fpath = fpath;
        return *this;
    }

    //! add another checker to ensure naive implementation is correct
    Checker& set_extra_opr_impl(const ExtraOprImpl& chk) {
        m_extra_opr_impl = chk;
        return *this;
    }

    //! set a callback to be invoked before executing the operator
    Checker& set_before_exec_callback(const BeforeExecCallback& cb) {
        m_before_exec_callback = cb;
        return *this;
    }

    Checker& reset_before_exec_callback() {
        m_before_exec_callback = nullptr;
        return *this;
    }

    //! set a tensors constraints function, for the purpose of manipulating
    //! tensors when testing.
    Checker& set_tensors_constraint(const TensorsConstriant& tensor_constraint) {
        m_tensor_constraint = tensor_constraint;
        return *this;
    }

    /*!
     * \brief set that exec() on opr should fail, so naive is not called and
     * exec() returns directly after opr is called.
     *
     * This is only valid for next exec() call. It is usually used for
     * testing megcore::AsyncErrorInfo.
     *
     * \param cb callback to be invoked after opr exec (so error would not
     *           be passed to destructor)
     */
    Checker& set_expect_exec_fail(const thin_function<void()>& cb) {
        m_expect_exec_fail = cb;
        return *this;
    }

    /*!
     * \brief set a function to canonize the outputs
     *
     * For some oprs maybe multiple outputs can be accepted; we can use a
     * function to transform them into a canonized form before comparing.
     *
     * The arguments are tensors on CPU and should be modified in-place.
     */
    Checker& set_output_canonizer(OutputCanonizer canonizer) {
        m_output_canonizer = std::move(canonizer);
        return *this;
    }

    //! get the opr impl so setting other than param() can be modified
    Opr* opr() {
        if (!m_opr_cur) {
            m_opr_cur = m_handle_cur->create_operator<Opr>();
        }
        return m_opr_cur.get();
    }

    //! whether previous exec succeeds
    bool prev_succ() const { return m_prev_succ; }

private:
    BeforeExecCallback m_before_exec_callback;
    Param m_param;
    Proxy m_naive_proxy, m_cur_proxy;
    std::unique_ptr<Opr> m_opr_cur;
};

::testing::AssertionResult __assert_tensor_eq(
        const char* expr0, const char* expr1, const char* expr_maxerr,
        const char* expr_maxerr_avg, const char* expr_maxerr_avg_biased,
        const TensorND& v0, const TensorND& v1, float maxerr, float maxerr_avg,
        float maxerr_avg_biased, bool allow_invalid = false);

::testing::AssertionResult __assert_tensor_eq_allow_invalid(
        const char* expr0, const char* expr1, const char* expr_maxerr,
        const char* expr_maxerr_avg, const char* expr_maxerr_avg_biased,
        const TensorND& v0, const TensorND& v1, float maxerr, float maxerr_avg,
        float maxerr_avg_biased);

#define MEGDNN_ASSERT_TENSOR_EQ_EPS_AVG(v0, v1, maxerr, maxerr_avg, maxerr_avg_biased) \
    ASSERT_PRED_FORMAT5(                                                               \
            ::megdnn::test::__assert_tensor_eq, v0, v1, maxerr, maxerr_avg,            \
            maxerr_avg_biased)

#define MEGDNN_ASSERT_TENSOR_EQ_EPS_AVG_ALLOW_INVALID(                        \
        v0, v1, maxerr, maxerr_avg, maxerr_avg_biased)                        \
    ASSERT_PRED_FORMAT5(                                                      \
            ::megdnn::test::__assert_tensor_eq_allow_invalid, v0, v1, maxerr, \
            maxerr_avg, maxerr_avg_biased)

#define MEGDNN_ASSERT_TENSOR_EQ_EPS(v0, v1, maxerr) \
    MEGDNN_ASSERT_TENSOR_EQ_EPS_AVG(v0, v1, maxerr, maxerr, maxerr)

#define MEGDNN_ASSERT_TENSOR_EQ(v0, v1) MEGDNN_ASSERT_TENSOR_EQ_EPS(v0, v1, 1e-3)

template <typename Opr, typename Proxy>
void Checker<Opr, Proxy>::exec(TensorLayoutArray layouts) {
    auto opr_naive = m_handle_naive->create_operator<Opr>();
    auto opr_relayout = m_handle_naive->create_operator<RelayoutForward>();

    auto opr_cur = this->opr();
    opr_naive->param() = m_param;
    opr_cur->param() = m_param;
    bool deduce_layout = layouts.back().ndim == 0;
    if (deduce_layout || m_force_deduce_dst) {
        m_naive_proxy.deduce_layout(opr_naive.get(), layouts);
    }
    auto exec_naive = [this, &opr_naive, &layouts,
                       &opr_relayout](const TensorValueArray& values) {
        TensorValueArray contig_values = values;
        TensorValueArray real_values = values;
        std::shared_ptr<TensorValueArray> tensors_naive_contig_storage;
        if (m_enable_contig_naive) {
            TensorLayoutArray contig_layouts;
            for (auto&& layout : layouts) {
                contig_layouts.emplace_back(TensorLayout{
                        static_cast<const TensorShape&>(layout), layout.dtype});
            }
            m_naive_proxy.deduce_layout(opr_naive.get(), contig_layouts);
            tensors_naive_contig_storage =
                    alloc_tensors(m_handle_naive.get(), contig_layouts, m_offset);
            contig_values = *tensors_naive_contig_storage;
            //! relayout value to the contig_values
            for (size_t i = 0; i < contig_values.size(); ++i) {
                if (real_values[i].layout.ndim == 0)
                    continue;
                real_values[i].layout.format = {};
                opr_relayout->exec(
                        real_values[i], contig_values[i], m_handle_naive.get());
            }
        }

        m_naive_proxy.exec(opr_naive.get(), contig_values);

        if (m_enable_contig_naive) {
            //! relayout to the values
            for (size_t i = 0; i < contig_values.size(); ++i) {
                if (real_values[i].layout.ndim == 0)
                    continue;
                opr_relayout->exec(
                        contig_values[i], real_values[i], m_handle_naive.get());
            }
        }
    };
    auto exec_opr = [this, opr_cur](const TensorValueArray& values) {
        if (m_before_exec_callback) {
            m_before_exec_callback(opr_cur, values);
        }
        m_cur_proxy.exec(opr_cur, values);
    };
    auto user_layouts = layouts;
    do_exec(user_layouts, layouts, exec_naive, exec_opr);
}

template <typename Opr, typename Proxy>
Checker<Opr, Proxy>& Checker<Opr, Proxy>::exect(
        const TensorValueArray& testcase_in, const TensorValueArray& testcase_out) {
    auto opr_cur = this->opr();
    opr_cur->param() = m_param;
    auto exec_opr = [this, opr_cur](const TensorValueArray& values) {
        if (m_before_exec_callback) {
            m_before_exec_callback(opr_cur, values);
        }
        m_cur_proxy.exec(opr_cur, values);
    };
    do_exec_with_testcases(testcase_in, testcase_out, exec_opr);
    return *this;
}

template <typename T, typename U>
TensorND TensorValue(
        const TensorShape& shape, T dtype, std::initializer_list<U> values) {
    TensorND tensor;
    tensor.layout = {shape, dtype};
    tensor.raw_ptr = static_cast<dt_byte*>(malloc(tensor.layout.span().dist_byte()));
    megdnn_assert(
            values.size() == tensor.layout.total_nr_elems(), "%zu == %zu",
            values.size(), tensor.layout.total_nr_elems());
    auto ptr = tensor.ptr<typename DTypeTrait<T>::ctype>();
    for (const auto& v : values) {
        *ptr++ = typename DTypeTrait<T>::ctype(v);
    }
    return tensor;
}

template <typename T, typename U>
TensorND TensorValueLowbit4(const TensorShape& shape, T dtype, std::vector<U> values) {
    TensorND tensor;
    tensor.layout = {shape, dtype};
    tensor.raw_ptr = static_cast<dt_byte*>(malloc(tensor.layout.span().dist_byte()));
    megdnn_assert(values.size() == tensor.layout.total_nr_elems());
    auto ptr = tensor.ptr<typename DTypeTrait<T>::ctype>();
    auto layout = tensor.layout;
    auto dim_in = shape[layout.ndim - 1];
    auto elems = tensor.layout.total_nr_elems();
    auto dim_out = elems / dim_in;
    auto stride_out = div_ceil(dim_in, 2_z);
    size_t in_offset = 0;
    for (size_t i = 0; i < dim_out; ++i) {
        for (size_t j = 0; j < dim_in; j += 2) {
            U a = values[in_offset + j];
            U b = 0;
            if (j + 1 < dim_in)
                b = values[in_offset + j + 1];
            megdnn_assert(a >= DTypeTrait<T>::min());
            megdnn_assert(a <= DTypeTrait<T>::max());
            megdnn_assert(b >= DTypeTrait<T>::min());
            megdnn_assert(b <= DTypeTrait<T>::max());
            ptr[j / 2] = (a & 0xF) | (b << 4);
        }
        in_offset += dim_in;
        ptr += stride_out;
    }
    return tensor;
}

class Testcase : public SmallVector<TensorND> {
public:
    using SmallVector<TensorND>::SmallVector;
    ~Testcase() {
        // Suicide
        for (const auto& tensor : *this) {
            if (tensor.raw_ptr) {
                free(tensor.raw_ptr);
            }
        }
    }

    Testcase(const Testcase&) = delete;
    Testcase operator=(const Testcase&) = delete;
};

struct ExecutionPolicyAlgoName {
    std::string name;
    std::vector<ExecutionPolicyAlgoName> sub_policy_names;

    ExecutionPolicyAlgoName(const char* name) : name{name} {}

    ExecutionPolicyAlgoName(
            const char* name, const std::vector<ExecutionPolicyAlgoName>& sub_policy)
            : name{name}, sub_policy_names{sub_policy} {}
};
/*!
 * \brief a callable to check that given algorithm is used for heuristic
 * \param require_algo if its value is true, then requires
 *      get_algorithm_heuristic() to return the expected algo; otherwise the
 *      expected algo must exist in get_all_algorithms_safe() and it would be set to
 *      be used
 */
template <class Opr, typename OprAlgoProxy = OprAlgoProxy<Opr>>
class AlgoChecker {
public:
    AlgoChecker(ExecutionPolicyAlgoName name, bool* require_algo = nullptr)
            : m_policy_name{name}, m_require_algo{require_algo} {}

    AlgoChecker(ExecutionPolicy policy, bool* require_algo = nullptr)
            : m_policy{policy}, m_require_algo{require_algo} {}

    static ExecutionPolicy construct_execution_policy_from_name(
            const ExecutionPolicyAlgoName& policy_name,
            const TensorLayoutArray& layouts, const std::string& param,
            Handle* handle) {
        ExecutionPolicy ret;
        megdnn_assert(layouts.size() == OprTrait<Opr>::arity);
        auto opr = handle->create_operator<Opr>();
        opr->param() = Algorithm::deserialize_read_pod<typename Opr::Param>(param);
        for (auto algo_info :
             AlgoProxy<Opr, OprTrait<Opr>::arity>::get_all_algorithms_info_safe(
                     opr.get(), layouts)) {
            if (std::regex_match(
                        algo_info.desc.name,
                        std::regex("(" + policy_name.name + ")(.*)"))) {
                ret.algo = algo_info.desc;
            } else {
                continue;
            }

            Algorithm* algo = opr->get_algorithm_from_desc(algo_info.desc);
            std::vector<Algorithm::SearchItem>&& sub_items =
                    algo->get_subopr_list(layouts, opr.get());
            if (sub_items.size() != policy_name.sub_policy_names.size()) {
                printf("Invalid sub_policy_names in %s, expected %zu but got "
                       "%zu\n",
                       algo_info.desc.name.c_str(), sub_items.size(),
                       policy_name.sub_policy_names.size());
                return {};
            }
            FOREACH_OPR_TYPE_DISPATCH(sub_items, {
                ExecutionPolicy policy =
                        AlgoChecker<_Opr>::construct_execution_policy_from_name(
                                policy_name.sub_policy_names[_item_idx], _item.layouts,
                                _item.param, handle);
                ret.sub_policy.push_back(policy);
            });
            return ret;
        }
        return ret;
    }

    void operator()(Opr* opr, const CheckerHelper::TensorValueArray& arr) {
        TensorLayoutArray layouts;
        for (auto&& val : arr) {
            layouts.push_back(val.layout);
        }
        if (!m_policy_name.name.empty()) {
            std::string param_str;
            Algorithm::serialize_write_pod(opr->param(), param_str);
            m_policy = construct_execution_policy_from_name(
                    m_policy_name, layouts, param_str, opr->handle());
            ASSERT_TRUE(m_policy.algo.valid())
                    << "algorithm " << m_policy_name.name << " not found";
        }
        if (m_require_algo && *m_require_algo) {
            auto algo = OprAlgoProxy::get_algorithm_info_heuristic(opr, layouts);
            ASSERT_STREQ(
                    opr->get_algorithm_from_desc(m_policy.algo)->name(),
                    algo.desc.name.c_str());
        } else {
            opr->execution_policy() = m_policy;
        }
    }

private:
    ExecutionPolicyAlgoName m_policy_name;
    ExecutionPolicy m_policy;
    bool* m_require_algo;
};

template <typename Opr>
void construct_sub_execution_policy_heuristic(
        ExecutionPolicy& policy, const TensorLayoutArray& layouts,
        const std::string& param, Handle* handle) {
    megdnn_assert(layouts.size() == OprTrait<Opr>::arity);
    auto opr = handle->create_operator<Opr>();
    opr->param() = Algorithm::deserialize_read_pod<typename Opr::Param>(param);
    if (!policy.algo.valid()) {
        policy.algo =
                AlgoProxy<Opr, OprTrait<Opr>::arity>::get_algorithm_info_heuristic(
                        opr.get(), layouts)
                        .desc;
    }

    Algorithm* algo = opr->get_algorithm_from_desc(policy.algo);
    std::vector<Algorithm::SearchItem>&& sub_items =
            algo->get_subopr_list(layouts, opr.get());
    FOREACH_OPR_TYPE_DISPATCH(sub_items, {
        policy.sub_policy.push_back(ExecutionPolicy{});
        construct_sub_execution_policy_heuristic<_Opr>(
                policy.sub_policy.back(), _item.layouts, _item.param, handle);
    });
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
