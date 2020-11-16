/**
 * \file dnn/test/common/checker.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
    float_t m_epsilon = 1e-3, m_max_avg_error = 1e-3,
            m_max_avg_biased_error = 1e-3;
    float_t m_perf_check_threshold = -1;
    bool m_perf_check = false;
    ExtraOprImpl m_extra_opr_impl;
    OutputCanonizer m_output_canonizer;
    TensorsConstriant m_tensor_constraint;
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

    void do_exec_with_testcases(const TensorValueArray& testcase_in,
                                const TensorValueArray& testcase_out,
                                const OprExec& exec_opr);

    void do_exec(const TensorLayoutArray& user_layouts,
                 const TensorLayoutArray& deduced_layouts,
                 const OprExec& exec_naive, const OprExec& exec_opr);

    void enable_contig_naive() { m_enable_contig_naive = true; }

private:
    std::shared_ptr<TensorValueArray> m_tensors_naive;

    void init_naive_values();
    void copy_tensors_to_device(const TensorValueArray& dest,
                                const TensorValueArray& src);
    void copy_tensors_from_device(const TensorValueArray& dest,
                                  const TensorValueArray& src);
    void check_tensors(const TensorValueArray& expected,
                       const TensorValueArray& computed);
};

template <typename Opr, typename Proxy = OprProxy<Opr>>
class Checker : public CheckerHelper {
public:
    using Param = typename Opr::Param;
    using BeforeExecCallback =
            std::function<void(Opr*, const TensorValueArray&)>;
    Checker(Handle* handle, bool check_dispatch = true)
            : CheckerHelper(handle, check_dispatch), m_param(Param()) {}

    TensorLayoutArray make_layouts(const TensorShapeArray& shapes) {
        TensorLayoutArray layouts(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            DType dt = (m_dtype.find(i) != m_dtype.end() ? m_dtype[i]
                                                         : dtype::Float32());
            TensorFormat fmt =
                    (m_fmt.find(i) != m_fmt.end() ? m_fmt[i] : TensorFormat{});
            layouts[i] = TensorLayout(shapes[i], dt, fmt);
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

    Checker& exect(const TensorValueArray& testcase_in,
                   const TensorValueArray& testcase_out);

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

    //! set a tensors constraints function, for the purpose of manipulating
    //! tensors when testing.
    Checker& set_tensors_constraint(
            const TensorsConstriant& tensor_constraint) {
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
        float maxerr_avg_biased);

#define MEGDNN_ASSERT_TENSOR_EQ_EPS_AVG(v0, v1, maxerr, maxerr_avg,         \
                                        maxerr_avg_biased)                  \
    ASSERT_PRED_FORMAT5(::megdnn::test::__assert_tensor_eq, v0, v1, maxerr, \
                        maxerr_avg, maxerr_avg_biased)

#define MEGDNN_ASSERT_TENSOR_EQ_EPS(v0, v1, maxerr) \
    MEGDNN_ASSERT_TENSOR_EQ_EPS_AVG(v0, v1, maxerr, maxerr, maxerr)

#define MEGDNN_ASSERT_TENSOR_EQ(v0, v1) \
    MEGDNN_ASSERT_TENSOR_EQ_EPS(v0, v1, 1e-3)

template <typename Opr, typename Proxy>
void Checker<Opr, Proxy>::exec(TensorLayoutArray layouts) {
    auto opr_naive = m_handle_naive->create_operator<Opr>();
    auto opr_relayout = m_handle_naive->create_operator<RelayoutForward>();

    auto opr_cur = this->opr();
    opr_naive->param() = m_param;
    opr_cur->param() = m_param;
    m_naive_proxy.deduce_layout(opr_naive.get(), layouts);
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
            tensors_naive_contig_storage = alloc_tensors(
                    m_handle_naive.get(), contig_layouts, m_offset);
            contig_values = *tensors_naive_contig_storage;
            //! relayout value to the contig_values
            for (size_t i = 0; i < contig_values.size(); ++i) {
                if (real_values[i].layout.ndim == 0)
                    continue;
                real_values[i].layout.format = {};
                opr_relayout->exec(real_values[i], contig_values[i],
                                   m_handle_naive.get());
            }
        }

        m_naive_proxy.exec(opr_naive.get(), contig_values);

        if (m_enable_contig_naive) {
            //! relayout to the values
            for (size_t i = 0; i < contig_values.size(); ++i) {
                if (real_values[i].layout.ndim == 0)
                    continue;
                opr_relayout->exec(contig_values[i], real_values[i],
                                   m_handle_naive.get());
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
        const TensorValueArray& testcase_in,
        const TensorValueArray& testcase_out) {
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
TensorND TensorValue(const TensorShape& shape, T dtype,
                     std::initializer_list<U> values) {
    TensorND tensor;
    tensor.layout = {shape, dtype};
    tensor.raw_ptr =
            static_cast<dt_byte*>(malloc(tensor.layout.span().dist_byte()));
    megdnn_assert(values.size() == tensor.layout.total_nr_elems(), "%zu == %zu",
                  values.size(), tensor.layout.total_nr_elems());
    auto ptr = tensor.ptr<typename DTypeTrait<T>::ctype>();
    for (const auto& v : values) {
        *ptr++ = typename DTypeTrait<T>::ctype(v);
    }
    return tensor;
}

template <typename T, typename U>
TensorND TensorValueLowbit4(const TensorShape& shape, T dtype,
                            std::vector<U> values) {
    TensorND tensor;
    tensor.layout = {shape, dtype};
    tensor.raw_ptr =
            static_cast<dt_byte*>(malloc(tensor.layout.span().dist_byte()));
    megdnn_assert(values.size() == tensor.layout.total_nr_elems());
    auto ptr = static_cast<U*>(tensor.raw_ptr);
    for (size_t i = 0; i < values.size(); i += 2) {
        U val0 = values[i], val1 = values[i + 1];
        megdnn_assert(val0 >= DTypeTrait<T>::min());
        megdnn_assert(val1 <= DTypeTrait<T>::max());
        ptr[i / 2] = (val0 & 0xF) | (val1 << 4);
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

/*!
 * \brief a callable to check that given algorithm is used for heuristic
 * \param require_algo if its value is true, then requires
 *      get_algorithm_heuristic() to return the expected algo; otherwise the
 *      expected algo must exist in get_all_algorithms() and it would be set to
 *      be used
 */
template <class Opr, typename OprAlgoProxy = OprAlgoProxy<Opr>>
class AlgoChecker {
    std::string m_name;
    typename Opr::Algorithm* m_algo = nullptr;
    bool* m_require_algo;

public:
    AlgoChecker(const char* name, bool* require_algo = nullptr)
            : m_name{name}, m_require_algo{require_algo} {}

    AlgoChecker(typename Opr::Algorithm* algo, bool* require_algo = nullptr)
            : m_algo{algo}, m_require_algo{require_algo} {}

    void operator()(Opr* opr, const CheckerHelper::TensorValueArray& arr) {
        TensorLayoutArray layouts;
        for (auto&& val : arr) {
            layouts.push_back(val.layout);
        }
        if (m_require_algo && *m_require_algo) {
            auto algo =
                    OprAlgoProxy::get_algorithm_info_heuristic(opr, layouts);
            if (m_name.empty()) {
                ASSERT_EQ(m_algo->name(), algo.name.c_str());
            } else {
                ASSERT_TRUE(std::regex_match(
                        algo.name.c_str(), std::regex("(" + m_name + ")(.*)")));
            }
        } else {
            if (m_name.empty()) {
                opr->execution_policy().algo = m_algo->info();
                return;
            } else {
                for (auto i :
                     OprAlgoProxy::get_all_algorithms_info(opr, layouts)) {
                    if (std::regex_match(i.name,
                                         std::regex("(" + m_name + ")(.*)"))) {
                        opr->execution_policy().algo = i;
                        return;
                    }
                }
            }
            ASSERT_TRUE(false) << "algorithm " << m_name << " not found";
        }
    }
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
