/**
 * \file dnn/test/common/checker.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./checker.h"
#include "megdnn/tensor_format.h"
#include "megdnn/tensor_iter.h"
#include "test/common/tensor.h"
#include "test/common/timer.h"
using namespace megdnn;
using namespace test;

namespace {
template <typename ctype, class Iter>
::testing::AssertionResult assert_tensor_eq_with_iter(
        const char* expr0, const char* expr1, Iter it0, Iter it1,
        const TensorLayout& layout, float maxerr, float maxerr_avg,
        float maxerr_avg_biased, bool allow_invalid) {
    auto nr_elem = layout.total_nr_elems();
    double error_sum = 0;
    double error_sum_biased = 0;
    for (size_t i = 0; i < nr_elem; ++i) {
        ctype iv0 = *it0, iv1 = *it1;
        float err = diff(iv0, iv1);
        error_sum += std::abs(err);
        error_sum_biased += err;
        if (!allow_invalid &&
            (!good_float(iv0) || !good_float(iv1) || std::abs(err) > maxerr)) {
            Index index(layout, i);
            return ::testing::AssertionFailure()
                << "Unequal value\n"
                << "Value of: " << expr1 << "\n"
                << "  Actual: " << (iv1 + 0) << "\n"
                << "Expected: " << expr0 << "\n"
                << "Which is: " << (iv0 + 0) << "\n"
                << "At index: " << index.to_string() << "/"
                << layout.TensorShape::to_string() << "\n"
                << "   DType: " << layout.dtype.name() << "\n"
                << "   error: " << std::abs(err) << "/" << maxerr;
        }

        ++it0;
        ++it1;
    }

    float error_avg = error_sum / nr_elem;
    if (error_avg > maxerr_avg) {
        return ::testing::AssertionFailure()
            << "Average error exceeds the upper limit\n"
            << "Value of: " << expr1 << "\n"
            << "Expected: " << expr0 << "\n"
            << "Average error: " << error_avg << "/" << maxerr_avg << "\n"
            << "Num of elements: " << nr_elem;
    }

    float error_avg_biased = error_sum_biased / nr_elem;
    if (std::abs(error_avg_biased) > maxerr_avg_biased) {
        return ::testing::AssertionFailure()
            << "Average biased error exceeds the upper limit\n"
            << "Value of: " << expr1 << "\n"
            << "Expected: " << expr0 << "\n"
            << "Average biased error: " << error_avg_biased << "/" << maxerr_avg_biased
            << "\n"
            << "Num of elements: " << nr_elem;
    }

    return ::testing::AssertionSuccess();
}

template <typename ctype>
::testing::AssertionResult assert_tensor_eq_with_dtype(
        const char* expr0, const char* expr1, const TensorND& v0, const TensorND& v1,
        float maxerr, float maxerr_avg, float maxerr_avg_biased, bool allow_invalid) {
    if (!std::is_same<ctype, dt_qint4>::value &&
        !std::is_same<ctype, dt_quint4>::value) {
        if (v0.layout.is_physical_contiguous() && v1.layout.is_physical_contiguous()) {
            return assert_tensor_eq_with_iter<ctype>(
                    expr0, expr1, v0.ptr<ctype>(), v1.ptr<ctype>(), v0.layout, maxerr,
                    maxerr_avg, maxerr_avg_biased, allow_invalid);
        }
    }

    auto it0 = megdnn::tensor_iter_valonly<ctype>(v0).begin(),
         it1 = megdnn::tensor_iter_valonly<ctype>(v1).begin();

    return assert_tensor_eq_with_iter<ctype>(
            expr0, expr1, it0, it1, v0.layout, maxerr, maxerr_avg, maxerr_avg_biased,
            allow_invalid);
}

template <class Impl>
void memcpy_noncontig(
        void* dst, const void* src, const TensorLayout& layout, const Impl& impl) {
    auto span = layout.span();
    dst = static_cast<dt_byte*>(dst) + span.low_byte;
    src = static_cast<const dt_byte*>(src) + span.low_byte;
    impl(dst, src, span.dist_byte());
}

template <typename Impl>
void copy_tensors(
        const CheckerHelper::TensorValueArray& dest,
        const CheckerHelper::TensorValueArray& src, const Impl& copy_impl) {
    megdnn_assert(dest.size() == src.size());
    for (size_t i = 0; i < src.size(); i++) {
        auto&& tensor = src[i];
        if (tensor.layout.ndim == 0)
            continue;
        memcpy_noncontig(dest[i].raw_ptr, tensor.raw_ptr, tensor.layout, copy_impl);
    }
}

void copy_tensors(
        const CheckerHelper::TensorValueArray& dest,
        const CheckerHelper::TensorValueArray& src) {
    copy_tensors(dest, src, memcpy);
}
}  // anonymous namespace

::testing::AssertionResult test::__assert_tensor_eq(
        const char* expr0, const char* expr1, const char* /*expr_maxerr*/,
        const char* /*expr_maxerr_avg*/, const char* /*expr_maxerr_avg*/,
        const TensorND& v0, const TensorND& v1, float maxerr, float maxerr_avg,
        float maxerr_avg_biased, bool allow_invalid) {
    if (!v0.layout.eq_shape(v1.layout)) {
        return ::testing::AssertionFailure()
            << "Shape mismatch\n"
            << "Value of: " << expr1 << "\n"
            << "  Actual: " << v1.layout.TensorShape::to_string() << "\n"
            << "Expected: " << expr0 << "\n"
            << "Which is: " << v0.layout.TensorShape::to_string() << "\n";
    }
    auto dtype = v0.layout.dtype;
    if (dtype != v1.layout.dtype) {
        return ::testing::AssertionFailure()
            << "Data type mismatch\n"
            << "Value of: " << expr1 << "\n"
            << "  Actual: " << v1.layout.dtype.name() << "\n"
            << "Expected: " << expr0 << "\n"
            << "Which is: " << v0.layout.dtype.name() << "\n";
    }

    switch (dtype.enumv()) {
#define cb(_dt)                                                              \
    case DTypeTrait<_dt>::enumv:                                             \
        return assert_tensor_eq_with_dtype<DTypeTrait<_dt>::ctype>(          \
                expr0, expr1, v0, v1, maxerr, maxerr_avg, maxerr_avg_biased, \
                allow_invalid);
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
        //! In order to avoid an unnecessary increase in binary size, we just
        //! use QuantizedS16 dtype in winograd_filter_preprocess now.
        cb(::megdnn::dtype::QuantizedS16) MEGDNN_FOREACH_QUANTIZED_LOWBIT_DTYPE(cb)
                cb(::megdnn::dtype::Uint16)
#undef cb
                        default : megdnn_trap();
    }
}

::testing::AssertionResult test::__assert_tensor_eq_allow_invalid(
        const char* expr0, const char* expr1, const char* expr_maxerr,
        const char* expr_maxerr_avg, const char* expr_maxerr_avg_biased,
        const TensorND& v0, const TensorND& v1, float maxerr, float maxerr_avg,
        float maxerr_avg_biased) {
    return __assert_tensor_eq(
            expr0, expr1, expr_maxerr, expr_maxerr_avg, expr_maxerr_avg_biased, v0, v1,
            maxerr, maxerr_avg, maxerr_avg_biased, true);
};

CheckerHelper::CheckerHelper(Handle* handle, bool check_dispatch)
        : m_handle_cur(handle), m_default_rng(new NormalRNG()) {
    //! set MGB_NO_NAIVE_CHECK=1 to close megdnn test check with naive
    const char* env_p = std::getenv("MGB_NO_NAIVE_CHECK");
    if (env_p) {
        int no_naive_flag = atoi(env_p);
        m_no_naive_and_check = no_naive_flag > 0 ? true : false;
        check_dispatch = false;
    } else {
        m_no_naive_and_check = false;
    }
    auto tmp_handle = create_cpu_handle(2, check_dispatch);
    m_handle_naive = std::move(tmp_handle);
}

CheckerHelper::~CheckerHelper() noexcept = default;

void CheckerHelper::do_exec_with_testcases(
        const TensorValueArray& testcase_in, const TensorValueArray& testcase_out,
        const OprExec& exec_opr) {
    m_prev_succ = false;

    // Validate layouts of tensors in testcase_in and testcase_out.
    // It must be possible to aggregate the layouts of inputs and outputs.
    TensorLayoutArray layouts;
    for (size_t i = 0; i < testcase_in.size(); i++) {
        // ndim == 0 means does not apply.
        ASSERT_TRUE(
                testcase_in[i].layout.ndim == 0 || testcase_out[i].layout.ndim == 0 ||
                testcase_in[i].layout.eq_layout(testcase_out[i].layout));
        layouts.emplace_back(
                testcase_in[i].layout.ndim > 0 ? testcase_in[i].layout
                                               : testcase_out[i].layout);
    }

    auto tensors_cur_storage = alloc_tensors(m_handle_cur, layouts, m_offset);
    auto tensors_cur_host_storage =
            alloc_tensors(m_handle_naive.get(), layouts, m_offset);
    auto&& tensors_cur = *tensors_cur_storage;
    auto&& tensors_cur_host = *tensors_cur_host_storage;
    copy_tensors_to_device(tensors_cur, testcase_in);

    exec_opr(tensors_cur);
    if (m_expect_exec_fail) {
        m_expect_exec_fail();
        m_expect_exec_fail = {};
        return;
    }

    copy_tensors_from_device(tensors_cur_host, tensors_cur);
    check_tensors(testcase_out, tensors_cur_host);
    m_prev_succ = !::testing::Test::HasFailure();
}

void CheckerHelper::do_exec(
        const TensorLayoutArray& user_layouts, const TensorLayoutArray& deduced_layouts,
        const OprExec& exec_naive, const OprExec& exec_opr) {
    m_prev_succ = false;

    // check if user provided layouts are correct
    for (size_t i = 0; i < deduced_layouts.size(); ++i) {
        if (user_layouts[i].ndim > 0) {
            ASSERT_TRUE(deduced_layouts[i].eq_shape(user_layouts[i]))
                    << "User provided shape is "
                    << user_layouts[i].TensorShape::to_string()
                    << "\nExpected shape is "
                    << deduced_layouts[i].TensorShape::to_string();
        }
    }
    auto layouts = user_layouts;
    for (size_t i = 0; i < layouts.size(); ++i) {
        if (layouts[i].ndim == 0) {
            //! in some opr, such as conv_bias has ndim==0
            layouts[i] = deduced_layouts[i];
        }
    }

    // allocate
    m_tensors_naive = alloc_tensors(m_handle_naive.get(), layouts, m_offset);
    auto tensors_cur_storage = alloc_tensors(m_handle_cur, layouts, m_offset);
    auto tensors_cur_host_storage =
            alloc_tensors(m_handle_naive.get(), layouts, m_offset);
    auto&& tensors_naive = *m_tensors_naive;
    auto&& tensors_cur = *tensors_cur_storage;
    auto&& tensors_cur_host = *tensors_cur_host_storage;
    std::shared_ptr<TensorValueArray> tensors_extra_opr_impl;
    if (m_extra_opr_impl) {
        tensors_extra_opr_impl = alloc_tensors(m_handle_naive.get(), layouts, m_offset);
    }

    init_naive_values();

    copy_tensors_to_device(tensors_cur, tensors_naive);
    if (m_extra_opr_impl) {
        copy_tensors(*tensors_extra_opr_impl, tensors_naive);
    }

    // execute

    exec_opr(tensors_cur);
    if (m_expect_exec_fail) {
        m_expect_exec_fail();
        m_expect_exec_fail = {};
        return;
    }
    if (m_stable_check) {
        auto tensors_bak_host_storage =
                alloc_tensors(m_handle_naive.get(), layouts, m_offset);
        auto&& tensors_bak_host = *tensors_bak_host_storage;
        copy_tensors_from_device(tensors_bak_host, tensors_cur);
        for (int i = 0; i < 10; i++) {
            exec_opr(tensors_cur);
            copy_tensors_from_device(tensors_cur_host, tensors_cur);
            check_tensors(tensors_bak_host, tensors_cur_host);
        }
    }
    if (m_no_naive_and_check) {
        m_prev_succ = !::testing::Test::HasFailure();
        return;
    }
    exec_naive(tensors_naive);
    if (m_extra_opr_impl) {
        m_extra_opr_impl(*tensors_extra_opr_impl);
    }

    // see if we need performance regression test
    if (m_perf_check) {
        ASSERT_GT(m_perf_check_threshold, 0) << "perf_check_threshold should be "
                                                "set ahead of time.";
        Timer timer_naive, timer_cur;

        megdnn_sync(m_handle_naive.get());
        timer_naive.start();
        exec_naive(tensors_naive);
        megdnn_sync(m_handle_naive.get());
        timer_naive.stop();

        megdnn_sync(m_handle_cur);
        timer_cur.start();
        exec_opr(tensors_cur);
        megdnn_sync(m_handle_cur);
        timer_cur.stop();

        size_t time_in_us_naive = timer_naive.get_time_in_us(),
               time_in_us_cur = timer_cur.get_time_in_us();
        EXPECT_GE(time_in_us_naive, static_cast<size_t>(100))
                << "Running time smaller than 100us "
                << "might be imprecise. naive_time=" << time_in_us_naive << "us.";
        float speedup_ratio = static_cast<float>(time_in_us_naive) / time_in_us_cur;
        EXPECT_GE(speedup_ratio, m_perf_check_threshold)
                << "speedup_ratio=" << speedup_ratio
                << " threshold=" << m_perf_check_threshold
                << " naive_time=" << time_in_us_naive
                << "us cur_time=" << time_in_us_cur << "us";
    }

    copy_tensors_from_device(tensors_cur_host, tensors_cur);
    if (m_output_canonizer) {
        m_output_canonizer(tensors_cur_host);
        m_output_canonizer(tensors_naive);
    }
    check_tensors(tensors_naive, tensors_cur_host);
    if (m_extra_opr_impl) {
        check_tensors(tensors_naive, *tensors_extra_opr_impl);
    }
    m_prev_succ = !::testing::Test::HasFailure();
}

std::shared_ptr<CheckerHelper::TensorValueArray> CheckerHelper::alloc_tensors(
        Handle* handle, const TensorLayoutArray& layouts, const size_t offset) {
    auto deleter = [handle, offset](TensorValueArray* ptr) {
        for (auto&& i : *ptr) {
            auto pdata = static_cast<dt_byte*>(i.raw_ptr) + i.layout.span().low_byte -
                         offset;
            megdnn_free(handle, pdata);
        }
        delete ptr;
    };
    std::shared_ptr<TensorValueArray> ret{new TensorValueArray, deleter};
    for (size_t i = 0; i < layouts.size(); ++i) {
        auto span = layouts[i].span();
        ret->emplace_back(
                static_cast<dt_byte*>(
                        megdnn_malloc(handle, span.dist_byte() + offset)) -
                        span.low_byte + offset,
                layouts[i]);
    }
    return ret;
}

void CheckerHelper::init_naive_values() {
    auto&& tensors_naive = *m_tensors_naive;
    megdnn_assert(!m_input_tensors_fpath || !m_tensor_constraint);
    if (m_input_tensors_fpath) {
        auto load = load_tensors(m_input_tensors_fpath);
        m_input_tensors_fpath = nullptr;
        megdnn_assert(load.size() <= tensors_naive.size());

        for (size_t i = 0; i < load.size(); ++i) {
            auto&& src = load[i];
            auto&& dst = tensors_naive[i];
            megdnn_assert(src->layout.eq_layout(dst.layout));
            memcpy_noncontig(dst.raw_ptr, src->raw_ptr, dst.layout, memcpy);
        }
        return;
    }

    for (size_t i = 0; i < tensors_naive.size(); ++i) {
        auto&& tensor = tensors_naive[i];
        auto rng = m_rng[i];
        if (!rng)
            rng = m_default_rng.get();
        rng->gen(tensor);
    }

    if (m_tensor_constraint) {
        m_tensor_constraint(tensors_naive);
    }
}

void CheckerHelper::copy_tensors_from_device(
        const TensorValueArray& dest, const TensorValueArray& src) {
    auto impl_d2h = [this](void* dst, const void* src, size_t sz) {
        megdnn_memcpy_D2H(m_handle_cur, dst, src, sz);
    };
    copy_tensors(dest, src, impl_d2h);
}

void CheckerHelper::check_tensors(
        const TensorValueArray& expected, const TensorValueArray& computed) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].layout.ndim == 0 || m_bypass.find(i) != m_bypass.end())
            continue;
        if (m_allow_invalid_check) {
            MEGDNN_ASSERT_TENSOR_EQ_EPS_AVG_ALLOW_INVALID(
                    expected[i], computed[i], m_epsilon, m_max_avg_error,
                    m_max_avg_biased_error);
        } else {
            MEGDNN_ASSERT_TENSOR_EQ_EPS_AVG(
                    expected[i], computed[i], m_epsilon, m_max_avg_error,
                    m_max_avg_biased_error);
        }
    }
}

void CheckerHelper::copy_tensors_to_device(
        const TensorValueArray& dest, const TensorValueArray& src) {
    auto impl_h2d = [this](void* dst, const void* src, size_t sz) {
        megdnn_memcpy_H2D(m_handle_cur, dst, src, sz);
    };
    copy_tensors(dest, src, impl_h2d);
}

// vim: syntax=cpp.doxygen
