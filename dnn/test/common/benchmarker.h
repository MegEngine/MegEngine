/**
 * \file dnn/test/common/benchmarker.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include <map>
#include <memory>
#include <regex>
#include <vector>
#include "megdnn/basic_types.h"
#include "megdnn/tensor_format.h"
#include "test/common/opr_algo_proxy.h"
#include "test/common/opr_proxy.h"
#include "test/common/rng.h"
#include "test/common/timer.h"

namespace megdnn {
namespace test {

template <typename Opr, typename T>
class BenchmarkerBase {
public:
    using Param = typename Opr::Param;
    using TensorValueArray = TensorNDArray;
    using BeforeExecCallback =
            std::function<void(Opr*, const TensorValueArray&)>;
    using TensorsConstriant = std::function<void(TensorValueArray& tensors)>;

    BenchmarkerBase(Handle* handle, T timer)
            : m_timer(timer),
              m_handle_naive(create_cpu_handle(2, false)),
              m_handle(handle),
              m_default_rng(new NormalRNG()),
              m_param(Param()),
              m_proxy{new OprProxy<Opr>()} {}

    const Handle* handle() const { return m_handle; }

    /*!
     * \brief benchmark opr on current param/dtype/rng config
     * \returns elapsed time in ms
     *
     * Benchmarker would construct TensorLayout vectors from shapes and
     * dtypes and call exec(TensorLayoutArray &).
     */
    float exec(const TensorShapeArray& shapes) {
        return exec(make_layouts(shapes));
    }
    float exec(TensorLayoutArray layouts);

    float exect(const TensorValueArray& testcase_in);

    //! disabiguate overloaded exec
    float execs(const TensorShapeArray& shapes) { return exec(shapes); }
    float execl(const TensorLayoutArray& layouts) { return exec(layouts); }
    BenchmarkerBase& set_param(Param param) {
        m_param = param;
        return *this;
    }
    BenchmarkerBase& set_dtype(size_t idx, DType dtype) {
        m_dtype[idx] = dtype;
        return *this;
    }
    BenchmarkerBase& set_rng(size_t idx, RNG* rng) {
        m_rng[idx] = rng;
        return *this;
    }
    BenchmarkerBase& set_fmt(size_t idx, TensorFormat fmt) {
        m_fmt[idx] = fmt;
        return *this;
    }
    BenchmarkerBase& set_tensors_constraint(
            const TensorsConstriant& tensor_constraint) {
        m_tensor_constraint = tensor_constraint;
        return *this;
    }
    TensorLayoutArray make_layouts(const TensorShapeArray& shapes) {
        TensorLayoutArray layouts(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            DType dt = (m_dtype.find(i) != m_dtype.end() ? m_dtype[i]
                                                         : dtype::Float32());
            TensorFormat fmt = (m_fmt.find(i) != m_fmt.end()
                                        ? m_fmt[i]
                                        : DefaultTensorFormat::make());
            layouts[i] = TensorLayout(shapes[i], dt, fmt);
        }
        return layouts;
    }
    BenchmarkerBase& set_proxy(std::unique_ptr<OprProxy<Opr>>& proxy) {
        m_proxy.reset(nullptr);
        m_proxy = std::move(proxy);
        return *this;
    }
    std::unique_ptr<OprProxy<Opr>>& proxy() { return m_proxy; }
    BenchmarkerBase& set_times(size_t times) {
        m_times = times;
        return *this;
    }
    BenchmarkerBase& set_display(bool display) {
        m_display = display;
        return *this;
    }
    //! set a callback to be invoked before executing the operator
    BenchmarkerBase& set_before_exec_callback(const BeforeExecCallback& cb) {
        m_before_exec_callback = cb;
        return *this;
    }

    /*!
     * \brief set adaptive benchmarking: ignore set_times() and find
     * suitable times to run for given duration;
     *
     * Note: the value returned by exec() would be average time per run,
     * rather than total elapsed time, if this is enabled.
     */
    BenchmarkerBase& set_adaptive_benchmark(float tot_time_in_secs) {
        m_adaptive_secs = tot_time_in_secs;
        return *this;
    }

    //! get the opr impl so setting other than param() can be modified
    Opr* opr() {
        if (!m_opr) {
            m_opr = m_handle->create_operator<Opr>();
        }
        return m_opr.get();
    }

    const Param& param() const { return m_param; }

private:
    T m_timer;
    bool m_display = true;
    size_t m_times = 1;
    float m_adaptive_secs = 0;
    std::unique_ptr<Handle> m_handle_naive;
    Handle* m_handle;
    std::unique_ptr<RNG> m_default_rng;
    std::map<size_t, RNG*> m_rng;
    std::map<size_t, DType> m_dtype;
    std::map<size_t, TensorFormat> m_fmt;
    Param m_param;
    std::unique_ptr<OprProxy<Opr>> m_proxy;
    BeforeExecCallback m_before_exec_callback;
    std::unique_ptr<Opr> m_opr;
    TensorsConstriant m_tensor_constraint;
};

template <typename Opr, typename T>
float BenchmarkerBase<Opr, T>::exec(TensorLayoutArray layouts) {
    auto opr = this->opr();
    opr->param() = m_param;
    auto user_layouts = layouts;
    m_proxy->deduce_layout(opr, layouts);
    for (size_t i = 0; i < layouts.size(); ++i)
        if (user_layouts[i].ndim > 0) {
            auto run = [&]() {
                ASSERT_TRUE(layouts[i].eq_shape(user_layouts[i]))
                        << "User provided shape is "
                        << user_layouts[i].TensorShape::to_string()
                        << "\nExpected shape is "
                        << layouts[i].TensorShape::to_string();
            };
            run();
        }
    auto allocate = [&layouts](Handle* handle) {
        TensorNDArray tensors(layouts.size());
        auto trans_func = [handle](const TensorLayout& layout) {
            auto span = layout.span();
            TensorND res;
            res.raw_ptr = static_cast<uint8_t*>(
                                  megdnn_malloc(handle, span.dist_byte())) +
                          span.low_byte;
            res.layout = layout;
            return res;
        };
        std::transform(layouts.begin(), layouts.end(), tensors.begin(),
                       trans_func);
        return tensors;
    };
    auto tensors_cur = allocate(m_handle);
    auto tensors_cur_host = allocate(m_handle_naive.get());
    // init
    for (size_t i = 0; i < tensors_cur_host.size(); ++i) {
        TensorND& tensor = tensors_cur_host[i];
        auto rng = m_rng[i];
        if (!rng)
            rng = m_default_rng.get();
        rng->gen(tensor);
    }
    if (m_tensor_constraint) {
        m_tensor_constraint(tensors_cur_host);
    }
    for (size_t i = 0; i < tensors_cur_host.size(); ++i) {
        TensorND& tensor = tensors_cur_host[i];
        if (tensor.layout.ndim == 0)
            continue;
        auto size = tensor.layout.span().high_byte;
        megdnn_memcpy_H2D(m_handle, tensors_cur[i].raw_ptr, tensor.raw_ptr,
                          size);
    }
    if (m_before_exec_callback) {
        m_before_exec_callback(opr, tensors_cur);
    }
    // run
    // warm up
    m_proxy->exec(opr, tensors_cur);
    megcoreSynchronize(m_handle->megcore_computing_handle());

    if (m_adaptive_secs) {
        // find m_times for adaptive benchmarking
        m_times = 0;
        int cur_times = 1;
        auto remain_time = m_adaptive_secs * 1e6;
        while (remain_time > 0) {
            m_timer.reset();
            m_timer.start();
            for (int i = 0; i < cur_times; ++i)
                m_proxy->exec(opr, tensors_cur);
            megcoreSynchronize(m_handle->megcore_computing_handle());
            m_timer.stop();
            m_times += cur_times;
            auto this_run_time = m_timer.get_time_in_us();
            remain_time -= this_run_time;
            cur_times = std::min(
                    cur_times * 2,
                    std::max<int>(1, remain_time / this_run_time * cur_times));
        }
    }
    m_timer.reset();
    m_timer.start();
    for (size_t t = 0; t < m_times; ++t)
        m_proxy->exec(opr, tensors_cur);
    megcoreSynchronize(m_handle->megcore_computing_handle());
    m_timer.stop();
    auto time_in_ms = m_timer.get_time_in_us() / 1e3;
    if (m_display) {
        std::cout << "Total time is " << time_in_ms << "ms "
                  << "for " << m_times << " run(s)." << std::endl;
    }
    auto free = [](Handle* handle, TensorNDArray& tensors) {
        std::for_each(tensors.begin(), tensors.end(),
                      [handle](const TensorND& tensor) {
                          megdnn_free(handle, tensor.raw_ptr);
                      });
    };
    free(m_handle, tensors_cur);
    free(m_handle_naive.get(), tensors_cur_host);
    if (m_adaptive_secs)
        time_in_ms /= m_times;
    return time_in_ms;
}

template <typename Opr, typename T>
float BenchmarkerBase<Opr, T>::exect(const TensorValueArray& testcase_in) {
    auto opr = this->opr();
    opr->param() = m_param;
    TensorLayoutArray layouts;
    TensorNDArray tensors_cur_host;
    for (auto& inp : testcase_in) {
        layouts.push_back(inp.layout);
        tensors_cur_host.emplace_back(inp);
    }
    auto user_layouts = layouts;
    m_proxy->deduce_layout(opr, layouts);
    for (size_t i = 0; i < layouts.size(); ++i)
        if (user_layouts[i].ndim > 0) {
            auto run = [&]() {
                ASSERT_TRUE(layouts[i].eq_shape(user_layouts[i]))
                        << "User provided shape is "
                        << user_layouts[i].TensorShape::to_string()
                        << "\nExpected shape is "
                        << layouts[i].TensorShape::to_string();
            };
            run();
        }
    auto allocate = [&layouts](Handle* handle) {
        TensorNDArray tensors(layouts.size());
        auto trans_func = [handle](const TensorLayout& layout) {
            auto span = layout.span();
            TensorND res;
            res.raw_ptr = static_cast<uint8_t*>(
                                  megdnn_malloc(handle, span.dist_byte())) +
                          span.low_byte;
            res.layout = layout;
            return res;
        };
        std::transform(layouts.begin(), layouts.end(), tensors.begin(),
                       trans_func);
        return tensors;
    };
    auto tensors_cur = allocate(m_handle);
    //! init
    for (size_t i = 0; i < tensors_cur_host.size(); ++i) {
        TensorND& tensor = tensors_cur_host[i];
        auto size = tensor.layout.span().high_byte;
        if (tensor.layout.ndim == 0)
            continue;
        megdnn_memcpy_H2D(m_handle, tensors_cur[i].raw_ptr, tensor.raw_ptr,
                          size);
    }
    if (m_before_exec_callback) {
        m_before_exec_callback(opr, tensors_cur);
    }
    //! run
    //! warm up
    m_proxy->exec(opr, tensors_cur);
    megcoreSynchronize(m_handle->megcore_computing_handle());

    if (m_adaptive_secs) {
        //! find m_times for adaptive benchmarking
        m_times = 0;
        int cur_times = 1;
        auto remain_time = m_adaptive_secs * 1e6;
        while (remain_time > 0) {
            m_timer.reset();
            m_timer.start();
            for (int i = 0; i < cur_times; ++i)
                m_proxy->exec(opr, tensors_cur);
            megcoreSynchronize(m_handle->megcore_computing_handle());
            m_timer.stop();
            m_times += cur_times;
            auto this_run_time = m_timer.get_time_in_us();
            remain_time -= this_run_time;
            cur_times = std::min(
                    cur_times * 2,
                    std::max<int>(1, remain_time / this_run_time * cur_times));
        }
    }
    m_timer.reset();
    m_timer.start();
    for (size_t t = 0; t < m_times; ++t)
        m_proxy->exec(opr, tensors_cur);
    megcoreSynchronize(m_handle->megcore_computing_handle());
    m_timer.stop();
    auto time_in_ms = m_timer.get_time_in_us() / 1e3;
    if (m_display) {
        std::cout << "Total time is " << time_in_ms << "ms "
                  << "for " << m_times << " run(s)." << std::endl;
    }
    auto free = [](Handle* handle, TensorNDArray& tensors) {
        std::for_each(tensors.begin(), tensors.end(),
                      [handle](const TensorND& tensor) {
                          megdnn_free(handle, tensor.raw_ptr);
                      });
    };
    free(m_handle, tensors_cur);
    if (m_adaptive_secs)
        time_in_ms /= m_times;
    return time_in_ms;
}

template <typename Opr, typename T = Timer>
class Benchmarker;

template <typename Opr>
class Benchmarker<Opr, Timer> : public BenchmarkerBase<Opr, Timer> {
public:
    Benchmarker(Handle* handle)
            : BenchmarkerBase<Opr, Timer>{handle, Timer{}} {}
};

////////////////// Algo Benchmark ////////////////////////
template <typename Opr, typename Proxy = OprProxy<Opr>, typename T = Timer>
float algo_benchmark(Benchmarker<Opr, T>& benchmark, TensorLayoutArray layouts,
                     const std::string& algo_base) {
    Proxy proxy;
    auto opr = benchmark.opr();
    opr->param() = benchmark.param();
    proxy.deduce_layout(opr, layouts);
    auto algos = OprAlgoProxy<Opr>::get_all_algorithms_info(opr, layouts);
    float min_used = std::numeric_limits<float>::max();
    bool execed = false;
    for (auto i : algos) {
        if (std::regex_match(i.name,
                             std::regex("(" + algo_base + ")(.*)"))) {
            opr->execution_policy().algo = i;
            auto used = benchmark.exec(layouts);
            min_used = std::min(min_used, used);
            printf("run algo: %s used: %f ms min_used: %f ms\n", i.name.c_str(),
                   used, min_used);
            execed = true;
        }
    }
    megdnn_assert(execed, "no algo start with %s\n", algo_base.c_str());
    return min_used;
}

template <typename Opr, typename Proxy = OprProxy<Opr>, typename T = Timer>
float algo_benchmark(Benchmarker<Opr, T>& benchmark, TensorShapeArray shapes,
                     const std::string& algo_base) {
    return algo_benchmark(benchmark, benchmark.make_layouts(shapes), algo_base);
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
