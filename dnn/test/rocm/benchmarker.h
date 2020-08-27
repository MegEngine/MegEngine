/**
 * \file dnn/test/rocm/benchmarker.h
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
#include "test/common/opr_proxy.h"
#include "megdnn/tensor_format.h"
#include "test/common/rng.h"
#include "test/rocm/fixture.h"
#include "src/rocm/utils.h"

#include "hip_header.h"

#include <map>

namespace megdnn {
namespace test {

template <typename Opr>
class ROCMBenchmarker {
public:
    using Param = typename Opr::Param;
    ROCMBenchmarker(Handle* handle_rocm, Handle* handle_naive);

    const Handle* handle() const { return m_handle_rocm; }

    /*!
     * \brief benchmark opr on current param/dtype/rng config
     * \returns elapsed time in ms
     *
     * ROCMBenchmarker would construct TensorLayout vectors from shapes and
     * dtypes and call exec(TensorLayoutArray &).
     */
    float exec(const TensorShapeArray& shapes);
    float exec(TensorLayoutArray layouts);

    //! disabiguate overloaded exec
    float execs(const TensorShapeArray& shapes) {
        return exec(make_layouts(shapes));
    }
    float execl(const TensorLayoutArray& layouts) { return exec(layouts); }
    ROCMBenchmarker& set_param(Param param) {
        m_param = param;
        return *this;
    }
    ROCMBenchmarker& set_dtype(size_t idx, DType dtype) {
        m_dtype[idx] = dtype;
        return *this;
    }
    ROCMBenchmarker& set_rng(size_t idx, RNG* rng) {
        m_rng[idx] = rng;
        return *this;
    }
    ROCMBenchmarker& set_proxy(const OprProxy<Opr>& proxy) {
        m_proxy = proxy;
        return *this;
    }
    ROCMBenchmarker& set_display(bool display) {
        m_display = display;
        return *this;
    }
    ROCMBenchmarker& set_fmt(size_t idx, TensorFormat fmt) {
        m_fmt[idx] = fmt;
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

private:
    class ROCMTimer {
    private:
        bool m_started, m_stopped;
        hipEvent_t m_event_start, m_event_end;
        hipStream_t m_stream;

    public:
        ROCMTimer() = delete;
        ROCMTimer(hipStream_t strm) : m_stream{strm} {
            hip_check(hipEventCreate(&m_event_start));
            hip_check(hipEventCreate(&m_event_end));
            reset();
        }
        ~ROCMTimer() {
            hip_check(hipEventDestroy(m_event_start));
            hip_check(hipEventDestroy(m_event_end));
        }
        void start() {
            megdnn_assert(!m_started);
            megdnn_assert(!m_stopped);
            m_started = true;
            hip_check(hipEventRecord(m_event_start, m_stream));
        }
        void stop() {
            megdnn_assert(m_started);
            megdnn_assert(!m_stopped);
            m_stopped = true;
            hip_check(hipEventRecord(m_event_end, m_stream));
        }
        float get_time_in_ms() const {
            megdnn_assert(m_started);
            megdnn_assert(m_stopped);
            hip_check(hipEventSynchronize(m_event_end));
            float ms;
            hip_check(hipEventElapsedTime(&ms, m_event_start, m_event_end));
            return ms;
        }
        void reset() {
            m_started = false;
            m_stopped = false;
        }
    };

    bool m_display = true;
    Handle* m_handle_naive;
    Handle* m_handle_rocm;
    std::unique_ptr<RNG> m_default_rng;
    std::map<size_t, RNG*> m_rng;
    std::map<size_t, DType> m_dtype;
    std::map<size_t, TensorFormat> m_fmt;
    Param m_param;
    OprProxy<Opr> m_proxy;
    ROCMTimer m_device_timer;
};

}  // namespace test
}  // namespace megdnn

#include "test/rocm/benchmarker.inl"

// vim: syntax=cpp.doxygen
