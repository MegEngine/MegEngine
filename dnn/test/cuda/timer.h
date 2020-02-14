/**
 * \file dnn/test/cuda/timer.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include "test/common/utils.h"

namespace megdnn {
namespace test {

class CUTimer {
public:
    CUTimer(cudaStream_t& stream, cudaEvent_t& evt0, cudaEvent_t& evt1)
            : m_stream{stream}, m_evt0{evt0}, m_evt1{evt1} {
        reset();
    }

    void reset() {
        m_started = false;
        m_stopped = false;
    }
    void start() {
        megdnn_assert(!m_started);
        megdnn_assert(!m_stopped);
        m_started = true;
        cudaProfilerStart();
        cudaEventRecord(m_evt0, m_stream);
    }
    void stop() {
        megdnn_assert(m_started);
        megdnn_assert(!m_stopped);
        m_stopped = true;
        cudaEventRecord(m_evt1, m_stream);
        cudaProfilerStop();
    }
    size_t get_time_in_us() const {
        cudaStreamSynchronize(m_stream);
        float t = -1;
        cudaEventElapsedTime(&t, m_evt0, m_evt1);
        return static_cast<size_t>(t * 1e3);
    }

private:
    bool m_started, m_stopped;
    size_t m_start_point, m_stop_point;
    cudaStream_t& m_stream;
    cudaEvent_t &m_evt0, &m_evt1;
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
