/**
 * \file dnn/test/cuda/benchmark.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "test/common/rng.h"
#include "test/common/benchmarker.h"
#include "test/cuda/timer.h"
#include "megcore_cuda.h"

namespace megdnn {
namespace test {

template <typename Opr>
class Benchmarker<Opr, CUTimer> : public BenchmarkerBase<Opr, CUTimer> {
public:
    Benchmarker(Handle* handle)
            : BenchmarkerBase<Opr, CUTimer>{handle,
                                            CUTimer{m_stream, m_evt0, m_evt1}} {
        cudaEventCreate(&m_evt0);
        cudaEventCreate(&m_evt1);
        megcoreGetCUDAStream(handle->megcore_computing_handle(), &m_stream);
    };
    ~Benchmarker() {
        cudaEventDestroy(m_evt0);
        cudaEventDestroy(m_evt1);
    }

private:
    cudaStream_t m_stream;
    cudaEvent_t m_evt0, m_evt1;
};

template <typename Opr>
using CUBenchmarker = Benchmarker<Opr, CUTimer>;

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
