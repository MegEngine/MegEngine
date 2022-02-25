#pragma once

#include "megcore_cuda.h"
#include "test/common/benchmarker.h"
#include "test/common/rng.h"
#include "test/cuda/timer.h"

namespace megdnn {
namespace test {

template <typename Opr>
class Benchmarker<Opr, CUTimer> : public BenchmarkerBase<Opr, CUTimer> {
public:
    Benchmarker(Handle* handle)
            : BenchmarkerBase<Opr, CUTimer>{handle, CUTimer{m_stream, m_evt0, m_evt1}} {
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
