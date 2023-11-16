#pragma once

#include "megcore_cambricon.h"
#include "src/cambricon/utils.mlu.h"
#include "test/cambricon/timer.h"
#include "test/common/benchmarker.h"
#include "test/common/rng.h"

namespace megdnn {
namespace test {

template <typename Opr>
class Benchmarker<Opr, CnrtTimer> : public BenchmarkerBase<Opr, CnrtTimer> {
public:
    Benchmarker(Handle* handle)
            : BenchmarkerBase<Opr, CnrtTimer>{
                      handle, CnrtTimer{m_queue, m_evt0, m_evt1}} {
        cnrt_check(cnrtNotifierCreate(&m_evt0));
        cnrt_check(cnrtNotifierCreate(&m_evt1));
        megcoreGetCNRTQueue(handle->megcore_computing_handle(), &m_queue);
    };
    ~Benchmarker() {
        cnrt_check(cnrtNotifierDestroy(m_evt0));
        cnrt_check(cnrtNotifierDestroy(m_evt1));
    }

private:
    cnrtQueue_t m_queue;
    cnrtNotifier_t m_evt0, m_evt1;
};

template <typename Opr>
using CnBenchmarker = Benchmarker<Opr, CnrtTimer>;

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
