#pragma once
#include "src/cambricon/utils.mlu.h"
#include "test/common/utils.h"

namespace megdnn {
namespace test {

class CnrtTimer {
public:
    CnrtTimer(cnrtQueue_t& queue, cnrtNotifier_t& evt0, cnrtNotifier_t& evt1)
            : m_queue{queue}, m_evt0{evt0}, m_evt1{evt1} {
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
        cnrt_check(cnrtPlaceNotifier(m_evt0, m_queue));
    }
    void stop() {
        megdnn_assert(m_started);
        megdnn_assert(!m_stopped);
        m_stopped = true;
        cnrt_check(cnrtPlaceNotifier(m_evt1, m_queue));
    }
    size_t get_time_in_us() const {
        cnrt_check(cnrtQueueSync(m_queue));
        float t = -1;
        cnrtNotifierElapsedTime(m_evt0, m_evt1, &t);
        return static_cast<size_t>(t * 1e3);
    }

private:
    bool m_started, m_stopped;
    size_t m_start_point, m_stop_point;
    cnrtQueue_t& m_queue;
    cnrtNotifier_t &m_evt0, &m_evt1;
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
