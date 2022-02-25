#include "test/x86/fixture.h"

#include "test/common/memory_manager.h"
#include "test/common/random_state.h"
#include "test/common/utils.h"

namespace megdnn {
namespace test {

void X86::TearDown() {
    m_handle.reset();
    m_fallback_handle.reset();
    MemoryManagerHolder::instance()->clear();
}

Handle* X86::fallback_handle() {
    if (!m_fallback_handle) {
        m_fallback_handle = create_cpu_handle(1);
    }
    return m_fallback_handle.get();
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
