#include "test/fallback/fixture.h"

#include "test/common/memory_manager.h"
#include "test/common/random_state.h"
#include "test/common/utils.h"

namespace megdnn {
namespace test {

void FALLBACK::SetUp() {
    RandomState::reset();
    m_handle = create_cpu_handle(1);
}

void FALLBACK::TearDown() {
    m_handle.reset();
    MemoryManagerHolder::instance()->clear();
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
