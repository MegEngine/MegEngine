#include "test/atlas/fixture.h"
#include "src/atlas/handle.h"
#include "src/atlas/megcore/device_context.hpp"
#include "src/atlas/utils.h"

#include "test/common/memory_manager.h"
#include "test/common/random_state.h"
#include "test/common/utils.h"

#include "acl/acl.h"

#include <cstdlib>

using namespace megdnn;
using namespace test;

void ATLAS::SetUp() {
    RandomState::reset();

    // use card 0
    megcore_check(megcoreCreateDeviceHandle(&m_dev_handle, megcorePlatformAtlas, 0));

    megcoreActivate(m_dev_handle);
    megcoreComputingHandle_t comp_handle;
    megcore_check(megcoreCreateComputingHandle(&comp_handle, m_dev_handle));
    m_handle_atlas = Handle::make(comp_handle);
    megdnn_assert(m_handle_atlas);
}

Handle* ATLAS::handle_naive() {
    if (!m_handle_naive)
        m_handle_naive = create_cpu_handle(2);
    return m_handle_naive.get();
}

void ATLAS::TearDown() {
    m_handle_naive.reset();
    m_handle_atlas.reset();
    MemoryManagerHolder::instance()->clear();
    megcoreDeactivate(m_dev_handle);
}

// vim: syntax=cpp.doxygen
