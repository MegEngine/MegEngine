#pragma once
#include "src/common/utils.h"
#if defined(MGB_ENABLE_CPUINFO_CHECK) && MGB_ENABLE_CPUINFO

#include <cpuinfo.h>

namespace megdnn {

const char* vendor_to_string(enum cpuinfo_vendor vendor);
const char* uarch_to_string(enum cpuinfo_uarch uarch);

}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
