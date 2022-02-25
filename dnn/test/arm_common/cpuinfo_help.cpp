#include "test/arm_common/cpuinfo_help.h"
#include "src/common/utils.h"
#if MGB_ENABLE_CPUINFO
std::mutex CpuInfoTmpReplace::m_cpuinfo_lock;
#endif
// vim: syntax=cpp.doxygen