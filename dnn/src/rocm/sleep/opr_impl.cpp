#include "hcc_detail/hcc_defs_prologue.h"

#include "./kern.h.hip"
#include "./opr_impl.h"

#include "src/rocm/handle.h"

namespace megdnn {
namespace rocm {

void SleepForwardImpl::exec() {
    double seconds = m_param.time;
    megdnn_assert(seconds > 0);
    auto hdl = static_cast<HandleImpl*>(handle());
    sleep(hdl->stream(), hdl->device_prop().clockRate * 1000 * seconds);
}

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
