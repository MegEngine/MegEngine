#include "./opr_impl.h"
#include "./kern.cuh"

#include "src/cuda/handle.h"

namespace megdnn {
namespace cuda {

void SleepForwardImpl::exec() {
    double seconds = m_param.time;
    megdnn_assert(seconds > 0);
    auto hdl = static_cast<HandleImpl*>(handle());
    sleep(hdl->stream(), hdl->device_prop().clockRate * 1e3 * seconds * 1.2);
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
