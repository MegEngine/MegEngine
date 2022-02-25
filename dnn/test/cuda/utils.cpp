#include "./utils.h"

namespace megdnn {
namespace test {
bool check_compute_capability(int major, int minor) {
    int dev;
    cuda_check(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, dev));

    //! we just skip sm_62 here, which means jetson tx2
    //! unless require sm_62 explicitly
    if (prop.major == 6 && prop.minor == 2) {
        return prop.major == major && prop.minor == minor;
    }

    return prop.major > major || (prop.major == major && prop.minor >= minor);
}

bool check_compute_capability_eq(int major, int minor) {
    int dev;
    cuda_check(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, dev));
    return (prop.major == major && prop.minor == minor);
}
const cudaDeviceProp current_cuda_device_prop() {
    int dev;
    cuda_check(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, dev));
    return prop;
}
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
