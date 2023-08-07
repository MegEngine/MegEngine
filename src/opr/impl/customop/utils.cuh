#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define cuda_check(_x)                        \
    do {                                      \
        cudaError_t _err = (_x);              \
        if (_err != cudaSuccess) {            \
            std::string x = std::string(#_x); \
            char line[10];                    \
            sprintf(line, "%d", __LINE__);    \
        }                                     \
    } while (0)

#define after_kernel_launch()           \
    do {                                \
        cuda_check(cudaGetLastError()); \
    } while (0)