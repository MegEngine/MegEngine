#pragma once

#if __CUDACC_VER_MAJOR__ >= 9
#define __shfl(x, y, z)      __shfl_sync(0xffffffffu, x, y, z)
#define __shfl_up(x, y, z)   __shfl_up_sync(0xffffffffu, x, y, z)
#define __shfl_down(x, y, z) __shfl_down_sync(0xffffffffu, x, y, z)
#define __shfl_xor(x, y, z)  __shfl_xor_sync(0xffffffffu, x, y, z)
#endif

// vim: syntax=cpp.doxygen
