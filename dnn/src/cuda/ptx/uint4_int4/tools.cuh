#include <cuda_runtime.h>

#if (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)
extern "C" {
//
// This NVVM intrinsic is subject to change in future versions of CUDA.
// Clients should not call it directly. Rather, they should use the
// cutlass::arch::ldsm<>() template.
//
__device__ uint32_t __nvvm_get_smem_pointer(void*);
}
#endif

inline __device__ unsigned get_smem_pointer(void* ptr) {
#if (defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 11)
    //
    // This NVVM intrinsic converts an address in shared memory to a plain
    // unsigned integer. This is necessary to pass to shared memory instructions
    // in inline PTX.
    //
    // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only
    // available in 10.2].
    //
    //__device__ size_t __cvta_generic_to_shared(void* ptr);

    /// CUTLASS helper to get SMEM pointer
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));

#elif (defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ == 10 && \
       __CUDACC_VER_MINOR__ >= 2)

    return __nvvm_get_smem_pointer(ptr);

#elif defined(__CUDA_ARCH__)

    uint32_t smem_ptr;

    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));

    return smem_ptr;

#else

    return 0;
#endif
}
