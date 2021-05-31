
#if __CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ >= 2)                 
// ignore warning of cutlass
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "src/cuda/matrix_mul/cutlass_matrix_mul_wrapper_batched_gemv_strided.cuinl"


  // Gemm operator cutlass_simt_sgemv_batched_strided_1x128_32_tt_align4x4
  using Operation_cutlass_simt_sgemv_batched_strided_1x128_32_tt_align4x4 = cutlass::gemm::kernel::DefaultGemv<
    cutlass::gemm::GemmShape<1, 128, 32>, 
    cutlass::gemm::GemmShape<1, 4, 4>, 
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor
  >;


template void megdnn::cuda::cutlass_wrapper::
  cutlass_vector_matrix_mul_batched_strided_wrapper<Operation_cutlass_simt_sgemv_batched_strided_1x128_32_tt_align4x4>(
      BatchedGemmCoord const& problem_size,
      const typename Operation_cutlass_simt_sgemv_batched_strided_1x128_32_tt_align4x4::ElementA* d_A, size_t lda, size_t batch_stride_a, 
      const typename Operation_cutlass_simt_sgemv_batched_strided_1x128_32_tt_align4x4::ElementB* d_B, size_t ldb, size_t batch_stride_b, 
      typename Operation_cutlass_simt_sgemv_batched_strided_1x128_32_tt_align4x4::ElementCD* d_C, size_t ldc, size_t batch_stride_c,
      cudaStream_t stream);

#pragma GCC diagnostic pop
#endif
