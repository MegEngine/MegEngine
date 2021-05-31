
#if __CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ >= 2)                 
// ignore warning of cutlass
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "src/cuda/matrix_mul/cutlass_matrix_mul_wrapper.cuinl"


  // Gemm operator cutlass_simt_sgemm_split_k_parallel_32x64_8x2_tt_align1
  using Operation_cutlass_simt_sgemm_split_k_parallel_32x64_8x2_tt_align1 = cutlass::gemm::device::GemmSplitKParallel<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >
  >;


template void megdnn::cuda::cutlass_wrapper::cutlass_matrix_mul_wrapper<Operation_cutlass_simt_sgemm_split_k_parallel_32x64_8x2_tt_align1>(
  const typename Operation_cutlass_simt_sgemm_split_k_parallel_32x64_8x2_tt_align1::ElementA* d_A, size_t lda, 
  const typename Operation_cutlass_simt_sgemm_split_k_parallel_32x64_8x2_tt_align1::ElementB* d_B, size_t ldb,  
  typename Operation_cutlass_simt_sgemm_split_k_parallel_32x64_8x2_tt_align1::ElementC* d_C, size_t ldc,  
  int* workspace, 
  cutlass::gemm::GemmCoord const& problem_size,   
  typename Operation_cutlass_simt_sgemm_split_k_parallel_32x64_8x2_tt_align1::EpilogueOutputOp::Params const& epilogue, 
  cudaStream_t stream, int split_k_slices);

#pragma GCC diagnostic pop
#endif
