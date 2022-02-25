#include "./algos.h"
#include <cuda.h>
#include "src/common/algo_base.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/conv_bias/opr_impl.h"
#include "src/cuda/utils.h"
#if CUDA_VERSION >= 10010
#include <cublasLt.h>
#endif

using namespace megdnn;
using namespace cuda;

MatrixMulForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&cublas);
#if CUDA_VERSION >= 10000
    all_algos.push_back(&wmma_uint4x4x32);
#endif
#if CUDA_VERSION >= 10010
    all_algos.push_back(&cublas_lt);
#endif
#if !MEGDNN_DISABLE_FLOAT16
    all_algos.push_back(&bfloat16);
#endif
#if CUDA_VERSION >= 9020
    fill_cutlass_algos();
    for (auto&& algo : simt_float32) {
        all_algos.push_back(&algo);
    }
    for (auto&& algo : simt_float32_split_k) {
        all_algos.push_back(&algo);
    }
    for (auto&& algo : simt_float32_gemv_batched_strided) {
        all_algos.push_back(&algo);
    }
#if CUDA_VERSION >= 10010
    for (auto&& algo : tensorop_float16) {
        all_algos.push_back(&algo);
    }
    for (auto&& algo : tensorop_float16_split_k) {
        all_algos.push_back(&algo);
    }
#endif
#endif

    all_algos.push_back(&naive);

    std::vector<cudnnConvolutionFwdAlgo_t> cudnn_conv_enum;
    for (auto&& algo : CudnnAlgoPack::conv_fwd_algos()) {
        cudnn_conv_enum.push_back(algo.first);
    }

    for (auto&& algo : cudnn_conv_enum) {
        conv1x1.push_back(AlgoConv1X1CUDNN(algo));
    }
    for (size_t i = 0; i < conv1x1.size(); ++i) {
        all_algos.push_back(&conv1x1[i]);
    }

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

#if CUDA_VERSION >= 9020
void MatrixMulForwardImpl::AlgoPack::fill_cutlass_algos() {
    using AlgoParam = AlgoCutlassMatrixMulBase::AlgoParam;
    simt_float32.emplace_back(AlgoParam{64, 256, 8, 32, 64, 8});
    simt_float32.emplace_back(AlgoParam{256, 64, 8, 64, 32, 8});
    simt_float32.emplace_back(AlgoParam{32, 256, 8, 16, 64, 8});
    simt_float32.emplace_back(AlgoParam{256, 32, 8, 64, 16, 8});
    simt_float32.emplace_back(AlgoParam{128, 128, 8, 32, 64, 8});
    simt_float32.emplace_back(AlgoParam{128, 64, 8, 64, 32, 8});
    simt_float32.emplace_back(AlgoParam{64, 128, 8, 32, 64, 8});
    simt_float32.emplace_back(AlgoParam{128, 32, 8, 64, 32, 8});
    simt_float32.emplace_back(AlgoParam{32, 128, 8, 32, 64, 8});
    simt_float32.emplace_back(AlgoParam{64, 64, 8, 32, 64, 8});
    simt_float32.emplace_back(AlgoParam{32, 64, 8, 32, 64, 8});
    simt_float32.emplace_back(AlgoParam{64, 32, 8, 64, 32, 8});
    simt_float32.emplace_back(AlgoParam{32, 32, 8, 32, 32, 8});
    simt_float32.emplace_back(AlgoParam{8, 32, 8, 8, 32, 8});
    simt_float32.emplace_back(AlgoParam{16, 32, 8, 16, 32, 8});
    simt_float32.emplace_back(AlgoParam{16, 64, 8, 16, 64, 8});
    simt_float32.emplace_back(AlgoParam{16, 128, 8, 16, 64, 8});
    simt_float32_split_k.emplace_back(AlgoParam{64, 256, 8, 32, 64, 8});
    simt_float32_split_k.emplace_back(AlgoParam{256, 64, 8, 64, 32, 8});
    simt_float32_split_k.emplace_back(AlgoParam{32, 256, 8, 16, 64, 8});
    simt_float32_split_k.emplace_back(AlgoParam{256, 32, 8, 64, 16, 8});
    simt_float32_split_k.emplace_back(AlgoParam{128, 128, 8, 32, 64, 8});
    simt_float32_split_k.emplace_back(AlgoParam{128, 64, 8, 64, 32, 8});
    simt_float32_split_k.emplace_back(AlgoParam{64, 128, 8, 32, 64, 8});
    simt_float32_split_k.emplace_back(AlgoParam{128, 32, 8, 64, 32, 8});
    simt_float32_split_k.emplace_back(AlgoParam{32, 128, 8, 32, 64, 8});
    simt_float32_split_k.emplace_back(AlgoParam{64, 64, 8, 32, 64, 8});
    simt_float32_split_k.emplace_back(AlgoParam{32, 64, 8, 32, 64, 8});
    simt_float32_split_k.emplace_back(AlgoParam{64, 32, 8, 64, 32, 8});
    simt_float32_split_k.emplace_back(AlgoParam{32, 32, 8, 32, 32, 8});
    simt_float32_split_k.emplace_back(AlgoParam{8, 32, 8, 8, 32, 8});
    simt_float32_split_k.emplace_back(AlgoParam{16, 32, 8, 16, 32, 8});
    simt_float32_split_k.emplace_back(AlgoParam{16, 64, 8, 16, 64, 8});
    simt_float32_split_k.emplace_back(AlgoParam{16, 128, 8, 16, 64, 8});
    simt_float32_gemv_batched_strided.emplace_back(128);
    simt_float32_gemv_batched_strided.emplace_back(64);
    simt_float32_gemv_batched_strided.emplace_back(32);
#define FOREACH_CUTLASS_MATMUL_MMA_SM70_SHAPES(cb) \
    cb(256, 128, 32, 64, 64, 32, 8, 8, 4);         \
    cb(128, 256, 32, 64, 64, 32, 8, 8, 4);         \
    cb(128, 128, 32, 64, 64, 32, 8, 8, 4);
#define FOREACH_CUTLASS_MATMUL_MMA_SM75_SHAPES(cb) \
    cb(256, 128, 32, 64, 64, 32, 16, 8, 8);        \
    cb(128, 256, 32, 64, 64, 32, 16, 8, 8);        \
    cb(128, 128, 32, 64, 64, 32, 16, 8, 8);
#define cb(...)                                            \
    tensorop_float16.emplace_back(AlgoParam{__VA_ARGS__}); \
    tensorop_float16_split_k.emplace_back(AlgoParam{__VA_ARGS__});
#if CUDA_VERSION >= 10010
    FOREACH_CUTLASS_MATMUL_MMA_SM70_SHAPES(cb)
#endif
#if CUDA_VERSION >= 10020
    FOREACH_CUTLASS_MATMUL_MMA_SM75_SHAPES(cb)
#endif
#undef cb
#undef FOREACH_CUTLASS_MATMUL_MMA_SM70_SHAPES
#undef FOREACH_CUTLASS_MATMUL_MMA_SM75_SHAPES
}
#endif

MatrixMulForwardImpl::AlgoPack MatrixMulForwardImpl::sm_algo_pack;

MEGDNN_DEF_GET_ALGO_FROM_DESC(MatrixMulForwardImpl)

MatrixMulForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        MatrixMulForwardImpl* o, const TensorLayout& A, const TensorLayout& B,
        const TensorLayout& C)
        : opr{o}, layout_a{A}, layout_b{B}, layout_c{C} {}

MatrixMulForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        MatrixMulForwardImpl* opr, _megdnn_tensor_in A, _megdnn_tensor_in B,
        _megdnn_tensor_out C, _megdnn_workspace workspace)
        : SizeArgs(opr, A.layout, B.layout, C.layout),
          tensor_a{A},
          tensor_b{B},
          tensor_c{C},
          workspace{workspace} {}

std::string MatrixMulForwardImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& param = opr->param();
    size_t m = layout_a.shape[0], n = layout_b.shape[1],
           k = layout_a.shape[param.transposeA ? 0 : 1];
    MEGDNN_MARK_USED_VAR(m);
    MEGDNN_MARK_USED_VAR(n);
    MEGDNN_MARK_USED_VAR(k);
    return ssprintf(
            "A={%zux%zu},B={%zux%zu},C={%zux%zu},Transpose A=%d,Transpose "
            "B=%d,ldA=%zu,ldB=%zu,ldC=%zu",
            m, k, k, n, m, n, param.transposeA, param.transposeB, layout_a.stride[0],
            layout_b.stride[0], layout_c.stride[0]);
}

// vim: syntax=cpp.doxygen
