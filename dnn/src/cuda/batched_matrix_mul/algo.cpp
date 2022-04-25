#include "./algo.h"
#include <cuda.h>
#include "src/cuda/utils.h"
#if CUDA_VERSION >= 10010
#include <cublasLt.h>
#endif

using namespace megdnn;
using namespace cuda;

BatchedMatrixMulForwardImpl::AlgoPack BatchedMatrixMulForwardImpl::sm_algo_pack;

std::string BatchedMatrixMulForwardImpl::AlgoBase::SizeArgs::to_string() const {
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

BatchedMatrixMulForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        BatchedMatrixMulForwardImpl* o, const TensorLayout& A, const TensorLayout& B,
        const TensorLayout& C)
        : opr(o), layout_a(A), layout_b(B), layout_c(C){};

BatchedMatrixMulForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        BatchedMatrixMulForwardImpl* o, _megdnn_tensor_in A, _megdnn_tensor_in B,
        _megdnn_tensor_in C, _megdnn_workspace workspace)
        : SizeArgs(o, A.layout, B.layout, C.layout),
          tensor_a{A},
          tensor_b{B},
          tensor_c{C},
          workspace{workspace} {}

BatchedMatrixMulForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&cublas);
#if CUDA_VERSION >= 10010
    all_algos.push_back(&cublasLt);
#endif
    all_algos.push_back(&int8x8x32);
    all_algos.push_back(&brute_force);
    all_algos.push_back(&naive_bmm);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(BatchedMatrixMulForwardImpl)

// vim: syntax=cpp.doxygen
