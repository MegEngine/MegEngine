#include "src/atlas/matrix_mul/algo.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_permute.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/matrix_mul/opr_impl.h"
#include "src/atlas/utils.h"

namespace megdnn {
namespace atlas {

MatrixMulForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        MatrixMulForwardImpl* o, const TensorLayout& A, const TensorLayout& B,
        const TensorLayout& C)
        : handle{concrete_handle(o->handle())},
          opr{o},
          layout_a{A},
          layout_b{B},
          layout_c{C} {}

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

MatrixMulForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        MatrixMulForwardImpl* opr, _megdnn_tensor_in A, _megdnn_tensor_in B,
        _megdnn_tensor_out C, _megdnn_workspace workspace)
        : SizeArgs(opr, A.layout, B.layout, C.layout),
          tensor_a{A},
          tensor_b{B},
          tensor_c{C},
          workspace{workspace} {}

bool MatrixMulForwardImpl::AlgoACL::is_available(const SizeArgs& args) const {
    //! TODO: to be complete
    MEGDNN_MARK_USED_VAR(args);
    return true;
}

size_t MatrixMulForwardImpl::AlgoACL::get_workspace_in_bytes(const SizeArgs&) const {
    return 0;
}

const char* MatrixMulForwardImpl::AlgoACL::name() const {
    return m_algo_name.c_str();
}

void MatrixMulForwardImpl::AlgoACL::exec(const ExecArgs& args) const {
    auto&& param = args.opr->param();

    AclTensor A(args.tensor_a);
    AclTensor B(args.tensor_b);
    AclTensor C(args.tensor_c);

    AclMem lhs_mem, rhs_mem;
    AclTensor lhs, rhs;

    if (param.transposeA) {
        TensorLayout lhs_lyt = args.tensor_a.layout.dimshuffle({1, 0});
        lhs_lyt.init_contiguous_stride();
        lhs_mem.alloc(lhs_lyt.access_bytes(), args.handle);
        lhs.init(lhs_mem.ptr(), lhs_lyt);
        AclIntArray permutation({1, 0});

        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        aclnn_check(aclnnPermuteGetWorkspaceSize(
                A.get(), permutation.get(), lhs.get(), &ws_size, &executor));
        AclMem ws(ws_size, args.handle);
        aclnn_check(aclnnPermute(ws.ptr(), ws_size, executor, args.handle->stream()));
    } else {
        lhs.init(args.tensor_a);
    }

    if (param.transposeB) {
        TensorLayout rhs_lyt = args.tensor_b.layout.dimshuffle({1, 0});
        rhs_lyt.init_contiguous_stride();
        rhs_mem.alloc(rhs_lyt.access_bytes(), args.handle);
        rhs.init(rhs_mem.ptr(), rhs_lyt);
        AclIntArray permutation({1, 0});

        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        aclnn_check(aclnnPermuteGetWorkspaceSize(
                B.get(), permutation.get(), rhs.get(), &ws_size, &executor));
        AclMem ws(ws_size, args.handle);
        aclnn_check(aclnnPermute(ws.ptr(), ws_size, executor, args.handle->stream()));
    } else {
        rhs.init(args.tensor_b);
    }

    int8_t cube_math_type = CUBE_KEEP_DTYPE;
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnn_check(aclnnMatmulGetWorkspaceSize(
            lhs.get(), rhs.get(), C.get(), cube_math_type, &ws_size, &executor));

    AclMem ws(ws_size, args.handle);
    aclnn_check(aclnnMatmul(ws.ptr(), ws_size, executor, args.handle->stream()));
}

MatrixMulForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_acl);
    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(MatrixMulForwardImpl)

}  // namespace atlas
}  // namespace megdnn
