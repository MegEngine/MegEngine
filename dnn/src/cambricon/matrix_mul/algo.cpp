#include "src/cambricon/matrix_mul/algo.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/handle.h"
#include "src/cambricon/utils.h"

namespace megdnn {
namespace cambricon {

MatrixMulForwardImpl::AlgoPack MatrixMulForwardImpl::sm_algo_pack;

MatrixMulForwardImpl::AlgoPack::AlgoPack() {
    all_algos.emplace_back(&default_matmul);
    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

struct MatMulCnnlDescs {
    CnnlTensorDescriptor a_desc, b_desc, c_desc;

    MatMulCnnlDescs(const MatrixMulForwardImpl::AlgoBase::SizeArgs& args) {
        cnnlDataType_t compute_dtype =
                convert_to_cnnl_datatype(args.layout_c.dtype.enumv());
        a_desc.set(args.layout_a);
        b_desc.set(args.layout_b);
        c_desc.set(args.layout_c);
    }
};

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

bool MatrixMulForwardImpl::AlgoDefault::is_available(const SizeArgs& args) const {
    auto&& layout_a = args.layout_a;
    auto&& layout_b = args.layout_b;
    auto&& layout_c = args.layout_c;
    return layout_a.dtype.enumv() == layout_b.dtype.enumv() &&
           (layout_a.dtype.enumv() == DTypeEnum::Float32 ||
            layout_a.dtype.enumv() == DTypeEnum::Float16) &&
           (layout_c.dtype.enumv() == DTypeEnum::Float32 ||
            layout_c.dtype.enumv() == DTypeEnum::Float16) &&
           args.opr->param().format == param::MatrixMul::Format::DEFAULT;
}

void MatrixMulForwardImpl::AlgoDefault::exec(const ExecArgs& args) const {
    auto handle = concrete_handle(args.opr->handle())->cnnl_handle();

    MatMulCnnlDescs descs(args);
    float cnnl_alpha = 1.0, cnnl_beta = 0.0;
    cnnl_check(cnnlMatMul(
            handle, args.opr->param().transposeA, args.opr->param().transposeB,
            &cnnl_alpha, descs.a_desc.desc(), args.tensor_a.raw_ptr(),
            descs.b_desc.desc(), args.tensor_b.raw_ptr(), &cnnl_beta,
            descs.c_desc.desc(), args.tensor_c.raw_ptr()));
}

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen