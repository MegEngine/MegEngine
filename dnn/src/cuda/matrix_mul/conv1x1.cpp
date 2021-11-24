#include <cuda.h>
#include "./algos.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/conv_bias/opr_impl.h"
#include "src/cuda/handle.h"
#include "src/cuda/relayout/opr_impl.h"
#include "src/cuda/transpose/opr_impl.h"
#include "src/cuda/utils.h"
using namespace megdnn;
using namespace cuda;

namespace {

std::unique_ptr<ConvBiasForward> prepare_conv_opr(
        const MatrixMulForwardImpl::AlgoBase::SizeArgs& args) {
    auto conv_bias_opr_ptr = args.opr->handle()->create_operator<ConvBiasForward>();

    auto conv_param_computemode =
            (args.opr->param().compute_mode == param::MatrixMul::ComputeMode::DEFAULT)
                    ? param::Convolution::ComputeMode::DEFAULT
                    : param::Convolution::ComputeMode::FLOAT32;
    conv_bias_opr_ptr->param() = {
            param::ConvBias::NonlineMode::IDENTITY,
            param::Convolution::Mode::CROSS_CORRELATION,
            param::Convolution::Sparse::DENSE,
            param::Convolution::Format::NCHW,
            0,  // pad_h
            0,  // pad_w
            1,  // stride_h
            1,  // stride_w
            1,  // dilate_h
            1,  // dilate_w
            conv_param_computemode};

    return conv_bias_opr_ptr;
}
std::tuple<size_t, size_t, size_t> gen_matrixmul_shape(
        const MatrixMulForwardImpl::AlgoBase::SizeArgs& args) {
    size_t m, k, n;
    if (!args.opr->param().transposeA) {
        m = args.layout_a.shape[0];
        k = args.layout_a.shape[1];
    } else {
        m = args.layout_a.shape[1];
        k = args.layout_a.shape[0];
    }
    if (!args.opr->param().transposeB) {
        megdnn_assert(k == args.layout_b.shape[0]);
        n = args.layout_b.shape[1];
    } else {
        megdnn_assert(k == args.layout_b.shape[1]);
        n = args.layout_b.shape[0];
    }
    return std::tuple<size_t, size_t, size_t>{m, k, n};
}

}  // namespace

bool MatrixMulForwardImpl::AlgoConv1X1CUDNN::is_available(const SizeArgs& args) const {
    if (!(args.layout_a.ndim == 2 && args.layout_b.ndim == 2 &&
          args.layout_c.ndim == 2))
        return false;

    auto conv_opr_ptr = prepare_conv_opr(args);

    size_t m, k, n;
    std::tie(m, k, n) = gen_matrixmul_shape(args);

    TensorLayout src_layout({1, k, 1, n}, args.layout_b.dtype);
    TensorLayout filter_layout({m, k, 1, 1}, args.layout_a.dtype);
    TensorLayout bias_layout(args.layout_a.dtype);
    TensorLayout z_layout(args.layout_a.dtype);
    TensorLayout dst_layout({1, m, 1, n}, args.layout_c.dtype);
    ConvBiasForwardImpl::AlgoBase::SizeArgs conv_size_args(
            static_cast<ConvBiasForwardImpl*>(conv_opr_ptr.get()), src_layout,
            filter_layout, bias_layout, z_layout, dst_layout);

    return m_impl->is_available(conv_size_args);
}

WorkspaceBundle MatrixMulForwardImpl::AlgoConv1X1CUDNN::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    SmallVector<size_t> sizes;
    auto conv_opr_ptr = prepare_conv_opr(args);

    size_t m, k, n;
    std::tie(m, k, n) = gen_matrixmul_shape(args);

    TensorLayout src_layout({1, k, 1, n}, args.layout_b.dtype);
    TensorLayout filter_layout({m, k, 1, 1}, args.layout_a.dtype);
    TensorLayout bias_layout(args.layout_a.dtype);
    TensorLayout z_layout(args.layout_a.dtype);
    TensorLayout dst_layout({1, m, 1, n}, args.layout_c.dtype);
    ConvBiasForwardImpl::AlgoBase::SizeArgs conv_size_args(
            static_cast<ConvBiasForwardImpl*>(conv_opr_ptr.get()), src_layout,
            filter_layout, bias_layout, z_layout, dst_layout);

    sizes.push_back(m_impl->get_workspace_in_bytes(conv_size_args));

    auto get_trans_layout = [](const TensorLayout& ly) {
        size_t m = ly[0], n = ly[1];
        TensorLayout trans{{n, m}, ly.dtype};
        return trans;
    };
    if (args.opr->param().transposeA) {
        sizes.push_back(get_trans_layout(args.layout_a).span().dist_byte());
    }
    if (args.opr->param().transposeB) {
        sizes.push_back(get_trans_layout(args.layout_b).span().dist_byte());
    }

    return {ptr, std::move(sizes)};
}

size_t MatrixMulForwardImpl::AlgoConv1X1CUDNN::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void MatrixMulForwardImpl::AlgoConv1X1CUDNN::exec(const ExecArgs& args) const {
    SizeArgs size_args(args.opr, args.layout_a, args.layout_b, args.layout_c);

    auto conv_opr_ptr = prepare_conv_opr(size_args);

    size_t m, k, n;
    std::tie(m, k, n) = gen_matrixmul_shape(size_args);

    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, size_args);
    auto A_dst_tensor = args.tensor_a;
    auto B_dst_tensor = args.tensor_b;
    if (args.opr->param().transposeA || args.opr->param().transposeB) {
        auto trans = args.opr->handle()->create_operator<RelayoutForward>();

        auto trans_tensor = [&](size_t workspace_pos, const TensorND& ori_tensor,
                                TensorND& dst_tensor) {
            TensorLayout dst_layout(
                    {ori_tensor.layout.shape[1], ori_tensor.layout.shape[0]},
                    ori_tensor.layout.dtype);
            dst_tensor = TensorND(bundle.get(workspace_pos), dst_layout);
            TensorND src_tensor(ori_tensor.raw_ptr(), dst_layout);
            src_tensor.layout.stride[0] = ori_tensor.layout.stride[1];
            src_tensor.layout.stride[1] = ori_tensor.layout.stride[0];

            trans->exec(src_tensor, dst_tensor, args.opr->handle());
        };

        if (args.opr->param().transposeA) {
            trans_tensor(1, args.tensor_a, A_dst_tensor);
        }
        if (args.opr->param().transposeB) {
            trans_tensor(bundle.nr_workspace() - 1, args.tensor_b, B_dst_tensor);
        }
    }

    TensorLayout src_layout({1, k, 1, n}, args.layout_b.dtype);
    TensorLayout filter_layout({m, k, 1, 1}, args.layout_a.dtype);
    TensorLayout dst_layout({1, m, 1, n}, args.layout_c.dtype);

    TensorND src(B_dst_tensor.raw_ptr(), src_layout);
    TensorND filter(A_dst_tensor.raw_ptr(), filter_layout);
    TensorND z(nullptr, TensorLayout(src_layout.dtype));
    TensorND bias(nullptr, TensorLayout(src_layout.dtype));
    TensorND dst(args.tensor_c.raw_ptr(), dst_layout);

    ConvBiasForwardImpl::AlgoBase::ExecArgs conv_exec_args(
            static_cast<ConvBiasForwardImpl*>(conv_opr_ptr.get()), src, filter, bias, z,
            dst, bundle.get_workspace(0));
    m_impl->exec(conv_exec_args);
}
