#include "./algo.h"
#include <string.h>
#include <vector>
#include "aclnnop/aclnn_avgpool2d.h"
#include "aclnnop/aclnn_avgpool2d_backward.h"
#include "aclnnop/aclnn_max_pool.h"
#include "aclnnop/aclnn_max_pool2d_with_indices.h"
#include "aclnnop/aclnn_max_pool2d_with_indices_backward.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

namespace megdnn {
namespace atlas {

PoolingForwardImpl::AlgoPack PoolingForwardImpl::sm_algo_pack;

PoolingForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_acl);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(PoolingForwardImpl)

PoolingForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        PoolingForwardImpl* o, const TensorLayout& src, const TensorLayout& dst)
        : handle{concrete_handle(o->handle())},
          opr{o},
          layout_src{&src},
          layout_dst{&dst} {}

PoolingForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        PoolingForwardImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, dst.layout),
          src_tensor{&src},
          dst_tensor{&dst},
          workspace{workspace} {}

std::string PoolingForwardImpl::AlgoBase::SizeArgs::to_string() const {
    return ssprintf(
            "src=%s, dst=%s", layout_src->to_string().c_str(),
            layout_dst->to_string().c_str());
}

size_t PoolingForwardImpl::AlgoACL::get_workspace_in_bytes(const SizeArgs&) const {
    return 0;
}

bool PoolingForwardImpl::AlgoACL::is_available(const SizeArgs& args) const {
    auto src_dtype = args.layout_src->dtype.enumv();
    auto dst_dtype = args.layout_dst->dtype.enumv();
    auto src_layout = args.layout_src;
    auto dst_layout = args.layout_dst;
    size_t stride_h = args.opr->param().stride_h;
    size_t stride_w = args.opr->param().stride_w;
    size_t window_h = args.opr->param().window_h;
    size_t window_w = args.opr->param().window_w;

    return src_layout->is_contiguous() && dst_layout->is_contiguous() &&
           args.opr->param().format == Param::Format::NCHW &&
           (src_layout->shape[0] > 0 && src_layout->shape[1] > 0 &&
            src_layout->shape[2] > 0 && src_layout->shape[3] > 0) &&
           (dst_layout->shape[0] > 0 && dst_layout->shape[1] > 0 &&
            dst_layout->shape[2] > 0 && dst_layout->shape[3] > 0) &&
           (stride_h >= 1 && stride_w >= 1 && window_h >= 1 && window_w >= 1) &&
           ((src_dtype == DTypeEnum::Float16 && dst_dtype == DTypeEnum::Float16) ||
            (src_dtype == DTypeEnum::Float32 && dst_dtype == DTypeEnum::Float32));
}

void PoolingForwardImpl::AlgoACL::exec(const ExecArgs& args) const {
    auto&& param = args.opr->param();
    AclIntArray kernel({param.window_h, param.window_w});
    AclIntArray stride({param.stride_h, param.stride_w});
    AclIntArray padding({param.pad_h, param.pad_w});

    megdnn_assert(
            param.format == Param::Format::NCHW, "ascend pooling only support NCHW");
    aclFormat fmt = aclFormat::ACL_FORMAT_NCHW;
    AclTensor src(args.src_tensor->raw_ptr(), args.src_tensor->layout, fmt);
    AclTensor dst(args.dst_tensor->raw_ptr(), args.dst_tensor->layout, fmt);

    bool ceilmode = false;
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    using Mode = param::Pooling::Mode;

    if (param.mode == Mode::MAX) {
        AclIntArray dilation({1, 1, 1, 1});
        aclnn_check(aclnnMaxPoolGetWorkspaceSize(
                src.get(), kernel.get(), stride.get(), /* autoPads= */ 0, padding.get(),
                dilation.get(), ceilmode, dst.get(), &ws_size, &executor));
        auto ws = AclMem(ws_size, args.handle);
        aclnn_check(aclnnMaxPool(ws.ptr(), ws_size, executor, args.handle->stream()));
    } else {
        megdnn_assert(
                param.mode == Mode::AVERAGE ||
                param.mode == Mode::AVERAGE_COUNT_EXCLUDE_PADDING);
        int8_t cube_math_type = CUBE_KEEP_DTYPE;
        int64_t divider = 0;
        bool count_include_pad = param.mode == Mode::AVERAGE ? true : false;

        aclnn_check(aclnnAvgPool2dGetWorkspaceSize(
                src.get(), kernel.get(), stride.get(), padding.get(), ceilmode,
                count_include_pad, divider, cube_math_type, dst.get(), &ws_size,
                &executor));
        auto ws = AclMem(ws_size, args.handle);
        aclnn_check(aclnnAvgPool2d(ws.ptr(), ws_size, executor, args.handle->stream()));
    }
}

PoolingBackwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_acl);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

PoolingBackwardImpl::AlgoPack PoolingBackwardImpl::sm_algo_pack;
MEGDNN_DEF_GET_ALGO_FROM_DESC(PoolingBackwardImpl)

PoolingBackwardImpl::AlgoBase::SizeArgs::SizeArgs(
        PoolingBackwardImpl* o, const TensorLayout& src, const TensorLayout& dst,
        const TensorLayout& diff, const TensorLayout& grad)
        : handle{concrete_handle(o->handle())},
          opr{o},
          layout_src{&src},
          layout_dst{&dst},
          layout_diff{&diff},
          layout_grad{&grad} {}

PoolingBackwardImpl::AlgoBase::ExecArgs::ExecArgs(
        PoolingBackwardImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_in dst,
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, dst.layout, diff.layout, grad.layout),
          src_tensor{&src},
          dst_tensor{&dst},
          diff_tensor{&diff},
          grad_tensor{&grad},
          workspace{workspace} {}

std::string PoolingBackwardImpl::AlgoBase::SizeArgs::to_string() const {
    return ssprintf(
            "src=%s, dst=%s, diff=%s, grad=%s", layout_src->to_string().c_str(),
            layout_dst->to_string().c_str(), layout_diff->to_string().c_str(),
            layout_grad->to_string().c_str());
}

size_t PoolingBackwardImpl::AlgoACL::get_workspace_in_bytes(const SizeArgs&) const {
    return 0;
}

bool PoolingBackwardImpl::AlgoACL::is_available(const SizeArgs& args) const {
    auto src_dtype = args.layout_src->dtype.enumv();
    auto dst_dtype = args.layout_dst->dtype.enumv();
    auto src_layout = args.layout_src;
    auto dst_layout = args.layout_dst;
    size_t stride_h = args.opr->param().stride_h;
    size_t stride_w = args.opr->param().stride_w;
    size_t window_h = args.opr->param().window_h;
    size_t window_w = args.opr->param().window_w;

    return src_layout->is_contiguous() && dst_layout->is_contiguous() &&
           args.opr->param().format == Param::Format::NCHW &&
           (src_layout->shape[0] > 0 && src_layout->shape[1] > 0 &&
            src_layout->shape[2] > 0 && src_layout->shape[3] > 0) &&
           (dst_layout->shape[0] > 0 && dst_layout->shape[1] > 0 &&
            dst_layout->shape[2] > 0 && dst_layout->shape[3] > 0) &&
           (window_h >= 1 && window_w >= 1 && stride_h >= 1 && stride_w >= 1) &&
           ((src_dtype == DTypeEnum::Float16 && dst_dtype == DTypeEnum::Float16) ||
            (src_dtype == DTypeEnum::Float32 && dst_dtype == DTypeEnum::Float32));
}

void PoolingBackwardImpl::AlgoACL::exec(const ExecArgs& args) const {
    auto&& param = args.opr->param();
    AclIntArray kernel({param.window_h, param.window_w});
    AclIntArray stride({param.stride_h, param.stride_w});
    AclIntArray padding({param.pad_h, param.pad_w});
    AclIntArray dilation({1, 1});

    megdnn_assert(
            param.format == Param::Format::NCHW, "ascend pooling only support NCHW");
    aclFormat fmt = aclFormat::ACL_FORMAT_NCHW;
    AclTensor src(args.src_tensor->raw_ptr(), args.src_tensor->layout, fmt);
    AclTensor dst(args.dst_tensor->raw_ptr(), args.dst_tensor->layout, fmt);
    AclTensor diff(args.diff_tensor->raw_ptr(), args.diff_tensor->layout, fmt);
    AclTensor grad(args.grad_tensor->raw_ptr(), args.grad_tensor->layout, fmt);

    bool ceilmode = false;
    using Mode = param::Pooling::Mode;
    if (param.mode == Mode::MAX) {
        uint64_t fwd_ws_size = 0, bwd_ws_size = 0;
        aclOpExecutor *fwd_executor = nullptr, *bwd_executor = nullptr;

        TensorLayout indices_layout(args.dst_tensor->layout, dtype::Int32());
        auto indices_buf = AclMem(indices_layout.access_bytes(), args.handle);
        auto indices = AclTensor(indices_buf.ptr(), indices_layout, fmt);

        aclnn_check(aclnnMaxPool2dWithIndicesGetWorkspaceSize(
                src.get(), kernel.get(), stride.get(), padding.get(), dilation.get(),
                ceilmode, dst.get(), indices.get(), &fwd_ws_size, &fwd_executor));
        auto fwd_ws = AclMem(fwd_ws_size, args.handle);
        aclnn_check(aclnnMaxPool2dWithIndices(
                fwd_ws.ptr(), fwd_ws_size, fwd_executor, args.handle->stream()));

        aclnn_check(aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(
                diff.get(), src.get(), indices.get(), kernel.get(), stride.get(),
                padding.get(), dilation.get(), ceilmode, grad.get(), &bwd_ws_size,
                &bwd_executor));
        auto bwd_ws = AclMem(bwd_ws_size, args.handle);
        aclnn_check(aclnnMaxPool2dWithIndicesBackward(
                bwd_ws.ptr(), bwd_ws_size, bwd_executor, args.handle->stream()));
    } else {
        megdnn_assert(
                param.mode == Mode::AVERAGE ||
                param.mode == Mode::AVERAGE_COUNT_EXCLUDE_PADDING);
        int8_t cube_math_type = CUBE_KEEP_DTYPE;
        int64_t divider = 0;
        bool count_include_pad = param.mode == Mode::AVERAGE ? true : false;

        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        aclnn_check(aclnnAvgPool2dBackwardGetWorkspaceSize(
                diff.get(), src.get(), kernel.get(), stride.get(), padding.get(),
                ceilmode, count_include_pad, divider, cube_math_type, grad.get(),
                &ws_size, &executor));
        auto ws = AclMem(ws_size, args.handle);
        aclnn_check(aclnnAvgPool2dBackward(
                ws.ptr(), ws_size, executor, args.handle->stream()));
    }
}

}  // namespace atlas
}  // namespace megdnn
