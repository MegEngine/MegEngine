#include "./algo.h"
#include "hcc_detail/hcc_defs_prologue.h"
#include "src/rocm/utils.h"

using namespace megdnn;
using namespace rocm;

PoolingForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_miopen);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

PoolingForwardImpl::AlgoPack PoolingForwardImpl::sm_algo_pack;
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

bool PoolingForwardImpl::AlgoMIOpen::is_available(const SizeArgs& args) const {
    return true;
}

void PoolingForwardImpl::AlgoMIOpen::init_mode(
        const ExecArgs& args, miopenPoolingMode_t& mode) const {
    switch (args.opr->param().mode) {
        case param::Pooling::Mode::MAX:
            mode = miopenPoolingMax;
            break;
        case param::Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            mode = miopenPoolingAverage;
            break;
        case param::Pooling::Mode::AVERAGE:
            mode = miopenPoolingAverageInclusive;
            break;
        default:
            megdnn_throw(ssprintf(
                    "Unspport pooling mode : {%d}",
                    static_cast<int>(args.opr->param().mode)));
    }
}

size_t PoolingForwardImpl::AlgoMIOpen::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return 0;
}

void PoolingForwardImpl::AlgoMIOpen::exec(const ExecArgs& args) const {
    auto handle = miopen_handle(args.handle);
    TensorDesc src_desc, dst_desc;
    args.init_desc(src_desc, dst_desc);
    miopenPoolingMode_t mode;
    init_mode(args, mode);

    miopenPoolingDescriptor_t miopen_desc;
    miopen_check(miopenCreatePoolingDescriptor(&miopen_desc));
    miopen_check(miopenSet2dPoolingDescriptor(
            miopen_desc, mode, args.opr->param().window_h, args.opr->param().window_w,
            args.opr->param().pad_h, args.opr->param().pad_w,
            args.opr->param().stride_h, args.opr->param().stride_w));

    dt_float32 alpha = 1.0f, beta = 0.0f;
    miopen_check(miopenPoolingForward(
            handle, miopen_desc, &alpha, src_desc.desc, args.src_tensor->raw_ptr(),
            &beta, dst_desc.desc, args.dst_tensor->raw_ptr(), false, nullptr, 0_z));
    miopen_check(miopenDestroyPoolingDescriptor(miopen_desc));
}

PoolingBackwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_miopen);

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

bool PoolingBackwardImpl::AlgoMIOpen::is_available(const SizeArgs&) const {
    return true;
}

size_t PoolingBackwardImpl::AlgoMIOpen::get_workspace_in_bytes(
        const SizeArgs& args) const {
    TensorDesc dst_desc;
    dst_desc.set(*args.layout_dst);

    size_t ws_size = 0_z;
    miopenPoolingGetWorkSpaceSize(dst_desc.desc, &ws_size);
    return ws_size;
}

void PoolingBackwardImpl::AlgoMIOpen::init_mode(
        const ExecArgs& args, miopenPoolingMode_t& mode) const {
    switch (args.opr->param().mode) {
        case param::Pooling::Mode::MAX:
            mode = miopenPoolingMax;
            break;
        case param::Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            mode = miopenPoolingAverage;
            break;
        case param::Pooling::Mode::AVERAGE:
            mode = miopenPoolingAverageInclusive;
            break;
        default:
            megdnn_throw(ssprintf(
                    "Unspport pooling mode : {%d}",
                    static_cast<int>(args.opr->param().mode)));
    }
}

void PoolingBackwardImpl::AlgoMIOpen::exec(const ExecArgs& args) const {
    auto handle = miopen_handle(args.handle);
    TensorDesc src_desc, dst_desc, diff_desc, grad_desc;
    args.init_desc(src_desc, dst_desc, diff_desc, grad_desc);
    miopenPoolingMode_t mode;
    init_mode(args, mode);

    miopenPoolingDescriptor_t miopen_desc;
    miopen_check(miopenCreatePoolingDescriptor(&miopen_desc));
    miopen_check(miopenSet2dPoolingDescriptor(
            miopen_desc, mode, args.opr->param().window_h, args.opr->param().window_w,
            args.opr->param().pad_h, args.opr->param().pad_w,
            args.opr->param().stride_h, args.opr->param().stride_w));

    float alpha = 1.0f, beta = 0.0f;
    if (args.opr->param().mode == param::Pooling::Mode::MAX) {
        //! FIXME: when using max pooling opr, the backward opr need the indices
        //! of the forward opr which stored in workspace. We have to recompute
        //! the indices by calling miopenPoolingForward again.
        miopen_check(miopenPoolingForward(
                handle, miopen_desc, &alpha, src_desc.desc, args.src_tensor->raw_ptr(),
                &beta, dst_desc.desc, args.dst_tensor->raw_ptr(), true,
                args.workspace.raw_ptr, args.workspace.size));
    }
    miopen_check(miopenPoolingBackward(
            handle, miopen_desc, &alpha, dst_desc.desc, args.dst_tensor->raw_ptr(),
            diff_desc.desc, args.diff_tensor->raw_ptr(), src_desc.desc,
            args.src_tensor->raw_ptr(), &beta, grad_desc.desc,
            args.grad_tensor->raw_ptr(), args.workspace.raw_ptr));
}