#include "algo.h"
#include "aclnnop/aclnn_convolution.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

namespace megdnn {
namespace atlas {

ConvBiasForwardImpl::AlgoPack ConvBiasForwardImpl::sm_algo_pack;

ConvBiasForwardImpl::AlgoPack::AlgoPack() {
    all_algos.emplace_back(&default_conv);
    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

using NonlineMode = param::ConvBias::NonlineMode;

ConvBiasForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvBiasForwardImpl* opr, const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst)
        : SizeArgs(
                  opr, src, filter, opr->make_canonized_filter_meta(src.ndim, filter),
                  bias, z, dst) {}

ConvBiasForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvBiasForwardImpl* o, const TensorLayout& src, const TensorLayout& filter,
        const CanonizedFilterMeta& filter_meta, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst)
        : opr{o},
          src_layout(&src),
          filter_layout(&filter),
          filter_meta(filter_meta),
          bias_layout(&bias),
          z_layout(&z),
          dst_layout(&dst) {}

ConvBiasForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        ConvBiasForwardImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_in filter,
        _megdnn_tensor_in bias, _megdnn_tensor_in z, _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, filter.layout, bias.layout, z.layout, dst.layout),
          src_tensor{&src},
          filter_tensor{&filter},
          bias_tensor{&bias},
          z_tensor{&z},
          dst_tensor{&dst},
          workspace{workspace} {}

std::string ConvBiasForwardImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return ssprintf(
            "src=%s, filter=%u{%u,%u,%u,%u}, dst=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
            src_layout->to_string().c_str(), fm.group, fm.ocpg, fm.icpg, fm.spatial[0],
            fm.spatial[1], dst_layout->to_string().c_str(), fm.padding[0],
            fm.padding[1], fm.stride[0], fm.stride[1], fm.dilation[0], fm.dilation[1],
            !fm.should_flip, src_layout->dtype.name(), dst_layout->dtype.name());
}

bool ConvBiasForwardImpl::AlgoDefault::is_available(const SizeArgs& args) const {
    if (args.opr->param().format != param::ConvBias::Format::NCHW) {
        return false;
    }
    if (!args.dst_layout->is_contiguous()) {
        return false;
    }
    auto bias_layout = *(args.bias_layout);
    if (bias_layout.ndim != 0) {
        if (bias_layout.collapse_contiguous().ndim != 1) {
            return false;
        }
    }
    auto input_dtype = args.src_layout->dtype.enumv();
    auto filter_dtype = args.filter_layout->dtype.enumv();
    auto bias_dtype = args.bias_layout->dtype.enumv();
    auto output_dtype = args.dst_layout->dtype.enumv();
    bool input_dtype_valid =
            input_dtype == DTypeEnum::Float32 || input_dtype == DTypeEnum::Float16;
    bool filter_dtype_valid =
            filter_dtype == DTypeEnum::Float32 || filter_dtype == DTypeEnum::Float16;
    bool bias_dtype_valid =
            bias_dtype == DTypeEnum::Float32 || bias_dtype == DTypeEnum::Float16;
    DTypeEnum high_precision_dtype = DTypeEnum::Float16;
    if (input_dtype == DTypeEnum::Float32 || filter_dtype == DTypeEnum::Float32) {
        high_precision_dtype = DTypeEnum::Float32;
    }
    bool output_dtype_valid = output_dtype == high_precision_dtype;
    bool conv_mode_valid =
            args.opr->param().mode == param::Convolution::Mode::CROSS_CORRELATION;
    bool comput_mode_valid =
            args.opr->param().compute_mode == param::ConvBias::ComputeMode::DEFAULT;

    return input_dtype_valid && filter_dtype_valid && bias_dtype_valid &&
           output_dtype_valid && comput_mode_valid && conv_mode_valid;
}

size_t ConvBiasForwardImpl::AlgoDefault::get_workspace_in_bytes(
        const SizeArgs& args) const {
    if (args.opr->param().sparse == param::ConvBias::Sparse::GROUP) {
        TensorLayout filter_layout(*(args.filter_layout));
        filter_layout.ndim = 4;
        filter_layout.shape[0] = filter_layout.shape[0] * filter_layout.shape[1];
        for (size_t i = 1; i <= 3; ++i) {
            filter_layout.shape[i] = filter_layout.shape[i + 1];
        }
        for (size_t i = 0; i < 4; ++i) {
            filter_layout.stride[i] = filter_layout.stride[i + 1];
        }
        return filter_layout.span().dist_byte();
    }
    return 0;
}

void ConvBiasForwardImpl::AlgoDefault::exec(const ExecArgs& args) const {
    auto handle = concrete_handle(args.opr->handle());
    bool is_group_conv = args.opr->param().sparse == param::ConvBias::Sparse::GROUP;
    TensorND collapsed_bias = *(args.bias_tensor);
    if (collapsed_bias.layout.ndim != 0) {
        collapsed_bias.layout = collapsed_bias.layout.collapse_contiguous();
    }

    TensorND filter = *(args.filter_tensor);
    // TODO: extract the func.
    if (is_group_conv) {
        TensorLayout filter_layout(*(args.filter_layout));
        filter_layout.ndim = 4;
        filter_layout.shape[0] = filter_layout.shape[0] * filter_layout.shape[1];
        for (size_t i = 1; i <= 3; ++i) {
            filter_layout.shape[i] = filter_layout.shape[i + 1];
        }
        for (size_t i = 0; i < 4; ++i) {
            filter_layout.stride[i] = filter_layout.stride[i + 1];
        }
        filter = TensorND(static_cast<void*>(args.workspace.raw_ptr), filter_layout);

        auto relayout_opr = args.opr->handle()->create_operator<RelayoutForward>();
        relayout_opr->exec(*(args.filter_tensor), filter, handle);
    }

    AclTensor acl_src(*(args.src_tensor), aclFormat::ACL_FORMAT_NCHW),
            acl_filter(filter, aclFormat::ACL_FORMAT_NCHW),
            acl_bias(collapsed_bias, aclFormat::ACL_FORMAT_NCHW),
            acl_z(*(args.z_tensor)),
            acl_dst(*(args.dst_tensor), aclFormat::ACL_FORMAT_NCHW);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    AclIntArray acl_strides({args.opr->param().stride_h, args.opr->param().stride_w});
    AclIntArray acl_paddings({args.opr->param().pad_h, args.opr->param().pad_w});
    AclIntArray acl_dilations({args.opr->param().dilate_h, args.opr->param().dilate_w});
    AclIntArray acl_dst_paddings({0, 0});
    int64_t group = 1;
    if (is_group_conv) {
        group = args.filter_meta.group;
    }

    aclnn_check(aclnnConvolutionGetWorkspaceSize(
            acl_src.get(), acl_filter.get(),
            args.bias_tensor->raw_ptr() == nullptr ? nullptr : acl_bias.get(),
            acl_strides.get(), acl_paddings.get(), acl_dilations.get(), false,
            acl_dst_paddings.get(), group, acl_dst.get(), 0, &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnConvolution(ws.ptr(), ws_size, executor, handle->stream()));

    //! create elemwise opr
    auto elemwise_opr = handle->create_operator<ElemwiseForward>();

    //! add z
    if (args.z_layout->ndim) {
        elemwise_opr->param().mode = param::Elemwise::Mode::ADD;
        elemwise_opr->exec({*args.dst_tensor, *args.z_tensor}, *args.dst_tensor);
    }

    auto nonlinemode = args.opr->param().nonlineMode;
    //! nonlinear
    if (nonlinemode == NonlineMode::RELU) {
        elemwise_opr->param().mode = param::Elemwise::Mode::RELU;
        elemwise_opr->exec({*args.dst_tensor}, *args.dst_tensor);
    } else if (nonlinemode == NonlineMode::SIGMOID) {
        elemwise_opr->param().mode = param::Elemwise::Mode::SIGMOID;
        elemwise_opr->exec({*args.dst_tensor}, *args.dst_tensor);
    } else if (nonlinemode == NonlineMode::H_SWISH) {
        elemwise_opr->param().mode = param::Elemwise::Mode::H_SWISH;
        elemwise_opr->exec({*args.dst_tensor}, *args.dst_tensor);
    }
}

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
