#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

using namespace megdnn;

namespace {
std::string get_errmsg(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst,
        const Convolution3D::Param& param) {
    MEGDNN_MARK_USED_VAR(src);
    MEGDNN_MARK_USED_VAR(filter);
    MEGDNN_MARK_USED_VAR(dst);
    return megdnn_layout_msg(src) + ", " + megdnn_layout_msg(filter) + ", " +
           megdnn_layout_msg(dst) + ", " + "is_ncdhw=" +
           std::to_string(param.format == param::Convolution3D::Format::NCDHW) + ", " +
           "is_xcorr=" +
           std::to_string((param.mode == Convolution3D::Mode::CROSS_CORRELATION)) +
           ", " + "pad_d=" + std::to_string(param.pad_d) + ", " +
           "pad_h=" + std::to_string(param.pad_h) + ", " +
           "pad_w=" + std::to_string(param.pad_w) + ", " +
           "stride_d=" + std::to_string(param.stride_d) + ", " +
           "stride_h=" + std::to_string(param.stride_h) + ", " +
           "stride_w=" + std::to_string(param.stride_w) + ", " +
           "dilate_d=" + std::to_string(param.dilate_d) + ", " +
           "dilate_h=" + std::to_string(param.dilate_h) + ", " +
           "dilate_w=" + std::to_string(param.dilate_w);
}
}  // namespace

Convolution3DBase::CanonizedFilterMeta Convolution3DBase::
        make_canonized_filter_meta_impl(
                size_t src_ndim, const TensorLayout& filter, const Param& param) {
    megdnn_assert_contiguous(filter);
    auto img_ndim = src_ndim - 2;
    CanonizedFilterMeta ret;
    ret.dtype_enum = filter.dtype.enumv();
    ret.format = param.format;
    if (param.mode == Mode::CONVOLUTION) {
        ret.should_flip = true;
    } else {
        megdnn_assert(param.mode == Mode::CROSS_CORRELATION, "invalid conv mode");
        ret.should_flip = false;
    }
    size_t flt_start, flt_spatial_start, ocpg_pos, icpg_pos;
    MEGDNN_MARK_USED_VAR(flt_spatial_start);
    MEGDNN_MARK_USED_VAR(ocpg_pos);
    MEGDNN_MARK_USED_VAR(icpg_pos);

    if (param.sparse == Param::Sparse::DENSE) {
        megdnn_assert(
                filter.ndim == img_ndim + 2,
                "bad filter ndim for dense convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);
        ret.group = 1;
        flt_start = 0;
    } else {
        megdnn_assert(
                param.sparse == Param::Sparse::GROUP,
                "invalid convolution sparse type");
        megdnn_assert(
                filter.ndim == img_ndim + 3,
                "bad filter ndim for group convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);
        ret.group = filter[0];
        flt_start = 1;
    }

    if (param.format == Param::Format::NCDHW) {
        // filter should be (oc, ic, fd, fh, fw)
        flt_spatial_start = 2;
        ocpg_pos = 0;
        icpg_pos = 1;
    } else {
        megdnn_assert(
                param.format == Param::Format::NDHWC, "invalid conv tensor format");
        // filter should be (oc, fd, fh, fw, ic)
        flt_spatial_start = 1;
        ocpg_pos = 0;
        icpg_pos = 4;
    }
    ret.spatial_ndim = src_ndim - 2;
    megdnn_assert(
            ret.spatial_ndim == 3,
            "only 3D convolution is supported, and input should be 5-dim; "
            "got input dim = %zu",
            src_ndim);
    ret.stride[0] = param.stride_d;
    ret.stride[1] = param.stride_h;
    ret.stride[2] = param.stride_w;
    ret.padding[0] = param.pad_d;
    ret.padding[1] = param.pad_h;
    ret.padding[2] = param.pad_w;
    ret.dilation[0] = param.dilate_d;
    ret.dilation[1] = param.dilate_h;
    ret.dilation[2] = param.dilate_w;
    ret.ocpg = filter[flt_start + ocpg_pos];
    ret.icpg = filter[flt_start + icpg_pos];
    for (size_t i = 0; i < ret.spatial_ndim; ++i) {
        megdnn_assert(
                ret.dilation[i] > 0, "invalid dilation on spatial dim %zu: %u", i,
                ret.dilation[i]);
        ret.spatial[i] = filter[i + flt_start + flt_spatial_start];
        ret.dilated_spatial[i] = (ret.spatial[i] - 1) * ret.dilation[i] + 1;
    }
    return ret;
}

Convolution3DBase::CanonizedFilterMeta Convolution3DBase::make_canonized_filter_meta(
        size_t src_ndim, const TensorLayout& filter) const {
    return make_canonized_filter_meta_impl(src_ndim, filter, param());
}

Convolution3DBase::CanonizedFilterMeta Convolution3DBase::deduce_layout_fwd(
        const TensorLayout& src, const TensorLayout& filter, TensorLayout& dst) const {
    auto errmsg = [&]() { return get_errmsg(src, filter, dst, param()); };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert(src.ndim >= 5_z, "%s", errmsg().c_str());
    megdnn_assert(src.dtype == filter.dtype, "%s", errmsg().c_str());
    if (param().data_type == Param::DataType::FLOAT) {
        megdnn_assert(
                src.dtype == dtype::Float32()
                                     DNN_INC_FLOAT16(|| src.dtype == dtype::Float16()),
                "invalid src dtype for conv: %s", src.dtype.name());
        dst.dtype = src.dtype;
    } else {
        megdnn_assert(param().data_type == Param::DataType::FLOAT_IO16xC32);
        DNN_INC_FLOAT16(megdnn_assert(
                src.dtype == dtype::Float16(), "invalid src dtype for conv: %s",
                src.dtype.name()));
        DNN_INC_FLOAT16(dst.dtype = dtype::Float16());
    }
    auto img_dim = src.ndim - 2;
    megdnn_assert(img_dim == 3, "this is the convolution for 3D image");
    megdnn_assert(
            filter.ndim == img_dim + 2 || filter.ndim == img_dim + 3, "%s",
            errmsg().c_str());
    auto cflt = make_canonized_filter_meta(src.ndim, filter);
    size_t src_or_dst_c_pos = 0;
    size_t src_or_dst_spatial_start = 0;
    if (param().format == Param::Format::NCDHW) {
        src_or_dst_c_pos = 1;
        src_or_dst_spatial_start = 2;
    } else {
        megdnn_assert(param().format == Param::Format::NDHWC, "invalid conv format");
        src_or_dst_c_pos = 4;
        src_or_dst_spatial_start = 1;
    }
    megdnn_assert(
            cflt.icpg * cflt.group == src[src_or_dst_c_pos], "%s", errmsg().c_str());
    dst.ndim = src.ndim;
    dst[0] = src[0];
    dst[src_or_dst_c_pos] = cflt.ocpg * cflt.group;
    for (size_t i = 0; i < cflt.spatial_ndim; ++i) {
        dst[i + src_or_dst_spatial_start] = infer_conv_shape(
                src[i + src_or_dst_spatial_start], cflt.dilated_spatial[i],
                cflt.stride[i], cflt.padding[i]);
    }
    dst.init_contiguous_stride();
    return cflt;
}

Convolution3DBase::CanonizedFilterMeta Convolution3DBase::check_layout_fwd(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst) const {
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(filter);
    TensorLayout dst_expected;
    auto ret = deduce_layout_fwd(src, filter, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
    return ret;
}

void Convolution3DForward::deduce_layout(
        const TensorLayout& src, const TensorLayout& filter, TensorLayout& dst) {
    deduce_layout_fwd(src, filter, dst);
}

Convolution3DBase::CanonizedFilterMeta Convolution3DForward::check_exec(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst,
        size_t workspace_in_bytes) {
    auto src_fwd = src;
    auto dst_fwd = dst;
    src_fwd.init_contiguous_stride();
    dst_fwd.init_contiguous_stride();

    auto ret = check_layout_fwd(src_fwd, filter, dst_fwd);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, filter, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}

Convolution3DBase::CanonizedFilterMeta Convolution3DBackwardData::check_exec(
        const TensorLayout& filter, const TensorLayout& diff, const TensorLayout& grad,
        size_t workspace_in_bytes) {
    megdnn_assert(
            param().data_type == Param::DataType::FLOAT,
            "only float type is supported for conv backward");
    auto diff_fwd = diff;
    auto grad_fwd = grad;
    diff_fwd.init_contiguous_stride();
    grad_fwd.init_contiguous_stride();

    auto ret = check_layout_fwd(grad_fwd, filter, diff_fwd);
    auto required_workspace_in_bytes = get_workspace_in_bytes(filter, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}

void Convolution3DBackwardData::deduce_layout_impl(
        const TensorLayout& filter, const TensorLayout& diff, const Param& param,
        TensorLayout& grad) {
    megdnn_assert(
            param.data_type == Param::DataType::FLOAT,
            "only float type is supported for conv backward");
    auto errmsg = [&]() { return get_errmsg(filter, diff, grad, param); };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(filter);
    megdnn_assert_contiguous(diff);
    megdnn_assert(filter.ndim == 5_z || filter.ndim == 6_z, "%s", errmsg().c_str());
    megdnn_assert(diff.ndim == 5_z, "%s", errmsg().c_str());
    megdnn_assert(filter.dtype == diff.dtype, "%s", errmsg().c_str());

    auto cflt = make_canonized_filter_meta_impl(diff.ndim, filter, param);
    megdnn_assert(cflt.ocpg * cflt.group == diff[1], "%s", errmsg().c_str());

    auto deduce = [&errmsg](size_t out, size_t filter, size_t stride, size_t pad) {
        MEGDNN_MARK_USED_VAR(errmsg);
        auto i = (out - 1) * stride + filter;
        megdnn_assert(i > pad * 2, "%s", errmsg().c_str());
        return i - pad * 2;
    };

    grad.ndim = diff.ndim;
    grad[0] = diff[0];
    grad[1] = cflt.group * cflt.icpg;
    grad.dtype = diff.dtype;
    for (size_t i = 0; i < cflt.spatial_ndim; ++i) {
        grad[i + 2] = deduce(
                diff[i + 2], cflt.dilated_spatial[i], cflt.stride[i], cflt.padding[i]);
    }
    grad.init_contiguous_stride();
}

void Convolution3DBackwardData::deduce_layout(
        const TensorLayout& filter, const TensorLayout& diff, TensorLayout& grad) {
    deduce_layout_impl(filter, diff, param(), grad);
}

Convolution3DBase::CanonizedFilterMeta Convolution3DBackwardFilter::check_exec(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad,
        size_t workspace_in_bytes) {
    megdnn_assert(
            param().data_type == Param::DataType::FLOAT,
            "only float type is supported for conv backward");
    auto src_fwd = src;
    auto diff_fwd = diff;
    src_fwd.init_contiguous_stride();
    diff_fwd.init_contiguous_stride();

    auto ret = check_layout_fwd(src_fwd, grad, diff_fwd);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}
// vim: syntax=cpp.doxygen
