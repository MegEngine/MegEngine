#include "megdnn/oprs/nn.h"
#include "src/common/utils.cuh"
#include "src/common/utils.h"

using namespace megdnn;
namespace {
template <typename Param>
std::string get_errmsg(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst,
        const Param& param) {
    MEGDNN_MARK_USED_VAR(src);
    MEGDNN_MARK_USED_VAR(filter);
    MEGDNN_MARK_USED_VAR(dst);
    return megdnn_layout_msg(src) + ", " + megdnn_layout_msg(filter) + ", " +
           megdnn_layout_msg(dst) + ", " + "is_nchw=" +
           std::to_string(param.format == param::Convolution::Format::NCHW) + ", " +
           "is_xcorr=" +
           std::to_string((param.mode == Convolution::Mode::CROSS_CORRELATION)) + ", " +
           "pad_h=" + std::to_string(param.pad_h) + ", " +
           "pad_w=" + std::to_string(param.pad_w) + ", " +
           "stride_h=" + std::to_string(param.stride_h) + ", " +
           "stride_w=" + std::to_string(param.stride_w) + ", " +
           "dilate_h=" + std::to_string(param.dilate_h) + ", " +
           "dilate_w=" + std::to_string(param.dilate_w);
}

}  // namespace

namespace megdnn {

void RegionRestrictedConvolutionForward::deduce_dtype(
        DType src, DType filter, DType rin, DType rout, DType& dst) {
    check_or_deduce_dtype_fwd(src, filter, dst);
    megdnn_assert(
            src.category() == DTypeCategory::FLOAT &&
                    filter.category() == DTypeCategory::FLOAT &&
                    dst.category() == DTypeCategory::FLOAT,
            "only float type is supported for region_restricted_conv forward");
    megdnn_assert(
            rin == rout && (rin == dtype::Int32() || rin == dtype::Uint8()),
            "the dtype of rin/rout should be Int32 or Uint8, got %s.", rin.name());
}

void RegionRestrictedConvolutionForward::deduce_layout(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& rin,
        const TensorLayout& rout, TensorLayout& dst) {
    MEGDNN_MARK_USED_VAR(rin);
    MEGDNN_MARK_USED_VAR(rout);
    megdnn_assert_not_empty(src, RegionRestrictedConvolution);
    megdnn_assert_not_empty(filter, RegionRestrictedConvolution);
    deduce_layout_fwd(src, filter, dst);
}

RegionRestrictedConvolutionForward::CanonizedFilterMeta
RegionRestrictedConvolutionForward::check_exec(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& rin,
        const TensorLayout& rout, const TensorLayout& dst, size_t workspace_in_bytes) {
    auto ret = check_layout_fwd(src, filter, dst);
    megdnn_assert(
            param().format == Param::Format::NCHW,
            "RegionRestrictedConv only support NCHW format mow.");
    megdnn_assert(
            param().stride_h == 1 && param().stride_w == 1,
            "RegionRestrictedConv only support stride 1.");
#define err_msg(lhs, rhs) \
    megdnn_assert(lhs == rhs, "shape mismatch, #lhs:%zu, #rhs:%zu", lhs, rhs);

    err_msg(rin.shape[0], src.shape[0]);
    err_msg(rin.shape[1], src.shape[2]);
    err_msg(rin.shape[2], src.shape[3]);
    err_msg(rout.shape[0], dst.shape[0]);
    err_msg(rout.shape[1], dst.shape[2]);
    err_msg(rout.shape[2], dst.shape[3]);
#undef err_msg
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, filter, rin, rout, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}

RegionRestrictedConvolutionBackwardData::CanonizedFilterMeta
RegionRestrictedConvolutionBackwardData::check_exec(
        const TensorLayout& filter, const TensorLayout& diff, const TensorLayout& rin,
        const TensorLayout& rout, const TensorLayout& grad, size_t workspace_in_bytes) {
    auto grad_fwd = grad;
    auto filter_fwd = filter;
    auto diff_fwd = diff;

    std::swap(grad_fwd.dtype, diff_fwd.dtype);

    grad_fwd.init_contiguous_stride();
    diff_fwd.init_contiguous_stride();
    auto ret = check_layout_fwd(grad_fwd, filter_fwd, diff_fwd);
#define err_msg(lhs, rhs) \
    megdnn_assert(lhs == rhs, "shape mismatch, #lhs:%zu, #rhs:%zu", lhs, rhs);
    err_msg(rin.shape[0], grad_fwd.shape[0]);   // batch
    err_msg(rin.shape[1], grad_fwd.shape[2]);   // ih
    err_msg(rin.shape[2], grad_fwd.shape[3]);   // iw
    err_msg(rout.shape[0], diff_fwd.shape[0]);  // batch
    err_msg(rout.shape[1], diff_fwd.shape[2]);  // oh
    err_msg(rout.shape[2], diff_fwd.shape[3]);  // ow
#undef err_msg
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(filter, diff, rin, rout, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}

void RegionRestrictedConvolutionBackwardData::deduce_dtype(
        DType filter, DType diff, DType rin, DType rout, DType& grad) {
    // FIXME: infering dtype of grad via naive impl only support fp32
    // (lack of quantized dtype infering or others) may not suitable in the furture
#if !MEGDNN_DISABLE_FLOAT16
    if (diff.enumv() == DTypeEnum::Float32 || diff.enumv() == DTypeEnum::Float16) {
        grad = diff;
    }
#endif
    megdnn_assert(grad.valid(), "dtype of grad requires deducing of assigned");
    megdnn_assert(
            diff.category() == DTypeCategory::FLOAT &&
                    filter.category() == DTypeCategory::FLOAT &&
                    grad.category() == DTypeCategory::FLOAT,
            "only float type is supported for region_restricted_conv backward data");
    megdnn_assert(
            rin == rout && (rin == dtype::Int32() || rin == dtype::Uint8()),
            "the dtype of rin/rout should be Int32 or Uint8, got %s.", rin.name());
}

void RegionRestrictedConvolutionBackwardData::deduce_layout(
        const TensorLayout& filter, const TensorLayout& diff, const TensorLayout& rin,
        const TensorLayout& rout, TensorLayout& grad) {
    auto errmsg = [&]() { return get_errmsg(filter, diff, grad, param()); };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(filter);
    megdnn_assert_contiguous(diff);
    megdnn_assert(filter.ndim == 4_z || filter.ndim == 5_z, "%s", errmsg().c_str());
    megdnn_assert(diff.ndim == 4_z || diff.ndim == 5_z, "%s", errmsg().c_str());

    deduce_dtype(filter.dtype, diff.dtype, rin.dtype, rout.dtype, grad.dtype);

    auto cflt = make_canonized_filter_meta(diff.ndim, filter);

    auto deduce = [&errmsg](size_t out, size_t filter, size_t stride, size_t pad) {
        MEGDNN_MARK_USED_VAR(errmsg);
        auto i = (out - 1) * stride + filter;
        megdnn_assert(i > pad * 2, "%s", errmsg().c_str());
        return i - pad * 2;
    };

    megdnn_assert(
            param().format == Param::Format::NCHW,
            "RegionRestrictedConvolutionBackwardData only support NCHW format mow.");
    size_t src_or_dst_c_pos = 1;
    size_t src_or_dst_spatial_start = 2;
    megdnn_assert(
            cflt.ocpg * cflt.group == diff[src_or_dst_c_pos], "%s", errmsg().c_str());
    grad.ndim = diff.ndim;
    grad[0] = diff[0];
    grad[src_or_dst_c_pos] = cflt.icpg * cflt.group;
    for (size_t i = 0; i < cflt.spatial_ndim; ++i) {
        grad[i + src_or_dst_spatial_start] =
                deduce(diff[i + src_or_dst_spatial_start], cflt.dilated_spatial[i],
                       cflt.stride[i], cflt.padding[i]);
    }
    grad.format = diff.format;
    grad.init_contiguous_stride();
}

RegionRestrictedConvolutionBackwardFilter::CanonizedFilterMeta
RegionRestrictedConvolutionBackwardFilter::check_exec(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout& rin,
        const TensorLayout& rout, const TensorLayout& grad, size_t workspace_in_bytes) {
    megdnn_assert(
            src.dtype.category() == DTypeCategory::FLOAT &&
                    diff.dtype.category() == DTypeCategory::FLOAT &&
                    grad.dtype.category() == DTypeCategory::FLOAT,
            "only float type is supported for conv backward filter");
    auto src_fwd = src;
    auto diff_fwd = diff;

    src_fwd.init_contiguous_stride();
    diff_fwd.init_contiguous_stride();
    auto ret = check_layout_fwd(src_fwd, grad, diff_fwd);
    megdnn_assert(
            param().format == Param::Format::NCHW,
            "RegionRestrictedConvolutionBackwardFilter only support NCHW format mow.");
#define err_msg(lhs, rhs) \
    megdnn_assert(lhs == rhs, "shape mismatch, #lhs:%zu, #rhs:%zu", lhs, rhs);
    err_msg(rin.shape[0], src_fwd.shape[0]);
    err_msg(rin.shape[1], src_fwd.shape[2]);
    err_msg(rin.shape[2], src_fwd.shape[3]);
    err_msg(rout.shape[0], diff_fwd.shape[0]);
    err_msg(rout.shape[1], diff_fwd.shape[2]);
    err_msg(rout.shape[2], diff_fwd.shape[3]);
#undef err_msg
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, diff, rin, rout, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
