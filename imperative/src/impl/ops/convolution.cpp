#include "megbrain/opr/dnn/convolution.h"
#include "../algo_chooser.h"
#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

namespace mgb {
namespace imperative {

namespace {

size_t infer_conv_shape(size_t inp, size_t flt, size_t stride, size_t pad) {
    mgb_assert(inp + 2 * pad >= flt, "input=%zu padding=%zu filter=%zu", inp, pad, flt);
    return (inp + 2 * pad - flt) / stride + 1;
}

namespace convolution {
std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Convolution>();
    return Convolution::make(node->param(), node->execution_policy());
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& conv = static_cast<const Convolution&>(def);
    OperatorNodeConfig config{conv.make_name()};
    return opr::Convolution::make(
            inputs[0], inputs[1], conv.param(), conv.policy(), config);
}

TensorLayout do_shape_infer(
        const OpDef& def, size_t src_ndim, TensorLayout src, TensorLayout filter) {
    auto&& conv = static_cast<const Convolution&>(def);
    using Param = ::megdnn::param::Convolution;

    auto img_ndim = src_ndim - 2;
    mgb_assert(
            img_ndim == 2,
            "only 2D convolution is supported, and input should be 4-dim; "
            "got input dim = %zu",
            src_ndim);
    size_t group = 1;
    size_t flt_start, flt_spatial_start, ocpg_pos, icpg_pos;
    if (conv.sparse == Param::Sparse::DENSE) {
        mgb_assert(
                filter.ndim == img_ndim + 2 || filter.ndim == img_ndim + 4,
                "bad filter ndim for dense convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);
        group = 1;
        flt_start = 0;
    } else {  // Param::Sparse::GROUP
        mgb_assert(
                filter.ndim == img_ndim + 3 || filter.ndim == img_ndim + 5,
                "bad filter ndim for group convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);
        // grp, oc, ic, dims[]
        group = filter[0];
        flt_start = 1;
    }

    uint32_t ic_block_size = 1, oc_block_size = 1;
    size_t src_or_dst_c_pos = 0;
    size_t src_or_dst_spatial_start = 0;
    if (conv.format == Param::Format::NCHW) {
        // filter should be (oc, ic, fh, fw)
        flt_spatial_start = 2;
        ocpg_pos = 0;
        icpg_pos = 1;
        src_or_dst_c_pos = 1;
        src_or_dst_spatial_start = 2;
    } else {  // Param::Format::NHWC
        // filter should be (oc, fh, fw, ic)
        flt_spatial_start = 1;
        ocpg_pos = 0;
        icpg_pos = 3;
        src_or_dst_c_pos = 3;
        src_or_dst_spatial_start = 1;
    }
    size_t ocpg = filter[flt_start + ocpg_pos] * oc_block_size;
    size_t icpg = filter[flt_start + icpg_pos] * ic_block_size;
    uint32_t dilation[2], dilated_spatial[2], stride[2], padding[2];
    dilation[0] = conv.dilate_h;
    dilation[1] = conv.dilate_w;
    stride[0] = conv.stride_h;
    stride[1] = conv.stride_w;
    padding[0] = conv.pad_h;
    padding[1] = conv.pad_w;
    for (size_t i = 0; i < img_ndim; ++i) {
        mgb_assert(
                dilation[i] > 0, "invalid dilation on spatial dim %zu: %u", i,
                dilation[i]);
        dilated_spatial[i] =
                (filter[i + flt_start + flt_spatial_start] - 1) * dilation[i] + 1;
    }
    mgb_assert(icpg * group == src[src_or_dst_c_pos], "group conv invalid");

    TensorLayout dst{src.dtype};
    dst.ndim = src_ndim;
    dst[0] = src[0];
    dst[src_or_dst_c_pos] = ocpg * group;
    for (size_t i = 0; i < img_ndim; ++i) {
        dst[i + src_or_dst_spatial_start] = infer_conv_shape(
                src[i + src_or_dst_spatial_start], dilated_spatial[i], stride[i],
                padding[i]);
    }
    dst.init_contiguous_stride();
    return dst;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    SmallVector<LogicalTensorDesc> dests(1);
    auto&& desc = dests[0];
    desc.comp_node = inputs[0].comp_node;

    TensorLayout src = inputs[0].layout;
    TensorLayout filter = inputs[1].layout;
    size_t src_ndim = src.ndim;
    if (src_ndim == 0 || filter.ndim == 0) {
        desc.layout = TensorLayout{{}, src.dtype};
        return {dests, false};
    }

    desc.layout = do_shape_infer(def, src_ndim, src, filter);
    return {dests, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    // create megdnn opr
    auto&& conv = static_cast<const Convolution&>(def);
    CompNode cn = inputs[0]->comp_node();

    TensorLayout out_layout = output_descs[0].layout;
    if (!validated)
        out_layout = do_shape_infer(
                def, inputs[0]->layout().ndim, inputs[0]->layout(),
                inputs[1]->layout());

    DeviceTensorND out =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, out_layout);

    using TensorND = megdnn::TensorND;
    SmallVector<TensorND> inp_tensornds(inputs.size());
    TensorLayoutArray inp_shapes(inputs.size()), oup_shapes(output_descs.size());
    for (unsigned i = 0; i < inputs.size(); ++i) {
        inp_tensornds[i] = inputs[i]->dnn_tensor();
        inp_shapes[i] = inputs[i]->layout();
    }
    oup_shapes[0] = out_layout;
    DnnOprCaller<megdnn::ConvBiasForward> dnn_opr(cn);
    auto&& param = dnn_opr.op->param();
    param.pad_h = conv.pad_h;
    param.pad_w = conv.pad_w;
    param.stride_h = conv.stride_h;
    param.stride_w = conv.stride_w;
    param.dilate_h = conv.dilate_h;
    param.dilate_w = conv.dilate_w;
    param.sparse = conv.sparse;
    param.compute_mode = conv.compute_mode;
    param.format = conv.format;

    // shape infer
    TensorLayout empty_shp({0}, inputs[0]->dtype());
    empty_shp.ndim = 0;

    size_t sz = setup_algo<megdnn::ConvBiasForward>(
            {inp_shapes[0], inp_shapes[1], empty_shp, empty_shp, oup_shapes[0]},
            dnn_opr.op.get(), 0, false, false, cn, conv.policy(), false);

    // alloc memory
    DeviceTensorND empty_bias =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, empty_shp);

    TensorLayout w_layout({sz}, dtype::Byte());
    auto dnn_wk = dnn_opr.create_workspace(w_layout);

    // exeucte
    dnn_opr.op->exec(
            inp_tensornds[0], inp_tensornds[1], empty_bias.as_megdnn(),
            empty_bias.as_megdnn(), out.as_megdnn(), nullptr, dnn_wk);
    return {Tensor::make(out)};
}

OP_TRAIT_REG(Convolution, Convolution, opr::Convolution)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace convolution
}  // namespace

namespace {
namespace conv_bias {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& conv = static_cast<const ConvBias&>(def);
    cg::OperatorNodeConfig config{conv.dtype};
    config.name(conv.make_name());
    if (inputs.size() == 2) {
        return opr::ConvBias::make(
                inputs[0], inputs[1], conv.param(), conv.policy(), config);
    } else if (inputs.size() == 3) {
        return opr::ConvBias::make(
                inputs[0], inputs[1], inputs[2], conv.param(), conv.policy(), config);
    } else if (inputs.size() == 4) {
        return opr::ConvBias::make(
                inputs[0], inputs[1], inputs[2], inputs[3], conv.param(), conv.policy(),
                config);
    }
    mgb_assert(0);
}

OP_TRAIT_REG(ConvBias, ConvBias).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace conv_bias
}  // namespace

namespace {
namespace convolution_backward_data {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& conv = static_cast<const ConvolutionBackwardData&>(def);
    OperatorNodeConfig config{conv.make_name()};
    DType output_dtype = conv.dtype;
    if (output_dtype.valid()) {
        config.output_dtype(output_dtype);
    }

    if (inputs.size() == 2) {
        return opr::ConvolutionBackwardData::make(
                inputs[0], inputs[1], conv.param(), conv.policy(), config);
    } else {
        mgb_assert(inputs.size() == 3);
        return opr::ConvolutionBackwardData::make(
                inputs[0], inputs[1], inputs[2], conv.param(), conv.policy(), config);
    }
}

TensorLayout convbwd_do_shape_infer(
        const OpDef& def, size_t diff_ndim, TensorLayout filter, TensorLayout diff,
        CompNode cn) {
    auto&& bwd_conv = static_cast<const ConvolutionBackwardData&>(def);
    DnnOprCaller<megdnn::ConvolutionBackwardData> caller(cn);
    auto&& dnn_opr = caller.op;
    using Param = ::megdnn::param::Convolution;
    // using Param1 = ::megdnn::param::ConvolutionBackwardData;

    auto img_ndim = diff_ndim - 2;
    mgb_assert(
            img_ndim == 2,
            "only 2D convolution is supported, and input should be 4-dim; "
            "got input dim = %zu",
            diff_ndim);
    size_t group = 1;
    size_t flt_start, flt_spatial_start, ocpg_pos, icpg_pos;
    if (bwd_conv.sparse == Param::Sparse::DENSE) {
        mgb_assert(
                filter.ndim == img_ndim + 2 || filter.ndim == img_ndim + 4,
                "bad filter ndim for dense convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);
        group = 1;
        flt_start = 0;
    } else {  // Param::Sparse::GROUP
        mgb_assert(
                filter.ndim == img_ndim + 3 || filter.ndim == img_ndim + 5,
                "bad filter ndim for group convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);
        // grp, oc, ic, dims[]
        group = filter[0];
        flt_start = 1;
    }

    uint32_t ic_block_size = 1, oc_block_size = 1;
    size_t src_or_dst_c_pos = 0;
    size_t src_or_dst_spatial_start = 0;
    if (bwd_conv.format == Param::Format::NCHW) {
        // filter should be (oc, ic, fh, fw)
        flt_spatial_start = 2;
        ocpg_pos = 0;
        icpg_pos = 1;
        src_or_dst_c_pos = 1;
        src_or_dst_spatial_start = 2;
    } else {  // Param::Format::NHWC
        // filter should be (oc, fh, fw, ic)
        flt_spatial_start = 1;
        ocpg_pos = 0;
        icpg_pos = 3;
        src_or_dst_c_pos = 3;
        src_or_dst_spatial_start = 1;
    }
    size_t ocpg = filter[flt_start + ocpg_pos] * oc_block_size;
    size_t icpg = filter[flt_start + icpg_pos] * ic_block_size;
    uint32_t dilation[2], dilated_spatial[2], stride[2], padding[2];
    dilation[0] = bwd_conv.dilate_h;
    dilation[1] = bwd_conv.dilate_w;
    stride[0] = bwd_conv.stride_h;
    stride[1] = bwd_conv.stride_w;
    padding[0] = bwd_conv.pad_h;
    padding[1] = bwd_conv.pad_w;
    for (size_t i = 0; i < img_ndim; ++i) {
        mgb_assert(
                dilation[i] > 0, "invalid dilation on spatial dim %zu: %u", i,
                dilation[i]);
        dilated_spatial[i] =
                (filter[i + flt_start + flt_spatial_start] - 1) * dilation[i] + 1;
    }
    mgb_assert(ocpg * group == diff[src_or_dst_c_pos], "group conv invalid");

    auto deduce = [](size_t out, size_t filter, size_t stride, size_t pad) {
        auto i = (out - 1) * stride + filter;
        mgb_assert(i > pad * 2);
        return i - pad * 2;
    };

    DType dst_dtype = bwd_conv.dtype;
    dnn_opr->deduce_dtype(filter.dtype, diff.dtype, dst_dtype);
    TensorLayout dst{dst_dtype};
    dst.ndim = diff_ndim;
    dst[0] = diff[0];
    dst[src_or_dst_c_pos] = icpg * group;
    for (size_t i = 0; i < img_ndim; ++i) {
        dst[i + src_or_dst_spatial_start] =
                deduce(diff[i + src_or_dst_spatial_start], dilated_spatial[i],
                       stride[i], padding[i]);
    }
    dst.init_contiguous_stride();
    return dst;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    SmallVector<LogicalTensorDesc> dests(1);
    auto&& desc = dests[0];
    desc.comp_node = inputs[0].comp_node;

    TensorLayout filter = inputs[0].layout;
    TensorLayout diff = inputs[1].layout;
    size_t diff_ndim = diff.ndim;
    if (diff_ndim == 0 || filter.ndim == 0) {
        desc.layout = TensorLayout{{}, diff.dtype};
        return {dests, false};
    }

    desc.layout =
            convbwd_do_shape_infer(def, diff_ndim, filter, diff, inputs[0].comp_node);
    return {dests, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    // create megdnn opr
    auto&& convbwd = static_cast<const ConvolutionBackwardData&>(def);
    CompNode cn = inputs[0]->comp_node();

    TensorLayout out_layout = output_descs[0].layout;
    if (!validated)
        out_layout = convbwd_do_shape_infer(
                def, inputs[1]->layout().ndim, inputs[0]->layout(), inputs[1]->layout(),
                cn);

    DeviceTensorND out =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, out_layout);

    using TensorND = megdnn::TensorND;
    SmallVector<TensorND> inp_tensornds(inputs.size());
    TensorLayoutArray inp_shapes(inputs.size()), oup_shapes(output_descs.size());
    for (unsigned i = 0; i < inputs.size(); ++i) {
        inp_tensornds[i] = inputs[i]->dnn_tensor();
        inp_shapes[i] = inputs[i]->layout();
    }
    oup_shapes[0] = out_layout;
    DnnOprCaller<megdnn::ConvolutionBackwardData> dnn_opr(cn);
    dnn_opr.op->param() = convbwd.param();

    size_t sz = setup_algo<megdnn::ConvolutionBackwardData>(
            {inp_shapes[0], inp_shapes[1], oup_shapes[0]}, dnn_opr.op.get(), 0, false,
            false, cn, convbwd.policy(), false);

    auto wk = Blob::make(cn, sz);
    auto ptr = wk->storage().get();
    megdnn::Workspace dnn_wk(ptr, sz);

    // exeucte
    dnn_opr.op->exec(inp_tensornds[0], inp_tensornds[1], out.as_megdnn(), dnn_wk);
    return {Tensor::make(out)};
}

OP_TRAIT_REG(ConvolutionBackwardData, ConvolutionBackwardData)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace convolution_backward_data
}  // namespace

namespace {
namespace convolution3d {
std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Convolution3D>();
    return Convolution3D::make(node->param(), node->execution_policy());
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& conv = static_cast<const Convolution3D&>(def);
    return opr::Convolution3D::make(inputs[0], inputs[1], conv.param(), conv.policy());
}

TensorLayout do_shape_infer(
        const OpDef& def, size_t src_ndim, TensorLayout src, TensorLayout filter) {
    auto&& conv = static_cast<const Convolution3D&>(def);
    using Param = ::megdnn::param::Convolution3D;
    auto img_ndim = src_ndim - 2;
    mgb_assert(
            img_ndim == 3,
            "only 3D convolution is supported, and input should be 5-dim; "
            "got input dim = %zu",
            src_ndim);

    size_t group = 1;
    size_t flt_start, flt_spatial_start, ocpg_pos, icpg_pos;
    if (conv.sparse == Param::Sparse::DENSE) {
        mgb_assert(
                filter.ndim == img_ndim + 2 || filter.ndim == img_ndim + 4,
                "bad filter ndim for dense convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);
        group = 1;
        flt_start = 0;
    } else {  // Param::Sparse::GROUP
        mgb_assert(
                filter.ndim == img_ndim + 3 || filter.ndim == img_ndim + 5,
                "bad filter ndim for group convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);

        // grp, oc, ic, dims[]
        group = filter[0];
        flt_start = 1;
    }

    uint32_t ic_block_size = 1, oc_block_size = 1;
    size_t src_or_dst_c_pos = 0;
    size_t src_or_dst_spatial_start = 0;
    if (conv.format == Param::Format::NCDHW) {
        // filter should be (oc, ic, fd, fh, fw)
        flt_spatial_start = 2;
        ocpg_pos = 0;
        icpg_pos = 1;
        src_or_dst_c_pos = 1;
        src_or_dst_spatial_start = 2;
    } else {  // Param::Format::NDHWC
        // filter should be (oc, fd, fh, fw, ic)
        flt_spatial_start = 1;
        ocpg_pos = 0;
        icpg_pos = 4;
        src_or_dst_c_pos = 4;
        src_or_dst_spatial_start = 1;
    }
    size_t ocpg = filter[flt_start + ocpg_pos] * oc_block_size;
    size_t icpg = filter[flt_start + icpg_pos] * ic_block_size;
    uint32_t dilation[3], dilated_spatial[3], stride[3], padding[3];
    dilation[0] = conv.dilate_d;
    dilation[1] = conv.dilate_h;
    dilation[2] = conv.dilate_w;
    stride[0] = conv.stride_d;
    stride[1] = conv.stride_h;
    stride[2] = conv.stride_w;
    padding[0] = conv.pad_d;
    padding[1] = conv.pad_h;
    padding[2] = conv.pad_w;
    for (size_t i = 0; i < img_ndim; ++i) {
        mgb_assert(
                dilation[i] > 0, "invalid dilation on spatial dim %zu: %u", i,
                dilation[i]);
        dilated_spatial[i] =
                (filter[i + flt_start + flt_spatial_start] - 1) * dilation[i] + 1;
    }
    mgb_assert(icpg * group == src[src_or_dst_c_pos], "group conv invalid");

    TensorLayout dst{src.dtype};
    dst.ndim = src_ndim;
    dst[0] = src[0];
    dst[src_or_dst_c_pos] = ocpg * group;
    for (size_t i = 0; i < img_ndim; ++i) {
        dst[i + src_or_dst_spatial_start] = infer_conv_shape(
                src[i + src_or_dst_spatial_start], dilated_spatial[i], stride[i],
                padding[i]);
    }
    dst.init_contiguous_stride();

    return dst;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    SmallVector<LogicalTensorDesc> dests(1);
    auto&& desc = dests[0];
    desc.comp_node = inputs[0].comp_node;

    TensorLayout src = inputs[0].layout;
    TensorLayout filter = inputs[1].layout;
    size_t src_ndim = src.ndim;
    if (src_ndim == 0 || filter.ndim == 0) {
        desc.layout = TensorLayout{{}, src.dtype};
        return {dests, false};
    }

    desc.layout = do_shape_infer(def, src_ndim, src, filter);
    return {dests, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    // create megdnn opr
    auto&& conv = static_cast<const Convolution3D&>(def);

    TensorLayout out_layout = output_descs[0].layout;
    if (!validated)
        out_layout = do_shape_infer(
                def, inputs[0]->layout().ndim, inputs[0]->layout(),
                inputs[1]->layout());

    using TensorND = megdnn::TensorND;
    CompNode cn = inputs[0]->comp_node();
    SmallVector<TensorND> inp_tensornds(inputs.size());
    TensorLayoutArray inp_shapes(inputs.size()), oup_shapes(output_descs.size());
    for (unsigned i = 0; i < inputs.size(); ++i) {
        inp_tensornds[i] = inputs[i]->dnn_tensor();
        inp_shapes[i] = inputs[i]->layout();
    }
    oup_shapes[0] = out_layout;
    DnnOprCaller<megdnn::Convolution3D> dnn_opr(cn);
    dnn_opr.op->param() = conv.param();

    // shape infer
    size_t sz = setup_algo<megdnn::Convolution3D>(
            {inp_shapes[0], inp_shapes[1], oup_shapes[0]}, dnn_opr.op.get(), 0, false,
            false, cn, conv.policy(), false);

    // alloc memory
    DeviceTensorND out =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, out_layout);

    megdnn::Workspace dnn_wk;
    if (sz != 0) {
        TensorLayout w_layout({sz}, dtype::Byte());
        dnn_wk = dnn_opr.create_workspace(w_layout);
    }

    // exeucte
    dnn_opr.op->exec(inp_tensornds[0], inp_tensornds[1], out.as_megdnn(), dnn_wk);
    return {Tensor::make(out)};
}

OP_TRAIT_REG(Convolution3D, Convolution3D, opr::Convolution3D)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace convolution3d
}  // namespace

namespace {
namespace convolution3d_backward_data {

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    mgb_assert(
            inputs.size() == 2,
            "inputs num of conv_transpose3d should be 2 but you give %zu",
            inputs.size());

    auto&& op_def = def.cast_final_safe<Convolution3DBackwardData>();
    auto&& weight = inputs[0];
    auto&& diff = inputs[1];
    auto& cn = weight.comp_node;

    if (weight.layout.ndim == 0 || diff.layout.ndim == 0) {
        return {{{TensorLayout{weight.layout.dtype}, cn, {}}}, false};
    }

    TensorLayout oup_layout;
    megdnn::Convolution3DBackwardData::deduce_layout_impl(
            weight.layout, diff.layout, op_def.param(), oup_layout);
    return {{{oup_layout, cn, {}}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op_def = def.cast_final_safe<Convolution3DBackwardData>();
    auto cn = inputs[0]->comp_node();
    megdnn::TensorND weight = inputs[0]->dnn_tensor();
    megdnn::TensorND diff = inputs[1]->dnn_tensor();

    DnnOprCaller<megdnn::Convolution3DBackwardData> caller(cn);
    auto&& dnn_opr = caller.op;
    dnn_opr->param() = op_def.param();

    TensorLayout& oup_layout = output_descs[0].layout;
    if (!validated) {
        megdnn::Convolution3DBackwardData::deduce_layout_impl(
                weight.layout, diff.layout, op_def.param(), oup_layout);
    }
    DeviceTensorND oup =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, oup_layout);

    size_t wk_size = setup_algo<megdnn::Convolution3DBackwardData>(
            {weight.layout, diff.layout, oup_layout}, dnn_opr.get(), 0, false, false,
            cn, op_def.policy(), false);
    megdnn::Workspace dnn_wk;
    if (wk_size != 0) {
        TensorLayout w_layout({wk_size}, dtype::Byte());
        dnn_wk = caller.create_workspace(w_layout);
    }

    dnn_opr->exec(weight, diff, oup.as_megdnn(), dnn_wk);
    return {Tensor::make(oup)};
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& conv = static_cast<const Convolution3DBackwardData&>(def);
    OperatorNodeConfig config{conv.make_name()};
    mgb_assert(inputs.size() == 2);
    return opr::Convolution3DBackwardData::make(
            inputs[0], inputs[1], conv.param(), conv.policy(), config);
}

OP_TRAIT_REG(Convolution3DBackwardData, Convolution3DBackwardData)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace convolution3d_backward_data
}  // namespace

}  // namespace imperative
}  // namespace mgb
