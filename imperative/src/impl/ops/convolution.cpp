#include "megbrain/opr/dnn/convolution.h"
#include "../algo_chooser.h"
#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"
#include "megbrain/common.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/tensor_gen.h"
#include "megdnn/oprs/nn.h"

namespace mgb {
namespace imperative {
namespace {
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

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& conv = def.cast_final_safe<Convolution>();
    DnnOprHelper<megdnn::ConvolutionForward> dnn_opr(conv.param());
    auto&& data = inputs[0].layout;
    auto&& filter = inputs[1].layout;
    TensorLayout output_layout{data.dtype};
    if (data.ndim && filter.ndim) {
        // deduce_layout won't override existing dtype
        dnn_opr.opr().deduce_layout(data, filter, output_layout);
    }
    return {{{output_layout, inputs[0].comp_node}}, output_layout.ndim != 0};
}

// Convolution::Param -> ConvBias::Param
auto conv_bias_param_from_convolution(const Convolution& conv) {
    megdnn::ConvBias::Param param;
    param.pad_h = conv.pad_h;
    param.pad_w = conv.pad_w;
    param.stride_h = conv.stride_h;
    param.stride_w = conv.stride_w;
    param.dilate_h = conv.dilate_h;
    param.dilate_w = conv.dilate_w;
    param.sparse = conv.sparse;
    param.compute_mode = conv.compute_mode;
    param.format = conv.format;
    return param;
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    // create megdnn opr
    auto&& conv = def.cast_final_safe<Convolution>();
    CompNode cn = inputs[0]->comp_node();

    // calling dnn ConvolutionForward when device is rocm
    // because there is no dnn ConvBiasForward on rocm
    if (cn.device_type() == CompNode::DeviceType::ROCM) {
        DnnOprCaller<megdnn::ConvolutionForward> dnn_opr(
                cn, conv.param(), conv.policy());
        auto out_layout = [&] {
            if (validated) {
                return output_descs[0].layout;
            } else {
                return dnn_opr.deduce_layout(inputs[0]->layout(), inputs[1]->layout());
            }
        }();

        // alloc memory
        auto out = Tensor::make(out_layout, cn);
        dnn_opr.exec_fastrun(inputs[0], inputs[1], out);
        return {out};
    }

    // calling dnn ConvBiasForward on cuda because it's faster then ConvolutionForward
    // ConvolutionForward internally uses ConvBiasForward to calculate the result
    auto&& param = conv_bias_param_from_convolution(conv);
    DnnOprCaller<megdnn::ConvBiasForward> dnn_opr(cn, param, conv.policy());

    megdnn::TensorND empty_bias;
    empty_bias.layout.dtype = inputs[0]->dtype();
    empty_bias.layout.ndim = 0;

    auto out_layout = [&] {
        if (validated) {
            return output_descs[0].layout;
        } else {
            TensorLayout out_layout{inputs[0]->dtype()};
            dnn_opr.op()->deduce_layout(
                    inputs[0]->layout(), inputs[1]->layout(), empty_bias.layout,
                    empty_bias.layout, out_layout);
            return out_layout;
        }
    }();

    // alloc memory
    auto out = Tensor::make(out_layout, cn);
    dnn_opr.exec_fastrun(inputs[0], inputs[1], empty_bias, empty_bias, out);
    return {out};
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
        auto* src_for_shape =
                opr::Alloc::make(inputs[2], inputs[0]->dtype(), {}).node();
        return opr::ConvolutionBackwardData::make(
                inputs[0], inputs[1], src_for_shape, conv.param(), conv.policy(),
                config);
    }
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& convbwd = def.cast_final_safe<ConvolutionBackwardData>();
    DnnOprHelper<megdnn::ConvolutionBackwardData> dnn_opr(convbwd.param());
    // force set dtype
    auto&& filter = inputs[0].layout;
    auto&& diff = inputs[1].layout;
    TensorLayout output_layout{convbwd.dtype};
    if (filter.ndim && diff.ndim) {
        // deduce_layout won't override existing dtype
        dnn_opr.opr().deduce_layout(filter, diff, output_layout);
    } else {
        dnn_opr.opr().deduce_dtype(filter.dtype, diff.dtype, output_layout.dtype);
    }
    if (inputs.size() == 3) {
        if (!inputs[2].value.empty()) {
            cg::copy_tensor_value_to_shape(output_layout, inputs[2].value);
            output_layout.init_contiguous_stride();
        } else {
            output_layout.ndim = 0;
        }
    }
    return {{{output_layout, inputs[0].comp_node}}, output_layout.ndim != 0};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    // create megdnn opr
    auto&& convbwd = def.cast_final_safe<ConvolutionBackwardData>();
    CompNode cn = inputs[0]->comp_node();
    DnnOprCaller<megdnn::ConvolutionBackwardData> dnn_opr(
            cn, convbwd.param(), convbwd.policy());
    auto out_layout = [&] {
        if (validated) {
            return output_descs[0].layout;
        } else {
            TensorLayout out_layout{inputs[0]->dtype()};
            if (inputs.size() == 3) {
                cg::copy_tensor_value_to_shape(
                        out_layout, inputs[2]->get_value().proxy_to_default_cpu());
            } else {
                dnn_opr.op()->deduce_layout(
                        inputs[0]->layout(), inputs[1]->layout(), out_layout);
            }
            out_layout.init_contiguous_stride();
            return out_layout;
        }
    }();
    auto out = Tensor::make(out_layout, cn);
    dnn_opr.exec_fastrun(inputs[0], inputs[1], out);
    return {out};
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

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& conv = def.cast_final_safe<Convolution3D>();
    TensorLayout src = inputs[0].layout;
    TensorLayout filter = inputs[1].layout;
    if (src.ndim == 0 || filter.ndim == 0) {
        return {{{TensorLayout{src.dtype}, inputs[0].comp_node}}, false};
    }
    DnnOprHelper<megdnn::Convolution3DForward> dnn_opr(conv.param());
    auto output = dnn_opr.deduce_layout(src, filter);
    return {{{output, inputs[0].comp_node}}, false};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    // create megdnn opr
    auto&& conv = def.cast_final_safe<Convolution3D>();
    CompNode cn = inputs[0]->comp_node();
    DnnOprCaller<megdnn::Convolution3D> dnn_opr(cn, conv.param(), conv.policy());
    auto out_layout = [&] {
        if (validated) {
            return output_descs[0].layout;
        } else {
            return dnn_opr.deduce_layout(inputs[0]->layout(), inputs[1]->layout());
        }
    }();
    // alloc memory
    auto out = Tensor::make(out_layout, cn);
    dnn_opr.exec_fastrun(inputs[0], inputs[1], out);
    return {out};
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
            inputs.size() == 2 || inputs.size() == 3,
            "inputs num of conv_transpose3d should be 2 or 3 but you give %zu",
            inputs.size());
    auto&& conv3dbwd = def.cast_final_safe<Convolution3DBackwardData>();
    DnnOprHelper<megdnn::Convolution3DBackwardData> dnn_opr(conv3dbwd.param());
    auto&& filter = inputs[0];
    auto&& diff = inputs[1];

    if (!(filter.layout.ndim && diff.layout.ndim)) {
        return {{{TensorLayout{filter.layout.dtype}, filter.comp_node}}, false};
    }

    TensorLayout output_layout = dnn_opr.deduce_layout(filter.layout, diff.layout);
    if (filter.layout.ndim && diff.layout.ndim) {
        if (inputs.size() == 3) {
            if (!inputs[2].value.empty()) {
                cg::copy_tensor_value_to_shape(output_layout, inputs[2].value);
                output_layout.init_contiguous_stride();
            } else {
                output_layout.ndim = 0;
            }
        }
    }
    return {{{output_layout, inputs[0].comp_node}}, output_layout.ndim != 0};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& conv3dbwd = def.cast_final_safe<Convolution3DBackwardData>();
    CompNode cn = inputs[0]->comp_node();
    DnnOprCaller<megdnn::Convolution3DBackwardData> dnn_opr(
            cn, conv3dbwd.param(), conv3dbwd.policy());
    auto out_layout = [&] {
        if (validated) {
            return output_descs[0].layout;
        } else {
            TensorLayout out_layout{inputs[0]->dtype()};
            dnn_opr.op()->deduce_layout(
                    inputs[0]->layout(), inputs[1]->layout(), out_layout);
            if (inputs.size() == 3) {
                cg::copy_tensor_value_to_shape(
                        out_layout, inputs[2]->get_value().proxy_to_default_cpu());
                out_layout.init_contiguous_stride();
            }
            return out_layout;
        }
    }();
    auto out = Tensor::make(out_layout, cn);
    dnn_opr.exec_fastrun(inputs[0], inputs[1], out);
    return {out};
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& conv = static_cast<const Convolution3DBackwardData&>(def);
    OperatorNodeConfig config{conv.make_name()};
    if (inputs.size() == 2) {
        return opr::Convolution3DBackwardData::make(
                inputs[0], inputs[1], conv.param(), conv.policy(), config);
    } else {
        mgb_assert(inputs.size() == 3);
        // The output shape is calculated in advance and given as input
        auto* src_for_shape =
                opr::Alloc::make(inputs[2], inputs[0]->dtype(), {}).node();
        return opr::Convolution3DBackwardData::make(
                inputs[0], inputs[1], src_for_shape, conv.param(), conv.policy(),
                config);
    }
}

OP_TRAIT_REG(Convolution3DBackwardData, Convolution3DBackwardData)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace convolution3d_backward_data
}  // namespace

namespace {
namespace region_restricted_conv {
std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::RegionRestrictedConvolution>();
    return RegionRestrictedConvolution::make(node->param());
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& conv = static_cast<const RegionRestrictedConvolution&>(def);
    OperatorNodeConfig config{conv.make_name()};
    return opr::RegionRestrictedConvolution::make(
            inputs[0], inputs[1], inputs[2], inputs[3], conv.param(), config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& region_restricted_conv =
            def.cast_final_safe<mgb::imperative::RegionRestrictedConvolution>();
    DnnOprHelper<megdnn::RegionRestrictedConvolutionForward> dnn_opr(
            region_restricted_conv.param());

    auto&& src = inputs[0].layout;
    auto&& filter = inputs[1].layout;
    auto&& rin = inputs[2].layout;
    auto&& rout = inputs[3].layout;
    TensorLayout output_layout{src.dtype};
    if (src.ndim && filter.ndim) {
        dnn_opr.opr().deduce_layout(src, filter, rin, rout, output_layout);
    }

    return {{{output_layout, inputs[0].comp_node}}, output_layout.ndim != 0};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    // create megdnn opr
    auto&& region_restricted_conv = def.cast_final_safe<RegionRestrictedConvolution>();
    CompNode cn = inputs[0]->comp_node();

    auto&& param = region_restricted_conv.param();
    DnnOprCaller<megdnn::RegionRestrictedConvolutionForward> dnn_opr(cn, param);

    auto srclo = inputs[0]->layout();
    auto filterlo = inputs[1]->layout();
    auto rinlo = inputs[2]->layout();
    auto routlo = inputs[3]->layout();

    auto out_layout = [&] {
        if (validated) {
            return output_descs[0].layout;
        } else {
            TensorLayout out_layout{inputs[0]->dtype()};
            dnn_opr.op()->deduce_layout(srclo, filterlo, rinlo, routlo, out_layout);
            return out_layout;
        }
    }();

    auto out = Tensor::make(out_layout, cn);
    dnn_opr.exec_with_ws(inputs[0], inputs[1], inputs[2], inputs[3], out);
    return {out};
}

OP_TRAIT_REG(
        RegionRestrictedConvolution, RegionRestrictedConvolution,
        opr::RegionRestrictedConvolution)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace region_restricted_conv
}  // namespace

namespace {
namespace region_restricted_conv_backward_data {

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node =
            &node_->cast_final_safe<opr::RegionRestrictedConvolutionBackwardData>();
    return RegionRestrictedConvolutionBackwardData::make(node->param());
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& conv = static_cast<const RegionRestrictedConvolutionBackwardData&>(def);
    OperatorNodeConfig config{conv.make_name()};
    // output_dtype may infered from input within rrconv bwd data(deduce_dtype api)
    CompNode cn = inputs[0]->comp_node();
    DType output_dtype;
    DnnOprCaller<megdnn::RegionRestrictedConvolutionBackwardData> dnn_opr(cn);
    dnn_opr.op()->deduce_dtype(
            inputs[0]->dtype(), inputs[1]->dtype(), inputs[2]->dtype(),
            inputs[3]->dtype(), output_dtype);
    if (output_dtype.valid())
        config.output_dtype(output_dtype);
    if (inputs.size() == 4) {
        return opr::RegionRestrictedConvolutionBackwardData::make(
                inputs[0], inputs[1], inputs[2], inputs[3], conv.param(), config);
    } else if (inputs.size() == 5) {
        return opr::RegionRestrictedConvolutionBackwardData::make(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], conv.param(),
                config);
    }
    mgb_assert(0);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& convbwd = def.cast_final_safe<
            mgb::imperative::RegionRestrictedConvolutionBackwardData>();
    DnnOprHelper<megdnn::RegionRestrictedConvolutionBackwardData> dnn_opr(
            convbwd.param());

    TensorLayout filter = inputs[0].layout;
    TensorLayout diff = inputs[1].layout;
    TensorLayout rin = inputs[2].layout;
    TensorLayout rout = inputs[3].layout;

    DType output_dtype;
    dnn_opr.opr().deduce_dtype(
            inputs[0].layout.dtype, inputs[1].layout.dtype, inputs[2].layout.dtype,
            inputs[3].layout.dtype, output_dtype);
    TensorLayout output_layout{output_dtype};
    if (diff.ndim && filter.ndim) {
        dnn_opr.opr().deduce_layout(filter, diff, rin, rout, output_layout);
    }
    return {{{output_layout, inputs[0].comp_node}}, output_layout.ndim != 0};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& convbwd = def.cast_final_safe<RegionRestrictedConvolutionBackwardData>();
    CompNode cn = inputs[0]->comp_node();
    DnnOprCaller<megdnn::RegionRestrictedConvolutionBackwardData> dnn_opr(
            cn, convbwd.param());

    auto filterlo = inputs[0]->layout();
    auto difflo = inputs[1]->layout();
    auto rinlo = inputs[2]->layout();
    auto routlo = inputs[3]->layout();

    auto out_layout = [&] {
        if (validated) {
            return output_descs[0].layout;
        } else {
            TensorLayout out_layout{inputs[0]->dtype()};
            dnn_opr.op()->deduce_layout(filterlo, difflo, rinlo, routlo, out_layout);
            return out_layout;
        }
    }();

    auto out = Tensor::make(out_layout, cn);
    dnn_opr.exec_with_ws(inputs[0], inputs[1], inputs[2], inputs[3], out);
    return {out};
}

OP_TRAIT_REG(
        RegionRestrictedConvolutionBackwardData,
        RegionRestrictedConvolutionBackwardData,
        opr::RegionRestrictedConvolutionBackwardData)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace region_restricted_conv_backward_data
}  // namespace

}  // namespace imperative
}  // namespace mgb
