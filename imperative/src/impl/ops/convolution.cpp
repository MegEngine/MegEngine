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
        return opr::ConvolutionBackwardData::make(
                inputs[0], inputs[1], inputs[2], conv.param(), conv.policy(), config);
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
            dnn_opr.op()->deduce_layout(
                    inputs[0]->layout(), inputs[1]->layout(), out_layout);
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
            inputs.size() == 2,
            "inputs num of conv_transpose3d should be 2 but you give %zu",
            inputs.size());
    auto&& op_def = def.cast_final_safe<Convolution3DBackwardData>();
    auto&& weight = inputs[0];
    auto&& diff = inputs[1];
    if (!(weight.layout.ndim && diff.layout.ndim)) {
        return {{{TensorLayout{weight.layout.dtype}, weight.comp_node}}, false};
    }
    DnnOprHelper<megdnn::Convolution3DBackwardData> dnn_opr(op_def.param());
    auto oup_layout = dnn_opr.deduce_layout(weight.layout, diff.layout);
    return {{{oup_layout, weight.comp_node}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& conv = def.cast_final_safe<Convolution3DBackwardData>();
    auto cn = inputs[0]->comp_node();

    auto&& wlayout = inputs[0]->layout();
    auto&& dlayout = inputs[1]->layout();

    DnnOprCaller<megdnn::Convolution3DBackwardData> dnn_op(
            cn, conv.param(), conv.policy());

    auto oup_layout = [&] {
        if (validated) {
            return output_descs[0].layout;
        } else {
            return dnn_op.deduce_layout(wlayout, dlayout);
        }
    }();
    auto oup = Tensor::make(oup_layout, cn);
    dnn_op.exec_fastrun(inputs[0], inputs[1], oup);
    return {oup};
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
