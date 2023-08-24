#include "megbrain/imperative/ops/custom_opdef.h"

#if MGB_CUSTOM_OP

#include "../op_trait.h"
#include "megbrain/custom/adaptor.h"
#include "megbrain/opr/custom_opnode.h"

namespace mgb {
namespace imperative {

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CustomOpDef);

CustomOpDef::CustomOpDef(const std::shared_ptr<const custom::CustomOp>& op)
        : m_op(op), m_param(op->param_info()) {}

CustomOpDef::CustomOpDef(
        const std::shared_ptr<const custom::CustomOp>& op, const custom::Param& param)
        : m_op(op), m_param(param) {}

void CustomOpDef::param(const custom::Param& rhs) {
    m_param = rhs;
}

custom::Param& CustomOpDef::param(void) {
    return m_param;
}

custom::Param CustomOpDef::param(void) const {
    return m_param;
}

size_t CustomOpDef::input_num(void) const {
    return m_op->input_num();
}

size_t CustomOpDef::output_num(void) const {
    return m_op->output_num();
}

std::string CustomOpDef::name(void) const {
    return m_op->op_type();
}

custom::RunTimeId CustomOpDef::runtime_id(void) const {
    return m_op->runtime_id();
}

const std::shared_ptr<const custom::CustomOp>& CustomOpDef::impl(void) const {
    return m_op;
}

void CustomOpDef::compute(
        std::shared_ptr<SmallVector<DeviceTensorND>> inputs,
        std::shared_ptr<SmallVector<DeviceTensorND>> outputs) const {
    custom::dispatch_custom_op(m_op, m_param, inputs, outputs);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> CustomOpDef::infer_output_attrs(
        const SmallVector<TensorPtr>& inputs) const {
    SmallVector<LogicalTensorDesc> input_descs(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        input_descs[i].comp_node = inputs[i]->comp_node();
        input_descs[i].layout = inputs[i]->layout();
    }
    return std::move(this->infer_output_attrs(input_descs));
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> CustomOpDef::infer_output_attrs(
        const SmallVector<LogicalTensorDesc>& inputs) const {
    SmallVector<CompNode> i_devices(inputs.size());
    SmallVector<TensorShape> i_shapes(inputs.size());
    SmallVector<megdnn::DType> i_dtypes(inputs.size());
    SmallVector<TensorFormat> i_formats(inputs.size());

    for (size_t i = 0; i < inputs.size(); i++) {
        i_devices[i] = inputs[i].comp_node;
        i_shapes[i] = inputs[i].layout;  // TensorLayout is derived from TensorShape
        i_dtypes[i] = inputs[i].layout.dtype;
        i_formats[i] = inputs[i].layout.format;
    }

    bool success = true;
    for (auto i_shape : i_shapes) {
        if (i_shape.ndim == 0) {
            success = false;
            break;
        }
    }

    SmallVector<CompNode> o_devices;
    SmallVector<megdnn::DType> o_dtypes;
    SmallVector<TensorFormat> o_formats;
    SmallVector<TensorShape> o_shapes;

    o_devices = custom::to_builtin<CompNode, custom::Device>(m_op->infer_output_device(
            custom::to_custom<CompNode, custom::Device>(i_devices), this->m_param));
    o_dtypes =
            custom::to_builtin<megdnn::DType, custom::DType>(m_op->infer_output_dtype(
                    custom::to_custom<megdnn::DType, custom::DType>(i_dtypes),
                    this->m_param));
    o_formats =
            custom::to_builtin<TensorFormat, custom::Format>(m_op->infer_output_format(
                    custom::to_custom<TensorFormat, custom::Format>(i_formats),
                    this->m_param));

    if (success) {
        o_shapes =
                custom::to_builtin<TensorShape, custom::Shape>(m_op->infer_output_shape(
                        custom::to_custom<TensorShape, custom::Shape>(i_shapes),
                        this->m_param));
    } else {
        o_shapes = SmallVector<TensorShape>(this->output_num());
    }

    SmallVector<LogicalTensorDesc> outputs(this->output_num());
    for (size_t i = 0; i < this->output_num(); i++) {
        outputs[i].comp_node = std::move(o_devices[i]);
        outputs[i].layout =
                std::move(TensorLayout(o_shapes[i], o_dtypes[i], o_formats[i]));
    }
    return std::tuple<SmallVector<LogicalTensorDesc>, bool>(outputs, success);
}

CustomOpDefFactory* CustomOpDefFactory::inst(void) {
    static CustomOpDefFactory factory;
    return &factory;
}

bool CustomOpDefFactory::is_custom_op(const OpDef& op) {
    return op.dyn_typeinfo() == CustomOpDef::typeinfo();
}

CustomOpDefFactory::CustomOpDefFactory() {
    ops = custom::CustomOpManager::inst();
}

std::vector<std::string> CustomOpDefFactory::op_list(void) const {
    return ops->op_name_list();
}

std::shared_ptr<OpDef> CustomOpDefFactory::create_opdef(
        const std::string& op_type) const {
    auto op = ops->find(op_type);
    return std::make_shared<CustomOpDef>(op);
}

std::shared_ptr<OpDef> CustomOpDefFactory::create_opdef(
        const custom::RunTimeId& op_id) const {
    auto op = ops->find(op_id);
    return std::make_shared<CustomOpDef>(op);
}

std::shared_ptr<OpDef> CustomOpDefFactory::create_opdef(
        const std::string& op_type, const custom::Param& param) const {
    auto op = ops->find(op_type);
    return std::make_shared<CustomOpDef>(op, param);
}

std::shared_ptr<OpDef> CustomOpDefFactory::create_opdef(
        const custom::RunTimeId& op_id, const custom::Param& param) const {
    auto op = ops->find(op_id);
    return std::make_shared<CustomOpDef>(op, param);
}

namespace custom_opdef {  // avoid name conflict

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    SmallVector<TensorPtr> outputs(output_descs.size());

    if (validated == false) {
        auto&& op = static_cast<const CustomOpDef&>(def);
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto [output_descs, success] = op.infer_output_attrs(inputs);
            mgb_assert(success == true, "infer output attributes fall\n");
        }
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto& output = outputs[i];
        output = Tensor::make(output_descs[i].layout, output_descs[i].comp_node);
    }

    std::shared_ptr<SmallVector<DeviceTensorND>> inp_tensornds =
            std::make_shared<SmallVector<DeviceTensorND>>();
    std::shared_ptr<SmallVector<DeviceTensorND>> oup_tensornds =
            std::make_shared<SmallVector<DeviceTensorND>>();
    for (size_t i = 0; i < inputs.size(); ++i) {
        inp_tensornds->emplace_back(inputs[i]->dev_tensor(true));
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        oup_tensornds->emplace_back(outputs[i]->dev_tensor(true));
    }

    auto&& op = static_cast<const CustomOpDef&>(def);
    op.compute(inp_tensornds, oup_tensornds);
    return outputs;
}

VarNodeArray apply_on_var_node(const OpDef& def, const cg::VarNodeArray& inputs) {
    auto&& op = static_cast<const CustomOpDef&>(def);
    OperatorNodeConfig config;
    VarNodeArray outputs =
            opr::CustomOpNode::make(op.impl(), inputs, op.param(), config);
    return outputs;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op = static_cast<const CustomOpDef&>(def);
    return op.infer_output_attrs(inputs);
}

size_t hash(const OpDef& def) {
    auto&& op = static_cast<const CustomOpDef&>(def);
    const custom::Param& param = op.param();
    size_t val = mgb::hash(op.runtime_id());
    std::string hash_str = "";
    for (auto&& val : param.raw()) {
        hash_str += val.first;
        hash_str += val.second.str();
    }

    val = mgb::hash_pair_combine(val, mgb::hash(hash_str));
    return val;
}

bool is_same_st(const OpDef& lhs, const OpDef& rhs) {
    auto &&a = static_cast<const CustomOpDef&>(lhs),
         &&b = static_cast<const CustomOpDef&>(rhs);
    return a.param() == b.param() && a.runtime_id() == b.runtime_id();
}

std::vector<std::pair<const char*, std::string>> props(const OpDef& def) {
    auto&& custom_opdef = def.cast_final_safe<CustomOpDef>();
    auto&& param_raw = custom_opdef.param().raw();
    std::vector<std::pair<const char*, std::string>> props_;
    for (auto&& kv : param_raw) {
        props_.emplace_back(kv.first.c_str(), kv.second.str());
    }
    return props_;
}

std::string make_name(const OpDef& def) {
    auto&& op = static_cast<const CustomOpDef&>(def);
    return op.name();
}

OP_TRAIT_REG(CustomOpDef, CustomOpDef)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .hash(hash)
        .is_same_st(is_same_st)
        .props(props)
        .make_name(make_name)
        .fallback();

}  // namespace custom_opdef

}  // namespace imperative
}  // namespace mgb

#endif
