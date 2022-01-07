#include "megbrain/imperative/transformations/format.h"

#include "megbrain/imperative/ops/autogen.h"

namespace mgb {
namespace imperative {

using FT = Format::Type;

TypedValueRef<FormattedTensorValue> FormattedTensorValue::as(const FT& target) const {
    return FormattedTensorValue::make(m_value, target);
}

TypedValueRef<FormattedTensorValue> FormattedTensorValue::to(
        const FT& target, const std::string& scope) const {
    std::vector<int32_t> pattern;
    if (m_format == FT::NHWC && target == FT::NCHW) {
        pattern = {0, 3, 1, 2};
    } else if (m_format == FT::NCHW && target == FT::NHWC) {
        pattern = {0, 2, 3, 1};
    } else {
        mgb_throw(
                MegBrainError, "Unsupport format conversion from %s to %s",
                m_format.to_string().c_str(), Format(target).to_string().c_str());
    }
    auto output = imperative::apply(
            *Dimshuffle::make(pattern, scope), std::vector<ValueRef>{m_value})[0];
    return FormattedTensorValue::make(output, target);
}

namespace {

ValueRef unwrap_input(const ValueRef& input) {
    if (auto format_input = input.as_ref<FormattedTensorValue>()) {
        return format_input->value();
    } else {
        return input;
    }
}

std::vector<ValueRef> unwrap_inputs(const Span<ValueRef>& inputs) {
    std::vector<ValueRef> unwrapped_inputs;
    for (auto&& input : inputs) {
        unwrapped_inputs.push_back(unwrap_input(input));
    }
    return unwrapped_inputs;
}

std::vector<ValueRef> wrap_outputs(
        const std::vector<ValueRef>& outputs, FT type = FT::DEFAULT) {
    std::vector<ValueRef> wrapped_outputs;
    for (auto&& output : outputs) {
        wrapped_outputs.push_back(FormattedTensorValue::make(output, type));
    }
    return wrapped_outputs;
}

ValueShape convert_nhwc2nchw_shape(const ValueShape& shape) {
    mgb_assert(shape.ndim == 4);
    auto out = ValueShape(shape);
    out[3] = shape[2];
    out[2] = shape[1];
    out[1] = shape[3];
    return out;
}

using FormatRule = std::function<std::vector<ValueRef>(
        const OpDef&, Span<ValueRef>&, const bool&)>;
static std::unordered_map<Typeinfo*, FormatRule> format_rules;

template <typename T>
void register_format_rule(
        std::vector<ValueRef> (*rule)(const T&, Span<ValueRef>&, const bool&)) {
    format_rules[T::typeinfo()] = [rule](const OpDef& def, Span<ValueRef>& inputs,
                                         const bool& auto_convert) {
        return (*rule)(def.cast_final_safe<T>(), inputs, auto_convert);
    };
}

auto convert_nchw2nhwc_pattern(const std::vector<int32_t>& pattern) {
    mgb_assert(pattern.size() == 4);
    auto nhwc_pattern = pattern;
    for (size_t idx = 0; idx < 4; ++idx) {
        auto dim = pattern[idx];
        if (dim == 1) {
            nhwc_pattern[idx] = 3;
        } else if (dim == 2) {
            nhwc_pattern[idx] = 1;
        } else if (dim == 3) {
            nhwc_pattern[idx] = 2;
        }
    }
    return nhwc_pattern;
}

std::vector<ValueRef> dimshuffle_rule(
        const Dimshuffle& op, Span<ValueRef>& inputs, const bool& auto_convert) {
    mgb_assert(inputs.size() == 1);
    auto& src = inputs[0].cast<FormattedTensorValue>();
    // Only support converting pattern from NCHW to NHWC currently.
    if (auto_convert && src.format() == FT::NHWC) {
        auto pattern = convert_nchw2nhwc_pattern(op.pattern);
        // dimshuffle will not maintain NHWC Format
        return wrap_outputs(imperative::apply(
                *Dimshuffle::make(std::move(pattern), op.scope()),
                unwrap_inputs(inputs)));
    }
    return wrap_outputs(imperative::apply(op, unwrap_inputs(inputs)));
}

ValueRef convert_nchw2nhwc_tensornd(const HostTensorND& shape) {
    mgb_assert(shape.layout().total_nr_elems() == 4);
    auto* old_ptr = shape.ptr<dt_int32>();
    auto cn = shape.comp_node();
    auto layout = shape.layout();
    auto nhwc_shape = HostTensorND(cn, layout);
    auto* new_ptr = nhwc_shape.ptr<dt_int32>();
    new_ptr[0] = old_ptr[0];
    new_ptr[1] = old_ptr[2];
    new_ptr[2] = old_ptr[3];
    new_ptr[3] = old_ptr[1];
    auto hv = HostStorage::make(nhwc_shape.storage());
    auto nhwc_shape_input =
            imperative::apply(CreateTensor(CreateTensor::Const, cn, layout), hv)[0];
    return nhwc_shape_input;
}

std::vector<ValueRef> reshape_rule(
        const Reshape& op, Span<ValueRef>& inputs, const bool& auto_convert) {
    mgb_assert(inputs.size() == 2);
    auto& src = inputs[0].cast<FormattedTensorValue>();
    if (auto_convert && src.format() == FT::NHWC) {
        auto shape = unwrap_input(inputs[1]).numpy().cast<HostValue>().as_nd();
        if (shape.layout().total_nr_elems() == 4) {
            // output is still NHWC format
            auto nhwc_shape = convert_nchw2nhwc_tensornd(shape);
            auto outputs = imperative::apply(
                    op, std::vector<ValueRef>{unwrap_input(inputs[0]), nhwc_shape});
            return wrap_outputs(outputs, FT::NHWC);
        } else {
            // will not maintain src's format
            auto nchw_src = src.to(FT::NCHW, op.scope())->value();
            auto outputs = imperative::apply(
                    op, std::vector<ValueRef>{nchw_src, unwrap_input(inputs[1])});
            return wrap_outputs(outputs);
        }
    }
    return wrap_outputs(imperative::apply(op, unwrap_inputs(inputs)));
}

std::vector<ValueRef> broadcast_rule(
        const Broadcast& op, Span<ValueRef>& inputs, const bool& auto_convert) {
    mgb_assert(inputs.size() == 2);
    auto& src = inputs[0].cast<FormattedTensorValue>();
    if (auto_convert && src.format() == FT::NHWC) {
        auto shape = unwrap_input(inputs[1]).numpy().cast<HostValue>().as_nd();
        if (shape.layout().total_nr_elems() == 4) {
            // output is still NHWC format
            auto nhwc_shape = convert_nchw2nhwc_tensornd(shape);
            auto outputs = imperative::apply(
                    op, std::vector<ValueRef>{unwrap_input(inputs[0]), nhwc_shape});
            return wrap_outputs(outputs, FT::NHWC);
        } else {
            // will not maintain src's format
            auto nchw_src = src.to(FT::NCHW, op.scope())->value();
            auto outputs = imperative::apply(
                    op, std::vector<ValueRef>{nchw_src, unwrap_input(inputs[1])});
            return wrap_outputs(outputs);
        }
    }
    return wrap_outputs(imperative::apply(op, unwrap_inputs(inputs)));
}

bool is_reduce_ndim_idx_items(
        const std::vector<std::tuple<int8_t, bool, bool, bool, bool>>& items,
        const Span<ValueRef>& inputs) {
    for (auto i = 0; i < items.size(); ++i) {
        auto&& [axis, begin, end, step, idx] = items[i];
        if (idx) {
            // if inputs[i] contains more than one value, ndim will not be reduced.
            return inputs[i].is_scalar();
        }
    }
    return false;
}

auto convert_nchw2nhwc_idx_items(
        const std::vector<std::tuple<int8_t, bool, bool, bool, bool>>& items) {
    auto nhwc_items = items;
    for (auto i = 0; i < nhwc_items.size(); ++i) {
        auto&& [axis, begin, end, step, idx] = nhwc_items[i];
        if (axis == 2 || axis == 3) {
            nhwc_items[i] = {axis - 1, begin, end, step, idx};
        } else if (axis == 1) {
            nhwc_items[i] = {3, begin, end, step, idx};
        }
    }
    return nhwc_items;
}

template <typename T>
std::vector<ValueRef> subtensor_rule(
        const T& op, Span<ValueRef>& inputs, const bool& auto_convert) {
    mgb_assert(inputs.size() >= 1);
    auto& src = inputs[0].cast<FormattedTensorValue>();
    bool is_reduce_ndim = is_reduce_ndim_idx_items(
            op.items, {&inputs[1], &inputs[inputs.size() - 1]});
    if (!is_reduce_ndim) {
        // only support NHWC2NCHW convert, otherwise maintain src's format
        if (!(auto_convert && src.format() == FT::NHWC)) {
            return {FormattedTensorValue::make(
                    imperative::apply(op, unwrap_inputs(inputs))[0], src.format())};
        }
        auto nhwc_items = convert_nchw2nhwc_idx_items(op.items);
        auto outputs = imperative::apply(
                *T::make(std::move(nhwc_items), op.scope()), unwrap_inputs(inputs));
        return wrap_outputs(outputs, FT::NHWC);
    }
    return wrap_outputs(imperative::apply(op, unwrap_inputs(inputs)));
}

template <typename T>
std::vector<ValueRef> setsubtensor_rule(
        const T& op, Span<ValueRef>& inputs, const bool& auto_convert) {
    mgb_assert(inputs.size() >= 2);
    auto& src = inputs[0].cast<FormattedTensorValue>();
    bool is_reduce_ndim = is_reduce_ndim_idx_items(
            op.items, {&inputs[2], &inputs[inputs.size() - 1]});
    if (!is_reduce_ndim) {
        // only support NHWC2NCHW convert, otherwise maintain src's format
        if (!(auto_convert && src.format() == FT::NHWC)) {
            return {FormattedTensorValue::make(
                    imperative::apply(op, unwrap_inputs(inputs))[0], src.format())};
        }
        // value has been broadcasted to src's fake NCHW shape.
        auto& value = inputs[1].cast<FormattedTensorValue>();
        auto& format = value.format();
        auto nhwc_inputs = std::vector<ValueRef>(inputs.size());
        if (format == FT::DEFAULT || format == FT::NCHW) {
            // value for setsubtensor should transpose to match shape.
            auto nhwc_value = value.as(FT::NCHW)->to(FT::NHWC);
            // make new inputs for setsubtensor
            nhwc_inputs[0] = src.value();
            nhwc_inputs[1] = nhwc_value->value();
            for (auto i = 2; i < inputs.size(); ++i) {
                nhwc_inputs[i] = inputs[i].as_ref<FormattedTensorValue>()->value();
            }
        } else if (format != FT::NHWC) {
            mgb_throw(
                    MegBrainError, "Unsupported format(%s) of value for setsubtensor.",
                    format.to_string().c_str());
        }
        auto nhwc_items = convert_nchw2nhwc_idx_items(op.items);
        auto outputs = imperative::apply(
                *T::make(std::move(nhwc_items), op.scope()), nhwc_inputs);
        return wrap_outputs(outputs, FT::NHWC);
    }
    return wrap_outputs(imperative::apply(op, unwrap_inputs(inputs)));
}

FT get_inputs_format(Span<ValueRef>& inputs) {
    FT format(FT::DEFAULT);
    for (auto& inp : inputs) {
        auto& inp_format = inp.cast<FormattedTensorValue>().format();
        if (inp_format != FT::DEFAULT) {
            mgb_assert(format == FT::DEFAULT || inp_format == format);
            format = inp_format.type();
        }
    }
    return format;
}

std::vector<ValueRef> concat_rule(
        const Concat& op, Span<ValueRef>& inputs, const bool& auto_convert) {
    FT format = get_inputs_format(inputs);
    if (!(format == FT::NHWC && auto_convert)) {
        return wrap_outputs(imperative::apply(op, unwrap_inputs(inputs)), format);
    }
    // TODO: handle 5D NHWC Tensor from group conv
    auto axis = op.axis;
    if (axis == 2 || axis == 3) {
        axis = axis - 1;
    } else if (axis == 1) {
        axis = 3;
    }
    return wrap_outputs(
            imperative::apply(
                    *Concat::make(axis, op.comp_node, op.scope()),
                    unwrap_inputs(inputs)),
            format);
}

std::vector<ValueRef> elemwise_rule(
        const Elemwise& op, Span<ValueRef>& inputs, const bool& auto_convert) {
    FT format = get_inputs_format(inputs);
    return wrap_outputs(imperative::apply(op, unwrap_inputs(inputs)), format);
}

std::vector<ValueRef> identity_rule_helper(
        const OpDef& op, const Span<ValueRef>& inputs) {
    // mgb_assert(inputs.size() == 1);
    auto& src = inputs[0].cast<FormattedTensorValue>();
    return wrap_outputs(
            imperative::apply(op, unwrap_inputs(inputs)), src.format().type());
}

// clang-format off
#define FOREACH_IDENTITY_OP(cb) \
    cb(Copy)                    \
    cb(FastpathCopy)            \
    cb(TypeCvt)                 \
    cb(Pooling)                 \
    cb(AdaptivePooling)         \
    cb(Dropout)                 \
    cb(Convolution)             \
    cb(BatchNorm)               \
    cb(Resize)                  \
    cb(Identity)
// clang-format on

#define CREATE_IDENTITY_OP_RULE(op)                                            \
    std::vector<ValueRef> op##_rule(                                           \
            const op& _op, Span<ValueRef>& inputs, const bool& auto_convert) { \
        return identity_rule_helper(_op, inputs);                              \
    }
FOREACH_IDENTITY_OP(CREATE_IDENTITY_OP_RULE)
#undef CREATE_IDENTITY_OP_RULE

#define REGISTER_IDENTITY_OP_RULE(op) register_format_rule(op##_rule);
struct FormatRuleRegistry {
    FormatRuleRegistry() {
        register_format_rule(dimshuffle_rule);
        register_format_rule(reshape_rule);
        register_format_rule(broadcast_rule);
        register_format_rule(subtensor_rule<Subtensor>);
        register_format_rule(subtensor_rule<IndexingMultiAxisVec>);
        register_format_rule(setsubtensor_rule<SetSubtensor>);
        register_format_rule(setsubtensor_rule<IndexingSetMultiAxisVec>);
        register_format_rule(concat_rule);
        register_format_rule(elemwise_rule);
        FOREACH_IDENTITY_OP(REGISTER_IDENTITY_OP_RULE)
    }
} _;
#undef REGISTER_IDENTITY_OP_RULE
}  // namespace

std::vector<ValueRef> FormatTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    if (auto* apply_op = op.as<ApplyOp>()) {
        // all inputs should be FormattedTensorValue
        auto iter = format_rules.find(apply_op->op().dyn_typeinfo());
        if (iter != format_rules.end()) {
            return iter->second(apply_op->op(), inputs, m_auto_convert);
        } else {
            return wrap_outputs(imperative::apply(op, unwrap_inputs(inputs)));
        }
    } else if (auto* create_tensor = op.as<CreateTensor>()) {
        auto format = create_tensor->format();
        return {FormattedTensorValue::make(imperative::apply(op, inputs)[0], format)};
    } else if (auto* get_attr = op.as<GetAttr>()) {
        auto* src = inputs.as_array<1>()[0].as<FormattedTensorValue>();
        if (!m_auto_convert || !src || src->format() != FT::NHWC) {
            return imperative::apply(op, unwrap_inputs(inputs));
        }
        switch (get_attr->attr()) {
            case GetAttr::Shape: {
                auto output = imperative::apply(op, unwrap_inputs(inputs))[0];
                auto shape = convert_nhwc2nchw_shape(output.cast<ShapeValue>());
                return {ShapeValue::make(shape)};
            }
            case GetAttr::Value: {
                auto nchw_src = unwrap_input(src->to(FT::NCHW, ""));
                return imperative::apply(op, std::vector<ValueRef>{nchw_src});
            }
            default:
                return imperative::apply(op, unwrap_inputs(inputs));
        }
    } else if (op.is<GetFormat>()) {
        bool is_formatted_tensor = inputs.as_array<1>()[0].is<FormattedTensorValue>();
        if (is_formatted_tensor) {
            return {FormatValue::make(inputs[0].cast<FormattedTensorValue>().format())};
        } else {
            mgb_log_warn(
                    "Not FormattedTensorValue input for GetFormat op: %s",
                    inputs[0].to_string().c_str());
            return {FormatValue::make(FT::DEFAULT)};
        }
    } else if (op.is<Operator::IdentityLike>()) {
        bool is_formatted_tensor = inputs.as_array<1>()[0].is<FormattedTensorValue>();
        if (is_formatted_tensor) {
            auto& format = inputs[0].cast<FormattedTensorValue>().format();
            return wrap_outputs(
                    imperative::apply(op, unwrap_inputs(inputs)), format.type());
        } else {
            mgb_log_warn(
                    "Not FormattedTensorValue input for IdentityLike op: %s",
                    inputs[0].to_string().c_str());
            return imperative::apply(op, inputs);
        }
    } else {
        return imperative::apply(op, unwrap_inputs(inputs));
    }
};

}  // namespace imperative
}  // namespace mgb
