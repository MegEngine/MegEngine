#include "megbrain/imperative/transformations/format.h"
#include "megbrain/imperative/transformations/grad.h"
#include "megbrain/imperative/transformations/symbol.h"

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/utility.h"

#include "megbrain/imperative/utils/helper.h"

namespace mgb {
namespace imperative {

using FT = Format::Type;

TypedValueRef<FormattedTensorValue> FormatTransformation::as(
        const FormattedTensorValue& tensor, const FT& target) const {
    return m_value_type.make(tensor.value(), target);
}

TypedValueRef<FormattedTensorValue> FormatTransformation::to(
        const FormattedTensorValue& tensor, const FT& target,
        const std::string& scope) const {
    std::vector<int32_t> pattern;
    Format format = tensor.format();
    if (format == target)
        return as(tensor, target);

    auto&& shape = tensor.value().shape().cast<ShapeValue>();
    if (format == FT::NHWC && (target == FT::NCHW || target == FT::DEFAULT)) {
        // FIXME(czh): temporary fast path for group conv 5D weight.
        if (shape.ndim == 5) {
            pattern = {0, 1, 4, 2, 3};
        } else if (shape.ndim == 4) {
            pattern = {0, 3, 1, 2};
        } else {
            mgb_throw(
                    MegBrainError,
                    "Unsupport format conversion for tensor %s(shape=%s) from %s to %s",
                    tensor.to_string().c_str(), shape.to_string().c_str(),
                    format.to_string().c_str(), Format(target).to_string().c_str());
        }
    } else if ((format == FT::NCHW || format == FT::DEFAULT) && target == FT::NHWC) {
        if (shape.ndim == 5) {
            pattern = {0, 1, 3, 4, 2};
        } else if (shape.ndim == 4) {
            pattern = {0, 2, 3, 1};
        } else {
            mgb_throw(
                    MegBrainError,
                    "Unsupport format conversion for tensor %s(shape=%s) from %s to %s",
                    tensor.to_string().c_str(), shape.to_string().c_str(),
                    format.to_string().c_str(), Format(target).to_string().c_str());
        }
    } else {
        mgb_throw(
                MegBrainError,
                "Unsupport format conversion for tensor %s(shape=%s) from %s to %s",
                tensor.to_string().c_str(), shape.to_string().c_str(),
                format.to_string().c_str(), Format(target).to_string().c_str());
    }
    mgb_log_debug(
            "Change tensor %s from %s to %s", tensor.to_string().c_str(),
            format.to_string().c_str(), Format(target).to_string().c_str());
    auto output =
            imperative::apply(*Dimshuffle::make(pattern, scope), {tensor.value()})[0];
    return m_value_type.make(output, target);
}

inline ValueRef FormatTransformation::unwrap_input(const ValueRef& input) const {
    if (auto format_input = input.as_ref(m_value_type)) {
        return format_input->value();
    } else {
        return input;
    }
}

inline ValueRefList FormatTransformation::unwrap_inputs(
        const Span<ValueRef>& inputs) const {
    ValueRefList unwrapped_inputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        unwrapped_inputs[i] = unwrap_input(inputs[i]);
    }
    return unwrapped_inputs;
}

inline ValueRef FormatTransformation::wrap_output(
        const ValueRef& output, Format format) const {
    return m_value_type.make(output, format);
}

inline ValueRefList FormatTransformation::wrap_outputs(
        const ValueRefList& outputs, Format format) const {
    ValueRefList wrapped_outputs(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        wrapped_outputs[i] = wrap_output(outputs[i], format);
    }
    return wrapped_outputs;
}

inline bool FormatTransformation::check_all_format_value(
        const Span<ValueRef>& inputs) const {
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i].as_ref(m_value_type)) {
            return false;
        }
    }
    return true;
}

namespace {

ValueShape convert_nhwc2nchw_shape(const ValueShape& shape) {
    auto out = ValueShape(shape);
    if (shape.ndim == 4) {
        out[1] = shape[3];
        out[2] = shape[1];
        out[3] = shape[2];
        return out;
    } else if (shape.ndim == 5) {
        out[2] = shape[4];
        out[3] = shape[2];
        out[4] = shape[3];
        return out;
    } else {
        mgb_throw(
                MegBrainError, "Unsupported shape ndim %lu in GetAttr(Shape).",
                shape.ndim);
    }
}

std::vector<int32_t> convert_nchw2nhwc_vector(const std::vector<int32_t>& shape) {
    auto out = std::vector<int32_t>(shape);
    if (shape.size() == 4) {
        out[1] = shape[2];
        out[2] = shape[3];
        out[3] = shape[1];
        return out;
    } else if (shape.size() == 5) {
        // GIOHW -> GIHWO
        out[2] = shape[3];
        out[3] = shape[4];
        out[4] = shape[2];
        return out;
    } else {
        mgb_throw(
                MegBrainError,
                "Unsupported shape ndim %lu in convert NCHW shape to NHWC.",
                shape.size());
    }
}

using FormatRule = std::function<ValueRefList(
        const OpDef&, Span<ValueRef>&, const bool&, const FormatTransformation&)>;
static std::unordered_map<Typeinfo*, FormatRule> format_rules;

template <typename T>
void register_format_rule(ValueRefList (*rule)(
        const T&, Span<ValueRef>&, const bool&, const FormatTransformation&)) {
    format_rules[T::typeinfo()] = [rule](const OpDef& def, Span<ValueRef>& inputs,
                                         const bool& auto_convert,
                                         const FormatTransformation& t) {
        return (*rule)(def.cast_final_safe<T>(), inputs, auto_convert, t);
    };
}

inline auto convert_nchw2nhwc_pattern(const std::vector<int32_t>& pattern) {
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

ValueRefList dimshuffle_rule(
        const Dimshuffle& op, Span<ValueRef>& inputs, const bool& auto_convert,
        const FormatTransformation& t) {
    mgb_assert(inputs.size() == 1);
    auto& src = inputs[0].cast(t.value_type());
    // Only support converting pattern from NCHW to NHWC currently.
    if (auto_convert && src.format() == FT::NHWC) {
        auto pattern = convert_nchw2nhwc_pattern(op.pattern);
        // dimshuffle will not maintain NHWC Format
        return t.wrap_outputs(imperative::apply(
                *Dimshuffle::make(std::move(pattern), op.scope()),
                t.unwrap_inputs(inputs)));
    }
    return t.wrap_outputs(imperative::apply(op, t.unwrap_inputs(inputs)));
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

ValueRefList reshape_rule(
        const Reshape& op, Span<ValueRef>& inputs, const bool& auto_convert,
        const FormatTransformation& t) {
    mgb_assert(inputs.size() >= 1);
    auto& src = inputs[0].cast(t.value_type());
    if (auto_convert && src.format() == FT::NHWC) {
        if (inputs.size() == 1) {
            if (op.shape.size() == 4) {
                // output is still NHWC format
                auto nhwc_shape = convert_nchw2nhwc_vector(op.shape);
                auto outputs = imperative::apply(
                        *Reshape::make(op.axis, nhwc_shape),
                        {t.unwrap_input(inputs[0])});
                return t.wrap_outputs(outputs, FT::NHWC);
            } else {
                // will not maintain src's format
                auto nchw_src = t.to(src, FT::DEFAULT, op.scope())->value();
                auto outputs = imperative::apply(op, {nchw_src});
                return t.wrap_outputs(outputs);
            }
        } else if (inputs.size() == 2) {
            auto shape = t.unwrap_input(inputs[1]).numpy()->as_nd();
            if (shape.layout().total_nr_elems() == 4) {
                // output is still NHWC format
                auto nhwc_shape = convert_nchw2nhwc_tensornd(shape);
                auto outputs = imperative::apply(
                        op,
                        SmallVector<ValueRef>{t.unwrap_input(inputs[0]), nhwc_shape});
                return t.wrap_outputs(outputs, FT::NHWC);
            } else {
                // will not maintain src's format
                auto nchw_src = t.to(src, FT::DEFAULT, op.scope())->value();
                auto outputs = imperative::apply(
                        op, SmallVector<ValueRef>{nchw_src, t.unwrap_input(inputs[1])});
                return t.wrap_outputs(outputs);
            }
        }
    }
    return t.wrap_outputs(imperative::apply(op, t.unwrap_inputs(inputs)));
}

ValueRefList broadcast_rule(
        const Broadcast& op, Span<ValueRef>& inputs, const bool& auto_convert,
        const FormatTransformation& t) {
    mgb_assert(inputs.size() >= 1);
    auto& src = inputs[0].cast(t.value_type());
    if (auto_convert && src.format() == FT::NHWC) {
        if (inputs.size() == 1) {
            if (op.shape.size() == 4) {
                // output is still NHWC format
                auto nhwc_shape = convert_nchw2nhwc_vector(op.shape);
                auto outputs = imperative::apply(
                        *Broadcast::make(nhwc_shape), {t.unwrap_input(inputs[0])});
                return t.wrap_outputs(outputs, FT::NHWC);
            } else {
                // will not maintain src's format
                auto nchw_src = t.to(src, FT::DEFAULT, op.scope())->value();
                auto outputs = imperative::apply(op, {nchw_src});
                return t.wrap_outputs(outputs);
            }
        } else if (inputs.size() == 2) {
            auto shape = t.unwrap_input(inputs[1]).numpy()->as_nd();
            if (shape.layout().total_nr_elems() == 4) {
                // output is still NHWC format
                auto nhwc_shape = convert_nchw2nhwc_tensornd(shape);
                auto outputs = imperative::apply(
                        op,
                        SmallVector<ValueRef>{t.unwrap_input(inputs[0]), nhwc_shape});
                return t.wrap_outputs(outputs, FT::NHWC);
            } else {
                // will not maintain src's format
                auto nchw_src = t.to(src, FT::DEFAULT, op.scope())->value();
                auto outputs = imperative::apply(
                        op, SmallVector<ValueRef>{nchw_src, t.unwrap_input(inputs[1])});
                return t.wrap_outputs(outputs);
            }
        }
    }
    return t.wrap_outputs(imperative::apply(op, t.unwrap_inputs(inputs)));
}

inline bool is_reduce_ndim_idx_items(
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

inline bool is_subtensor_reduce_ndim(
        const std::vector<std::tuple<int8_t, bool, bool, bool, bool>>& items,
        const std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t>> slice_items) {
    for (auto i = 0; i < items.size(); ++i) {
        auto&& [axis, begin, end, step, idx] = items[i];
        if (idx) {
            auto&& [b_val, e_val, s_val, ax_val] = slice_items[i];
            return ax_val != INT_MAX;
        }
    }
    return false;
}

inline auto convert_nchw2nhwc_idx_items(
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
ValueRefList subtensor_rule(
        const T& op, Span<ValueRef>& inputs, const bool& auto_convert,
        const FormatTransformation& t) {
    mgb_assert(inputs.size() >= 1);
    auto& src = inputs[0].cast(t.value_type());
    bool is_reduce_ndim = false;
    if (inputs.size() > 1) {
        is_reduce_ndim = is_reduce_ndim_idx_items(
                op.items, {&inputs[1], &inputs[inputs.size() - 1]});
    } else {
        is_reduce_ndim = is_subtensor_reduce_ndim(op.items, op.slice_items);
    }
    if (!is_reduce_ndim) {
        // only support NHWC2NCHW convert, otherwise maintain src's format
        if (!(auto_convert && src.format() == FT::NHWC)) {
            return {t.wrap_output(
                    imperative::apply(op, t.unwrap_inputs(inputs))[0], src.format())};
        }
        auto nhwc_items = convert_nchw2nhwc_idx_items(op.items);
        auto outputs = imperative::apply(
                *T::make(std::move(nhwc_items), op.slice_items, op.scope()),
                t.unwrap_inputs(inputs));
        return t.wrap_outputs(outputs, FT::NHWC);
    }
    return t.wrap_outputs(imperative::apply(op, t.unwrap_inputs(inputs)));
}

template <typename T>
ValueRefList indexing_rule(
        const T& op, Span<ValueRef>& inputs, const bool& auto_convert,
        const FormatTransformation& t) {
    mgb_assert(inputs.size() >= 1);
    auto& src = inputs[0].cast(t.value_type());
    bool is_reduce_ndim = is_reduce_ndim_idx_items(
            op.items, {&inputs[1], &inputs[inputs.size() - 1]});
    if (!is_reduce_ndim) {
        // only support NHWC2NCHW convert, otherwise maintain src's format
        if (!(auto_convert && src.format() == FT::NHWC)) {
            return {t.wrap_output(
                    imperative::apply(op, t.unwrap_inputs(inputs))[0], src.format())};
        }
        auto nhwc_items = convert_nchw2nhwc_idx_items(op.items);
        auto outputs = imperative::apply(
                *T::make(std::move(nhwc_items), op.scope()), t.unwrap_inputs(inputs));
        return t.wrap_outputs(outputs, FT::NHWC);
    }
    return t.wrap_outputs(imperative::apply(op, t.unwrap_inputs(inputs)));
}

template <typename T>
ValueRefList setsubtensor_rule(
        const T& op, Span<ValueRef>& inputs, const bool& auto_convert,
        const FormatTransformation& t) {
    mgb_assert(inputs.size() >= 2);
    auto& src = inputs[0].cast(t.value_type());
    bool is_reduce_ndim = is_reduce_ndim_idx_items(
            op.items, {&inputs[2], &inputs[inputs.size() - 1]});
    if (!is_reduce_ndim) {
        // only support NHWC2NCHW convert, otherwise maintain src's format
        if (!(auto_convert && src.format() == FT::NHWC)) {
            return {t.wrap_output(
                    imperative::apply(op, t.unwrap_inputs(inputs))[0], src.format())};
        }
        // value has been broadcasted to src's fake NCHW shape.
        auto& value = inputs[1].cast(t.value_type());
        auto& format = value.format();
        auto nhwc_inputs = ValueRefList(inputs.size());
        if (format == FT::DEFAULT || format == FT::NCHW) {
            // value for setsubtensor should transpose to match shape.
            auto nhwc_value = t.to(value, FT::NHWC);
            // make new inputs for setsubtensor
            nhwc_inputs[0] = src.value();
            nhwc_inputs[1] = nhwc_value->value();
            for (auto i = 2; i < inputs.size(); ++i) {
                nhwc_inputs[i] = t.unwrap_input(inputs[i]);
            }
        } else if (format != FT::NHWC) {
            mgb_throw(
                    MegBrainError, "Unsupported format(%s) of value for setsubtensor.",
                    format.to_string().c_str());
        }
        auto nhwc_items = convert_nchw2nhwc_idx_items(op.items);
        auto outputs = imperative::apply(
                *T::make(std::move(nhwc_items), op.scope()), nhwc_inputs);
        return t.wrap_outputs(outputs, FT::NHWC);
    }
    return t.wrap_outputs(imperative::apply(op, t.unwrap_inputs(inputs)));
}

inline FT get_inputs_format(Span<ValueRef>& inputs, const FormatTransformation& t) {
    FT format(FT::DEFAULT);
    for (auto& inp : inputs) {
        auto&& inp_format = inp.cast(t.value_type()).format();
        if (inp_format != FT::DEFAULT) {
            mgb_assert(format == FT::DEFAULT || inp_format == format);
            format = inp_format.type();
        }
    }
    return format;
}

inline ValueRefList unify_inputs_format(
        const Span<ValueRef>& inputs, const FT& dst_fmt, const std::string& scope,
        const FormatTransformation& t) {
    ValueRefList unified_inputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto&& inp = inputs[i].cast(t.value_type());
        if (inp.format() != dst_fmt) {
            unified_inputs[i] = t.to(inp, dst_fmt, scope);
        } else {
            unified_inputs[i] = inputs[i];
        }
    }
    return unified_inputs;
}

ValueRefList elemwise_rule(
        const Elemwise& op, Span<ValueRef>& inputs, const bool& auto_convert,
        const FormatTransformation& t) {
    FT format = get_inputs_format(inputs, t);
    if (format == FT::NHWC && auto_convert) {
        ValueRefList unified_inputs(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto&& inp = inputs[i].cast(t.value_type());
            if (inp.format() != FT::NHWC && inp.value().is_scalar()) {
                unified_inputs[i] = t.value_type().make(inp.value(), FT::NHWC);
            } else {
                unified_inputs[i] = inputs[i];
            }
        }
        unified_inputs = unify_inputs_format(unified_inputs, FT::NHWC, op.scope(), t);
        return t.wrap_outputs(
                imperative::apply(op, t.unwrap_inputs(unified_inputs)), format);
    }
    return t.wrap_outputs(imperative::apply(op, t.unwrap_inputs(inputs)), format);
}

ValueRefList concat_rule(
        const Concat& op, Span<ValueRef>& inputs, const bool& auto_convert,
        const FormatTransformation& t) {
    FT format = get_inputs_format(inputs, t);
    if (!(format == FT::NHWC && auto_convert)) {
        return t.wrap_outputs(imperative::apply(op, t.unwrap_inputs(inputs)), format);
    }
    ValueRefList unified_inputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto&& inp = inputs[i].cast(t.value_type());
        if (inp.format() != FT::NHWC && inp.value().is_scalar()) {
            unified_inputs[i] = t.value_type().make(inp.value(), FT::NHWC);
        } else {
            unified_inputs[i] = inputs[i];
        }
    }
    unified_inputs = unify_inputs_format(unified_inputs, FT::NHWC, op.scope(), t);
    // TODO: handle 5D NHWC Tensor from group conv
    auto axis = op.axis;
    if (axis == 2 || axis == 3) {
        axis = axis - 1;
    } else if (axis == 1) {
        axis = 3;
    }
    return t.wrap_outputs(
            imperative::apply(
                    *Concat::make(axis, op.comp_node, op.scope()),
                    t.unwrap_inputs(unified_inputs)),
            format);
}

ValueRefList identity_rule_helper(
        const OpDef& op, const Span<ValueRef>& inputs, const FormatTransformation& t) {
    // mgb_assert(inputs.size() == 1);
    if (auto& src = inputs[0].as_ref(t.value_type())) {
        return t.wrap_outputs(
                imperative::apply(op, t.unwrap_inputs(inputs)), src->format());
    } else {
        return t.wrap_outputs(imperative::apply(op, t.unwrap_inputs(inputs)));
    }
}

ValueRefList batchnorm_rule(
        const BatchNorm& op, Span<ValueRef>& inputs, const bool& auto_convert,
        const FormatTransformation& t) {
    auto&& inp_format = inputs[0].cast(t.value_type()).format();
    if (inp_format == FT::NHWC) {
        auto new_param = op.param();
        new_param.param_dim = BatchNorm::ParamDim::DIM_111C;
        auto new_op = BatchNorm::make(new_param);
        return identity_rule_helper(*new_op, inputs, t);
    }
    return identity_rule_helper(op, inputs, t);
}

ValueRefList adaptive_pooling_rule(
        const AdaptivePooling& op, Span<ValueRef>& inputs, const bool& auto_convert,
        const FormatTransformation& t) {
    auto&& inp_format = inputs[0].cast(t.value_type()).format();
    if (inp_format == FT::NHWC) {
        auto new_param = op.param();
        new_param.format = AdaptivePooling::Format::NHWC;
        auto new_op = AdaptivePooling::make(new_param, op.shape);
        return identity_rule_helper(*new_op, inputs, t);
    }
    return identity_rule_helper(op, inputs, t);
}

// clang-format off
#define FOREACH_MULTI_INPS_NO_PARAM_OP(cb)  \
    cb(CompiledOp)                          \
    cb(SubgraphOp)

#define FOREACH_IDENTITY_OP(cb)             \
    cb(Copy)                                \
    cb(FastpathCopy)                        \
    cb(TypeCvt)                             \
    cb(Dropout)                             \
    cb(FillLike)                            \
    cb(Identity)

#define FOREACH_FORMAT_OP(cb)               \
    cb(WarpAffine)                          \
    cb(Resize)

#define FOREACH_FORMAT_POLICY_OP(cb)        \
    cb(Pooling)                             \
    cb(Convolution)

#define FOREACH_BYPASS_OP(cb)               \
    cb(ParamPackSplit)                      \
    cb(ParamPackConcat)                     \
    cb(CollectiveComm)                      \
    cb(CheckNonFinite)
// clang-format on

// multi inputs op without params
#define CREATE_MULTI_INPS_NO_PARAM_OP_RULE(Op)                               \
    ValueRefList Op##_rule(                                                  \
            const Op& _op, Span<ValueRef>& inputs, const bool& auto_convert, \
            const FormatTransformation& t) {                                 \
        FT format = get_inputs_format(inputs, t);                            \
        return t.wrap_outputs(                                               \
                imperative::apply(_op, t.unwrap_inputs(inputs)), format);    \
    }
FOREACH_MULTI_INPS_NO_PARAM_OP(CREATE_MULTI_INPS_NO_PARAM_OP_RULE)
#undef CREATE_MULTI_INPS_NO_PARAM_OP_RULE

// identity op
#define CREATE_IDENTITY_OP_RULE(Op)                                          \
    ValueRefList Op##_rule(                                                  \
            const Op& _op, Span<ValueRef>& inputs, const bool& auto_convert, \
            const FormatTransformation& t) {                                 \
        return identity_rule_helper(_op, inputs, t);                         \
    }
FOREACH_IDENTITY_OP(CREATE_IDENTITY_OP_RULE)
#undef CREATE_IDENTITY_OP_RULE

// identity op with Format param
#define CREATE_FORMAT_OP_RULE(Op)                                            \
    ValueRefList Op##_rule(                                                  \
            const Op& _op, Span<ValueRef>& inputs, const bool& auto_convert, \
            const FormatTransformation& t) {                                 \
        auto&& inp_format = inputs[0].cast(t.value_type()).format();         \
        if (inp_format == FT::NHWC) {                                        \
            auto new_param = _op.param();                                    \
            new_param.format = Op::Format::NHWC;                             \
            auto new_op = Op::make(new_param);                               \
            return identity_rule_helper(*new_op, inputs, t);                 \
        }                                                                    \
        return identity_rule_helper(_op, inputs, t);                         \
    }
FOREACH_FORMAT_OP(CREATE_FORMAT_OP_RULE)
#undef CREATE_FORMAT_OP_RULE

// identity op with Format and policy param
#define CREATE_FORMAT_POLICY_OP_RULE(Op)                                     \
    ValueRefList Op##_rule(                                                  \
            const Op& _op, Span<ValueRef>& inputs, const bool& auto_convert, \
            const FormatTransformation& t) {                                 \
        auto&& inp_format = inputs[0].cast(t.value_type()).format();         \
        if (inp_format == FT::NHWC) {                                        \
            auto new_param = _op.param();                                    \
            new_param.format = Op::Format::NHWC;                             \
            auto new_op = Op::make(new_param, _op.policy());                 \
            return identity_rule_helper(*new_op, inputs, t);                 \
        }                                                                    \
        return identity_rule_helper(_op, inputs, t);                         \
    }
FOREACH_FORMAT_POLICY_OP(CREATE_FORMAT_POLICY_OP_RULE)

#define CREATE_BYPASS_OP_RULE(Op)                                               \
    ValueRefList Op##_rule(                                                     \
            const Op& _op, Span<ValueRef>& inputs, const bool& auto_convert,    \
            const FormatTransformation& t) {                                    \
        return t.wrap_outputs(imperative::apply(_op, t.unwrap_inputs(inputs))); \
    }
FOREACH_BYPASS_OP(CREATE_BYPASS_OP_RULE)
#undef CREATE_BYPASS_OP_RULE

#undef CREATE_FORMAT_OP_RULE
#define REGISTER_OP_RULE(op) register_format_rule(op##_rule);
struct FormatRuleRegistry {
    FormatRuleRegistry() {
        register_format_rule(dimshuffle_rule);
        register_format_rule(reshape_rule);
        register_format_rule(broadcast_rule);
        register_format_rule(subtensor_rule<Subtensor>);
        register_format_rule(indexing_rule<IndexingMultiAxisVec>);
        register_format_rule(setsubtensor_rule<SetSubtensor>);
        register_format_rule(setsubtensor_rule<IndexingSetMultiAxisVec>);
        register_format_rule(elemwise_rule);
        register_format_rule(concat_rule);
        register_format_rule(batchnorm_rule);
        register_format_rule(adaptive_pooling_rule);
        FOREACH_MULTI_INPS_NO_PARAM_OP(REGISTER_OP_RULE)
        FOREACH_IDENTITY_OP(REGISTER_OP_RULE)
        FOREACH_FORMAT_OP(REGISTER_OP_RULE)
        FOREACH_FORMAT_POLICY_OP(REGISTER_OP_RULE)
        FOREACH_BYPASS_OP(REGISTER_OP_RULE)
    }
} _;
#undef REGISTER_OP_RULE
}  // namespace

ValueRefList FormatTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    if (auto* apply_op = op.as<ApplyOp>()) {
        // bypass SymbolValue
        if (!check_all_format_value(inputs)) {
            return imperative::apply(op, unwrap_inputs(inputs));
        }
        // all inputs should be FormattedTensorValue
        auto iter = format_rules.find(apply_op->op().dyn_typeinfo());
        if (iter != format_rules.end()) {
            return iter->second(apply_op->op(), inputs, m_auto_convert, *this);
        } else {
            auto unified_inputs = unify_inputs_format(
                    inputs, FT::DEFAULT, apply_op->op().scope(), *this);
            return wrap_outputs(imperative::apply(op, unwrap_inputs(unified_inputs)));
        }
    } else if (auto* create_tensor = op.as<CreateTensor>()) {
        auto format = create_tensor->format();
        if (format == FT::NHWC) {
            auto output = wrap_output(imperative::apply(op, inputs)[0]);
            output = to(output.cast(m_value_type), FT::NHWC, "");
            return {output};
        } else {
            return {wrap_output(imperative::apply(op, inputs)[0], format)};
        }
    } else if (auto* get_attr = op.as<GetAttr>()) {
        auto&& input = inputs.item();
        if (!input.is(m_value_type)) {
            return imperative::apply(op, input);
        }
        auto& src = input.cast(m_value_type);
        if (!(m_auto_convert && src.format() == FT::NHWC)) {
            return imperative::apply(op, unwrap_inputs(inputs));
        }
        switch (get_attr->attr()) {
            case GetAttr::Shape: {
                auto output = imperative::apply(op, unwrap_inputs(inputs))[0];
                auto shape = convert_nhwc2nchw_shape(output.cast<ShapeValue>());
                return {ShapeValue::make(shape)};
            }
            case GetAttr::Value: {
                auto nchw_src = unwrap_input(to(src, FT::DEFAULT, ""));
                return imperative::apply(op, {nchw_src});
            }
            default:
                return imperative::apply(op, unwrap_inputs(inputs));
        }
    } else if (op.is<GetFormat>()) {
        auto&& inp_ref = inputs[0].as_ref(m_value_type);
        if (inp_ref) {
            return {FormatValue::make(inp_ref->format())};
        } else {
            MGE_CALL_ONCE(mgb_log_warn(
                    "Not FormattedTensorValue input for GetFormat op: %s, %s",
                    op.to_string().c_str(), inputs[0].to_string().c_str()));
            return {FormatValue::make(FT::DEFAULT)};
        }
    } else if (auto* _op = op.as<SetFormat>()) {
        auto&& inp_ref = inputs[0].as_ref(m_value_type);
        mgb_assert(inp_ref, "Cannot set format for non-format Tensor.");
        return {to(*inp_ref, _op->format().type(), "")};
    } else if (op.is<Operator::IdentityLike>()) {
        auto&& inp_ref = inputs[0].as_ref(m_value_type);
        if (inp_ref) {
            auto&& format = inp_ref->format();
            return wrap_outputs(imperative::apply(op, unwrap_inputs(inputs)), format);
        } else {
            return imperative::apply(op, inputs);
        }
    } else if (op.is<AttachGrad>()) {
        auto&& inp_ref = inputs[0].as_ref(m_value_type);
        if (inp_ref) {
            auto format = inp_ref->format();
            GenericFunction callback =
                    (GenericFunction&)inputs[1].cast<FunctionValue>();
            // make param grads as FormattedTensor
            GenericFunction new_callback =
                    [&, callback, format](Span<ValueRef> inputs_) -> ValueRefList {
                auto wrapped_inputs = SmallVector<ValueRef>{
                        m_value_type.make(inputs_.item(), format)};
                auto ret = callback(wrapped_inputs);
                return ret;
            };
            auto&& outputs = imperative::apply(
                    op, inp_ref->value(), FunctionValue::make(new_callback));
            // make params(GradValue) as FormattedTensor
            return wrap_outputs(outputs, format);
        } else {
            MGE_CALL_ONCE(mgb_log_warn(
                    "Not FormattedTensorValue input for AttachGrad op: %s, %s",
                    op.to_string().c_str(), inputs[0].to_string().c_str()));
            return imperative::apply(op, inputs);
        }
    } else if (auto* set_grad = op.as<SetGrad>()) {
        // make grads in Function backward as FormattedTensor
        size_t nr_inputs = set_grad->nr_inputs();
        size_t nr_outputs = inputs.size() - nr_inputs;
        Span<ValueRef> inputs_ = {inputs.data(), nr_inputs};
        Span<ValueRef> outputs_ = {inputs.data() + nr_inputs, nr_outputs};

        // run original apply.
        // grads needn't to unwrap and wrap, which will be unwrapped in GradTrans
        auto&& outputs = imperative::apply(op, unwrap_inputs(inputs));

        // handle output's formats
        auto wrapped_outputs = ValueRefList(nr_outputs);
        for (size_t i = 0; i < nr_outputs; ++i) {
            if (auto output_ref = outputs_[i].as_ref(m_value_type)) {
                wrapped_outputs[i] =
                        m_value_type.make(outputs[i], output_ref->format());
            } else {
                MGE_CALL_ONCE(mgb_log_warn(
                        "Not FormattedTensorValue outputs for SetGrad op: %s, %s",
                        op.to_string().c_str(), inputs_[i].to_string().c_str()));
                wrapped_outputs[i] = m_value_type.make(outputs[i], FT::DEFAULT);
            }
        }
        return wrapped_outputs;
    } else {
        return imperative::apply(op, unwrap_inputs(inputs));
    }
};

}  // namespace imperative
}  // namespace mgb
