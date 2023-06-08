#include "megbrain/imperative/transformations/grad.h"

#include <variant>

#include <range/v3/all.hpp>
#include "megbrain/imperative/graph_cache.h"
#include "megbrain/imperative/profiler.h"
#include "megbrain/imperative/resource_manager.h"

#include <range/v3/all.hpp>
namespace mgb {
namespace imperative {

static std::shared_ptr<OptimizedBackwardGraphResult> make_optimized_backward_graph(
        const OpDef& op, Span<ValueRef> inputs, Span<ValueRef> outputs,
        Span<bool> inputs_require_grad) {
    // hash
    using OptimizedBackwardGraphCache = OpMethResultCache<
            std::shared_ptr<OptimizedBackwardGraphResult>, SmallVector<bool>>;
    thread_local auto& cache =
            *ResourceManager::create_local<OptimizedBackwardGraphCache>();
    OptimizedBackwardGraphCache::key_t cache_key{op.shared_from_this()};
    SmallVector<LogicalTensorDesc>& input_descs = cache_key.inputs;
    cache_key.extra<0>() = inputs_require_grad.copy_into<SmallVector<bool>>();
    input_descs.resize(inputs.size());
    // some overhead, consider simplify LogicalTensorDesc
    for (size_t i = 0; i < inputs.size(); ++i) {
        input_descs[i].layout.dtype = *inputs[i].dtype();
        input_descs[i].comp_node = *inputs[i].device();
    }

    auto iter = cache.find(cache_key);
    if (iter != cache.end()) {
        return iter->second;
    }

    // slow path
    SmallVector<bool> output_has_grad(outputs.size(), true);
    std::shared_ptr<OptimizedBackwardGraphResult> ret;
    auto bg = OpDef::make_backward_graph(
            op, input_descs, std::get<0>(cache_key.extras), output_has_grad);
    if (!bg.graph.empty()) {
        ret = std::make_shared<OptimizedBackwardGraphResult>(bg);
    }
    cache.emplace(cache_key, ret);
    return ret;
}

BackwardGraphWithClosure::BackwardGraphWithClosure(
        std::shared_ptr<OptimizedBackwardGraphResult> backward_graph,
        std::shared_ptr<OpDef> op, Span<ValueRef> inputs, Span<ValueRef> outputs)
        : backward_graph(backward_graph),
          output_mask_offset(inputs.size()),
          grad_mask_offset(inputs.size() + outputs.size()),
          op(op) {
    auto& save_for_backward = backward_graph->save_for_backward;
    mgb_assert(save_for_backward.size() == inputs.size() + 2 * outputs.size());
    size_t count = std::count_if(
            save_for_backward.begin(), save_for_backward.end(), ranges::identity{});
    if (!backward_graph->precomp.empty()) {
        SmallVector<ValueRef> inputs_and_outputs(inputs.size() + outputs.size());
        auto it = inputs_and_outputs.begin();
        for (auto&& input : inputs) {
            *it++ = input;
        }
        for (auto&& output : outputs) {
            *it++ = output;
        }
        auto precomp = imperative::apply(backward_graph->precomp, inputs_and_outputs);
        closure.reserve(precomp.size() + count);
        std::copy(precomp.begin(), precomp.end(), std::back_inserter(closure));
    } else {
        closure.reserve(count);
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (save_for_backward[i]) {
            closure.push_back(inputs[i]);
        }
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (save_for_backward[inputs.size() + i]) {
            closure.push_back(outputs[i]);
        }
    }
    if (outputs.size() > 1) {
        output_descs.reserve(outputs.size());
        for (auto&& output : outputs) {
            auto symbolic_shape = imperative::apply(*GetVarShape::make(), output)[0];
            output_descs.push_back({symbolic_shape, output.dtype(), output.device()});
        }
    }
}
void BackwardGraphWithClosure::operator()(
        Span<ValueRef> grads, std::function<void(size_t, ValueRef)> receiver) {
    ValueRef args[closure.size() + grads.size()];
    size_t nargs = 0;
    for (auto&& value : closure) {
        args[nargs++] = value;
    }
    size_t null_grad = 0;
    size_t valid_grad = 0;
    for (size_t i = 0; i < grads.size(); ++i) {
        if (backward_graph->save_for_backward[grad_mask_offset + i]) {
            if (grads[i]) {
                valid_grad++;
                args[nargs++] = grads[i];
            } else {
                null_grad++;
                nargs++;
            }
        }
    }
    if (valid_grad == 0) {
        return;
    }
    if (null_grad > 0) {
        auto zeros_like = [](const OutputDesc& desc) {
            HostTensorStorage storage(*desc.device);
            storage.ensure_size(desc.dtype->size());
            std::memset(storage.ptr(), 0, desc.dtype->size());
            auto t = imperative::apply(
                    CreateTensor(
                            CreateTensor::Unique, *desc.device, *desc.dtype,
                            ValueShape()),
                    HostStorage::make(storage))[0];
            auto res = imperative::apply(*Broadcast::make(), t, desc.shape)[0];
            return res;
        };
        nargs = closure.size();
        for (size_t i = 0; i < grads.size(); ++i) {
            if (backward_graph->save_for_backward[grad_mask_offset + i]) {
                if (!grads[i]) {
                    args[nargs] = zeros_like(output_descs[i]);
                }
                nargs++;
            }
        }
    }
    auto igrads = imperative::apply(backward_graph->backward, Span(args, nargs));
    auto&& iter = igrads.begin();
    for (auto [i, p] : ranges::views::enumerate(backward_graph->input_has_grad)) {
        if (p) {
            receiver(i, std::move(*iter));
            ++iter;
        }
    }
}

void CustomBackward::operator()(
        Span<ValueRef> grads, std::function<void(size_t, ValueRef)> receiver) {
    size_t nargs = grads.size();
    ValueRef args[nargs];
    for (size_t i = 0; i < nargs; ++i) {
        args[i] = grads[i];
    }
    auto ret = m_backward({args, nargs});
    for (size_t i = 0; i < ret.size(); ++i) {
        if (auto&& t = ret[i]) {
            receiver(i, std::move(t));
        }
    }
}

std::string GradSlot::to_string() const {
    bool has_callback = bool(callback);
    return ssprintf(
            "GradSlot{grad=%s, has_callback=%d}", m_grad.to_string().c_str(),
            (int)has_callback);
}

std::string GradFn::to_string() const {
    return ssprintf("GradFn{dests=%s}", imperative::to_string(m_dests).c_str());
}

std::string GradSlotPtr::to_string() const {
    if (!m_fn) {
        return "<empty>";
    }
    return (*this)->to_string();
}

std::string GradValue::to_string() const {
    return ssprintf(
            "GradValue{key=\"%s\", slot=%s, value=%s}", m_key->name().c_str(),
            m_slot.to_string().c_str(), m_value.to_string().c_str());
}

static std::unordered_map<Typeinfo*, CustomBackward::BackwardRule>&
get_backward_rule_storage() {
    static std::unordered_map<Typeinfo*, CustomBackward::BackwardRule> sl_storage;
    return sl_storage;
}

bool CustomBackward::register_grad_rule(Typeinfo* typeinfo, BackwardRule rule) {
    return get_backward_rule_storage().insert({typeinfo, rule}).second;
}

auto CustomBackward::lookup_grad_rule(Typeinfo* typeinfo) -> BackwardRule {
    auto iter = get_backward_rule_storage().find(typeinfo);
    if (iter == get_backward_rule_storage().end()) {
        return {};
    }
    return iter->second;
}

void GradKey::backward() {
    mgb_assert(m_frozen);
    auto& tape = m_frozen_tape;
    for (std::ptrdiff_t k = tape.size() - 1; k >= 0; --k) {
        auto& [grad_fn, op] = tape[k];
        auto grad_receiver = [&, grad_fn = grad_fn](size_t i, ValueRef grad) {
            auto& dest = grad_fn->m_dests[i];
            if (dest) {
                auto& existing_grad = dest->m_grad;
                if (!existing_grad) {
                    existing_grad = grad;
                } else {
                    existing_grad = imperative::apply(
                            ApplyOp(*Elemwise::make(Elemwise::Mode::ADD)),
                            existing_grad, grad)[0];
                }
            }
        };
        // clang-format off
        std::visit([&, grad_fn = grad_fn, op = op](auto&& backward) {
            using T = std::decay_t<decltype(backward)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                mgb_throw(AssertionError, "invalid backward");
            } else {
                // mgb_assert(grad_fn->m_slots.size() > 0);
                SmallVector<ValueRef> grads (grad_fn->m_slots.size());
                auto iter = grads.begin();
                for (auto&& slot : grad_fn->m_slots) {
                    *iter++ = slot.m_grad;
                }
                std::string name = op ? op->name() + "Backward" : "CustomBackward";
                if (Profiler::is_profiling()) {
                imperative::apply(PushScope(name, ScopeType::BACKWARD), Span<ValueRef>(nullptr, nullptr));
                }
                backward(grads, grad_receiver);
                if (Profiler::is_profiling()) {
                imperative::apply(PopScope(name, ScopeType::BACKWARD), Span<ValueRef>(nullptr, nullptr));
                }
            }
        }, grad_fn->m_backward);
        // clang-format on
        for (auto&& dest : grad_fn->m_dests) {
            if (!dest) {
                continue;
            }
            if (!dest.m_producer_record.next && dest->callback) {
                // I'm the last grad producer, invoke callback
                if (dest->m_grad) {
                    dest->callback(dest->m_grad);
                }
            }
        }
        grad_fn->clear();
    }
    tape.clear();
}

GradValue::ref_t GradKey::attach(
        ValueRef tensor, std::function<void(ValueRef)> callback) {
    // always create a new grad value
    GradSlotPtr grad_slot;
    auto& grad_fn = grad_slot.m_fn;
    grad_fn = LocalPtr<GradFn>::make();
    grad_fn->m_key = shared_from_this();
    grad_fn->m_slots.resize(1);
    grad_fn->m_slots[0].callback = callback;
    grad_slot.m_index = 0;
    if (auto&& grad_value = tensor.as_ref(m_value_type)) {
        grad_fn->m_backward.emplace<IdentityBackward>();
        grad_fn->m_dests.push_back(grad_value->m_slot);
        tensor = grad_value->m_value;
        m_tape.emplace_back(grad_fn, nullptr);
    }
    return m_value_type.make(tensor, shared_from_this(), grad_slot);
}

void GradKey::freeze() {
    mgb_assert(m_frozen_tape.empty() && !m_frozen);
    for (auto&& [grad_fn, op] : m_tape) {
        if (auto valid_grad_fn = grad_fn.lock()) {
            m_frozen_tape.push_back({valid_grad_fn, op});
        }
    }
    m_tape.clear();
    m_frozen = true;
}

ValueRefList GradTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    auto fallback = [&] {
        SmallVector<ValueRef> unwrapped_inputs(inputs.size());
        {
            // overhead
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (auto&& grad_value = as_grad_value(inputs[i])) {
                    unwrapped_inputs[i] = grad_value->m_value;
                } else {
                    unwrapped_inputs[i] = inputs[i];
                }
            }
        }
        return imperative::apply(op, unwrapped_inputs);
    };
    if (op.is<GetAttr>()) {
        // overhead
        if (auto&& grad_value = as_grad_value(inputs.item())) {
            return imperative::apply(op, grad_value->m_value);
        } else {
            return imperative::apply(op, inputs);
        }
    }
    if (m_suppressed) {
        return fallback();
    }
    if (auto* op_val = op.as<ApplyOp>()) {
        size_t nr_require_grad = 0;
        SmallVector<bool> require_grads(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (is_grad_value(inputs[i])) {
                nr_require_grad++;
                require_grads[i] = true;
            } else {
                require_grads[i] = false;
            }
        }
        if (nr_require_grad == 0) {
            return imperative::apply(op, inputs);
        }
        SmallVector<ValueRef> captured_inputs(inputs.size());
        SmallVector<bool> inputs_require_grad(inputs.size());
        // capture value so that trace could assume input as same
        auto capture_value = [](const ValueRef& value) {
            // TODO: fastpath copy shouldn't be an OpDef
            static auto fastpath_copy = FastpathCopy::make();
            return imperative::apply(ApplyOp(*fastpath_copy), value)[0];
        };
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& input = inputs[i];
            if (auto&& grad_value = as_grad_value(input)) {
                captured_inputs[i] = capture_value(grad_value->m_value);
                inputs_require_grad[i] = true;
            } else {
                captured_inputs[i] = capture_value(input);
                inputs_require_grad[i] = false;
            }
        }
        // copy grad_fn->m_backward is expensive
        auto grad_fn = LocalPtr<GradFn>::make();
        auto& backward_storage = grad_fn->m_backward;
        auto outputs = [&] {
            auto backward_rule =
                    CustomBackward::lookup_grad_rule(op_val->op().dyn_typeinfo());
            if (backward_rule) {
                CustomBackward backward;
                auto optional_outputs = backward_rule(
                        op_val->op(), captured_inputs, inputs_require_grad, backward);
                if (optional_outputs) {
                    backward_storage = backward;
                    // backward by rule
                    return *optional_outputs;
                }
            }
            auto outputs = imperative::apply(op, captured_inputs);
            auto backward_graph = make_optimized_backward_graph(
                    op_val->op(), captured_inputs, outputs, inputs_require_grad);
            if (backward_graph) {
                backward_storage = BackwardGraphWithClosure(
                        backward_graph, op_val->op().shared_from_this(),
                        {captured_inputs.begin(), captured_inputs.end()},
                        {outputs.data(), outputs.size()});
                // backward by make_backward_graph
                return outputs;
            } else {
                // no backward
                return outputs;
            }
        }();
        if (std::holds_alternative<std::monostate>(backward_storage)) {
            return outputs;
        }
        grad_fn->m_key = m_key;
        grad_fn->m_slots.resize(outputs.size());
        mgb_assert(!outputs.empty());
        grad_fn->m_dests.reserve(inputs.size());
        // clang-format off
        auto visitor = [&](auto& backward) {
            using T = std::decay_t<decltype(backward)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                mgb_throw(AssertionError, "invalid backward");
            } else {
                // little overhead
                for (size_t i = 0; i < inputs.size(); ++i) {
                    if (backward.input_has_grad(i) && require_grads[i]) {
                        auto& input_grad_slot =
                                inputs[i].cast(m_value_type).slot();
                        grad_fn->m_dests.emplace_back(input_grad_slot);
                        grad_fn->m_dests.back().m_producer_record.insert_after(
                                input_grad_slot->m_producer_head);
                    } else {
                        grad_fn->m_dests.emplace_back();
                    }
                }
                for (size_t i = 0; i < outputs.size(); ++i) {
                    if (backward.output_requires_grad(i)) {
                        // little overhead: Value::make
                        auto grad_value = m_value_type.make(outputs[i], m_key, GradSlotPtr{grad_fn, i});
                        outputs[i] = record_grad(grad_value);
                    }
                }
            }
        };
        // std::visit may be slightly slower than direct if
        std::visit(visitor, backward_storage);
        // clang-format on
        mgb_assert(!grad_fn->m_slots.empty());
        m_key->m_tape.push_back({grad_fn, op_val->op().shared_from_this()});
        return outputs;
    } else if (auto* igc = op.as<InsertGradCallback>()) {
        auto grad_fn = LocalPtr<GradFn>::make();
        auto& backward =
                std::get<CustomBackward>(grad_fn->m_backward = CustomBackward());
        auto id = inputs[0];
        backward.m_backward = [id, callback = igc->callback()](
                                      Span<ValueRef> inputs) -> SmallVector<ValueRef> {
            callback({&id, (size_t)1});
            return {};
        };
        m_key->m_side_effects.push_back(grad_fn);
        m_key->m_tape.push_back({grad_fn, nullptr});
        auto next_id = IntegerValue::make((int)id.cast<IntegerValue>() + 1);
        auto prev_count =
                imperative::apply(InsertGradCallback(igc->callback()), next_id)[0];
        auto count = IntegerValue::make((int)prev_count.cast<IntegerValue>() + 1);
        return {count};
    } else if (op.is<CreateTensor>()) {
        return imperative::apply(op, inputs);
    } else if (auto* attach_grad = op.as<AttachGrad>()) {
        if (!has_key(attach_grad->key())) {
            return fallback();
        } else {
            GenericFunction callback =
                    (GenericFunction&)inputs[1].cast<FunctionValue>();
            auto output =
                    attach_grad->key()->attach(inputs[0], [callback](ValueRef grad) {
                        auto ret = callback({&grad, 1});
                        mgb_assert(ret.empty());
                    });
            return {record_grad(output)};
        }
    } else if (auto* grad_backward = op.as<GradBackward>()) {
        if (!has_key(grad_backward->key())) {
            return fallback();
        }
        size_t nr_grads = inputs.size() / 2;
        mgb_assert(nr_grads * 2 == inputs.size());
        auto values = inputs.sub(0, nr_grads);
        auto grads = inputs.sub(nr_grads, nr_grads);
        make_backward_closure(values)(grads);
        return {};
    } else if (auto* is_attached_to = op.as<IsAttachedTo>()) {
        if (has_key(is_attached_to->key())) {
            if (auto&& grad_value = as_grad_value(inputs[0])) {
                // TODO: assert grad_fn
                return {BoolValue::make(true)};
            }
        }
        return {BoolValue::make(false)};
    } else if (auto* set_grad = op.as<SetGrad>()) {
        // TODO: merge SetGrad and ApplyOp
        auto grad_fn = LocalPtr<GradFn>::make();
        auto& backward =
                std::get<CustomBackward>(grad_fn->m_backward = CustomBackward());
        size_t nr_inputs = set_grad->nr_inputs();
        mgb_assert(inputs.size() > nr_inputs);
        size_t nr_outputs = inputs.size() - nr_inputs;
        Span<ValueRef> inputs_ = {inputs.data(), nr_inputs};
        auto outputs_ = fallback();
        backward.m_input_has_grad.resize(nr_inputs, true);
        backward.m_output_attrs.resize(
                nr_outputs, CustomBackward::OutputAttr{true, true});
        backward.m_backward = [fn = set_grad->grad_fn()](Span<ValueRef> inputs) {
            auto result = fn(inputs);
            return SmallVector<ValueRef>(result.begin(), result.end());
        };
        ValueRefList outputs(nr_outputs);
        grad_fn->m_key = m_key;
        grad_fn->m_slots.resize(nr_outputs);
        grad_fn->m_dests.reserve(nr_inputs);
        for (size_t i = 0; i < nr_inputs; ++i) {
            if (auto&& grad_value = as_grad_value(inputs_[i])) {
                auto& input_grad_slot = grad_value->m_slot;
                grad_fn->m_dests.emplace_back(grad_value->m_slot);
                grad_fn->m_dests.back().m_producer_record.insert_after(
                        input_grad_slot->m_producer_head);
            } else {
                grad_fn->m_dests.emplace_back();
            }
        }
        for (size_t i = 0; i < nr_outputs; ++i) {
            auto& output = outputs_[i];
            auto grad_value = as_grad_value(output);
            if (grad_value) {
                grad_value = m_value_type.make(
                        grad_value->m_value, m_key, GradSlotPtr(grad_fn, i));
            } else {
                grad_value = m_value_type.make(output, m_key, GradSlotPtr(grad_fn, i));
            }
            outputs[i] = record_grad(grad_value);
        }
        m_key->m_tape.push_back({grad_fn, nullptr});
        return outputs;
    } else if (auto* gbc = op.as<GetBackwardColsure>()) {
        if (gbc->key() != m_key) {
            return fallback();
        }
        return {FunctionValue::make(make_backward_closure(inputs))};
    } else if (op.is<DetachGrad>()) {
        if (auto&& grad_value = as_grad_value(inputs[0])) {
            return {grad_value->m_value};
        } else {
            return {inputs[0]};
        }
    } else if (op.is<GetGradKey>()) {
        for (auto&& input : inputs) {
            if (auto&& grad_value = as_grad_value(input)) {
                return {GradKeyValue::make(grad_value->m_key)};
            }
        }
        return imperative::apply(op, inputs);
    } else if (op.is<GetGradSlot>()) {
        mgb_assert(inputs.size() == 1);
        if (auto&& grad_value = as_grad_value(inputs[0])) {
            return {GradSlotValue::make(grad_value->slot())};
        } else {
            return {};
        }
    } else if (op.kind() == Operator::IdentityLike) {
        mgb_assert(inputs.size() == 1);
        if (auto&& grad_value = as_grad_value(inputs[0])) {
            auto output = imperative::apply(op, grad_value->m_value)[0];
            auto grad_output = m_value_type.make(output, m_key, grad_value->slot());
            return {record_grad(grad_output)};
        } else {
            return imperative::apply(op, inputs);
        }
    } else {
        return fallback();
    }
}

GenericFunction GradTransformation::make_backward_closure(Span<ValueRef> ys) {
    // reset GradKey
    auto grad_key = m_key;
    std::vector<GradSlotPtr> y_slots;
    for (auto&& y : ys) {
        if (auto&& grad_value = as_grad_value(y)) {
            y_slots.push_back(grad_value->slot());
        } else {
            y_slots.emplace_back();
        }
    }
    GenericFunction closure = [grad_key, y_slots](Span<ValueRef> dys) -> ValueRefList {
        size_t nr_grads = y_slots.size();
        mgb_assert(dys.size() == nr_grads);
        for (size_t i = 0; i < nr_grads; ++i) {
            if (y_slots[i]) {
                y_slots[i]->m_grad = dys[i];
            }
        }
        grad_key->backward();
        return {};
    };
    grad_key->freeze();
    cleanup();
    return closure;
}

void GradTransformation::on_unregister() noexcept {
    cleanup();
}

void GradTransformation::cleanup() {
    for (auto&& weak_value : m_weak_values) {
        auto grad_value = weak_value.lock();
        if (grad_value) {
            mgb_assert(grad_value->m_key == m_key);
            grad_value.reset(grad_value->m_value);
        }
    }
    m_weak_values.clear();
    m_key = {};
}

void GradTransformation::suppress() {
    m_suppressed++;
}

void GradTransformation::resume() {
    m_suppressed--;
}

}  // namespace imperative
}  // namespace mgb
