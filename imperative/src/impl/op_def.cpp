#include "megbrain/imperative/op_def.h"

#include <sstream>

#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/resource_manager.h"

#include "./op_trait.h"

namespace mgb {
namespace imperative {

std::shared_ptr<OpDef> OpDef::make_from_op_node(cg::OperatorNodeBase* node) {
    OpTrait* trait;
    trait = OpTrait::find_by_typeinfo(node->dyn_typeinfo());
    if (!trait) {
        // TODO: register `make_from_op_node` for each OperatorNode
        // instead of forwarding to OprAttr
        trait = OpTrait::find_by_typeinfo(OprAttr::typeinfo());
    }
    mgb_assert(trait);
    return trait->make_from_op_node(node);
}

DispatchMode OpDef::decide_dispatch_mode(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    return def.trait()->decide_dispatch_mode(def, inputs);
}

SmallVector<TensorPtr> OpDef::apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    return def.trait()->apply_on_physical_tensor(
            def, std::move(inputs), output_descs, validated);
}
void OpDef::apply_on_device_tensornd(
        const OpDef& def, const SmallVector<DeviceTensorND>& inputs,
        SmallVector<DeviceTensorND>* outputs) {
    def.trait()->apply_on_device_tensornd(def, inputs, outputs);
    return;
}

VarNodeArray OpDef::apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    return def.trait()->apply_on_var_node(def, inputs);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> OpDef::infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    return def.trait()->infer_output_attrs_fallible(def, inputs);
}

SmallVector<VarNode::LayoutConstraintCallback> OpDef::get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    return def.trait()->get_input_layout_constraint(def, inputs);
}

EncodedSubgraph OpDef::make_backward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    using BackwardGraphCache =
            OpMethResultCache<EncodedSubgraph, SmallVector<bool>, SmallVector<bool>>;
    thread_local auto& cache = *ResourceManager::create_local<BackwardGraphCache>();
    BackwardGraphCache::key_t cache_key{
            const_cast<OpDef&>(def).shared_from_this(),
            inputs,
            {input_requires_grad, output_has_grad}};
    auto iter = cache.find(cache_key);
    if (iter == cache.end()) {
        iter = cache.insert({cache_key, def.trait()->make_backward_graph(
                                                def, inputs, input_requires_grad,
                                                output_has_grad)})
                       .first;
    }
    return iter->second;
}

std::vector<std::pair<const char*, std::string>> OpDef::props(const OpDef& def) {
    return def.trait()->props(def);
}

EncodedSubgraph OpDef::make_forward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    using ForwardGraphCache =
            OpMethResultCache<EncodedSubgraph, SmallVector<bool>, SmallVector<bool>>;
    thread_local auto& cache = *ResourceManager::create_local<ForwardGraphCache>();
    ForwardGraphCache::key_t cache_key{
            const_cast<OpDef&>(def).shared_from_this(), inputs};
    auto iter = cache.find(cache_key);
    if (iter == cache.end()) {
        iter = cache.insert({cache_key, def.trait()->make_forward_graph(def, inputs)})
                       .first;
    }
    return iter->second;
}

std::string OpDef::to_string() const {
    std::string builder = trait()->name;
    builder += "{";
    for (auto&& [name, value] : props(*this)) {
        builder += name;
        builder += ": ";
        builder += value;
        builder += ",";
    }
    return builder + "}";
}

std::string OpDef::name() const {
    return trait()->name;
}

size_t OpDef::hash() const {
    return trait()->hash(*this);
}

bool OpDef::is_same_st(const Hashable& rhs) const {
    return trait()->is_same_st(*this, static_cast<const OpDef&>(rhs));
}

const OpTrait* OpDef::trait() const {
    if (!m_trait) {
        m_trait = OpTrait::find_by_typeinfo(dyn_typeinfo());
        mgb_throw_if(
                !m_trait, MegBrainError, "can not find op_trait by %s",
                dyn_typeinfo()->name);
    }
    return m_trait;
}

const std::string OpDef::scope() const {
    return m_scope;
}

void OpDef::set_scope(const std::string& scope) {
    m_scope = scope;
}

const std::string OpDef::make_name() const {
    if (m_scope.empty())
        return trait()->make_name(*this);
    return m_scope + "." + trait()->make_name(*this);
}

const std::string OpDef::type_name() const {
    return trait()->name;
}

const std::string OpDef::py_traceback() const {
    return m_py_traceback;
}

void OpDef::set_py_traceback(const std::string& traceback) {
    m_py_traceback = traceback;
}

static thread_local OpDef::allocator_t local_allocator;

void OpDef::set_allocator(allocator_t allocator) {
    mgb_assert(!local_allocator, "allocator has been set before");
    local_allocator = allocator;
}

DeviceTensorStorage::RawStorage OpDef::allocate(CompNode device, size_t size) const {
    return local_allocator(device, size);
}

std::string Subgraph::repr() const {
    std::ostringstream buf;
    buf << "(";
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i > 0)
            buf << ", ";
        buf << "%" << inputs[i];
    }
    buf << ") => {\n";
    auto fmt_const = [](size_t i, const TensorPtr& t) {
        if (t->shape().ndim == 1 && t->shape()[0] == 1) {
            auto&& v = t->get_value();
            if (v.dtype() == dtype::Float32{}) {
                return std::to_string(*v.ptr<dt_float32>());
            } else if (v.dtype() == dtype::Int32{}) {
                return std::to_string(*v.ptr<int32_t>());
            }
        }
        return std::string("%c") + std::to_string(i);
    };
    std::unordered_map<size_t, std::string> const_reps;
    for (auto&& [i, t] : constants) {
        const_reps.emplace(i, fmt_const(i, t));
    }
    for (auto& [op, ins, outs] : exprs) {
        buf << "  ";
        if (outs.size()) {
            for (size_t i = 0; i < outs.size(); ++i) {
                if (i > 0)
                    buf << ", ";
                buf << "%" << outs[i];
            }
            buf << " = ";
        }
        if (auto* p = op->try_cast_final<OprAttr>()) {
            buf << p->type;
        } else {
            buf << op->to_string();
        }
        for (size_t i : ins) {
            buf << " ";
            auto&& it = const_reps.find(i);
            if (it != const_reps.end()) {
                buf << it->second;
            } else {
                buf << "%" << i;
            }
        }
        buf << "\n";
    }
    buf << "  ";
    if (outputs.size()) {
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (i > 0)
                buf << ", ";
            buf << "%" << outputs[i];
        }
    } else {
        buf << "()";
    }
    buf << "\n}\n";
    return buf.str();
}

bool Subgraph::is_single() const {
    if (exprs.size() != 1) {
        return false;
    }
    auto& expr = exprs.at(0);
    return expr.inputs == inputs && expr.outputs == outputs;
}

std::shared_ptr<OpDef> Subgraph::as_single() const {
    if (is_single()) {
        return exprs.at(0).op;
    } else {
        return nullptr;
    }
}

bool Subgraph::operator==(const Subgraph& rhs) const {
    mgb_assert(false, "Not Implemented");
}

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
