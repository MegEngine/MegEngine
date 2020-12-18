/**
 * \file imperative/src/impl/proxy_graph.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./blob_manager_impl.h"
#include "./proxy_graph.h"
#include "megbrain/graph/static_infer.h"
#include "megbrain/graph/operator_node.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/ops/backward_graph.h"

#if __cplusplus >= 201703L
#include <optional>
#endif

namespace mgb {
namespace imperative {

using cg::OperatorNodeBase;

template<bool p, typename T, typename F>
constexpr auto&& select(T&& t, F&& f) {
    if constexpr (p) {
        return std::forward<T>(t);
    } else {
        return std::forward<F>(f);
    }
}

MGB_DEFINE_OPR_CLASS(
        ProxyGraph::InputPlaceholder,
        cg::OperatorNodeBase) // {

    void on_output_comp_node_stream_changed() override {
        mgb_assert(0);
    }
    // TODO: consider implement following initialization method,
    // so InputPlaceholder can be initialized correctly during
    // operator insertion
    void init_output_comp_node() override {
    }
    void init_output_format() override {
    }
    void init_output_dtype() override {
    }
    void init_output_static_infer_desc() override {
    }
    void init_output_mem_plan(bool dynamic) override {
        MGB_MARK_USED_VAR(dynamic);
        mgb_assert(0);
    }
    void do_execute(ExecEnv &env) override {
        mgb_assert(0);
    }

public:
    Tensor* m_tensor;

    InputPlaceholder(ComputingGraph& graph, Tensor* tensor = nullptr,
                     const DeviceTensorND& static_infer_value = {})
            : Super(&graph, {}, "device_value", {}), m_tensor(tensor),
              m_static_infer_value(static_infer_value) {
        mgb_assert(m_static_infer_value.empty() ||
                   m_static_infer_value.comp_node() == CompNode::default_cpu());
        add_output(None)->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC);
        // never dedup
        add_equivalence_component<ScalarHash<void*>>(this);
    }

    static SymbolVar make(ComputingGraph& graph, Tensor& tensor) {
        auto opr = graph.insert_opr(
            std::make_unique<InputPlaceholder>(graph, &tensor));
        auto var = opr->output(0);
        auto&& dev_tensor = tensor.dev_tensor();
        var->m_comp_node = dev_tensor.comp_node();
        var->m_shape = dev_tensor.shape();
        var->m_dev_tensor = dev_tensor;
        var->reset_dev_tensor_from_tensor(dev_tensor);
        return var;
    }

    static SymbolVar make(ComputingGraph& graph, const LogicalTensorDesc& desc) {
        auto opr = graph.insert_opr(
            std::make_unique<InputPlaceholder>(graph, nullptr, desc.value));
        auto var = opr->output(0);
        var->m_comp_node = desc.comp_node;
        var->m_shape = desc.layout;
        var->m_dev_tensor.reset({}, TensorLayout(desc.layout.dtype));
        return var;
    }

    const DeviceTensorND* get_static_infer_value(bool may_sync) {
        if (!m_static_infer_value.empty()) {
            return &m_static_infer_value;
        }
        if (m_tensor && (may_sync || m_tensor->try_get_value())) {
            auto&& hv = m_tensor->get_value();
            mgb_assert(!hv.empty());
            m_static_infer_value = hv.proxy_to_default_cpu();
            // steal ownership from shared_ptr
            using SP = std::shared_ptr<dt_byte>;
            auto& sp = const_cast<SP&>(m_static_infer_value.storage().raw_storage());
            static auto dummy = std::make_shared<dt_byte>();
            sp = SP(dummy, sp.get());
            return &m_static_infer_value;
        }
        return nullptr;
    }

private:
    DeviceTensorND m_static_infer_value;
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(
        ProxyGraph::InputPlaceholder);

class ProxyGraph::ExecEnv final : public cg::GraphExecutable::ExecEnv {

public:
    void dispatch_on_comp_node(CompNode, Task&& task) override {
        task();
    }

    void dispatch_on_comp_node_with_mask(CompNode, Task&& task,
                                         cg::ExecutionMask* mask) override {
        mgb_throw_if(mask, GraphError,
                     "ExecutionMask not supported in imperative mode");
        task();
    }

    void pause_exec() override {}

    void resume_exec() override {}
};

class ProxyGraph::StaticInferManager : public cg::static_infer::StaticInferManager {
public:
    using Tag = cg::static_infer::Tag;
    using ShapeInferDesc = cg::static_infer::ShapeInferDesc;
    using ValueInferDesc = cg::static_infer::ValueInferDesc;
    using InferType = cg::static_infer::InferType;
    using DepVal = cg::static_infer::DepVal;
    using DepElement = cg::static_infer::DepElement;
    using DepType = cg::static_infer::DepType;
    using InpElement = cg::static_infer::InpElement;

    struct Result {
        TensorShape shape;
        DeviceTensorND value;
    };

    ProxyGraph* owner;
    cg::OperatorNodeBase* cur_opr = nullptr;
    std::vector<std::optional<ShapeInferDesc>> shape_descs;
    std::vector<std::optional<ValueInferDesc>> value_descs;
    std::vector<Result> inferred_outputs;

    StaticInferManager(ProxyGraph* owner_) : owner(owner_) {}

    size_t locate_output(VarNode* var) {
        mgb_assert(cur_opr);
        auto&& output_vars = cur_opr->output();
        mgb_assert(shape_descs.size() == output_vars.size());
        auto&& it = std::find(output_vars.begin(), output_vars.end(), var);
        mgb_assert(it != output_vars.end());
        return it - output_vars.begin();
    }

    void register_shape_infer(Tag dest, const ShapeInferDesc &desc) override {
        auto i = locate_output(dest);
        mgb_assert(!shape_descs[i]);
        shape_descs[i].emplace(desc);
    }

    void register_value_infer(Tag dest, const ValueInferDesc &desc) override {
        auto i = locate_output(dest);
        mgb_assert(!value_descs[i]);
        value_descs[i].emplace(desc);
    }

    InferType get_infer_type(Tag var) override {
        // may be called during get_proxy_opr or make_backward_graph

        // don't let opr apply any immediate optimization
        return {InferType::MISSING_INP, InferType::MISSING_INP};

        if (auto opr = var->owner_opr()->try_cast_final<InputPlaceholder>()) {
            return {var->shape().ndim ? InferType::CONST : InferType::MISSING_INP,
                    opr->m_tensor ? InferType::CONST : InferType::MISSING_INP};
        }
        if (cur_opr) {
            auto&& outputs = cur_opr->output();
            auto&& it = std::find(outputs.begin(), outputs.end(), var);
            if (it != outputs.end()) {
                return {infer_shape_fallible(var) ? InferType::CONST : InferType::MISSING_INP,
                        // value inference could be expensive
                        InferType::MISSING_INP};
            }
        }
        return {InferType::MISSING_INP, InferType::MISSING_INP};
    }

    void update() {
        if (cur_opr != owner->m_cur_opr) {
            clear();
            cur_opr = owner->m_cur_opr;
            if (cur_opr) {
                auto nout = cur_opr->output().size();
                shape_descs.resize(nout);
                value_descs.resize(nout);
                inferred_outputs.resize(nout);
                cur_opr->init_output_static_infer_desc();
            }
        }
    }

    void clear() {
        cur_opr = nullptr;
        shape_descs.clear();
        value_descs.clear();
        inferred_outputs.clear();
    }

    template<bool is_shape>
    auto do_infer(Tag dest, bool may_sync)
            -> const std::conditional_t<is_shape, TensorShape, DeviceTensorND>* {
        // Some infer_func does not use InpVal passed to them, but
        // call infer_* on their inputs instead, so dest could be an input.
        // It is also possible that an opr call infer_* on its inputs before it
        // is inserted
        if (auto opr = dest->owner_opr()->try_cast_final<InputPlaceholder>()) {
            if constexpr (is_shape) {
                auto* shp = &dest->shape();
                return shp->ndim ? shp : nullptr;
            } else {
                return opr->get_static_infer_value(may_sync);
            }
        }

        mgb_assert(cur_opr);
        mgb_assert(cur_opr->output().size() == shape_descs.size());

        // dest must be an output now
        auto i = locate_output(dest);
        auto& result = inferred_outputs[i];
        auto& desc = select<is_shape>(shape_descs[i], value_descs[i]);

        // return if no need to call infer_func
        if constexpr (is_shape) {
            if (result.shape.ndim != 0) {
                return &result.shape;
            }
        } else {
            if (!result.value.empty()) {
                return &result.value;
            }
        }
        if (!desc) {
            return nullptr;
        }

        // fill args for infer_func
        cg::static_infer::InpVal args{1};
        args.val.reserve(desc->deps.size());
        auto push_shape = [&args](const TensorShape* shape) {
            args.val.emplace_back();
            args.val.back().m_shape = shape;
        };
        auto push_value = [&args](const DeviceTensorND* value) {
            args.val.emplace_back();
            args.val.back().m_value = value;
        };

        for (auto&& dep : desc->deps) {
            if (auto opr = dep.dest->owner_opr()->template try_cast_final<InputPlaceholder>()) {
                if (dep.type == DepType::SHAPE) {
                    if (dep.dest->shape().ndim) {
                        push_shape(&dep.dest->shape());
                    } else {
                        return nullptr;
                    }
                } else {
                    if (auto* p = opr->get_static_infer_value(may_sync)) {
                        push_value(p);
                    } else {
                        return nullptr;
                    }
                }
                continue;
            }

            // dep must be an output
            if (dep.type == DepType::SHAPE) {
                if (auto* p = do_infer<true>(dep.dest, may_sync)) {
                    push_shape(p);
                } else {
                    return nullptr;
                }
            } else {
                if (auto* p = do_infer<false>(dep.dest, may_sync)) {
                    push_value(p);
                } else {
                    return nullptr;
                }
            }
        }

        // call infer_func
        if constexpr (is_shape) {
            if (!desc->infer_func(result.shape, args)) {
                mgb_log_warn("something is missing for shape inference of %s",
                             cur_opr->dyn_typeinfo()->name);
                return nullptr;
            }
            return &result.shape;
        } else {
            if (!desc->infer_func(result.value, args)) {
                mgb_log_warn("something is missing for value inference of %s",
                             cur_opr->dyn_typeinfo()->name);
                return nullptr;
            }
            return &result.value;
        }
    }

    const TensorShape& infer_shape(Tag var) override {
        auto* p = do_infer<true>(var, true);
        mgb_assert(p, "failed to infer shape for %s", var->name().c_str());
        return *p;
    }
    const TensorShape* infer_shape_fallible(Tag var) override {
        return do_infer<true>(var, false);
    }
    const DeviceTensorND& infer_value(Tag var) override {
        auto* p = do_infer<false>(var, true);
        mgb_assert(p, "failed to infer value for %s", var->name().c_str());
        return *p;
    }
    const DeviceTensorND* infer_value_fallible(Tag var) override {
        return do_infer<false>(var, false);
    }

    DepVal get_rt_static_source_deps(const DepElement&) override {mgb_assert(0);}
};

class ProxyGraph::SeqCompNodeOptimizer : public cg::SeqCompNodeOptimizer {
    void register_stream_var(VarNode*, StreamPropType) override {}
    void register_propagate_function(VarNode*, PropFunction) override {}
    StreamPropType stream_prop_type(VarNode*) override {mgb_assert(0);}
};

class ProxyGraph::ProxyGraphImpl : public cg::ComputingGraph {
    static std::atomic<size_t> m_node_id;
    ProxyGraph* m_owner;
    MemPool<VarNode> m_var_node_pool;
    std::vector<std::unique_ptr<OperatorNodeBase>> m_opr_refkeeper;
    CompNode::UnorderedSet m_used_comp_node;
    VarReceiverInfo m_var_receiver_info;
public:
    ~ProxyGraphImpl() {
        mgb_assert(!m_owner->m_cur_opr);
        if (is_finalized()) return;
        for (auto&& i : m_used_comp_node) {
            if (i.device_type() == CompNode::DeviceType::CUDA) continue;
            i.sync();
        }
    }

    ProxyGraphImpl(ProxyGraph* owner) : m_owner(owner) {
        options().imperative_proxy_graph = true;
        options().no_force_inplace = true;
        options().log_level = 0;
        m_var_receiver_info.dev_value = 1;
        m_var_receiver_info.allow_empty_value = 1;
    }

    static std::unique_ptr<ProxyGraphImpl> make(ProxyGraph* owner) {
        return std::make_unique<ProxyGraphImpl>(owner);
    }

    void add_used_comp_node(CompNode cn) {
        m_used_comp_node.insert(cn);
    }

    bool invalid() const {
        return is_finalized() || nr_oprs_in_graph() > m_owner->m_max_op_cnt;
    }

    size_t next_node_id() override {
        return m_node_id.fetch_add(1);
    }

    void* alloc_varnode_storage() override {
        return m_var_node_pool.alloc_raw();
    }

    void free_varnode_storage(void* ptr) override {
        m_var_node_pool.free_raw(ptr);
    }

    OperatorNodeBase* insert_opr(std::unique_ptr<OperatorNodeBase> opr_uniqp) override {
        mgb_assert(!is_finalized());
        auto opr = opr_uniqp.get();

        if (!opr->inserted_in_graph()) {
            m_opr_refkeeper.emplace_back(std::move(opr_uniqp));
            opr->set_inserted_in_graph();
            opr->init_output_comp_node();
            opr->init_output_dtype();
            opr->init_output_format();
        }
        return opr;
    }

    cg::static_infer::StaticInferManager& static_infer_manager() override {
        return *m_owner->m_static_infer_manager;
    }

    cg::SeqCompNodeOptimizer& seq_comp_node_optimizer() override {
        return *m_owner->m_seq_comp_node_optimizer;
    }

    std::shared_ptr<void> on_comp_node_finalize() override {
        // FIXME: mutex
        mgb_assert(!m_owner->m_cur_opr);
        // finalize would do sync first
        m_opr_refkeeper.clear();
        return {};
    }

    const VarReceiverInfo& var_receiver_in_current_comp_seq(
            const VarNode *var) const override {
        return m_var_receiver_info;
    }

    size_t nr_oprs_in_graph() const override {return m_opr_refkeeper.size();}

    std::unique_ptr<cg::AsyncExecutable> compile(const OutputSpec &out_spec) override {mgb_assert(0);}
    SmallVector<std::unique_ptr<cg::AsyncExecutable>> compile_multi_part(
            const SmallVector<OutputSpec>& out_specs) override {mgb_assert(0);}
    cg::AsyncExecutable* current_comp_seq() override {mgb_assert(0);}
    std::string get_mem_allocation_info() const override {mgb_assert(0);}
    VarNode* find_var_by_id(size_t id) const override {mgb_assert(0);}
    void share_device_memory_with(ComputingGraph &other) override {mgb_assert(0);}
    void set_device_memory_allocator(
            std::shared_ptr<cg::DeviceMemoryAllocator> allocator) override {mgb_assert(0);}
    size_t get_device_memory_size(CompNode cn) override {mgb_assert(0);}
    size_t clear_device_memory() override {mgb_assert(0);}
    void set_as_subgraph(ComputingGraph &par_graph) override {mgb_assert(0);}
    void record_async_error(std::unique_ptr<MegBrainError> async_exc) override {mgb_assert(0);}
};

std::atomic<size_t> ProxyGraph::ProxyGraphImpl::m_node_id = 0;

ProxyGraph::ProxyGraph() :
        m_graph(ProxyGraphImpl::make(this)),
        m_env{new ExecEnv},
        m_static_infer_manager(new StaticInferManager(this)),
        m_seq_comp_node_optimizer(new SeqCompNodeOptimizer()) {
}

void ProxyGraph::reset() {
    mgb_assert(!m_cur_opr);
    m_graph = ProxyGraphImpl::make(this);
}

ProxyGraph* ProxyGraph::get_default_graph() {
    static thread_local ProxyGraph inst;
    if (inst.m_graph->invalid()) {
        inst.reset();
    }
    return &inst;
}

class ProxyGraph::CurOprGuard {
public:
    CurOprGuard(ProxyGraph* owner, OperatorNodeBase* opr) : m_owner(owner) {
        mgb_assert(!owner->m_cur_opr);
        owner->m_cur_opr = opr;
    }
    CurOprGuard(const CurOprGuard&) = delete;
    ~CurOprGuard() {
        m_owner->cleanup();
    }
private:
    ProxyGraph* m_owner;
};

#define CUR_OPR_GUARD(opr) CurOprGuard MGB_TOKENPASTE2(__cur_opr_guard_, __LINE__)(this, opr)

/*********************** Physical Tensor Impl ***********************/

SmallVector<LogicalTensorDesc> ProxyGraph::infer_output_attrs(
        const OpDef& opdef,
        const SmallVector<Tensor*>& inputs) {
    SmallVector<LogicalTensorDesc> ret;
    CUR_OPR_GUARD(get_proxy_opr(opdef, inputs));
    do_shape_infer(true);
    for (auto&& i: m_cur_opr->usable_output()) {
        mgb_assert(i->dtype().valid() && i->comp_node().valid());
        mgb_assert(i->shape().ndim || i->contain_flag(VarNode::Flag::NO_SYS_MEM_ALLOC));
        ret.push_back({{i->shape(), i->dtype()}, i->comp_node()});
    }
    return ret;
}

void ProxyGraph::invoke_op(const OpDef& opdef,
        const SmallVector<Tensor*>& inputs,
        const SmallVector<Tensor*>& outputs) {
    CUR_OPR_GUARD(get_proxy_opr(opdef, inputs));
    init_output_tensor(outputs);
    for (auto oup : m_cur_opr->output()) {
        m_graph->add_used_comp_node(oup->comp_node());
    }
    m_cur_opr->execute(*m_env);
}

void ProxyGraph::cleanup() {
    if (m_cur_opr) {
        for (auto&& i : m_cur_opr->input()) {
            i->m_dev_tensor.storage({});
        }
        for (auto&& i : m_cur_opr->output()) {
            i->m_dev_tensor.storage({});
        }
        m_static_infer_manager->clear();
    }
    m_cur_opr = nullptr;
}

void ProxyGraph::init_output_tensor(const SmallVector<Tensor*>& outputs) {
    // get proxy opr
    auto proxy = m_cur_opr;

    do_shape_infer(true);

    size_t j = 0;
    for (auto&& var : proxy->output()) {
        auto &&chk = var->m_mem_plan.reset_from_owner_var().chunk();
        if (var->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
            // alloc workspace
            TensorLayout layout{var->shape(), var->dtype(), var->format()};
            DeviceTensorStorage storage;
            storage.comp_node(var->comp_node())
                   .ensure_size(layout.dtype.size(layout.total_nr_elems()));
            var->m_dev_tensor.reset(storage, layout);
        } else {
            mgb_assert(j < outputs.size());
            auto &&tensor = outputs[j];
            auto &&layout = tensor->layout();
            mgb_assert(var->comp_node() == tensor->comp_node() &&
                        var->shape().eq_shape(layout) &&
                        var->dtype() == layout.dtype);
            var->assign_dev_tensor_from_tensor(tensor->dev_tensor());
            ++ j;
        }
        chk.mem_alloc_status.set_from_owner_var();
    }
    mgb_assert(j == outputs.size());

    // Memory forwarding was bypassed in megbrain with graph option
    // imerative_proxy_graph on, here we call mem_plan_fwd_in2out_readonly
    // to initialize some opr(e.g. Subtensor)'s internal state
    // TODO: implement memory forwarding
    proxy->mem_plan_fwd_in2out_readonly();
    {
        // some opr (e.g. Reduce) rely on on_mem_status_changed to set
        // input/output tensor corretly, since we bypass var_node_mem_mgr
        // on_mem_status_changed should be called here
        auto&& cb = proxy->get_opr_event_callback().on_mem_status_changed;
        if (cb.valid()) {
            cb.val()();
        }
    }
}

cg::OperatorNodeBase* ProxyGraph::get_proxy_opr(
        const OpDef& opdef,
        const SmallVector<Tensor*>& inputs) {
    VarNodeArray vinputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++ i) {
        vinputs[i] = InputPlaceholder::make(*m_graph, *inputs[i]).node();
    }
    auto opr = OpDef::apply_on_var_node(opdef, vinputs)[0]->owner_opr();
    mgb_assert(!opr->same_type<InputPlaceholder>());
    for (auto &&i : opr->input()) {
        mgb_assert(i->owner_opr()->same_type<InputPlaceholder>());
    }
    return opr;
}

/*********************** Logical Tensor Impl ***********************/

size_t ProxyGraph::get_opr_output_size(const OpDef& opdef,
        const SmallVector<LogicalTensorDesc>& inputs) {
    return get_proxy_opr(opdef, inputs)->usable_output().size();
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> ProxyGraph::infer_output_attrs_fallible(
        const OpDef& opdef,
        const SmallVector<LogicalTensorDesc>& inputs) {
    auto opr = get_proxy_opr(opdef, inputs);
    CUR_OPR_GUARD(opr);
    SmallVector<LogicalTensorDesc> outputs;
    bool validated = do_shape_infer(false);
    for (auto&& i : opr->usable_output()) {
        outputs.push_back({{i->shape(), i->dtype()}, i->comp_node()});
    }
    bool need_check = opr->same_type<opr::Reshape>();
    return {outputs, validated && !need_check};
}

struct ProxyGraph::GradGraph {
    cg::VarNodeArray inputs;
    cg::VarNodeArray outputs;
    cg::VarNodeArray output_grads;
    cg::VarNode* grad;
};

BackwardGraphResult
ProxyGraph::make_backward_graph(
        const OpDef& opdef,
        const SmallVector<LogicalTensorDesc>& input_descs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    ThinHashMap<VarNode*, size_t> var2idx;
    auto push = [&var2idx, cnt=0](VarNode* var) mutable {
        auto&& ret = var2idx.emplace(var, cnt ++);
        mgb_assert(ret.second, "var %s has been already inserted", var->cname());
        return ret.first->second;
    };
    auto inputs = make_input_place_holders(input_descs);
    auto fwd = OpDef::apply_on_var_node(opdef, inputs)[0]->owner_opr();
    auto&& outputs = fwd->usable_output();
    SmallVector<LogicalTensorDesc> output_descs;
    for (auto&& i : outputs) {
        output_descs.push_back({TensorLayout{i->dtype()}, i->comp_node()});
    }
    auto output_grads = make_input_place_holders(output_descs);
    mgb_assert(output_grads.size() == output_has_grad.size());
    bool any_input_has_grad = false;
    for (size_t i = 0; i < output_grads.size(); ++ i) {
        if (!output_has_grad[i]) {
            output_grads[i] = nullptr;
        } else {
            any_input_has_grad = true;
        }
    }
    if (!any_input_has_grad) {
        return {};
    }
    auto* gfunc = cg::lookup_grad_func(fwd->dyn_typeinfo());

    BackwardGraphResult result;
    auto&& backward = BackwardGraph::make();
    auto&& igraph = backward->cast_final_safe<BackwardGraph>().graph();

    size_t nr_backward_graph_inputs = 0;
    auto gen_expr = [this, &var2idx, &igraph, &push, &fwd,
            &nr_backward_graph_inputs](cg::OperatorNodeBase* op) {
        if (auto t = as_tensor(op)) {
            mgb_assert(op->output().size() == 1);
            igraph.constants.emplace_back(push(op->output(0)), std::move(t));
        } else if (op->same_type<InputPlaceholder>()) {
            ++ nr_backward_graph_inputs;
            push(op->output(0));
        } else {
            std::vector<size_t> inputs, outputs;
            for (auto &&i : op->input()) {
                if (i->owner_opr() == fwd) {
                    if (var2idx.find(i) == var2idx.end()) {
                        ++ nr_backward_graph_inputs;
                        push(i);
                    }
                }
                inputs.push_back(var2idx.at(i));
            }
            for (auto &&i : op->usable_output()) {
                outputs.push_back(push(i));
            }
            igraph.exprs.emplace_back(OpDef::make_from_op_node(op), inputs, outputs);
        }
    };

    // set backward graph outputs
    cg::DepOprIter iter{gen_expr};
    iter.set_visited(fwd);
    result.input_has_grad.resize(inputs.size());

    VarNodeArray output_grads_with_unused_var;
    {
        auto iter = output_grads.begin();
        for (auto&& i : fwd->output()) {
            if (i->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                // the var node with VOLATILE_CONTENT(e.g. workspace
                // or an empty var) would not be considered as a normal
                // output, so its grad is always NULL
                output_grads_with_unused_var.push_back(nullptr);
            } else {
                output_grads_with_unused_var.push_back(*iter);
                ++ iter;
            }
        }
        mgb_assert(iter == output_grads.end());
    }

    Maybe<VarNodeArray> grad_results;
    for (size_t i = 0; i < inputs.size(); ++ i) {
        VarNode* grad;
        if (grad_results.valid()) {
            grad = grad_results.val()[i];
        } else {
            auto res = (*gfunc)(fwd, i, output_grads_with_unused_var);
            if (res.from_single()) {
                grad = res.single();
            } else {
                grad_results.emplace(res.all(fwd));
                grad = grad_results.val()[i];
            }
        }
        if (grad && !grad->owner_opr()->same_type<opr::InvalidGrad>()
            && input_requires_grad[i]) {
            mgb_assert(!grad->owner_opr()->same_type<opr::InvalidGrad>(),
                       "gradient of operator %s w.r.t. input #%lu is "
                       "either not well defined or not implemented",
                       fwd->dyn_typeinfo()->name, i);
            iter.add(grad);
            igraph.outputs.push_back(var2idx.at(grad));
            result.input_has_grad[i] = true;
        } else {
            result.input_has_grad[i] = false;
        }
    }
    if (igraph.outputs.empty()) {
        return {};
    }

    // set backward graph inputs
    igraph.inputs.reserve(nr_backward_graph_inputs);
    result.save_for_backward.reserve(nr_backward_graph_inputs);
    auto write_inputs = [&igraph, &var2idx, &result](const VarNodeArray& vars) {
        for (auto&& i: vars) {
            auto&& iter = var2idx.find(i);
            if (iter != var2idx.end()) {
                igraph.inputs.push_back(iter->second);
                result.save_for_backward.push_back(true);
            } else {
                result.save_for_backward.push_back(false);
            }
        }
    };
    write_inputs(inputs);
    write_inputs(outputs);
    write_inputs(output_grads);
    mgb_assert(igraph.inputs.size() == nr_backward_graph_inputs);

    auto treat_as_single = [](auto&& igraph) {
        if (igraph.exprs.size() != 1)
            return false;
        auto&& expr = igraph.exprs[0];
        auto&& expr_inputs = std::get<1>(expr);
        if (expr_inputs.size() != igraph.inputs.size()) {
            return false;
        }
        for (size_t i = 0; i < expr_inputs.size(); ++ i) {
            if (igraph.inputs[i] != expr_inputs[i]) {
                return false;
            }
        }
        auto&& expr_outputs = std::get<2>(expr);
        if (expr_outputs.size() != igraph.outputs.size()) {
            return false;
        }
        for (size_t i = 0; i < expr_outputs.size(); ++ i) {
            if (igraph.outputs[i] != expr_outputs[i]) {
                return false;
            }
        }
        return true;
    };
    if (treat_as_single(igraph)) {
        result.backward = std::get<0>(igraph.exprs[0]);
    } else {
        result.backward = backward;
    }
    return result;
}

cg::OperatorNodeBase* ProxyGraph::get_proxy_opr(const OpDef& opdef,
        const SmallVector<LogicalTensorDesc>& inputs) {
    mgb_assert(!m_cur_opr);
    auto vinputs = make_input_place_holders(inputs);
    return OpDef::apply_on_var_node(opdef, vinputs)[0]->owner_opr();
}

VarNodeArray ProxyGraph::make_input_place_holders(const SmallVector<LogicalTensorDesc>& inputs) {
    VarNodeArray vinputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++ i) {
        vinputs[i] = InputPlaceholder::make(*m_graph, inputs[i]).node();
    }
    return vinputs;
}

/*********************** Common Impl ***********************/

bool ProxyGraph::do_shape_infer(bool sync_value) {
    m_static_infer_manager->update();

    bool validated = true;
    for (auto* var : m_cur_opr->output()) {
        if (sync_value) {
            var->shape(m_static_infer_manager->infer_shape(var));
        } else if (auto* shape = m_static_infer_manager->infer_shape_fallible(var)) {
                var->shape(*shape);
        } else {
            validated = false;
        }
    }
    return validated;
}

TensorPtr ProxyGraph::as_tensor(cg::OperatorNodeBase* opr, bool share) {
    // TODO : maybe some tensor should copy value from origin opr rather than
    // share the RawStorage
    mgb_assert(share, "can't share memory with opr %s", opr->cname());
    if (opr->same_type<opr::ImmutableTensor>()) {
        auto&& dv = opr->cast_final_safe<opr::ImmutableTensor>().value();
        HostTensorND hv(dv.comp_node(), dv.shape(), dv.dtype());
        const DeviceTensorND* cpu_value;
        // get host value
        if (opr->owner_graph() == m_graph.get()) {
            CUR_OPR_GUARD(opr);
            m_static_infer_manager->update();
            cpu_value = m_static_infer_manager->infer_value_fallible(opr->output(0));
        } else {
            cpu_value = opr->owner_graph()->static_infer_manager().infer_value_fallible(opr->output(0));
        }
        mgb_assert(cpu_value);
        mgb_assert(cpu_value->comp_node() == CompNode::default_cpu());
        // default_cpu is synchronous with respect to caller
        hv.proxy_to_default_cpu().copy_from_fixlayout(*cpu_value);
        return Tensor::make(dv, hv);
    } else if (opr->same_type<opr::SharedDeviceTensor>()) {
        return Tensor::make(opr->cast_final_safe<opr::SharedDeviceTensor>().get_dev_tensor());
    } else {
        return {};
    }
}

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
