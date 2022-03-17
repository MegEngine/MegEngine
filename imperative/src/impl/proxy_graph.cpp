/**
 * \file imperative/src/impl/proxy_graph.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./proxy_graph.h"
#include "./blob_manager_impl.h"
#include "megbrain/graph.h"
#include "megbrain/graph/operator_node.h"
#include "megbrain/graph/static_infer.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/subgraph_detail.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

#if __cplusplus >= 201703L
#include <optional>
#endif

namespace mgb {
namespace imperative {

using cg::OperatorNodeBase;

template <bool p, typename T, typename F>
constexpr auto&& select(T&& t, F&& f) {
    if constexpr (p) {
        return std::forward<T>(t);
    } else {
        return std::forward<F>(f);
    }
}

MGB_DEFINE_OPR_CLASS(ProxyGraph::InputPlaceholder, cg::OperatorNodeBase) // {
    void on_output_comp_node_stream_changed() override { mgb_assert(0); }
    // TODO: consider implement following initialization method,
    // so InputPlaceholder can be initialized correctly during
    // operator insertion
    void init_output_comp_node() override {}
    void init_output_format() override {}
    void init_output_dtype() override {}
    void init_output_static_infer_desc() override {}
    void init_output_mem_plan(bool dynamic) override {
        MGB_MARK_USED_VAR(dynamic);
        mgb_assert(0);
    }
    void do_execute(ExecEnv& env) override { mgb_assert(0); }

public:
    Tensor* m_tensor;

    InputPlaceholder(
            ComputingGraph& graph, Tensor* tensor = nullptr,
            const DeviceTensorND& static_infer_value = {})
            : Super(&graph, {}, "device_value", {}),
              m_tensor(tensor),
              m_static_infer_value(static_infer_value) {
        mgb_assert(
                m_static_infer_value.empty() ||
                m_static_infer_value.comp_node() == CompNode::default_cpu());
        add_output(None)->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC);
        // never dedup
        add_equivalence_component<ScalarHash<void*>>(this);
    }

    static SymbolVar make(ComputingGraph& graph, Tensor& tensor) {
        auto opr = graph.insert_opr(std::make_unique<InputPlaceholder>(graph, &tensor));
        auto var = opr->output(0);
        auto&& dev_tensor = tensor.dev_tensor(false);
        var->m_comp_node = dev_tensor.comp_node();
        var->m_shape = dev_tensor.shape();
        if (dev_tensor.empty()) {
            auto layout = dev_tensor.layout();
            layout.init_contiguous_stride();
            dev_tensor.reset(dev_tensor.storage(), layout);
        }
        var->force_assign_dev_tensor_from_tensor(dev_tensor);
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
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ProxyGraph::InputPlaceholder);

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

    void register_shape_infer(Tag dest, const ShapeInferDesc& desc) override {
        auto i = locate_output(dest);
        mgb_assert(!shape_descs[i]);
        shape_descs[i].emplace(desc);
    }

    void register_value_infer(Tag dest, const ValueInferDesc& desc) override {
        auto i = locate_output(dest);
        mgb_assert(!value_descs[i]);
        value_descs[i].emplace(desc);
    }

    InferType get_infer_type(Tag var) override {
        // don't let opr apply any immediate optimization
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

    template <bool is_shape>
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
        auto push_shape = [&args](const TensorShape* shape) {
            args.val.emplace_back();
            args.val.back().m_shape = shape;
        };
        auto push_value = [&args](const DeviceTensorND* value) {
            args.val.emplace_back();
            args.val.back().m_value = value;
        };

        for (auto&& dep : desc->deps) {
            if (auto opr = dep.dest->owner_opr()
                                   ->template try_cast_final<InputPlaceholder>()) {
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
                mgb_log_warn(
                        "something is missing for shape inference of %s",
                        cur_opr->dyn_typeinfo()->name);
                return nullptr;
            }
            return &result.shape;
        } else {
            if (!desc->infer_func(result.value, args)) {
                mgb_log_warn(
                        "something is missing for value inference of %s",
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

    DepVal get_rt_static_source_deps(const DepElement&) override { mgb_assert(0); }
};

class ProxyGraph::SeqCompNodeOptimizer : public cg::SeqCompNodeOptimizer {
    void register_stream_var(VarNode*, StreamPropType) override {}
    void register_propagate_function(VarNode*, PropFunction) override {}
    StreamPropType stream_prop_type(VarNode*) override { mgb_assert(0); }
};

class ProxyGraph::ProxyGraphImpl : public cg::ComputingGraph {
    static std::atomic<size_t> m_node_id;
    ProxyGraph* m_owner;
    MemPool<VarNode> m_var_node_pool;
    std::vector<std::unique_ptr<OperatorNodeBase>> m_opr_refkeeper;
    std::mutex m_opr_refkeeper_mtx;
    CompNode::UnorderedSet m_used_comp_node;
    VarReceiverInfo m_var_receiver_info;

public:
    ~ProxyGraphImpl() {
        mgb_assert(!m_owner->m_cur_opr);
        if (is_finalized())
            return;
        for (auto&& i : m_used_comp_node) {
            if (i.device_type() == CompNode::DeviceType::CUDA)
                continue;
            if (i.device_type() == CompNode::DeviceType::ROCM)
                continue;
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

    void add_used_comp_node(CompNode cn) { m_used_comp_node.insert(cn); }

    bool invalid() const {
        return is_finalized() || nr_oprs_in_graph() > m_owner->m_max_op_cnt;
    }

    size_t next_node_id() override { return m_node_id.fetch_add(1); }

    void* alloc_varnode_storage() override { return m_var_node_pool.alloc_raw(); }

    void free_varnode_storage(void* ptr) override { m_var_node_pool.free_raw(ptr); }

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
        MGB_LOCK_GUARD(m_opr_refkeeper_mtx);
        mgb_assert(!m_owner->m_cur_opr);
        // finalize would do sync first
        m_opr_refkeeper.clear();
        return {};
    }

    const VarReceiverInfo& var_receiver_in_current_comp_seq(
            const VarNode* var) const override {
        return m_var_receiver_info;
    }

    size_t nr_oprs_in_graph() const override { return m_opr_refkeeper.size(); }

    void record_async_error(std::unique_ptr<MegBrainError> async_exc) override {
        if (!ProxyGraph::tm_async_error) {
            std::swap(async_exc, tm_async_error);
        }
    }

    std::unique_ptr<cg::AsyncExecutable> compile(const OutputSpec& out_spec) override {
        mgb_assert(0);
    }
    SmallVector<std::unique_ptr<cg::AsyncExecutable>> compile_multi_part(
            const SmallVector<OutputSpec>& out_specs) override {
        mgb_assert(0);
    }
    cg::AsyncExecutable* current_comp_seq() override { mgb_assert(0); }
    std::string get_mem_allocation_info() const override { mgb_assert(0); }
    VarNode* find_var_by_id(size_t id) const override { mgb_assert(0); }
    void share_device_memory_with(ComputingGraph& other) override { mgb_assert(0); }
    void set_device_memory_allocator(
            std::shared_ptr<cg::DeviceMemoryAllocator> allocator) override {
        mgb_assert(0);
    }
    size_t get_device_memory_size(CompNode cn) override { mgb_assert(0); }
    size_t clear_device_memory() override { mgb_assert(0); }
    void set_as_subgraph(ComputingGraph& par_graph) override { mgb_assert(0); }
};

std::atomic<size_t> ProxyGraph::ProxyGraphImpl::m_node_id = 0;

ProxyGraph::ProxyGraph()
        : m_graph(ProxyGraphImpl::make(this)),
          m_static_infer_manager(new StaticInferManager(this)),
          m_seq_comp_node_optimizer(new SeqCompNodeOptimizer()) {}

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
    ~CurOprGuard() { m_owner->cleanup(); }

private:
    ProxyGraph* m_owner;
};

#define CUR_OPR_GUARD(opr) \
    CurOprGuard MGB_TOKENPASTE2(__cur_opr_guard_, __LINE__)(this, opr)

/*********************** Physical Tensor Impl ***********************/

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

/*********************** Logical Tensor Impl ***********************/

EncodedSubgraph ProxyGraph::make_backward_graph(
        const OpDef& opdef, const SmallVector<LogicalTensorDesc>& input_descs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    using op_t = OperatorNodeBase*;
    using var_t = VarNode*;
    using vars_t = VarNodeArray;
    auto inputs = make_input_place_holders(input_descs);
    auto outputs = OpDef::apply_on_var_node(opdef, inputs);
    SmallVector<LogicalTensorDesc> output_descs;
    for (auto&& i : outputs) {
        output_descs.push_back({TensorLayout{i->dtype()}, i->comp_node()});
    }
    GradContext<op_t, var_t> grad_context{[&](VarNode* lhs, VarNode* rhs) -> VarNode* {
        auto add = opr::Elemwise::Mode::ADD;
        return opr::Elemwise::make(VarNodeArray{lhs, rhs}, add).node();
    }};
    cg::DepOprIter iter{[&](OperatorNodeBase* op) {
        grad_context.record_expr(op, op->input(), op->output());
    }};
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& input = inputs[i];
        iter.set_visited(input->owner_opr());
        if (input_requires_grad[i]) {
            grad_context.mark_require_grad(input);
        }
    }
    for (auto&& output : outputs) {
        iter.add(output);
    }
    auto output_grads = make_input_place_holders(output_descs);
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (!output_has_grad[i]) {
            output_grads[i] = nullptr;
        }
    }
    auto compute_input_grads = [&](op_t op, vars_t inputs, vars_t outputs,
                                   vars_t output_grads) {
        auto* gfunc = cg::lookup_grad_func(op->dyn_typeinfo());
        vars_t input_grads(inputs.size(), nullptr);
        bool any_grad = false;
        for (auto&& output_grad : output_grads) {
            if (output_grad) {
                any_grad = true;
            }
        }
        if (!gfunc || !any_grad) {
            return input_grads;
        }
        Maybe<VarNodeArray> grad_results;
        auto&& input_requires_grad = grad_context.get_require_grads(inputs);
        for (size_t i = 0; i < inputs.size(); ++i) {
            VarNode* grad;
            if (grad_results.valid()) {
                grad = grad_results.val()[i];
            } else {
                mgb_assert(gfunc, "could not find grad function");
                auto res = (*gfunc)(op, i, output_grads);
                if (res.from_single()) {
                    grad = res.single();
                } else {
                    grad_results.emplace(res.all(op));
                    grad = grad_results.val()[i];
                }
            }
            if (grad && !grad->owner_opr()->same_type<opr::InvalidGrad>()) {
                if (input_requires_grad[i]) {
                    input_grads[i] = grad;
                }
            }
        }
        return input_grads;
    };
    grad_context.backward(outputs, output_grads, compute_input_grads);
    auto input_grads = grad_context.get_grads(inputs);
    VarNodeArray bgraph_inputs;
    bgraph_inputs.insert(bgraph_inputs.end(), inputs.begin(), inputs.end());
    bgraph_inputs.insert(bgraph_inputs.end(), outputs.begin(), outputs.end());
    bgraph_inputs.insert(bgraph_inputs.end(), output_grads.begin(), output_grads.end());
    auto graph = subgraph_detail::make_from_computing_graph(bgraph_inputs, input_grads);
    return graph;
}

VarNodeArray ProxyGraph::make_input_place_holders(
        const SmallVector<LogicalTensorDesc>& inputs) {
    VarNodeArray vinputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        vinputs[i] = InputPlaceholder::make(*m_graph, inputs[i]).node();
    }
    return vinputs;
}

/*********************** Common Impl ***********************/

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
            cpu_value = opr->owner_graph()->static_infer_manager().infer_value_fallible(
                    opr->output(0));
        }
        mgb_assert(cpu_value);
        mgb_assert(cpu_value->comp_node() == CompNode::default_cpu());
        // default_cpu is synchronous with respect to caller
        hv.proxy_to_default_cpu().copy_from_fixlayout(*cpu_value);
        return Tensor::make(dv, hv);
    } else if (opr->same_type<opr::SharedDeviceTensor>()) {
        return Tensor::make(
                opr->cast_final_safe<opr::SharedDeviceTensor>().get_dev_tensor());
    } else {
        return {};
    }
}

thread_local std::unique_ptr<MegBrainError> ProxyGraph::tm_async_error;

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
