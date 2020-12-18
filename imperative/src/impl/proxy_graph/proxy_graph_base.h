#include "megbrain/graph/cg.h"

namespace mgb::imperative::proxy_graph {

using cg::VarNode;

struct ExecEnvBase : cg::GraphExecutable::ExecEnv {
    void dispatch_on_comp_node(CompNode, Task&& task) override {
        task();
    }

    void dispatch_on_comp_node_with_mask(CompNode, Task&&, cg::ExecutionMask*) override {mgb_assert(0);}
    void pause_exec() override {mgb_assert(0);}
    void resume_exec() override {mgb_assert(0);}
};

struct StaticInferManagerBase : cg::static_infer::StaticInferManager {
protected:
    void register_shape_infer(VarNode*, const cg::static_infer::ShapeInferDesc&) override {mgb_assert(0);};
    void register_value_infer(VarNode*, const cg::static_infer::ValueInferDesc&) override {mgb_assert(0);};
    cg::static_infer::InferType get_infer_type(VarNode*) override {mgb_assert(0);};
    const TensorShape& infer_shape(VarNode*) override {mgb_assert(0);}
    const TensorShape* infer_shape_fallible(VarNode*) override {mgb_assert(0);}
    const DeviceTensorND& infer_value(VarNode*) override {mgb_assert(0);}
    const DeviceTensorND* infer_value_fallible(VarNode*) override {mgb_assert(0);}
    cg::static_infer::DepVal get_rt_static_source_deps(const cg::static_infer::DepElement&) override {mgb_assert(0);}
};

struct SeqCompNodeOptimizerBase : cg::SeqCompNodeOptimizer {
protected:
    void register_stream_var(VarNode*, StreamPropType) override {}
    void register_propagate_function(VarNode*, PropFunction) override {}
    StreamPropType stream_prop_type(VarNode*) override {mgb_assert(0);}
};

struct ProxyGraphBase : cg::ComputingGraph {
private:
    VarReceiverInfo m_var_receiver_info;
    SeqCompNodeOptimizerBase m_seq_comp_node_optimizer;
    StaticInferManagerBase m_static_infer_manager;

protected:
    MemPool<VarNode> m_var_node_pool;

    ProxyGraphBase() {
        options().imperative_proxy_graph = true;
        options().no_force_inplace = true;
        options().log_level = 0;
        m_var_receiver_info.dev_value = 1;
        m_var_receiver_info.allow_empty_value = 1;
    }

    void* alloc_varnode_storage() override {
        return m_var_node_pool.alloc_raw();
    }

    void free_varnode_storage(void* ptr) override {
        m_var_node_pool.free_raw(ptr);
    }

    const VarReceiverInfo& var_receiver_in_current_comp_seq(const VarNode *var) const override {
        return m_var_receiver_info;
    }

    cg::static_infer::StaticInferManager& static_infer_manager() override {
        return m_static_infer_manager;
    }

    cg::SeqCompNodeOptimizer& seq_comp_node_optimizer() override {
        return m_seq_comp_node_optimizer;
    }

    std::shared_ptr<void> on_comp_node_finalize() override {
        return {};
    }

    std::unique_ptr<cg::AsyncExecutable> compile(const OutputSpec&) override {mgb_assert(0);}
    SmallVector<std::unique_ptr<cg::AsyncExecutable>> compile_multi_part(const SmallVector<OutputSpec>&) override {mgb_assert(0);}
    cg::AsyncExecutable* current_comp_seq() override {mgb_assert(0);}
    std::string get_mem_allocation_info() const override {mgb_assert(0);}
    VarNode* find_var_by_id(size_t) const override {mgb_assert(0);}
    void share_device_memory_with(ComputingGraph&) override {mgb_assert(0);}
    void set_device_memory_allocator(std::shared_ptr<cg::DeviceMemoryAllocator>) override {mgb_assert(0);}
    size_t get_device_memory_size(CompNode) override {mgb_assert(0);}
    size_t clear_device_memory() override {mgb_assert(0);}
    void set_as_subgraph(ComputingGraph&) override {mgb_assert(0);}
    void record_async_error(std::unique_ptr<MegBrainError>) override {mgb_assert(0);}
};

MGB_DEFINE_OPR_CLASS(
        ProxyGraph::InputPlaceholder,
        cg::OperatorNodeBase) // {

    void on_output_comp_node_stream_changed() override {mgb_assert(0);}
    void init_output_comp_node() override {}
    void init_output_format() override {}
    void init_output_dtype() override {}
    void init_output_static_infer_desc() override {}
    void init_output_mem_plan(bool) override {mgb_assert(0);}
    void do_execute(ExecEnv&) override {mgb_assert(0);}

public:
    InputPlaceholder(cg::ComputingGraph& graph)
            : Super(&graph, {}, "placeholder", {}) {
        add_output(None)->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC);
        // never dedup
        add_equivalence_component<ScalarHash<void*>>(this);
    }

    InputPlaceholder(cg::ComputingGraph& graph, DType dtype, CompNode cn)
            : InputPlaceholder(graph) {
        output(0)->dtype(dtype).comp_node(cn);
    }
};

using InputPlaceholder = ProxyGraph::InputPlaceholder;

} // namespace mgb::imperative::proxy_graph
