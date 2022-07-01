#include "megbrain/graph/operator_node.h"
#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/physical_tensor.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "../blob_manager_impl.h"
#include "./common.h"
#include "./proxy_graph_base.h"

#include <optional>
#include "megbrain/opr/utility.h"
#include "range/v3/all.hpp"

namespace mgb::imperative::proxy_graph {

using cg::OperatorNodeBase;

template <typename C, typename E>
std::pair<bool, size_t> find_index(const C& container, const E& item) {
    auto&& it = std::find(container.begin(), container.end(), item);
    return {it != container.end(), it - container.begin()};
}

template <typename T, typename = void>
class TensorAdaptor;

template <typename T, typename U>
using enable_if_same_upto_cv_t =
        std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<U>>>;

template <typename T>
class TensorAdaptor<T, enable_if_same_upto_cv_t<T, LogicalTensorDesc>> {
    T& wrapped;
    template <typename U>
    using maybe_add_const_t = std::conditional_t<std::is_const_v<T>, const U, U>;

public:
    using type = T;

    TensorAdaptor(T& desc) : wrapped(desc) {}
    TensorAdaptor(T* desc) : wrapped(*desc) {}

    DType dtype() { return wrapped.layout.dtype; }
    CompNode comp_node() { return wrapped.comp_node; }
    maybe_add_const_t<TensorShape>& shape() { return wrapped.layout; }
    bool has_value() { return wrapped.value.shape_valid(); }
    auto& value() { return wrapped.value; }

    auto* operator->() { return &wrapped; }
};

template <typename T>
class TensorAdaptor<T, enable_if_same_upto_cv_t<T, Tensor>> {
    Tensor& wrapped;

public:
    using type = Tensor;

    TensorAdaptor(Tensor& tensor) : wrapped(tensor) {}
    TensorAdaptor(Tensor* tensor) : wrapped(*tensor) {}

    DType dtype() { return wrapped.dtype(); }
    CompNode comp_node() { return wrapped.comp_node(); }
    const TensorShape& shape() { return wrapped.shape(); }

    type* operator->() { return &wrapped; }
};

// deduction guides
template <typename T>
TensorAdaptor(T&) -> TensorAdaptor<T, void>;
template <typename T>
TensorAdaptor(T*) -> TensorAdaptor<T, void>;

inline SmallVector<Tensor*> to_raw_ptr_array(
        const SmallVector<TensorPtr>& inputs, bool ensure_storage = true) {
    SmallVector<Tensor*> ret;
    for (auto&& i : inputs) {
        mgb_assert(i);
        ret.push_back(i.get());
        if (ensure_storage) {
            // apply lazy allocation
            i->blob()->storage();
        }
    }
    return ret;
}

static size_t get_workspace_limit(CompNode cn, size_t old_limit) {
    size_t free = cn.get_free_mem();
    size_t lmt = cn.get_max_block_size_available();
    return std::max(lmt, free);
}

// single opr graph, for static inference and execution
// contains static inference descs
class ProxyGraph::MiniGraph {
protected:
    struct InferDepItem {
        bool is_input : 1;
        size_t idx : 63;
        cg::static_infer::DepType type;
    };

    enum class InferStatus { UNKOWN, READY, FAILED };

    // inference desc and pre-allocated storage for a single var
    template <typename T>
    struct InferData {
        SmallVector<InferDepItem> deps;
        thin_function<bool(T&, const cg::static_infer::InpVal&)> infer_func;

        // pre-allocated infer states
        InferStatus status = InferStatus::UNKOWN;
        cg::static_infer::InpVal inp_val;
        T dest;

        void initialize(
                OperatorNodeBase* opr, const cg::static_infer::DepVal& dep_val,
                const thin_function<bool(T&, const cg::static_infer::InpVal&)>& func) {
            mgb_assert(!infer_func);
            infer_func = func;
            inp_val.val.resize(dep_val.size());

            for (auto&& dep : dep_val) {
                auto [found, i] = find_index(opr->input(), dep.dest);
                if (found) {
                    deps.push_back({true, i, dep.type});
                } else {
                    auto [found, i] = find_index(opr->output(), dep.dest);
                    mgb_assert(found);
                    deps.push_back({false, i, dep.type});
                }
            }
        }

        void reset() {
            status = InferStatus::UNKOWN;
            if constexpr (std::is_same_v<T, TensorShape>) {
                dest.ndim = 0;
            } else {
                static_assert(std::is_same_v<T, DeviceTensorND>);
                dest.storage({});
            }
        }
    };

    struct OutputData {
        InferData<TensorShape> shape_infer;
        InferData<DeviceTensorND> value_infer;
    };

    struct InferSessionBase {
        virtual const TensorShape& infer_shape(VarNode*) { mgb_assert(0); }
        virtual const TensorShape* infer_shape_fallible(VarNode*) { mgb_assert(0); }
        virtual const DeviceTensorND& infer_value(VarNode*) { mgb_assert(0); }
        virtual const DeviceTensorND* infer_value_fallible(VarNode*) { mgb_assert(0); }
    };

    size_t buf_size;
    SmallVector<size_t> hash_buf;

    OperatorNodeBase* m_opr = nullptr;
    SmallVector<std::unique_ptr<OperatorNodeBase>> opr_ref_keeper;

    size_t run_id = 0;
    SmallVector<OutputData> output_data;
    SmallVector<size_t> input_remap;
    SmallVector<size_t> output_remap;

    // pre-allocated buffer for converted inputs
    SmallVector<std::optional<DeviceTensorND>> input_value_storage;

    InferSessionBase* m_sess = nullptr;

    template <typename T>
    struct InputAdaptor {
        T& wrapped;
        SmallVector<std::optional<DeviceTensorND>>& value_storage;

        InputAdaptor(MiniGraph& owner, T& inputs)
                : wrapped(inputs), value_storage(owner.input_value_storage) {}
        ~InputAdaptor() {
            for (auto& i : value_storage) {
                i.reset();
            }
        }

        const TensorShape* shape(size_t i) {
            TensorAdaptor tensor(wrapped[i]);
            auto& shape = tensor.shape();
            return shape.ndim ? &shape : nullptr;
        }

        const DeviceTensorND* value(size_t i, bool sync) {
            TensorAdaptor tensor(wrapped[i]);
            using tensor_t = std::remove_cv_t<typename decltype(tensor)::type>;
            if constexpr (std::is_same_v<tensor_t, Tensor>) {
                auto& storage = value_storage[i];
                if (!storage) {
                    if (sync) {
                        return &storage.emplace(
                                tensor->get_value().proxy_to_default_cpu());
                    } else {
                        if (auto* hv = tensor->try_get_value()) {
                            return &storage.emplace(hv->proxy_to_default_cpu());
                        }
                        return nullptr;
                    }
                }
                return &storage.value();
            } else {
                auto& value = tensor.value();
                return value.shape_valid() ? &value : nullptr;
            }
        }
    };

public:
    template <typename I, typename G>
    MiniGraph(
            G& graph, const OpDef& opdef, const I& inputs, const size_t* hash_buf_,
            const size_t buf_size_)
            : buf_size(buf_size_), input_value_storage(inputs.size()) {
        mgb_assert(!m_opr);
        auto _ = graph.scoped_attach(this);
        cg::VarNodeArray vinputs(inputs.size());
        for (auto&& [i, t] : ranges::views::enumerate(inputs)) {
            auto tensor = TensorAdaptor(t);
            opr_ref_keeper.emplace_back(
                    new InputPlaceholder(graph, tensor.dtype(), tensor.comp_node()));
            vinputs[i] = opr_ref_keeper.back()->output(0);
        }
        auto ovars = OpDef::apply_on_var_node(opdef, vinputs);
        if (!m_opr) {
            // identity
            mgb_assert(vinputs.size() == 1 && ovars.size() == 1);
            mgb_assert(ovars[0] == vinputs[0]);
            auto&& input = vinputs[0];
            ovars[0] = opr::Identity::make(input).node();
        }
        mgb_assert(m_opr);
        output_data.resize(m_opr->output().size());
        for (auto* v : ovars) {
            mgb_assert(v->owner_opr() == m_opr);
        }
        m_opr->init_output_static_infer_desc();

        // fix permuted input: the order of m_opr->input() and vinputs may be
        // different, input_remap keeps the index map of m_opr->input() and vinputs
        for (auto* v : m_opr->input()) {
            auto [found, i] = find_index(vinputs, v);
            mgb_assert(found);
            input_remap.push_back(i);
        }
        auto fix_dep_idx = [&](SmallVector<InferDepItem>& deps) {
            for (auto& dep : deps) {
                if (dep.is_input) {
                    dep.idx = input_remap[dep.idx];
                }
            }
        };
        for (auto& data : output_data) {
            fix_dep_idx(data.shape_infer.deps);
            fix_dep_idx(data.value_infer.deps);
        }

        // fix permuted output
        for (auto* v : ovars) {
            auto [found, i] = find_index(m_opr->output(), v);
            mgb_assert(found);
            output_remap.push_back(i);
        }

        hash_buf.resize(buf_size);
        for (size_t i = 0; i < buf_size; ++i) {
            hash_buf[i] = hash_buf_[i];
        }
    }

    bool is_same_buf(const size_t hash_buf_[], const size_t buf_size_) {
        if (buf_size != buf_size_) {
            return false;
        }
        for (size_t i = 0; i < buf_size; i++) {
            if (hash_buf[i] != hash_buf_[i]) {
                return false;
            }
        }
        return true;
    }

    // methods for containing graph

    OperatorNodeBase* insert_opr(std::unique_ptr<OperatorNodeBase> opr_uniqp) {
        mgb_assert(!m_opr);
        m_opr = opr_uniqp.get();
        mgb_assert(opr_ref_keeper.back()->owner_graph() == m_opr->owner_graph());
        mgb_assert(!m_opr->inserted_in_graph());
        opr_ref_keeper.push_back(std::move(opr_uniqp));
        m_opr->set_inserted_in_graph();
        m_opr->init_output_comp_node();
        m_opr->init_output_dtype();
        return m_opr;
    }

    void init_input_tensor(const SmallVector<Tensor*>& inputs) {
        auto&& opr_inputs = m_opr->input();
        mgb_assert(opr_inputs.size() == inputs.size());
        size_t idx = 0;
        for (auto&& input : opr_inputs) {
            mgb_assert(input->owner_opr()->same_type<InputPlaceholder>());
            auto&& dev_tensor = inputs[input_remap[idx]]->dev_tensor(false);
            auto&& layout = dev_tensor.layout();
            input->shape(dev_tensor.shape());

            input->force_assign_dev_tensor_from_tensor(dev_tensor);

            mgb_assert(input->shape().eq_shape(layout));
            idx++;
        }
    }

    void init_output_tensor(const SmallVector<Tensor*>& outputs) {
        ::mgb::opr::intl::WorkspaceLimitHook::set_impl(
                m_opr->owner_graph(), get_workspace_limit);

        for (auto&& var : m_opr->output()) {
            auto&& chk = var->m_mem_plan.reset_from_owner_var().chunk();
            chk.mem_alloc_status.set_from_owner_var();
        }
        m_opr->mem_plan_fwd_in2out_readonly();
        size_t j = 0;
        for (auto&& var : m_opr->output()) {
            if (var->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                auto comp_node = var->comp_node();
                auto dtype = var->dtype();
                auto&& shape = var->shape();
                size_t size = dtype.size(shape.total_nr_elems());
                mgb_assert(
                        var->format().is_default(), "non default format for workspace");
                auto raw_storage = Blob::make(comp_node, size)->storage();
                DeviceTensorStorage storage;
                storage.reset(comp_node, size, raw_storage);
                var->m_dev_tensor.reset(storage, {shape, dtype});
            } else {
                mgb_assert(j < outputs.size());
                auto&& tensor = outputs[j];
                if (var->m_mem_plan.chunk().owner_var != var) {
                    tensor->assign_from_dev_tensor(
                            var->m_dev_tensor);  // memory forwarding
                } else {
                    var->assign_dev_tensor_from_tensor(tensor->dev_tensor());
                }
                ++j;
            }
        }
        mgb_assert(j == outputs.size());
        {
            // some opr (e.g. Reduce) rely on on_mem_status_changed to set
            // input/output tensor corretly, since we bypass var_node_mem_mgr
            // on_mem_status_changed should be called here
            auto&& cb = m_opr->get_opr_event_callback().on_mem_status_changed;
            if (cb.valid()) {
                cb.val()();
            }
        }
    }

    void execute(
            const SmallVector<Tensor*>& inputs, const SmallVector<Tensor*>& outputs,
            cg::GraphExecutable::ExecEnv& env) {
        init_input_tensor(inputs);
        init_output_tensor(outputs);
        m_opr->execute(env);
        for (auto&& i : m_opr->input()) {
            i->m_dev_tensor.storage({});
        }
        for (auto&& i : m_opr->output()) {
            i->m_dev_tensor.storage({});
        }
    }

    void register_shape_infer(
            VarNode* varnode, const cg::static_infer::ShapeInferDesc& desc) {
        auto [found, i] = find_index(m_opr->output(), varnode);
        mgb_assert(found);
        output_data[i].shape_infer.initialize(m_opr, desc.deps, desc.infer_func);
    }

    void register_value_infer(
            VarNode* varnode, const cg::static_infer::ValueInferDesc& desc) {
        auto [found, i] = find_index(m_opr->output(), varnode);
        mgb_assert(found);
        output_data[i].value_infer.initialize(m_opr, desc.deps, desc.infer_func);
    }

    const TensorShape& infer_shape(VarNode* var) {
        mgb_assert(m_sess);
        return m_sess->infer_shape(var);
    }

    const DeviceTensorND& infer_value(VarNode* var) {
        mgb_assert(m_sess);
        return m_sess->infer_value(var);
    }

    OperatorNodeBase* opr() { return m_opr; }

    // inference routine template for type of input
    template <typename I>
    class InferSession : protected InferSessionBase {
    public:
        MiniGraph& owner;
        SmallVector<OutputData>& output_data;
        InputAdaptor<I> inputs;

        template <typename T>
        const T* infer(InferData<T>& target, bool sync) {
            bool ret;
            if (target.status != InferStatus::UNKOWN) {
                ret = target.status == InferStatus::READY;
            } else {
                ret = target.infer_func && do_infer(target, sync);
                target.status = ret ? InferStatus::READY : InferStatus::FAILED;
            }
            return ret ? &target.dest : nullptr;
        }

        template <typename T>
        bool do_infer(InferData<T>& target, bool sync) {
            for (size_t i = 0; i < target.deps.size(); ++i) {
                target.inp_val.run_id = owner.run_id;
                auto& dep = target.deps[i];
                if (dep.is_input) {
                    if (dep.type == cg::static_infer::DepType::SHAPE) {
                        if (auto* val = inputs.shape(dep.idx)) {
                            target.inp_val.val[i].m_shape = val;
                        } else
                            return false;
                    } else {
                        if (auto* val = inputs.value(dep.idx, sync)) {
                            target.inp_val.val[i].m_value = val;
                        } else
                            return false;
                    }
                } else {
                    if (dep.type == cg::static_infer::DepType::SHAPE) {
                        // using opr->output()->shape when it's available
                        // otherwise infer it
                        if (!owner.m_opr->output(dep.idx)->shape().is_empty()) {
                            target.inp_val.val[i].m_shape =
                                    &owner.m_opr->output(dep.idx)->shape();
                        } else if (
                                auto* val =
                                        infer(output_data[dep.idx].shape_infer, sync)) {
                            target.inp_val.val[i].m_shape = val;
                        } else
                            return false;
                    } else {
                        if (auto* val = infer(output_data[dep.idx].value_infer, sync)) {
                            target.inp_val.val[i].m_value = val;
                        } else
                            return false;
                    }
                }
            }
            return target.infer_func(target.dest, target.inp_val);
        }

        // methods for owner mini graph
        // corresponding methods of containing ComputingGraph will be redirected here

        const TensorShape& infer_shape(VarNode* var) override {
            mgb_assert(owner.m_opr);
            auto [found, i] = find_index(owner.m_opr->input(), var);
            mgb_assert(found);
            i = owner.input_remap[i];
            auto* shape = inputs.shape(i);
            mgb_assert(shape);
            return *shape;
        }

        const DeviceTensorND& infer_value(VarNode* var) override {
            mgb_assert(owner.m_opr);
            auto [found, i] = find_index(owner.m_opr->input(), var);
            mgb_assert(found);
            i = owner.input_remap[i];
            auto* value = inputs.value(i, true);
            mgb_assert(value);
            return *value;
        }

    public:
        InferSession(MiniGraph& mgraph, I& inputs_)
                : owner(mgraph),
                  output_data(mgraph.output_data),
                  inputs(mgraph, inputs_) {
            mgraph.run_id++;
            mgb_assert(!owner.m_sess);
            owner.m_sess = this;
        }
        ~InferSession() {
            owner.m_sess = nullptr;
            for (auto& i : output_data) {
                i.shape_infer.reset();
                i.value_infer.reset();
            }
        }

        const TensorShape* infer_shape(size_t i, bool sync) {
            i = owner.output_remap[i];
            auto* p = infer(output_data[i].shape_infer, sync);
            if (sync)
                mgb_assert(p, "failed to infer shape");
            return p;
        }

        const DeviceTensorND* infer_value(size_t i, bool sync) {
            i = owner.output_remap[i];
            auto* p = infer(output_data[i].value_infer, sync);
            if (sync)
                mgb_assert(p, "failed to infer value");
            return p;
        }
    };

    template <typename T>
    InferSession<T> infer_session(T& inputs) {
        return InferSession(*this, inputs);
    }

    size_t output_size() { return output_remap.size(); }

    VarNode* output_var(size_t i) {
        i = output_remap[i];
        return m_opr->output(i);
    }
};

class CompNodeTracker {
    static constexpr size_t bucket_size = 100;
    static constexpr size_t bucket_count = 10;

    CompNode comp_node;
    std::array<std::unique_ptr<CompNode::Event>, bucket_count> events;

    size_t free_slots = bucket_size;
    size_t head = 0;  // events[head] is not recorded
    size_t tail = 0;  // events[tail] is not finished

    void rotate() {
        while (tail < head && events[tail % bucket_count]->finished()) {
            ++tail;
        }
        auto& ev = events[head % bucket_count];
        if (head == tail + bucket_count) {
            // do not wait if head == tail
            ev->host_wait();
            ++tail;
        }
        ev->record();
        ++head;
        free_slots = bucket_size;
    }

public:
    CompNodeTracker(CompNode cn) : comp_node(cn) {
        for (auto& e : events) {
            e = cn.create_event();
        }
    }

    size_t add_opr() {
        if (!free_slots)
            rotate();
        --free_slots;
        return head;
    }

    size_t progress() { return tail; }
};

class ExecMiniGraph : public ProxyGraph::MiniGraph {
    union BusyListItem {
        size_t finish_time;
        OperatorNodeBase* opr;
    };

    SmallVector<CompNodeTracker*> comp_node_trackers;
    std::deque<BusyListItem> busy_oprs;
    SmallVector<OperatorNodeBase*> idle_oprs;

    OperatorNodeBase* acquire_opr() {
        mgb_assert(!m_opr);
        if (!idle_oprs.empty()) {
            m_opr = idle_oprs.back();
            idle_oprs.pop_back();
            return m_opr;
        }
        mgb_assert(busy_oprs.size() > comp_node_trackers.size());
        bool can_pop = true;
        for (auto [item, tracker] : ranges::views::zip(busy_oprs, comp_node_trackers)) {
            if (item.finish_time >= tracker->progress()) {
                can_pop = false;
                break;
            }
        }
        if (can_pop) {
            for (auto _ : comp_node_trackers) {
                MGB_MARK_USED_VAR(_);
                busy_oprs.pop_front();
            }
            m_opr = busy_oprs.front().opr;
            busy_oprs.pop_front();
            return m_opr;
        }
        mgb_assert(false);
    }

    template <bool in_use>
    void release_opr() {
        if constexpr (in_use) {
            for (auto tracker : comp_node_trackers) {
                tracker->add_opr();
            }
        }
    }
};

class ProxyGraphTypeI : public ProxyGraphBase {
    class StaticInferManager : public StaticInferManagerBase {
        ProxyGraph::MiniGraph* target = nullptr;

        friend class ProxyGraphTypeI;

    public:
        void register_shape_infer(
                VarNode* var, const cg::static_infer::ShapeInferDesc& desc) override {
            mgb_assert(target);
            target->register_shape_infer(var, desc);
        };
        void register_value_infer(
                VarNode* var, const cg::static_infer::ValueInferDesc& desc) override {
            mgb_assert(target);
            target->register_value_infer(var, desc);
        };
        cg::static_infer::InferType get_infer_type(VarNode*) override {
            return {cg::static_infer::InferType::MISSING_INP,
                    cg::static_infer::InferType::MISSING_INP};
        }
        // some poorly written inference func would call infer_{shape,value}
        const TensorShape& infer_shape(VarNode* var) override {
            mgb_assert(target);
            return target->infer_shape(var);
        }
        const DeviceTensorND& infer_value(VarNode* var) override {
            mgb_assert(target);
            return target->infer_value(var);
        }
    };

    ProxyGraph::MiniGraph* target = nullptr;
    StaticInferManager m_static_infer_manager;
    std::unordered_multimap<size_t, ProxyGraph::MiniGraph> m_mini_graph_cache;
    std::mutex m_mini_graph_cache_mtx;
    size_t opr_count = 0;
    ExecEnvBase m_env;
    CompNode::UnorderedSet m_used_comp_node;

    static thread_local std::unique_ptr<ProxyGraphTypeI> sm_instance;

    friend class ProxyGraph::MiniGraph;

    size_t nr_oprs_in_graph() const override { return opr_count; }

    size_t next_node_id() override { return opr_count; }

    void add_used_comp_node(CompNode cn) { m_used_comp_node.insert(cn); }

    std::shared_ptr<void> on_comp_node_finalize() override {
        assert(!target);
        MGB_LOCK_GUARD(m_mini_graph_cache_mtx);
        m_mini_graph_cache.clear();
        return {};
    }

    cg::static_infer::StaticInferManager& static_infer_manager() override {
        return m_static_infer_manager;
    }

    void attach(ProxyGraph::MiniGraph* target_) {
        target = target_;
        m_static_infer_manager.target = target_;
    }

    struct AttachGuard {
        ProxyGraphTypeI* owner = nullptr;
        ProxyGraph::MiniGraph* target = nullptr;

        AttachGuard(
                ProxyGraphTypeI* owner_ = nullptr,
                ProxyGraph::MiniGraph* target_ = nullptr)
                : owner(owner_), target(target_) {}
        AttachGuard(AttachGuard&) = delete;
        AttachGuard& operator=(AttachGuard&) = delete;
        AttachGuard(AttachGuard&& rhs) : owner(rhs.owner), target(rhs.target) {
            rhs.owner = nullptr;
        }
        AttachGuard& operator=(AttachGuard&& rhs) = delete;
        ~AttachGuard() {
            if (owner)
                owner->attach(target);
        }
    };

    [[nodiscard]] AttachGuard scoped_attach(ProxyGraph::MiniGraph* target_) {
        attach(target_);
        return attach_guard();
    }

    [[nodiscard]] AttachGuard attach_guard(ProxyGraph::MiniGraph* target_ = nullptr) {
        return {this, target_};
    }

public:
    ~ProxyGraphTypeI() {
        if (is_finalized()) {
            return;
        }
        for (auto&& i : m_used_comp_node) {
            if (i.device_type() == CompNode::DeviceType::CUDA)
                continue;
            i.sync();
        }
    }

    OperatorNodeBase* insert_opr(std::unique_ptr<OperatorNodeBase> opr_uniqp) override {
        mgb_assert(target);
        return target->insert_opr(std::move(opr_uniqp));
    }

    static ProxyGraphTypeI& inst() {
        if (!sm_instance || sm_instance->is_finalized()) {
            sm_instance.reset(new ProxyGraphTypeI);
        }
        return *sm_instance;
    }

    template <typename T>
    ProxyGraph::MiniGraph& get_cached_minigraph(const OpDef& def, const T& inputs) {
        mgb_assert(!is_finalized());
        size_t buf_size = 2 * inputs.size() + 1;
        size_t buf[buf_size];
        size_t pos = 0;
        buf[pos++] = def.hash();
        for (auto&& inp : inputs) {
            auto tensor = TensorAdaptor(inp);
            buf[pos++] = mgb::hash(tensor.dtype().handle());
            buf[pos++] = mgb::hash(tensor.comp_node());
        }
        mgb_assert(pos == buf_size);
        auto key = XXHash{}.update(buf, buf_size * sizeof(size_t)).digest();
        auto its = m_mini_graph_cache.equal_range(key);
        auto it = its.first;
        for (; it != its.second; ++it) {
            if (it->second.is_same_buf(buf, buf_size)) {
                return it->second;
            }
            mgb_log_warn("hash collision occurs in minigraph cache with key: %lu", key);
        }
        auto&& result = m_mini_graph_cache.emplace(
                std::piecewise_construct, std::make_tuple(key),
                std::forward_as_tuple(
                        *this, def, inputs, static_cast<size_t*>(buf), buf_size));
        mgb_assert(result->first);
        return result->second;
    }

    std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
            const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
        auto& minigraph = get_cached_minigraph(def, inputs);
        auto _ = scoped_attach(&minigraph);
        auto sess = minigraph.infer_session(inputs);
        std::tuple<SmallVector<LogicalTensorDesc>, bool> ret;
        auto& [descs, noerr] = ret;
        for (size_t i = 0; i < minigraph.output_size(); ++i) {
            descs.emplace_back();
            auto& desc = descs.back();
            desc.layout.dtype = minigraph.output_var(i)->dtype();
            desc.layout.format = minigraph.output_var(i)->format();
            desc.comp_node = minigraph.output_var(i)->comp_node();
            if (auto* shape = sess.infer_shape(i, false)) {
                desc.layout.init_contiguous_stride(*shape);
                noerr = true;
            } else {
                noerr = false;
            }
        }
        return ret;
    }

    SmallVector<TensorPtr> apply_on_physical_tensor(
            const OpDef& def, SmallVector<TensorPtr> inputs,
            SmallVector<LogicalTensorDesc>& desc, const bool& validated) {
        auto raw_inputs = to_raw_ptr_array(inputs);
        auto& minigraph = get_cached_minigraph(def, raw_inputs);
        auto _ = scoped_attach(&minigraph);
        auto sess = minigraph.infer_session(raw_inputs);
        ::mgb::opr::intl::WorkspaceLimitHook::set_impl(
                minigraph.opr()->owner_graph(), get_workspace_limit);
        // some output var in minigraph.opr()->output() may not appears in
        // minigraph.opr()->usable_output() bug execution may use the attrs for those
        // output var, so we infer attrs for all outputs, but only return
        // LogicalTensorDesc for minigraph.opr()->usable_output()
        for (size_t i = 0; i < minigraph.opr()->output().size(); ++i) {
            auto* var = minigraph.opr()->output()[i];
            auto* shape = sess.infer(sess.output_data[i].shape_infer, true);
            mgb_assert(shape);
            var->shape(*shape);
        }

        SmallVector<TensorPtr> outputs(minigraph.output_size(), {});
        for (size_t i = 0; i < outputs.size(); i++) {
            auto* ovar = minigraph.output_var(i);
            mgb_assert(ovar->dtype().valid() && ovar->comp_node().valid());
            mgb_assert(
                    ovar->shape().ndim ||
                    ovar->contain_flag(VarNode::Flag::NO_SYS_MEM_ALLOC));
            outputs[i] = Tensor::make(
                    TensorLayout{ovar->shape(), ovar->dtype()}, ovar->comp_node());
        }

        auto raw_outputs = to_raw_ptr_array(outputs, false);
        CompNode::UnorderedSet used_cns;
        for (auto&& out : raw_outputs) {
            auto cn = out->comp_node();
            add_used_comp_node(cn);
            if (used_cns.insert(cn).second) {
                for (auto&& in : inputs) {
                    if (in->comp_node() != cn) {
                        auto e = in->get_ready_event();
                        device_wait_event(cn, in->comp_node(), e);
                    }
                }
            }
        }

        // some opr (e.g. Subtensor) may invoke infer_value during execution,
        // so we need create inference session here
        minigraph.execute(raw_inputs, raw_outputs, m_env);
        for (auto&& cn : used_cns) {
            bool should_record = false;
            for (auto&& in : inputs) {
                if (in->comp_node() != cn) {
                    should_record = true;
                    auto e = record_event(cn);
                    async_release(cn, e, *in);
                }
            }
            if (should_record) {
                record_event(cn, true);
            }
        }

        return outputs;
    }
};

}  // namespace mgb::imperative::proxy_graph
