#include "megbrain/graph/operator_node.h"
#include "megbrain/imperative/physical_tensor.h"
#include "megbrain/imperative/op_def.h"

#include "./common.h"
#include "./proxy_graph_base.h"

#include <optional>
#include "range/v3/all.hpp"


namespace mgb::imperative::proxy_graph {

using cg::OperatorNodeBase;


template<typename C, typename E>
std::pair<bool, size_t> find_index(const C& container, const E& item) {
    auto&& it = std::find(container.begin(), container.end(), item);
    return {it != container.end(), it - container.begin()};
}


template <typename T, typename = void> class TensorAdaptor;

template <typename T, typename U>
using enable_if_same_upto_cv_t = std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<U>>>;

template<typename T>
class TensorAdaptor<T, enable_if_same_upto_cv_t<T, LogicalTensorDesc>> {
    T& wrapped;
    template <typename U>
    using maybe_add_const_t = std::conditional_t<std::is_const_v<T>, const U, U>;

public:
    using type = T;

    TensorAdaptor(T& desc) : wrapped(desc) {}
    TensorAdaptor(T* desc) : wrapped(*desc) {}

    DType dtype() {return wrapped.layout.dtype;}
    CompNode comp_node() {return wrapped.comp_node;}
    maybe_add_const_t<TensorShape>& shape() {return wrapped.layout;}
    bool has_value() {return wrapped.value.shape_valid();}
    auto& value() {return wrapped.value;}

    auto* operator->() {return &wrapped;}
};

template<typename T>
class TensorAdaptor<T, enable_if_same_upto_cv_t<T, Tensor>> {
    Tensor& wrapped;

public:
    using type = Tensor;

    TensorAdaptor(Tensor& tensor) : wrapped(tensor) {}
    TensorAdaptor(Tensor* tensor) : wrapped(*tensor) {}

    DType dtype() {return wrapped.dtype();}
    CompNode comp_node() {return wrapped.comp_node();}
    const TensorShape& shape() {return wrapped.shape();}

    type* operator->() {return &wrapped;}
};

// deduction guides
template <typename T> TensorAdaptor(T&) -> TensorAdaptor<T, void>;
template <typename T> TensorAdaptor(T*) -> TensorAdaptor<T, void>;


// single opr graph, for static inference and execution
// contains static inference descs
class ProxyGraph::MiniGraph {
protected:
    struct InferDepItem {
        bool is_input : 1;
        size_t idx : 63;
        cg::static_infer::DepType type;
    };

    enum class InferStatus {
        UNKOWN,
        READY,
        FAILED
    };

    // inference desc and pre-allocated storage for a single var
    template <typename T>
    struct InferData {
        SmallVector<InferDepItem> deps;
        thin_function<bool(T&, const cg::static_infer::InpVal&)> infer_func;

        // pre-allocated infer states
        InferStatus status = InferStatus::UNKOWN;
        cg::static_infer::InpVal inp_val;
        T dest;

        void initialize(OperatorNodeBase* opr, const cg::static_infer::DepVal& dep_val,
                        const thin_function<bool(T&, const cg::static_infer::InpVal&)>& func) {
            mgb_assert(!infer_func);
            infer_func = func;
            inp_val.val.resize(dep_val.size());
            deps.reserve(dep_val.size());

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
        virtual const TensorShape& infer_shape(VarNode*) {mgb_assert(0);}
        virtual const TensorShape* infer_shape_fallible(VarNode*) {mgb_assert(0);}
        virtual const DeviceTensorND& infer_value(VarNode*) {mgb_assert(0);}
        virtual const DeviceTensorND* infer_value_fallible(VarNode*) {mgb_assert(0);}
    };

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

        InputAdaptor(MiniGraph& owner, T& inputs) : wrapped(inputs), value_storage(owner.input_value_storage) {}
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
                        return &storage.emplace(tensor->get_value().proxy_to_default_cpu());
                    } else {
                        if (auto* hv = tensor->try_get_value()) {
                            return &storage.emplace(hv->proxy_to_default_cpu());
                        }
                        return nullptr;
                    }
                }
            } else {
                auto& value = tensor.value();
                return value.shape_valid() ? &value : nullptr;
            }
        }
    };

public:
    template <typename I, typename G>
    MiniGraph(G& graph, const OpDef& opdef, const I& inputs) : input_value_storage(inputs.size()) {
        mgb_assert(!m_opr);
        auto _ = graph.scoped_attach(this);
        cg::VarNodeArray vinputs(inputs.size());
        for (auto&& [i, t] : ranges::views::enumerate(inputs)) {
            auto tensor = TensorAdaptor(t);
            opr_ref_keeper.emplace_back(new InputPlaceholder(graph, tensor.dtype(), tensor.comp_node()));
            vinputs[i] = opr_ref_keeper.back()->output(0);
        }
        auto ovars = OpDef::apply_on_var_node(opdef, vinputs);
        mgb_assert(m_opr);
        output_data.resize(m_opr->output().size());
        for (auto* v : ovars) {
            mgb_assert(v->owner_opr() == m_opr);
        }
        m_opr->init_output_static_infer_desc();

        // fix permuted input
        input_remap.reserve(m_opr->input().size());
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
        output_remap.reserve(ovars.size());
        for (auto* v : ovars) {
            auto [found, i] = find_index(m_opr->output(), v);
            mgb_assert(found);
            output_remap.push_back(i);
        }
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

    void register_shape_infer(VarNode* varnode, const cg::static_infer::ShapeInferDesc& desc) {
        auto [found, i] = find_index(m_opr->output(), varnode);
        mgb_assert(found);
        output_data[i].shape_infer.initialize(m_opr, desc.deps, desc.infer_func);
    }

    void register_value_infer(VarNode* varnode, const cg::static_infer::ValueInferDesc& desc) {
        auto [found, i] = find_index(m_opr->output(), varnode);
        mgb_assert(found);
        output_data[i].value_infer.initialize(m_opr, desc.deps, desc.infer_func);
    }

    const TensorShape& infer_shape(VarNode* var) {
        return m_sess->infer_shape(var);
    }

    const DeviceTensorND& infer_value(VarNode* var) {
        return m_sess->infer_value(var);
    }

    OperatorNodeBase* opr() {
        return m_opr;
    }

    // inference routine template for type of input
    template<typename I>
    class InferSession : protected InferSessionBase {
        MiniGraph& owner;
        SmallVector<OutputData>& output_data;
        InputAdaptor<I> inputs;

        template<typename T>
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

        template<typename T>
        bool do_infer(InferData<T>& target, bool sync) {
            for (size_t i = 0; i < target.deps.size(); ++i) {
                target.inp_val.run_id = owner.run_id;
                auto& dep = target.deps[i];
                if (dep.is_input) {
                    if (dep.type == cg::static_infer::DepType::SHAPE) {
                        if (auto* val = inputs.shape(dep.idx)) {
                            target.inp_val.val[i].m_shape = val;
                        } else return false;
                    } else {
                        if (auto* val = inputs.value(dep.idx, sync)) {
                            target.inp_val.val[i].m_value = val;
                        } else return false;
                    }
                } else {
                    if (dep.type == cg::static_infer::DepType::SHAPE) {
                        if (auto* val = infer(output_data[dep.idx].shape_infer, sync)) {
                            target.inp_val.val[i].m_shape = val;
                        } else return false;
                    } else {
                        if (auto* val = infer(output_data[dep.idx].value_infer, sync)) {
                            target.inp_val.val[i].m_value = val;
                        } else return false;
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
            auto* value = inputs.value(i, false);
            mgb_assert(value);
            return *value;
        }

    public:
        InferSession(MiniGraph& mgraph, I& inputs_)
                : owner(mgraph), output_data(mgraph.output_data), inputs(mgraph, inputs_) {
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
            return infer(output_data[i].shape_infer, sync);
        }

        const DeviceTensorND* infer_value(size_t i, bool sync) {
            i = owner.output_remap[i];
            return infer(output_data[i].shape_infer, sync);
        }
    };

    template <typename T>
    InferSession<T> infer_session(T& inputs) {return InferSession(*this, inputs);}

    size_t output_size() {
        return output_remap.size();
    }

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
    size_t head = 0; // events[head] is not recorded
    size_t tail = 0; // events[tail] is not finished

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
        if (!free_slots) rotate();
        --free_slots;
        return head;
    }

    size_t progress() {
        return tail;
    }
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
                busy_oprs.pop_front();
            }
            m_opr = busy_oprs.front().opr;
            busy_oprs.pop_front();
            return m_opr;
        }

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
        void register_shape_infer(VarNode* var, const cg::static_infer::ShapeInferDesc& desc) override {
            target->register_shape_infer(var, desc);
        };
        void register_value_infer(VarNode* var, const cg::static_infer::ValueInferDesc& desc) override {
            target->register_value_infer(var, desc);
        };
        cg::static_infer::InferType get_infer_type(VarNode*) override {
            return {cg::static_infer::InferType::MISSING_INP, cg::static_infer::InferType::MISSING_INP};
        }
        // some poorly written inference func would call infer_{shape,value}
        const TensorShape& infer_shape(VarNode* var) override {
            return target->infer_shape(var);
        }
        const DeviceTensorND& infer_value(VarNode* var) override {
            return target->infer_value(var);
        }
    };

    ProxyGraph::MiniGraph* target = nullptr;
    StaticInferManager m_static_infer_manager;
    std::unordered_map<size_t, ProxyGraph::MiniGraph> m_mini_graph_cache;
    size_t opr_count = 0;

    static thread_local std::unique_ptr<ProxyGraphTypeI> sm_instance;

    friend class ProxyGraph::MiniGraph;

    size_t nr_oprs_in_graph() const override {
        return opr_count;
    }

    size_t next_node_id() override {
        return opr_count;
    }

    std::shared_ptr<void> on_comp_node_finalize() override {
        sm_instance.reset();
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

        AttachGuard(ProxyGraphTypeI* owner_ = nullptr, ProxyGraph::MiniGraph* target_ = nullptr)
                : owner(owner_), target(target_) {}
        AttachGuard(AttachGuard&) = delete;
        AttachGuard& operator=(AttachGuard&) = delete;
        AttachGuard(AttachGuard&& rhs) : owner(rhs.owner), target(rhs.target) {rhs.owner = nullptr;}
        AttachGuard& operator=(AttachGuard&& rhs) = delete;
        ~AttachGuard() {if (owner) owner->attach(target);}
    };

    [[nodiscard]]
    AttachGuard scoped_attach(ProxyGraph::MiniGraph* target_) {
        attach(target_);
        return attach_guard();
    }

    [[nodiscard]]
    AttachGuard attach_guard(ProxyGraph::MiniGraph* target_ = nullptr) {
        return {this, target_};
    }

public:
    OperatorNodeBase* insert_opr(std::unique_ptr<OperatorNodeBase> opr_uniqp) override {
        return target->insert_opr(std::move(opr_uniqp));
    }

    static ProxyGraphTypeI& inst() {
        if (!sm_instance) {
            sm_instance.reset(new ProxyGraphTypeI);
        }
        return *sm_instance;
    }

    std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(const OpDef& def,
            const SmallVector<LogicalTensorDesc>& inputs) {
        size_t buf_size = 2 * inputs.size() + 1;
        size_t buf[buf_size];
        size_t pos = 0;
        buf[pos++] = def.hash();
        for (auto&& desc : inputs) {
            buf[pos++] = mgb::hash(desc.layout.dtype.handle());
            buf[pos++] = mgb::hash(desc.comp_node);
        }
        mgb_assert(pos == buf_size);
        auto key = XXHash{}.update(buf, buf_size*sizeof(size_t)).digest();
        auto it = m_mini_graph_cache.find(key);
        if (it == m_mini_graph_cache.end()) {
            auto&& result = m_mini_graph_cache.emplace(
                std::piecewise_construct,
                std::make_tuple(key),
                std::forward_as_tuple(*this, def, inputs));
            mgb_assert(result.second);
            it = result.first;
        }
        auto& minigraph = it->second;
        auto _ = scoped_attach(&minigraph);
        auto sess = minigraph.infer_session(inputs);
        std::tuple<SmallVector<LogicalTensorDesc>, bool> ret;
        auto& [descs, noerr] = ret;
        descs.reserve(minigraph.output_size());
        for (size_t i = 0; i < minigraph.output_size(); ++i) {
            descs.emplace_back();
            auto& desc = descs.back();
            desc.layout.dtype = minigraph.output_var(i)->dtype();
            desc.comp_node = minigraph.output_var(i)->comp_node();
            if (auto* shape = sess.infer_shape(i, false)) {
                desc.layout.init_contiguous_stride(*shape);
            } else {
                noerr = false;
            }
        }
        return ret;
    }
};

} // namespace mgb::imperative::proxy_graph
