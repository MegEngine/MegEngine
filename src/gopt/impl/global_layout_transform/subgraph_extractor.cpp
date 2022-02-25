#include "megbrain/gopt/subgraph_extractor.h"
#include <atomic>
#include "megbrain/serialization/opr_shallow_copy.h"

using namespace mgb;
using namespace cg;
using namespace gopt;

/* ================== GraphPartition::InputPlaceholder =================*/
// clang-format off
MGB_DEFINE_OPR_CLASS(GraphPartition::InputPlaceholder,
                     cg::SingleCNOperatorNodeBase) // {
public:
    InputPlaceholder(VarNode* src_var, const TensorShape& infer_shp,
                     std::unique_ptr<HostTensorND> infer_val = nullptr);

    static SymbolVar make(VarNode* src_var, const TensorShape& infer_shp,
                          std::unique_ptr<HostTensorND> infer_val = nullptr);

    size_t input_id() const { return m_id; }

private:
    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    void init_output_comp_node() override;

    const size_t m_id;
    TensorShape m_infer_shp;
    std::unique_ptr<HostTensorND> m_infer_val;
    static std::atomic_size_t sm_id;
};
// clang-format on

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GraphPartition::InputPlaceholder);

std::atomic_size_t GraphPartition::InputPlaceholder::sm_id{0};
GraphPartition::InputPlaceholder::InputPlaceholder(
        VarNode* src_var, const TensorShape& infer_shp,
        std::unique_ptr<HostTensorND> infer_val)
        : Super(src_var->owner_graph(), {}, {}, {}),
          m_id{sm_id.fetch_add(1, std::memory_order_relaxed)},
          m_infer_shp{infer_shp},
          m_infer_val{std::move(infer_val)} {
    name(ssprintf("InputPlaceholder@%zu", m_id));
    add_equivalence_component<ScalarHash<DTypeEnum>>(src_var->dtype().enumv());
    add_equivalence_component<ScalarHash<size_t>>(m_id);
    add_output(None)->dtype(src_var->dtype());
}

void GraphPartition::InputPlaceholder::init_output_comp_node() {
    output(0)->comp_node(CompNode::default_cpu());
}

void GraphPartition::InputPlaceholder::scn_do_execute() {
    mgb_throw(InternalError, "InputPlaceholder opr can not be executed");
}

void GraphPartition::InputPlaceholder::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    if (m_infer_shp.ndim == 0) {
        auto infer_shape = [](TensorShape&, const InpVal&) { return false; };
        mgr.register_shape_infer(output(0), {SourceType::MUTABLE, {}, infer_shape});
    } else {
        mgr.register_shape_infer(output(0), ShapeInferDesc::make_const(m_infer_shp));
    }

    if (m_infer_val == nullptr) {
        auto infer_value = [](DeviceTensorND&, const InpVal&) { return false; };
        mgr.register_value_infer(output(0), {SourceType::MUTABLE, {}, infer_value});
    } else {
        auto infer_value = [this](DeviceTensorND& dest, const InpVal&) {
            dest.copy_from(*m_infer_val).sync();
            return true;
        };
        mgr.register_value_infer(output(0), {SourceType::CONSTANT, {}, infer_value});
    }
}

SymbolVar GraphPartition::InputPlaceholder::make(
        VarNode* src_var, const TensorShape& infer_shp,
        std::unique_ptr<HostTensorND> infer_val) {
    return src_var->owner_graph()
            ->insert_opr(std::make_unique<InputPlaceholder>(
                    src_var, infer_shp, std::move(infer_val)))
            ->output(0);
}

/* ================== GraphPartition =================*/
#if MGB_ENABLE_JSON
std::shared_ptr<json::Value> GraphPartition::to_json() const {
    auto replaced_outputs = std::get<1>(replace_graph_by_placeholder());

    ThinHashSet<VarNode*> all_var_node;
    ThinHashSet<OperatorNodeBase*> all_opr_node;
    auto comp_seq = json::Array::make();

    auto cb = [&](OperatorNodeBase* opr) {
        comp_seq->add(json::String::make(opr->id_str()));
        for (const auto& i : opr->input()) {
            if (all_var_node.count(i) == 0) {
                all_var_node.insert(i);
            }
        }
        all_opr_node.insert(opr);
        for (const auto& o : opr->output()) {
            all_var_node.insert(o);
        }
    };
    cg::DepOprIter iter{cb};
    for (const auto& o : replaced_outputs)
        iter.add(o->owner_opr());

    auto dump_node_coll = [](auto&& collection) {
        auto objptr = json::Object::make();
        auto&& obj = *objptr;
        for (auto&& i : collection)
            obj[i->id_str()] = i->to_json();
        return objptr;
    };

    return json::Object::make(
            {{"operator", dump_node_coll(all_opr_node)},
             {"var", dump_node_coll(all_var_node)},
             {"comp_seq", comp_seq}});
}
#endif

std::pair<VarNodeArray, VarNodeArray> GraphPartition::replace_graph_by_placeholder()
        const {
    ThinHashMap<VarNode*, VarNode*> old2new;
    auto graph_partition_copy_opr_shallow = [](OperatorNodeBase* opr,
                                               const VarNodeArray& inps) {
        OperatorNodeConfig config = opr->config();
        return serialization::copy_opr_shallow(*opr, inps, config)->output(0);
    };

    OperatorNodeSet input_opr_set;
    for (const auto& i : m_inputs)
        input_opr_set.insert(i->owner_opr());

    VarNodeArray placeholders;
    VarNodeArray replaced_outputs;
    VarNodeArray new_i;
    auto cb = [&](OperatorNodeBase* opr) {
        for (const auto& o : opr->output()) {
            if (o->contain_flag(VarNode::Flag::VOLATILE_CONTENT) ||
                (input_opr_set.count(opr) && !m_inputs.count(o))) {
                continue;
            }
            VarNode* new_o;
            if (m_inputs.count(o)) {
                auto&& mgr = opr->owner_graph()->static_infer_manager();
                const TensorShape* shp_ptr = nullptr;
                if (cg::is_static_var_shape(o)) {
                    shp_ptr = mgr.infer_shape_fallible(o);
                }
                TensorShape infer_shp;
                if (shp_ptr)
                    infer_shp = *shp_ptr;
                std::unique_ptr<HostTensorND> hval = nullptr;
                const DeviceTensorND* dval_ptr = nullptr;
                if (cg::is_static_var_value(o)) {
                    dval_ptr = mgr.infer_value_fallible(o);
                }
                if (dval_ptr) {
                    hval.reset(new HostTensorND(
                            CompNode::default_cpu(), dval_ptr->dtype()));
                    hval->resize(dval_ptr->shape()).copy_from(*dval_ptr).sync();
                }
                new_o = InputPlaceholder::make(o, infer_shp, std::move(hval)).node();
                placeholders.push_back(new_o);
            } else {
                new_i.clear();
                for (const auto& i : opr->input()) {
                    new_i.push_back(old2new.at(i));
                }
                new_o = graph_partition_copy_opr_shallow(o->owner_opr(), new_i);
            }
            old2new[o] = new_o;
        }
    };
    cg::DepOprIter iter{cb};
    for (auto&& i : m_inputs) {
        for (auto&& j : i->owner_opr()->input()) {
            if (!input_opr_set.count(j->owner_opr()) &&
                !m_opr_set.count(j->owner_opr())) {
                iter.set_visited(j->owner_opr());
            }
        }
    }
    for (auto&& o : m_outputs)
        iter.add(o->owner_opr());
    for (auto&& o : m_outputs) {
        replaced_outputs.push_back(old2new.at(o));
    }
    return std::make_pair(placeholders, replaced_outputs);
}

/* ================== SubGraphExtractor =================*/
std::vector<GraphPartition> SubGraphExtractor::extract(
        const SymbolVarArray& endpoint_vars) const {
    ThinHashMap<OperatorNodeBase*, std::pair<OperatorNodeBase*, int>> parent;
    thin_function<OperatorNodeBase*(OperatorNodeBase*)> union_find;
    union_find = [&parent, &union_find](OperatorNodeBase* o) {
        if (parent[o].first == o)
            return o;
        else {
            auto p = union_find(parent[o].first);
            parent[o].first = p;
            return p;
        }
    };
    auto union_merge = [&parent, &union_find](
                               OperatorNodeBase* x, OperatorNodeBase* y) {
        auto root_x = union_find(x), root_y = union_find(y);
        if (root_x != root_y) {
            OperatorNodeBase *large, *small;
            if (parent[root_x].second < parent[root_y].second) {
                small = root_x, large = root_y;
            } else {
                small = root_y, large = root_x;
            }
            parent[small].first = large;
            if (parent[large].second == parent[small].second) {
                parent[large].second += 1;
            }
        }
    };

    std::vector<OperatorNodeBase*> topo;
    auto cb = [this, &parent, &union_merge, &topo](OperatorNodeBase* opr) {
        topo.push_back(opr);
        if (m_opr_list.count(opr->dyn_typeinfo()) == 0)
            return;
        auto find = parent.find(opr);
        if (find == parent.end()) {
            parent.insert(std::make_pair(opr, std::make_pair(opr, 0)));
        }
        for (auto&& i : opr->input()) {
            auto&& o = i->owner_opr();
            if (m_opr_list.count(o->dyn_typeinfo()) == 0)
                continue;
            union_merge(opr, o);
        }
    };
    cg::DepOprIter iter{cb};
    for (const auto& v : endpoint_vars)
        iter.add(v.node()->owner_opr());

    std::vector<GraphPartition> partitions;
    partitions.reserve(topo.size());
    ThinHashMap<OperatorNodeBase*, GraphPartition*> roots;
    /// backward pass
    for (const auto& opr : reverse_adaptor(topo)) {
        if (m_opr_list.count(opr->dyn_typeinfo()) > 0) {
            auto root = union_find(opr);
            auto find = roots.find(root);
            GraphPartition* partition = nullptr;
            if (find == roots.end()) {
                partitions.emplace_back(GraphPartition{});
                auto insert = roots.insert(std::make_pair(root, &partitions.back()));
                partition = insert.first->second;
                for (auto&& o : opr->output()) {
                    if (!o->contain_flag(cg::VarNode::Flag::VOLATILE_CONTENT))
                        partition->output().insert(o);
                }
            } else {
                partition = find->second;
                for (auto&& o : opr->output()) {
                    if (!o->contain_flag(cg::VarNode::Flag::VOLATILE_CONTENT)) {
                        auto erase = partition->input().erase(o);
                        if (erase == 0)
                            partition->output().insert(o);
                    }
                }
            }
            partition->opr_set().insert(opr);
            partition->all_oprs().push_back(opr);
            for (const auto& i : opr->input())
                partition->input().insert(i);
        }
    }
    /// forward pass
    for (auto&& opr : topo) {
        if (m_opr_list.count(opr->dyn_typeinfo()) == 0) {
            for (const auto& i : opr->input()) {
                if (m_opr_list.count(i->owner_opr()->dyn_typeinfo())) {
                    auto root = union_find(i->owner_opr());
                    GraphPartition* partition;
                    auto find = roots.find(root);
                    if (find != roots.end()) {
                        partition = find->second;
                        partition->output().insert(i);
                    }
                }
            }
        }
    }

    for (auto&& partition : partitions) {
        auto& all_oprs = partition.all_oprs();
        std::reverse(all_oprs.begin(), all_oprs.end());
    }
    return partitions;
}

// vim: syntax=cpp.doxygen
