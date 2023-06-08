#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/rdnn/profiler.h"
#include "megbrain/serialization/opr_load_dump.h"

#include "../op_trait.h"
#include "megbrain/imperative/proxy_graph_detail.h"

namespace mgb {
namespace imperative {

namespace {
class OprParamsLoadContext final : public serialization::OprLoadContextRawPOD {
public:
    bool strict = true;

private:
    const OprAttr::Param& m_param;
    size_t m_pos = 0;
    ComputingGraph* m_graph;

    void read_raw(void* dest, size_t size) override final {
        mgb_assert(m_pos + size <= m_param.size(), "too many bytes requested");
        memcpy(dest, m_param.data() + m_pos, size);
        m_pos += size;
    }

    std::shared_ptr<HostTensorND> load_tensor() override { mgb_assert(0); }

    std::shared_ptr<DeviceTensorND> load_tensor_shared(
            bool copy_immediatly = false) override {
        MGB_MARK_USED_VAR(copy_immediatly);
        mgb_assert(0);
    }

    const serialization::GraphLoadConfig& config() const override { mgb_assert(0); }

public:
    OprParamsLoadContext(const OprAttr::Param& param, ComputingGraph* graph)
            : serialization::OprLoadContextRawPOD(false),
              m_param(param),
              m_graph(graph) {}

    ~OprParamsLoadContext() {
        if (strict)
            mgb_assert(m_pos == m_param.size(), "param not fully consumed");
    }

    ComputingGraph& graph() override { return *m_graph; }
};

class OprParamsDumpContext final : public serialization::OprDumpContextRawPOD {
public:
    OprAttr::Param m_param;
    OprParamsDumpContext() : serialization::OprDumpContextRawPOD(false) {}
    void write_raw(const void* data, size_t size) {
        const char* src = static_cast<const char*>(data);
        m_param.insert(m_param.end(), src, src + size);
    }
    void dump_tensor(
            const std::string& name, const HostTensorND& tensor,
            TensorWriteMethod method, TensorFormat format = {}) {
        mgb_assert(0);
    }
    const serialization::GraphDumpConfig& config() const { mgb_assert(0); }
};

#define cb(FASTRUN_OPR)                                                           \
    megdnn::param::ExecutionPolicy get_strategy_##FASTRUN_OPR(                    \
            cg::OperatorNodeBase* opr) {                                          \
        auto policy =                                                             \
                opr->cast_final<opr::FASTRUN_OPR>().execution_policy_transient(); \
        return policy;                                                            \
    }                                                                             \
    void set_strategy_##FASTRUN_OPR(                                              \
            cg::OperatorNodeBase* opr, megdnn::param::ExecutionPolicy policy) {   \
        auto&& p = opr->cast_final<opr::FASTRUN_OPR>();                           \
        p.set_execution_policy(policy);                                           \
    }

DNN_FOREACH_FASTRUN_OPR(cb)
#undef cb

typedef thin_function<megdnn::param::ExecutionPolicy(cg::OperatorNodeBase*)> get_func;
typedef thin_function<void(cg::OperatorNodeBase*, megdnn::param::ExecutionPolicy)>
        set_func;

static const mgb::thin_hash_table::ThinHashMap<
        mgb::Typeinfo*, std::pair<get_func, set_func>>&
get_type2policy() {
    static mgb::thin_hash_table::ThinHashMap<
            mgb::Typeinfo*, std::pair<get_func, set_func>>
            sl_type2policy;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define cb(FASTRUN_OPR)                            \
    sl_type2policy[opr::FASTRUN_OPR::typeinfo()] = \
            std::make_pair(get_strategy_##FASTRUN_OPR, set_strategy_##FASTRUN_OPR);
        DNN_FOREACH_FASTRUN_OPR(cb)
    });
    return std::as_const(sl_type2policy);
}

VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& attr = def.cast_final_safe<OprAttr>();
    auto config = attr.config;
    config.name(attr.make_name());
    mgb_assert(!inputs.empty());
    auto registry = serialization::OprRegistry::find_by_name(attr.type);
    mgb_assert(registry, "operator %s not found", attr.type.c_str());
    OprParamsLoadContext ctx{attr.param, inputs[0]->owner_graph()};
    auto opr_with_accessor = registry->loader(ctx, inputs, config);
    auto&& opr = opr_with_accessor.opr();
    if (get_type2policy().find(opr->dyn_typeinfo()) != get_type2policy().end()) {
        get_type2policy().at(opr->dyn_typeinfo()).second(opr, attr.policy);
    }
    return opr_with_accessor.usable_output();
}

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* opr) {
    OprParamsDumpContext ctx;
    auto registry = serialization::OprRegistry::find_by_type(opr->dyn_typeinfo());
    mgb_assert(registry, "operator %s not found", opr->dyn_typeinfo()->name);
    mgb_assert(
            registry->dumper, "operator %s cannot be serialized",
            opr->dyn_typeinfo()->name);
    registry->dumper(ctx, *opr);
    megdnn::param::ExecutionPolicy policy;
    if (get_type2policy().find(opr->dyn_typeinfo()) != get_type2policy().end()) {
        policy = get_type2policy().at(opr->dyn_typeinfo()).first(opr);
    }
    return OprAttr::make(
            registry->name, std::move(ctx.m_param), policy, opr->config(),
            opr->dyn_typeinfo());
}

std::vector<std::pair<const char*, std::string>> props(const OpDef& def) {
    return {};
}

std::string make_name(const OpDef& def) {
    auto&& attr = def.cast_final_safe<OprAttr>();
    return attr.type;
}

OP_TRAIT_REG(OprAttr, OprAttr)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .props(props)
        .make_name(make_name)
        .fallback();

}  // anonymous namespace

bool OprAttr::is_same_st(const Hashable& rhs_) const {
    auto&& rhs = static_cast<const OprAttr&>(rhs_);
    return type == rhs.type && param == rhs.param &&
           policy.strategy == rhs.policy.strategy &&
           policy.workspace_limit == rhs.policy.workspace_limit &&
           config.comp_node() == rhs.config.comp_node() &&
           config.output_dtype() == rhs.config.output_dtype();
}

size_t OprAttr::hash() const {
    return hash_pair_combine(
            hash_pair_combine(
                    hash_pair_combine(
                            mgb::hash(type),
                            mgb::hash(static_cast<std::vector<char>>(param))),
                    hash_pair_combine(
                            static_cast<size_t>(policy.strategy),
                            policy.workspace_limit)),
            config.hash());
}

std::shared_ptr<json::Value> OprAttr::mgb_param(OprFootprint* footprint) {
    OprParamsLoadContext ctx{param, nullptr};
    ctx.strict = false;
    return footprint->get_serial_param_json(mgb_opr_type, ctx);
};

MGB_DYN_TYPE_OBJ_FINAL_IMPL(OprAttr);

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
