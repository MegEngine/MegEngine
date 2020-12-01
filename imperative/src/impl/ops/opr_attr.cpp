/**
 * \file imperative/src/impl/ops/opr_attr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/serialization/opr_load_dump.h"

#include "../op_trait.h"
#include "megbrain/imperative/proxy_graph_detail.h"

namespace mgb {
namespace imperative {

namespace {
class OprParamsLoadContext final: public serialization::OprLoadContextRawPOD {
    const OprAttr::Param& m_param;
    size_t m_pos = 0;
    ComputingGraph *m_graph;

    void read_raw(void *dest, size_t size) override final {
        mgb_assert(m_pos + size <= m_param.size(), "too many bytes requested");
        memcpy(dest, m_param.data() + m_pos, size);
        m_pos += size;
    }

    std::shared_ptr<HostTensorND> load_tensor() override {
        mgb_assert(0);
    }

    std::shared_ptr<DeviceTensorND> load_tensor_shared() override {
        mgb_assert(0);
    }

    const serialization::GraphLoadConfig& config() const override {
        mgb_assert(0);
    }

    public:
        OprParamsLoadContext(const OprAttr::Param& param,
                ComputingGraph *graph):
            serialization::OprLoadContextRawPOD(false), m_param(param), m_graph(graph)
        {}

        ~OprParamsLoadContext() {
            mgb_assert(m_pos == m_param.size(), "param not fully consumed");
        }

        ComputingGraph& graph() override {
            return *m_graph;
        }
};

class OprParamsDumpContext final: public serialization::OprDumpContextRawPOD {
public:
    OprAttr::Param m_param;
    OprParamsDumpContext() : serialization::OprDumpContextRawPOD(false) {}
    void write_raw(const void *data, size_t size) {
        const char* src = static_cast<const char*>(data);
        m_param.insert(m_param.end(), src, src + size);
    }
    void dump_tensor(
            const std::string &name,
            const HostTensorND &tensor,
            TensorWriteMethod method) {
        mgb_assert(0);
    }
    const serialization::GraphDumpConfig& config() const {
        mgb_assert(0);
    }
};

cg::OperatorNodeBase* apply_on_var_node(
        const OpDef& def, const VarNodeArray& inputs) {
    auto&& attr = def.cast_final_safe<OprAttr>();
    mgb_assert(!inputs.empty());
    auto registry = serialization::OprRegistry::find_by_name(attr.type);
    mgb_assert(registry, "operator %s not found", attr.type.c_str());
    OprParamsLoadContext ctx{attr.param, inputs[0]->owner_graph()};
    return registry->loader(ctx, inputs, attr.config);
}

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* opr) {
    OprParamsDumpContext ctx;
    auto registry = serialization::OprRegistry::find_by_type(opr->dyn_typeinfo());
    mgb_assert(registry, "operator %s not found", opr->dyn_typeinfo()->name);
    mgb_assert(registry->dumper, "operator %s cannot be serialized", opr->dyn_typeinfo()->name);
    registry->dumper(ctx, *opr);
    return OprAttr::make(registry->name, std::move(ctx.m_param), opr->config());
}

OP_TRAIT_REG(OprAttr, OprAttr)
    .make_from_op_node(make_from_op_node)
    .apply_on_var_node(apply_on_var_node)
    .fallback();

} // anonymous namespace

bool OprAttr::is_same_st(const Hashable& rhs_) const {
    auto&& rhs = static_cast<const OprAttr&>(rhs_);
    return type == rhs.type && param == rhs.param
        && config.comp_node() == rhs.config.comp_node()
        && config.output_dtype() == rhs.config.output_dtype();
}

size_t OprAttr::hash() const {
    return hash_pair_combine(
            hash_pair_combine(
                mgb::hash(type),
                mgb::hash(static_cast<std::vector<char>>(param))),
            config.hash());
}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(OprAttr);

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
