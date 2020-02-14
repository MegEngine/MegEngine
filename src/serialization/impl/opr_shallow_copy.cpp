/**
 * \file src/serialization/impl/opr_shallow_copy.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/serialization/opr_shallow_copy.h"

#include "megbrain/gopt/basic_arith.h"
#include "megbrain/serialization/opr_load_dump.h"
#include "megbrain/serialization/opr_registry.h"
#include "megbrain/utils/big_key_hashmap.h"

using namespace mgb;
using namespace serialization;

namespace {
//! dump single opr to memory for shallow copy
class OprDumpContextMemory final : public OprDumpContextRawPOD {
    std::vector<uint8_t> m_buf;

    void write_raw(const void* data, size_t size) override {
        auto pos = m_buf.size();
        auto end = pos + size;
        if (end > m_buf.capacity())
            m_buf.reserve(end * 2);
        m_buf.resize(end);
        memcpy(m_buf.data() + pos, data, size);
    }

    void dump_tensor(const std::string&, const HostTensorND&,
                     TensorWriteMethod) override {
        mgb_throw(GraphError,
                  "OprDumpContextMemory does not support dump tensor");
    }

    const GraphDumpConfig& config() const override {
        mgb_throw(GraphError, "OprDumpContextMemory has no associated config");
    }

public:
    OprDumpContextMemory() : OprDumpContextRawPOD(false) {}

    auto&& buf() const { return m_buf; }
};

//! load single opr from memory for shallow copy
class OprLoadContextMemory final : public OprLoadContextRawPOD {
    const uint8_t* m_ptr;
    size_t m_size, m_pos = 0;
    ComputingGraph* m_graph;

    void read_raw(void* dest, size_t size) override {
        auto end = m_pos + size;
        mgb_assert(end <= m_size);
        memcpy(dest, m_ptr + m_pos, size);
        m_pos = end;
    }

    ComputingGraph& graph() override { return *m_graph; }

    std::shared_ptr<HostTensorND> load_tensor() override { mgb_assert(0); }

    std::shared_ptr<DeviceTensorND> load_tensor_shared() override {
        mgb_assert(0);
    }

    const GraphLoadConfig& config() const override {
        mgb_throw(GraphError, "OprLoadContextMemory has no associated config");
    }

public:
    OprLoadContextMemory(ComputingGraph* graph,
                         const OprDumpContextMemory& dumper)
            : OprLoadContextRawPOD(false),
              m_ptr{dumper.buf().data()},
              m_size{dumper.buf().size()},
              m_graph{graph} {}

    ~OprLoadContextMemory() { mgb_assert(m_pos == m_size); }
};

class ShallowCopyCacheContainer final : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

    struct HashEq {
        template <typename T>
        static bool eq(const T& x, const T& y) {
            return x == y;
        }
        static bool eq(const OperatorNodeConfig& x,
                       const OperatorNodeConfig& y) {
            return x.is_same(y);
        }
        static size_t hash(const void* ptr) {
            return std::hash<const void*>{}(ptr);
        }
        static size_t hash(const VarNodeArray& inputs) {
            return PODHash<VarNode*>::perform(inputs.data(), inputs.size());
        }
        static size_t hash(const OperatorNodeConfig& config) {
            return config.hash();
        }
    };

public:
    big_key_hash_map::BigKeyHashMap<
            cg::OperatorNodeBase*, HashEq,
            big_key_hash_map::Copy<const cg::OperatorNodeBase*>,
            big_key_hash_map::Ref<VarNodeArray>,
            big_key_hash_map::Ref<OperatorNodeConfig>>
            cache;
};
MGB_TYPEINFO_OBJ_IMPL(ShallowCopyCacheContainer);

}  // anonymous namespace

ComputingGraph* serialization::OprShallowCopyContext::owner_graph(
        const cg::OperatorNodeBase& opr, const VarNodeArray& inputs) const {
    if (!m_owner_graph) {
        if (inputs.empty())
            return opr.owner_graph();
        return inputs[0]->owner_graph();
    }
    if (!inputs.empty())
        mgb_assert(m_owner_graph == inputs[0]->owner_graph());

    return m_owner_graph;
}

cg::OperatorNodeBase* serialization::copy_opr_shallow(
        const cg::OperatorNodeBase& opr, const VarNodeArray& inputs,
        const OperatorNodeConfig& config, const OprShallowCopyContext& ctx) {
    auto registry = OprRegistry::find_by_type(opr.dyn_typeinfo());
    mgb_assert(registry, "could not find OprReceiver to copy opr %s{%s}",
               opr.cname(), opr.dyn_typeinfo()->name);

    mgb_assert(inputs.size() == opr.input().size());
    auto dst_og = ctx.owner_graph(opr, inputs);
    auto do_copy = [&]() {
        auto nr_opr_before = opr.owner_graph()->nr_oprs_in_graph();
        auto ret = registry->shallow_copy(ctx, opr, inputs, config);

        if (dst_og != opr.owner_graph() ||
            opr.owner_graph()->nr_oprs_in_graph() != nr_opr_before) {
            auto&& attr = ret->node_prop().attribute();
            if (!attr.src_opr) {
                auto src = cg::get_opr_root_source_opr(
                        const_cast<cg::OperatorNodeBase*>(&opr));
                if (ret != src)
                    attr.src_opr = src;
            }
            if (!attr.priority) {
                // priority may have been changed by OprInserted event handlers
                // (like in python case)
                attr.priority = opr.node_prop().attribute().priority;
            }
        }
        return ret;
    };
    cg::OperatorNodeBase* ret;
    if (dst_og == opr.owner_graph()) {
        // use cache for copy in same graph
        auto&& cache =
                dst_og->options()
                        .user_data
                        .get_user_data_or_create<ShallowCopyCacheContainer>()
                        ->cache;
        auto ins = cache.get(&opr, inputs, config);
        if (ins.first) {
            *ins.second = do_copy();
        } else {
            cg::update_output_var_shapes(*ins.second);
        }
        ret = *ins.second;
    } else {
        ret = do_copy();
    }

    mgb_assert(gopt::has_inplace_basic_arith_opt(opr) ||
                       ((  // outputs match
                                opr.usable_output().size() ==
                                ret->usable_output().size()) &&
                        (  // new opr is returned
                                (&opr != ret) || opr.input() == inputs)),
               "bad opr copy: src=%s{%s} dst=%s{%s}", opr.cname(),
               opr.dyn_typeinfo()->name, ret->cname(),
               ret->dyn_typeinfo()->name);

    return ret;
}

cg::OperatorNodeBase* serialization::intl::copy_opr_shallow_default_impl(
        const OprShallowCopyContext& ctx, const cg::OperatorNodeBase& opr,
        const VarNodeArray& inputs, const OperatorNodeConfig& config) {
    MGB_MARK_USED_VAR(ctx);

    auto registry = OprRegistry::find_by_type(opr.dyn_typeinfo());
    mgb_assert(registry && registry->dumper && registry->loader,
               "can not shallow_copy operator %s{%s}: "
               "no dumper/loader registered",
               opr.cname(), opr.dyn_typeinfo()->name);
    OprDumpContextMemory dumper;
    registry->dumper(dumper, opr);

    OprLoadContextMemory loader{opr.owner_graph(), dumper};
    return registry->loader(loader, inputs, config);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
