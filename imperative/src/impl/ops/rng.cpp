/**
 * \file imperative/src/impl/ops/rng.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/rng.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/graph/helper.h"
#include "megbrain/opr/rand.h"

#include "../op_trait.h"
#include "../dnn_op_helper.h"

namespace mgb::imperative::rng {

namespace {

template <typename HandleFactory, typename THandle>
class DnnOpManagerT : public CompNodeDepedentObject, public NonCopyableObj {
public:
    using DT = CompNode::DeviceType;
    using Handle = THandle;
    using OpTypeInfo = size_t;

    template <typename... Args>
    Handle new_handle(Args&&... args) {
        return static_cast<HandleFactory*>(this)->do_new_handle(
                std::forward<Args>(args)...);
    }

    size_t delete_handle(Handle handle) {
        size_t removed = 0;
        if (!is_finalized()) {
            MGB_LOCK_GUARD(m_mtx);
            removed = m_handle2ops.erase(handle);
        }
        static_cast<HandleFactory*>(this)->do_delete_handle(handle);
        return removed;
    }

    template <typename DnnOp>
    auto get_dnn_op(Handle handle, OpTypeInfo tpinfo, CompNode cn) {
        mgb_assert(!is_finalized());
        DnnOpWithMutex* dnn_op_with_mtx;
        {
            MGB_LOCK_GUARD(m_mtx);
            dnn_op_with_mtx = &m_handle2ops[handle][tpinfo];
        }
        auto dnn_handle =
                MegDNNHandle::get(CompNodeEnv::from_comp_node(cn)).handle();
        std::unique_lock<std::mutex> lock(dnn_op_with_mtx->mtx);
        bool initialized = false;
        DnnOp* dnn_op = static_cast<DnnOp*>(dnn_op_with_mtx->op.get());
        if (dnn_op != nullptr) {
            mgb_assert(dnn_op->handle() == dnn_handle);
            initialized = true;
        } else {
            auto new_op = dnn_handle->create_operator<DnnOp>();
            dnn_op = new_op.get();
            dnn_op_with_mtx->op = std::move(new_op);
        }
        return std::make_tuple(initialized, dnn_op, std::move(lock));
    }

protected:
    using DnnOpManagerBase = DnnOpManagerT<HandleFactory, Handle>;
    DnnOpManagerT() = default;

private:
    struct DnnOpWithMutex {
        std::mutex mtx;
        std::unique_ptr<megdnn::OperatorBase> op;
        DnnOpWithMutex(): op{nullptr} {}
    };

    std::shared_ptr<void> on_comp_node_finalize() override {
        MGB_LOCK_GUARD(m_mtx);
        m_handle2ops.clear();
        return {};
    }

    std::unordered_map<Handle, std::unordered_map<OpTypeInfo, DnnOpWithMutex> > m_handle2ops;
    std::mutex m_mtx;
};

class RNGDnnOpManager final
        : public DnnOpManagerT<RNGDnnOpManager, Handle> {
public:
    Handle new_handle(CompNode comp_node, uint64_t seed) {
        MGB_LOCK_GUARD(sm_mtx);
        return DnnOpManagerBase::new_handle(comp_node, seed);
    }

    size_t delete_handle(Handle handle) {
        MGB_LOCK_GUARD(sm_mtx);
        return DnnOpManagerBase::delete_handle(handle);
    }

    Handle do_new_handle(CompNode comp_node, uint64_t seed) {
        auto handle = m_handle_pool.alloc(comp_node, seed);
        return reinterpret_cast<Handle>(handle);
    }

    void do_delete_handle(Handle handle) {
        m_handle_pool.free(reinterpret_cast<HandleData*>(handle));
    }

    static uint64_t get_seed(Handle handle) {
        if (!handle) { return glob_default_seed; }
        return reinterpret_cast<HandleData*>(handle)->seed;
    }

    static CompNode get_comp_node(Handle handle) {
        mgb_assert(handle, "invalid handle");
        return reinterpret_cast<HandleData*>(handle)->comp_node;
    }

    static Handle get_default_handle(CompNode comp_node) {
        mgb_assert(comp_node.valid());
        MGB_LOCK_GUARD(sm_mtx);
        auto&& glob_handle = glob_default_handles[comp_node];
        if (!glob_handle) {
            glob_handle = inst().do_new_handle(comp_node, glob_default_seed);
        } else if (get_seed(glob_handle) != glob_default_seed) {
            inst().DnnOpManagerBase::delete_handle(glob_handle);
            glob_handle = inst().do_new_handle(comp_node, glob_default_seed);
        }
        return glob_handle;
    }

    static RNGDnnOpManager& inst() {
        static RNGDnnOpManager mgr;
        return mgr;
    }

    static void set_glob_default_seed(uint64_t seed) {
        MGB_LOCK_GUARD(sm_mtx);
        glob_default_seed = seed;
    }

    static uint64_t get_glob_default_seed() {
        MGB_LOCK_GUARD(sm_mtx);
        return glob_default_seed;
    }

private:
    struct HandleData {
        CompNode comp_node;
        uint64_t seed;
        HandleData(CompNode cn, uint64_t seed) : comp_node(cn), seed(seed) {}
    };

    MemPool<HandleData> m_handle_pool;

    static std::mutex sm_mtx;
    static CompNode::UnorderedMap<Handle> glob_default_handles;
    static uint64_t glob_default_seed;
};

uint64_t RNGDnnOpManager::glob_default_seed = 0;
std::mutex RNGDnnOpManager::sm_mtx;
CompNode::UnorderedMap<Handle> RNGDnnOpManager::glob_default_handles;

template <typename Op>
struct OpMeth;

template <>
struct OpMeth<UniformRNG> {
    using DnnOp = megdnn::UniformRNG;
    using Param = DnnOp::Param;
    using OpNode = mgb::opr::UniformRNG;
    static Param make_param(const UniformRNG& rng) {
        auto handle_seed = RNGDnnOpManager::get_seed(rng.handle);
        mgb_assert(handle_seed == rng.seed,
            "inconsistent rng seed: rng op: %lu handle: %lu",
            handle_seed, rng.seed);
        return {handle_seed};
    }
};

template <>
struct OpMeth<GaussianRNG> {
    using DnnOp = megdnn::GaussianRNG;
    using Param = DnnOp::Param;
    using OpNode = mgb::opr::GaussianRNG;
    static Param make_param(const GaussianRNG& rng) {
        auto handle_seed = RNGDnnOpManager::get_seed(rng.handle);
        mgb_assert(handle_seed == rng.seed,
            "inconsistent rng seed: rng op: %lu handle: %lu",
            handle_seed, rng.seed);
        return {handle_seed, rng.mean, rng.std};
    }
};

template <typename Op>
void exec(const OpDef& op, const SmallVector<TensorPtr>& inputs,
          const SmallVector<TensorPtr>& outputs) {
    auto&& rng = op.cast_final_safe<Op>();
    auto dest = outputs[0];

    auto cn = dest->comp_node();
    auto handle = rng.handle;
    if (!handle) {
        handle = RNGDnnOpManager::get_default_handle(cn);
    }

    // retrieve dnn_op from glob cache
    auto dnn_op_thread_safe = RNGDnnOpManager::inst()
            .get_dnn_op<typename OpMeth<Op>::DnnOp>(
                handle, reinterpret_cast<size_t>(op.dyn_typeinfo()),
                cn);
    auto initialized = std::get<0>(dnn_op_thread_safe);
    auto dnn_op = std::get<1>(dnn_op_thread_safe);
    if (initialized) {
        auto handle_seed = RNGDnnOpManager::get_seed(handle);
        mgb_assert(dnn_op->param().seed == handle_seed,
            "inconsistent rng seed: handle: %lu, dnn_op: %lu",
            handle_seed, dnn_op->param().seed);
    }
    dnn_op->param() = OpMeth<Op>::make_param(rng);

    // allocate workspace
    size_t wk_size = dnn_op->get_workspace_in_bytes(dest->layout());
    auto workspace = Blob::make(cn, wk_size);
    megdnn::Workspace dnn_wk(workspace->storage().get(), wk_size);

    dnn_op->exec(dest->dev_tensor().as_megdnn(), dnn_wk);
}

template <typename Op>
SmallVector<LogicalTensorDesc> infer_output_attrs(
        const OpDef& op, const SmallVector<TensorPtr>& inputs) {
    LogicalTensorDesc dest;
    auto handle = op.cast_final_safe<Op>().handle;
    if (handle) {
        dest.comp_node = RNGDnnOpManager::get_comp_node(handle);
    } else {
        dest.comp_node = inputs[0]->comp_node();
    }

    auto hv = inputs[0]->get_value().proxy_to_default_cpu();
    TensorShape tshape;
    cg::copy_tensor_value_to_shape(tshape, hv);
    dest.layout = TensorLayout(tshape, dtype::Float32());
    return {dest};
}

template <typename Op>
SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    auto desc = infer_output_attrs<Op>(def, inputs);
    SmallVector<TensorPtr> outputs;
    for (auto&& i : desc) {
        outputs.push_back(Tensor::make(i.layout, i.comp_node));
    }
    exec<Op>(def, inputs, outputs);
    return outputs;
}

template<typename Op>
SymbolVar apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    size_t nr_inp = inputs.size();
    auto&& rng = def.cast_final_safe<Op>();
    mgb_assert(nr_inp == 1, "%s expects 1 inputs; got %lu actually",
               rng.dyn_typeinfo()->name,
               nr_inp);
    auto param = OpMeth<Op>::make_param(rng);
    OperatorNodeConfig config;
    if (rng.handle) {
        config = {rng.make_name(), RNGDnnOpManager::get_comp_node(rng.handle)};
    } else {
        config = {rng.make_name()};
    }
    return OpMeth<Op>::OpNode::make(inputs[0], param, config);
}

template<typename T>
std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& xxx_rng_def = def.cast_final_safe<T>();
    size_t nr_inp = inputs.size();
    mgb_assert(nr_inp == 1, "%s expects 1 inputs; got %lu actually",
               xxx_rng_def.dyn_typeinfo()->name,
               nr_inp);

    auto&& tshp = inputs[0];

    TensorLayout out_layout = tshp.layout;
    out_layout.dtype = dtype::Float32();
    if (tshp.layout.ndim == 0 || tshp.value.empty()) {
        out_layout.ndim = 0;
        return {{{out_layout, tshp.comp_node}}, true};
    }
    mgb_assert(
            tshp.layout.ndim == 1,
            "target shape of %s expects ndim=1; got ndim=%lu actually",
            xxx_rng_def.dyn_typeinfo()->name,
            tshp.layout.ndim);

    size_t target_ndim = tshp.layout.shape[0];
    out_layout.ndim = target_ndim;
    auto* ptr = tshp.value.ptr<dt_int32>();
    for (size_t i = 0; i < target_ndim; ++i) {
        out_layout.shape[i] = ptr[i];
    }

    return {{{out_layout, tshp.comp_node}}, true};
}

} // anonymous namespace

Handle new_handle(CompNode comp_node, uint64_t seed) {
    return RNGDnnOpManager::inst().new_handle(comp_node, seed);
}

size_t delete_handle(Handle handle) {
    return RNGDnnOpManager::inst().delete_handle(handle);
}

void set_global_rng_seed(uint64_t seed) {
    RNGDnnOpManager::set_glob_default_seed(seed);
}

uint64_t get_global_rng_seed() {
    return RNGDnnOpManager::get_glob_default_seed();
}

#define REG_RNG_OP(NAME)\
namespace { \
OP_TRAIT_REG(NAME, NAME, OpMeth<NAME>::OpNode) \
    .apply_on_var_node(apply_on_var_node<NAME>) \
    .apply_on_physical_tensor(apply_on_physical_tensor<NAME>) \
    .infer_output_attrs_fallible(infer_output_attrs_fallible<NAME>) \
    .fallback(); \
} \

REG_RNG_OP(UniformRNG)
REG_RNG_OP(GaussianRNG)

}  // namespace mgb::imperative::rng

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
