#include "megbrain/imperative/ops/rng.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/graph/helper.h"
#include "megbrain/opr/rand.h"

#include "../dnn_op_helper.h"
#include "../op_trait.h"

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
        auto dnn_handle = MegDNNHandle::get(CompNodeEnv::from_comp_node(cn)).handle();
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
        DnnOpWithMutex() : op{nullptr} {}
    };

    std::shared_ptr<void> on_comp_node_finalize() override {
        MGB_LOCK_GUARD(m_mtx);
        m_handle2ops.clear();
        return {};
    }

    std::unordered_map<Handle, std::unordered_map<OpTypeInfo, DnnOpWithMutex>>
            m_handle2ops;
    std::mutex m_mtx;
};

class RNGDnnOpManager final : public DnnOpManagerT<RNGDnnOpManager, Handle> {
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
        if (!handle) {
            return glob_default_seed;
        }
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
        }
        mgb_assert(get_seed(glob_handle) == glob_default_seed);
        return glob_handle;
    }

    static RNGDnnOpManager& inst() {
        static RNGDnnOpManager mgr;
        return mgr;
    }

    static void set_glob_default_seed(uint64_t seed) {
        MGB_LOCK_GUARD(sm_mtx);
        for (auto&& elem : glob_default_handles) {
            mgb_assert(elem.first.valid());
            if (elem.second) {
                inst().DnnOpManagerBase::delete_handle(elem.second);
            }
            elem.second = inst().do_new_handle(elem.first, seed);
        }
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
        mgb_assert(
                handle_seed == rng.seed,
                "inconsistent rng seed: rng op: %lu handle: %lu", handle_seed,
                rng.seed);
        return {handle_seed, rng.dtype.enumv()};
    }
};

template <>
struct OpMeth<PoissonRNG> {
    using DnnOp = megdnn::PoissonRNG;
    using Param = DnnOp::Param;
    using OpNode = mgb::opr::PoissonRNG;
    static Param make_param(const PoissonRNG& rng) {
        auto handle_seed = RNGDnnOpManager::get_seed(rng.handle);
        mgb_assert(
                handle_seed == rng.seed,
                "inconsistent rng seed: rng op: %lu handle: %lu", handle_seed,
                rng.seed);
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
        mgb_assert(
                handle_seed == rng.seed,
                "inconsistent rng seed: rng op: %lu handle: %lu", handle_seed,
                rng.seed);
        return {handle_seed, rng.mean, rng.std, rng.dtype.enumv()};
    }
};

template <>
struct OpMeth<GammaRNG> {
    using DnnOp = megdnn::GammaRNG;
    using Param = DnnOp::Param;
    using OpNode = mgb::opr::GammaRNG;
    static Param make_param(const GammaRNG& rng) {
        auto handle_seed = RNGDnnOpManager::get_seed(rng.handle);
        mgb_assert(
                handle_seed == rng.seed,
                "inconsistent rng seed: rng op: %lu handle: %lu", handle_seed,
                rng.seed);
        return {handle_seed};
    }
};

template <>
struct OpMeth<PermutationRNG> {
    using DnnOp = megdnn::PermutationRNG;
    using Param = DnnOp::Param;
    using OpNode = mgb::opr::PermutationRNG;
    static Param make_param(const PermutationRNG& rng) {
        auto handle_seed = RNGDnnOpManager::get_seed(rng.handle);
        mgb_assert(
                handle_seed == rng.seed,
                "inconsistent rng seed: rng op: %lu handle: %lu", handle_seed,
                rng.seed);
        return {handle_seed, rng.dtype.enumv()};
    }
};

template <>
struct OpMeth<BetaRNG> {
    using DnnOp = megdnn::BetaRNG;
    using Param = DnnOp::Param;
    using OpNode = mgb::opr::BetaRNG;
    static Param make_param(const BetaRNG& rng) {
        auto handle_seed = RNGDnnOpManager::get_seed(rng.handle);
        mgb_assert(
                handle_seed == rng.seed,
                "inconsistent rng seed: rng op: %lu handle: %lu", handle_seed,
                rng.seed);
        return {handle_seed};
    }
};

template <>
struct OpMeth<ShuffleRNG> {
    using DnnOp = megdnn::ShuffleRNG;
    using Param = DnnOp::Param;
    using OpNode = mgb::opr::ShuffleRNG;
    static Param make_param(const ShuffleRNG& rng) {
        auto handle_seed = RNGDnnOpManager::get_seed(rng.handle);
        mgb_assert(
                handle_seed == rng.seed,
                "inconsistent rng seed: rng op: %lu handle: %lu", handle_seed,
                rng.seed);
        return {handle_seed};
    }
};

template <>
struct OpMeth<ExponentialRNG> {
    using DnnOp = megdnn::ExponentialRNG;
    using Param = DnnOp::Param;
    using OpNode = mgb::opr::ExponentialRNG;
    static Param make_param(const ExponentialRNG& rng) {
        auto handle_seed = RNGDnnOpManager::get_seed(rng.handle);
        mgb_assert(
                handle_seed == rng.seed,
                "inconsistent rng seed: rng op: %lu handle: %lu", handle_seed,
                rng.seed);
        return {handle_seed};
    }
};

template <>
struct OpMeth<Dropout> {
    using DnnOp = megdnn::Dropout;
    using Param = DnnOp::Param;
    using OpNode = mgb::opr::Dropout;
    static Param make_param(const Dropout& opdef) {
        auto handle_seed = RNGDnnOpManager::get_seed(opdef.handle);
        mgb_assert(
                handle_seed == opdef.seed,
                "inconsistent dropout seed: dropout op: %lu handle: %lu", handle_seed,
                opdef.seed);
        return {opdef.drop_prob, handle_seed};
    }
};

template <>
struct OpMeth<MultiHeadAttn> {
    using DnnOp = megdnn::MultiHeadAttn;
    using Param = DnnOp::Param;
    using OpNode = mgb::opr::MultiHeadAttn;
    static Param make_param(const MultiHeadAttn& opdef) {
        auto handle_seed = RNGDnnOpManager::get_seed(opdef.handle);
        mgb_assert(
                handle_seed == opdef.seed,
                "inconsistent multiheadattn seed: dropout op: %lu handle: %lu",
                handle_seed, opdef.seed);

        return {opdef.num_heads,      opdef.embeding_size,
                opdef.k_size,         opdef.v_size,
                opdef.qproj_size,     opdef.kproj_size,
                opdef.vproj_size,     opdef.oproj_size,
                opdef.qbias,          opdef.kbias,
                opdef.vbias,          opdef.obias,
                opdef.sm_scaler,      opdef.input_order,
                opdef.attn_mask_type, opdef.tensor_combination_type,
                opdef.add_bias_kv,    opdef.add_zero_attn,  
                opdef.need_weights,   opdef.reslink,
                opdef.training,       handle_seed,
                opdef.attn_prob,      opdef.out_prob};
    }
};

template <bool>
struct _InferLayout;

template <int nr_in>
struct _RNGOprMaker;

template <int nr_in, int nr_out>
struct _RNGOprInvoker;

template <>
struct _InferLayout<true> {
    template <typename Op>
    static TensorLayout do_infer(const TensorPtr& inp, const Op& rng) {
        TensorShape tshape;
        auto hv = inp->get_value().proxy_to_default_cpu();
        cg::copy_tensor_value_to_shape(tshape, hv);
        return TensorLayout(tshape, rng.dtype);
    }

    template <typename Op>
    static TensorLayout do_infer(const LogicalTensorDesc& inp, const Op& rng) {
        TensorLayout out_layout = inp.layout;
        out_layout.dtype = rng.dtype;
        if (inp.layout.ndim == 0 || inp.value.empty()) {
            out_layout.ndim = 0;
            return out_layout;
        }
        mgb_assert(
                inp.layout.ndim == 1,
                "target shape of %s expects ndim=1; got ndim=%lu actually",
                rng.dyn_typeinfo()->name, inp.layout.ndim);
        size_t target_ndim = inp.layout.shape[0];
        out_layout.ndim = target_ndim;
        auto* ptr = inp.value.ptr<dt_int32>();
        for (size_t i = 0; i < target_ndim; ++i) {
            out_layout.shape[i] = ptr[i];
        }
        out_layout.init_contiguous_stride();
        return out_layout;
    }
};

template <>
struct _InferLayout<false> {
    template <typename Op>
    static TensorLayout do_infer(const TensorPtr& inp, const Op& rng) {
        return inp->layout();
    }

    template <typename Op>
    static TensorLayout do_infer(const LogicalTensorDesc& inp, const Op& rng) {
        mgb_assert(inp.layout.ndim);
        return inp.layout;
    }
};

#define _INST_RNG_INVOLKER(DNN_NR_INPUTS, DNN_NR_OUTPUTS)                  \
    template <>                                                            \
    struct _RNGOprInvoker<DNN_NR_INPUTS, DNN_NR_OUTPUTS> {                 \
        template <typename Opr>                                            \
        static void exec(                                                  \
                Opr* dnn_op, const SmallVector<TensorPtr>& inputs,         \
                const SmallVector<TensorPtr>& outputs) {                   \
            size_t wk_size = 0;                                            \
            wk_size = dnn_op->get_workspace_in_bytes(                      \
                    _FOR_EACH_IN(->layout()) _FOR_EACH_OUT(->layout()));   \
            auto workspace = Blob::make(outputs[0]->comp_node(), wk_size); \
            megdnn::Workspace dnn_wk(workspace->storage().get(), wk_size); \
            dnn_op->exec(                                                  \
                    _FOR_EACH_IN(->dev_tensor().as_megdnn())               \
                            _FOR_EACH_OUT(->dev_tensor().as_megdnn()),     \
                    dnn_wk);                                               \
        }                                                                  \
    };

#define _INST_RNG_MAKER(MGB_NR_INPUTS)                                                \
    template <>                                                                       \
    struct _RNGOprMaker<MGB_NR_INPUTS> {                                              \
        template <typename Op>                                                        \
        static auto make(const VarNodeArray& inputs, const Op& rng) {                 \
            auto param = OpMeth<Op>::make_param(rng);                                 \
            OperatorNodeConfig config;                                                \
            if (rng.handle) {                                                         \
                config = {                                                            \
                        rng.make_name(), RNGDnnOpManager::get_comp_node(rng.handle)}; \
            } else {                                                                  \
                config = {rng.make_name()};                                           \
            }                                                                         \
            return OpMeth<Op>::OpNode::make(_FOR_EACH_IN() param, config);            \
        }                                                                             \
    };

#define _FOR_EACH_IN(subfix)
#define _FOR_EACH_OUT(subfix) outputs[0] subfix
_INST_RNG_INVOLKER(0, 1)
#undef _FOR_EACH_OUT
#undef _FOR_EACH_IN

#define _FOR_EACH_IN(subfix)  inputs[0] subfix,
#define _FOR_EACH_OUT(subfix) outputs[0] subfix
_INST_RNG_INVOLKER(1, 1)
#undef _FOR_EACH_OUT

#define _FOR_EACH_OUT(subfix) outputs[0] subfix, outputs[1] subfix
_INST_RNG_INVOLKER(1, 2)
_INST_RNG_MAKER(1)
#undef _FOR_EACH_OUT
#undef _FOR_EACH_IN

#define _FOR_EACH_IN(subfix)  inputs[0] subfix, inputs[1] subfix,
#define _FOR_EACH_OUT(subfix) outputs[0] subfix
_INST_RNG_INVOLKER(2, 1)
_INST_RNG_MAKER(2)
#undef _FOR_EACH_OUT
#undef _FOR_EACH_IN

#define _FOR_EACH_IN(subfix) \
    inputs[0] subfix, inputs[1] subfix, inputs[2] subfix, inputs[3] subfix,
#define _FOR_EACH_OUT(subfix) outputs[0] subfix, outputs[1] subfix
_INST_RNG_INVOLKER(4, 2)
_INST_RNG_MAKER(4)
#undef _FOR_EACH_OUT
#undef _FOR_EACH_IN

#define _FOR_EACH_IN(subfix)                                                \
    inputs[0] subfix, inputs[1] subfix, inputs[2] subfix, inputs[3] subfix, \
            inputs[4] subfix,
_INST_RNG_MAKER(5)
#undef _FOR_EACH_IN

#define _FOR_EACH_IN(subfix)                                                \
    inputs[0] subfix, inputs[1] subfix, inputs[2] subfix, inputs[3] subfix, \
            inputs[4] subfix, inputs[5] subfix,
_INST_RNG_MAKER(6)
#undef _FOR_EACH_IN

#define _FOR_EACH_IN(subfix)                                                \
    inputs[0] subfix, inputs[1] subfix, inputs[2] subfix, inputs[3] subfix, \
            inputs[4] subfix, inputs[5] subfix, inputs[6] subfix,
#define _FOR_EACH_OUT(subfix) \
    outputs[0] subfix, outputs[1] subfix, outputs[2] subfix, outputs[3] subfix
_INST_RNG_INVOLKER(7, 4)
_INST_RNG_MAKER(7)
#undef _FOR_EACH_OUT
#undef _FOR_EACH_IN

#undef _INST_RNG_INVOLKER
#undef _INST_RNG_MAKER

template <typename Op>
void exec(
        const OpDef& op, const SmallVector<TensorPtr>& inputs,
        const SmallVector<TensorPtr>& outputs,
        const SmallVector<TensorPtr>& workspace) {
    auto&& rng = op.cast_final_safe<Op>();

    auto dest = outputs[0];
    if (dest->layout().is_empty())
        return;
    auto cn = dest->comp_node();
    auto handle = rng.handle;
    if (!handle) {
        handle = RNGDnnOpManager::get_default_handle(cn);
    }

    // retrieve dnn_op from glob cache
    auto dnn_op_thread_safe =
            RNGDnnOpManager::inst().get_dnn_op<typename OpMeth<Op>::DnnOp>(
                    handle, reinterpret_cast<size_t>(op.dyn_typeinfo()), cn);
    auto initialized = std::get<0>(dnn_op_thread_safe);
    auto dnn_op = std::get<1>(dnn_op_thread_safe);
    if (initialized) {
        auto handle_seed = RNGDnnOpManager::get_seed(handle);
        mgb_assert(
                dnn_op->param().seed == handle_seed,
                "inconsistent rng seed: handle: %lu, dnn_op: %lu", handle_seed,
                dnn_op->param().seed);
    }
    dnn_op->param() = OpMeth<Op>::make_param(rng);
    _RNGOprInvoker<OpMeth<Op>::DnnOp::NR_INPUTS, OpMeth<Op>::DnnOp::NR_OUTPUTS>::exec(
            dnn_op, inputs, outputs);
}

template <typename Op>
SmallVector<LogicalTensorDesc> infer_output_attrs(
        const OpDef& op, const SmallVector<TensorPtr>& inputs) {
    LogicalTensorDesc dest;
    auto&& rng = op.cast_final_safe<Op>();
    auto handle = rng.handle;
    if (handle) {
        dest.comp_node = RNGDnnOpManager::get_comp_node(handle);
    } else {
        dest.comp_node = inputs[0]->comp_node();
    }
    constexpr bool rng_with_shape = OpMeth<Op>::DnnOp::NR_INPUTS == 0;
    if (!rng_with_shape) {
        for (int i = 0; i < inputs.size(); ++i) {
            mgb_assert(
                    inputs[i]->comp_node() == dest.comp_node,
                    "%s expects the device of inputs[%d] to be same as the device of "
                    "handle; "
                    "got %s and %s actually",
                    rng.dyn_typeinfo()->name, i,
                    inputs[i]->comp_node().to_string().c_str(),
                    dest.comp_node.to_string().c_str());
        }
    }
    dest.layout = _InferLayout<rng_with_shape>::do_infer(inputs[0], rng);
    return {dest};
}

template <>
SmallVector<LogicalTensorDesc> infer_output_attrs<ShuffleRNG>(
        const OpDef& op, const SmallVector<TensorPtr>& inputs) {
    SmallVector<LogicalTensorDesc> dests(2);
    auto&& rng = op.cast_final_safe<ShuffleRNG>();
    auto handle = rng.handle;
    if (handle) {
        dests[0].comp_node = RNGDnnOpManager::get_comp_node(handle);
        dests[1].comp_node = RNGDnnOpManager::get_comp_node(handle);
    } else {
        dests[0].comp_node = inputs[0]->comp_node();
        dests[1].comp_node = inputs[0]->comp_node();
    }
    dests[0].layout = TensorLayout(inputs[0]->layout());
    dests[0].layout.dtype = inputs[0]->layout().dtype;
    dests[1].layout =
            TensorLayout(TensorShape({inputs[0]->layout()[0]}), dtype::Int32());
    return dests;
}

template <>
SmallVector<LogicalTensorDesc> infer_output_attrs<Dropout>(
        const OpDef& op, const SmallVector<TensorPtr>& inputs) {
    SmallVector<LogicalTensorDesc> dests(2);
    auto&& cn = inputs[0]->comp_node();

    dests[0].comp_node = cn;
    dests[0].layout = TensorLayout(inputs[0]->layout());
    dests[0].layout.dtype = inputs[0]->layout().dtype;

    auto get_mask_size = [&]() -> size_t {
        auto dnn_handle = MegDNNHandle::get(CompNodeEnv::from_comp_node(cn)).handle();
        return dnn_handle->create_operator<megdnn::Dropout>()->get_mask_size_in_bytes(
                inputs[0]->layout());
    };
    dests[1].comp_node = cn;
    dests[1].layout = TensorLayout(TensorShape({get_mask_size()}), dtype::Byte());
    return dests;
}

template <typename Op>
std::tuple<SmallVector<LogicalTensorDesc>, bool> _infer_output_attrs(
        const OpDef& op, const SmallVector<TensorLayout>& inputs, const CompNode cn){};

template <>
std::tuple<SmallVector<LogicalTensorDesc>, bool> _infer_output_attrs<MultiHeadAttn>(
        const OpDef& op, const SmallVector<TensorLayout>& inputs, const CompNode cn) {
    bool success = inputs[0].ndim != 0;

    SmallVector<LogicalTensorDesc> dests(4);

    // retrieve dnn_op from glob cache
    auto&& rng = op.cast_final_safe<MultiHeadAttn>();
    auto handle = rng.handle;
    if (!handle) {
        handle = RNGDnnOpManager::get_default_handle(cn);
    }
    auto dnn_op_thread_safe = RNGDnnOpManager::inst().get_dnn_op<megdnn::MultiHeadAttn>(
            handle, reinterpret_cast<size_t>(op.dyn_typeinfo()), cn);
    auto dnn_op = std::get<1>(dnn_op_thread_safe);
    dnn_op->param() = OpMeth<MultiHeadAttn>::make_param(rng);

    TensorLayout out, attn_weight, mask_layout, othr_layout;
    dnn_op->deduce_layout(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
            out, attn_weight, mask_layout, othr_layout);

    dests[0].comp_node = cn;
    dests[0].layout = out;
    dests[0].layout.dtype = inputs[0].dtype;
    dests[1].comp_node = cn;
    dests[1].layout = attn_weight;
    if (success) {
        dests[2].comp_node = cn;
        dests[2].layout = mask_layout;
        dests[3].comp_node = cn;
        dests[3].layout = othr_layout;
    } else {
        dests[2].comp_node = cn;
        dests[2].layout = TensorLayout(dtype::Byte());
        dests[3].comp_node = cn;
        dests[3].layout = TensorLayout(inputs[0].dtype);
    }

    return {dests, success};
}

template <>
SmallVector<LogicalTensorDesc> infer_output_attrs<MultiHeadAttn>(
        const OpDef& op, const SmallVector<TensorPtr>& inputs) {
    using InputType = opr::MultiHeadAttn::Param::TensorCombinationType;
    auto&& cn = inputs[0]->comp_node();
    auto input_type = op.cast_final_safe<MultiHeadAttn>().tensor_combination_type;

    std::tuple<SmallVector<LogicalTensorDesc>, bool> ret;
    TensorLayout empty_layout;
    if (input_type == InputType::NONE)
        ret = _infer_output_attrs<MultiHeadAttn>(
                op,
                {inputs[0]->layout(), inputs[1]->layout(), inputs[2]->layout(),
                 inputs[3]->layout(), empty_layout, empty_layout, empty_layout},
                cn);
    else if (input_type == InputType::ONLY_MASK)
        ret = _infer_output_attrs<MultiHeadAttn>(
                op,
                {inputs[0]->layout(), inputs[1]->layout(), inputs[2]->layout(),
                 inputs[3]->layout(), inputs[4]->layout(), empty_layout, empty_layout},
                cn);
    else if (input_type == InputType::ONLY_BIASKV)
        ret = _infer_output_attrs<MultiHeadAttn>(
                op,
                {inputs[0]->layout(), inputs[1]->layout(), inputs[2]->layout(),
                 inputs[3]->layout(), empty_layout, inputs[4]->layout(),
                 inputs[5]->layout()},
                cn);
    else
        ret = _infer_output_attrs<MultiHeadAttn>(
                op,
                {inputs[0]->layout(), inputs[1]->layout(), inputs[2]->layout(),
                 inputs[3]->layout(), inputs[4]->layout(), inputs[5]->layout(),
                 inputs[6]->layout()},
                cn);

    return std::get<0>(ret);
}

template <typename Op>
SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    SmallVector<TensorPtr> outputs;
    SmallVector<LogicalTensorDesc> desc = infer_output_attrs<Op>(def, inputs);
    for (auto&& i : desc) {
        outputs.push_back(Tensor::make(i.layout, i.comp_node));
    }
    exec<Op>(def, inputs, outputs, {});
    return outputs;
}

template <>
SmallVector<TensorPtr> apply_on_physical_tensor<MultiHeadAttn>(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    using InputType = opr::MultiHeadAttn::Param::TensorCombinationType;
    SmallVector<TensorPtr> outputs;
    SmallVector<LogicalTensorDesc> desc =
            infer_output_attrs<MultiHeadAttn>(def, inputs);
    for (auto&& i : desc) {
        outputs.push_back(Tensor::make(i.layout, i.comp_node));
    }

    auto&& rng = def.cast_final_safe<MultiHeadAttn>();
    auto dest = outputs[0];
    if (dest->layout().is_empty())
        return outputs;
    auto cn = dest->comp_node();
    auto handle = rng.handle;
    if (!handle) {
        handle = RNGDnnOpManager::get_default_handle(cn);
    }

    // retrieve dnn_op from glob cache
    auto dnn_op_thread_safe =
            RNGDnnOpManager::inst().get_dnn_op<typename OpMeth<MultiHeadAttn>::DnnOp>(
                    handle, reinterpret_cast<size_t>(def.dyn_typeinfo()), cn);
    auto initialized = std::get<0>(dnn_op_thread_safe);
    auto dnn_op = std::get<1>(dnn_op_thread_safe);
    if (initialized) {
        auto handle_seed = RNGDnnOpManager::get_seed(handle);
        mgb_assert(
                dnn_op->param().seed == handle_seed,
                "inconsistent rng seed: handle: %lu, dnn_op: %lu", handle_seed,
                dnn_op->param().seed);
    }
    dnn_op->param() = OpMeth<MultiHeadAttn>::make_param(rng);

    auto input_type = rng.tensor_combination_type;
    std::shared_ptr<Tensor> empty_dnn(nullptr);
    size_t wk_size = 0;
    TensorLayout empty_layout;
    megdnn::TensorND empty_tensor;

    if (input_type == InputType::ALL) {
        wk_size = dnn_op->get_workspace_in_bytes(
                inputs[0]->layout(), inputs[1]->layout(), inputs[2]->layout(),
                inputs[3]->layout(), inputs[4]->layout(), inputs[5]->layout(),
                inputs[6]->layout(), outputs[0]->layout(), outputs[1]->layout(),
                outputs[2]->layout(), outputs[3]->layout());
        auto workspace = Blob::make(outputs[0]->comp_node(), wk_size);
        megdnn::Workspace dnn_wk(workspace->storage().get(), wk_size);
        dnn_op->exec(
                inputs[0]->dev_tensor().as_megdnn(),
                inputs[1]->dev_tensor().as_megdnn(),
                inputs[2]->dev_tensor().as_megdnn(),
                inputs[3]->dev_tensor().as_megdnn(),
                inputs[4]->dev_tensor().as_megdnn(),
                inputs[5]->dev_tensor().as_megdnn(),
                inputs[6]->dev_tensor().as_megdnn(),
                outputs[0]->dev_tensor().as_megdnn(),
                outputs[1]->dev_tensor().as_megdnn(),
                outputs[2]->dev_tensor().as_megdnn(),
                outputs[3]->dev_tensor().as_megdnn(), dnn_wk);
    } else if (input_type == InputType::ONLY_MASK) {
        wk_size = dnn_op->get_workspace_in_bytes(
                inputs[0]->layout(), inputs[1]->layout(), inputs[2]->layout(),
                inputs[3]->layout(), inputs[4]->layout(), empty_layout, empty_layout,
                outputs[0]->layout(), outputs[1]->layout(), outputs[2]->layout(),
                outputs[3]->layout());
        auto workspace = Blob::make(outputs[0]->comp_node(), wk_size);
        megdnn::Workspace dnn_wk(workspace->storage().get(), wk_size);
        dnn_op->exec(
                inputs[0]->dev_tensor().as_megdnn(),
                inputs[1]->dev_tensor().as_megdnn(),
                inputs[2]->dev_tensor().as_megdnn(),
                inputs[3]->dev_tensor().as_megdnn(),
                inputs[4]->dev_tensor().as_megdnn(), empty_tensor, empty_tensor,
                outputs[0]->dev_tensor().as_megdnn(),
                outputs[1]->dev_tensor().as_megdnn(),
                outputs[2]->dev_tensor().as_megdnn(),
                outputs[3]->dev_tensor().as_megdnn(), dnn_wk);
    } else if (input_type == InputType::ONLY_BIASKV) {
        wk_size = dnn_op->get_workspace_in_bytes(
                inputs[0]->layout(), inputs[1]->layout(), inputs[2]->layout(),
                inputs[3]->layout(), empty_layout, inputs[4]->layout(),
                inputs[5]->layout(), outputs[0]->layout(), outputs[1]->layout(),
                outputs[2]->layout(), outputs[3]->layout());
        auto workspace = Blob::make(outputs[0]->comp_node(), wk_size);
        megdnn::Workspace dnn_wk(workspace->storage().get(), wk_size);
        dnn_op->exec(
                inputs[0]->dev_tensor().as_megdnn(),
                inputs[1]->dev_tensor().as_megdnn(),
                inputs[2]->dev_tensor().as_megdnn(),
                inputs[3]->dev_tensor().as_megdnn(), empty_tensor,
                inputs[4]->dev_tensor().as_megdnn(),
                inputs[5]->dev_tensor().as_megdnn(),
                outputs[0]->dev_tensor().as_megdnn(),
                outputs[1]->dev_tensor().as_megdnn(),
                outputs[2]->dev_tensor().as_megdnn(),
                outputs[3]->dev_tensor().as_megdnn(), dnn_wk);
    } else {
        wk_size = dnn_op->get_workspace_in_bytes(
                inputs[0]->layout(), inputs[1]->layout(), inputs[2]->layout(),
                inputs[3]->layout(), empty_layout, empty_layout, empty_layout,
                outputs[0]->layout(), outputs[1]->layout(), outputs[2]->layout(),
                outputs[3]->layout());
        auto workspace = Blob::make(outputs[0]->comp_node(), wk_size);
        megdnn::Workspace dnn_wk(workspace->storage().get(), wk_size);
        dnn_op->exec(
                inputs[0]->dev_tensor().as_megdnn(),
                inputs[1]->dev_tensor().as_megdnn(),
                inputs[2]->dev_tensor().as_megdnn(),
                inputs[3]->dev_tensor().as_megdnn(), empty_tensor, empty_tensor,
                empty_tensor, outputs[0]->dev_tensor().as_megdnn(),
                outputs[1]->dev_tensor().as_megdnn(),
                outputs[2]->dev_tensor().as_megdnn(),
                outputs[3]->dev_tensor().as_megdnn(), dnn_wk);
    }
    return outputs;
}

template <typename Op, typename Output>
Output apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    size_t nr_inp = inputs.size();
    constexpr size_t dnn_nr_inp = OpMeth<Op>::DnnOp::NR_INPUTS;
    auto&& rng = def.cast_final_safe<Op>();
    if (dnn_nr_inp == 0) {
        mgb_assert(
                nr_inp == 1, "%s expects 1 inputs; got %lu actually",
                rng.dyn_typeinfo()->name, nr_inp);
    }
    constexpr size_t mgb_nr_inp = dnn_nr_inp + !dnn_nr_inp;
    return _RNGOprMaker<mgb_nr_inp>::make(inputs, rng);
}

template <>
SymbolVarArray apply_on_var_node<MultiHeadAttn, SymbolVarArray>(
        const OpDef& def, const VarNodeArray& inputs) {
    auto&& rng = def.cast_final_safe<MultiHeadAttn>();
    using InputType = opr::MultiHeadAttn::Param::TensorCombinationType;
    auto input_type = rng.tensor_combination_type;
    if (input_type == InputType::ALL) {
        return _RNGOprMaker<7>::make(inputs, rng);
    } else if (input_type == InputType::ONLY_BIASKV) {
        return _RNGOprMaker<6>::make(inputs, rng);
    } else if (input_type == InputType::ONLY_MASK) {
        return _RNGOprMaker<5>::make(inputs, rng);
    } else {
        return _RNGOprMaker<4>::make(inputs, rng);
    }
}

template <typename Op>
std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    bool success = inputs[0].layout.ndim != 0;
    LogicalTensorDesc dest;
    auto&& xxx_rng_def = def.cast_final_safe<Op>();
    size_t nr_inp = inputs.size();
    constexpr bool rng_with_shape = OpMeth<Op>::DnnOp::NR_INPUTS == 0;
    if (rng_with_shape) {
        mgb_assert(
                nr_inp == 1, "%s expects 1 inputs; got %lu actually",
                xxx_rng_def.dyn_typeinfo()->name, nr_inp);
    }
    dest.comp_node = inputs[0].comp_node;
    if (success) {
        dest.layout = _InferLayout<rng_with_shape>::do_infer(inputs[0], xxx_rng_def);
    } else {
        dest.layout = TensorLayout(inputs[0].layout.dtype);
    }
    return {{dest}, inputs[0].layout.ndim != 0};
}

template <>
std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible<
        ShuffleRNG>(const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    bool success = inputs[0].layout.ndim != 0;

    SmallVector<LogicalTensorDesc> dests(2);
    dests[0].comp_node = inputs[0].comp_node;
    dests[0].layout = TensorLayout(inputs[0].layout);
    dests[0].layout.dtype = inputs[0].layout.dtype;
    dests[1].comp_node = inputs[0].comp_node;
    if (success) {
        dests[1].layout =
                TensorLayout(TensorShape({inputs[0].layout.shape[0]}), dtype::Int32());
    } else {
        dests[1].layout = TensorLayout(dtype::Int32());
    }
    return {dests, success};
}

template <>
std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible<Dropout>(
        const OpDef& op, const SmallVector<LogicalTensorDesc>& inputs) {
    bool success = inputs[0].layout.ndim != 0;

    SmallVector<LogicalTensorDesc> dests(2);
    auto cn = inputs[0].comp_node;
    dests[0].comp_node = cn;
    dests[0].layout = TensorLayout(inputs[0].layout);
    dests[0].layout.dtype = inputs[0].layout.dtype;

    auto get_mask_size = [&]() -> size_t {
        auto dnn_handle = MegDNNHandle::get(CompNodeEnv::from_comp_node(cn)).handle();
        return dnn_handle->create_operator<megdnn::Dropout>()->get_mask_size_in_bytes(
                inputs[0].layout);
    };
    dests[1].comp_node = cn;
    if (success) {
        dests[1].layout = TensorLayout(TensorShape({get_mask_size()}), dtype::Byte());
    } else {
        dests[1].layout = TensorLayout(dtype::Byte());
    }

    return {dests, success};
}

template <>
std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible<
        MultiHeadAttn>(const OpDef& op, const SmallVector<LogicalTensorDesc>& inputs) {
    using InputType = opr::MultiHeadAttn::Param::TensorCombinationType;
    auto&& cn = inputs[0].comp_node;
    auto input_type = op.cast_final_safe<MultiHeadAttn>().tensor_combination_type;

    std::tuple<SmallVector<LogicalTensorDesc>, bool> ret;
    TensorLayout empty_layout;
    if (input_type == InputType::NONE)
        ret = _infer_output_attrs<MultiHeadAttn>(
                op,
                {inputs[0].layout, inputs[1].layout, inputs[2].layout, inputs[3].layout,
                 empty_layout, empty_layout, empty_layout},
                cn);
    else if (input_type == InputType::ONLY_MASK)
        ret = _infer_output_attrs<MultiHeadAttn>(
                op,
                {inputs[0].layout, inputs[1].layout, inputs[2].layout, inputs[3].layout,
                 inputs[4].layout, empty_layout, empty_layout},
                cn);
    else if (input_type == InputType::ONLY_BIASKV)
        ret = _infer_output_attrs<MultiHeadAttn>(
                op,
                {inputs[0].layout, inputs[1].layout, inputs[2].layout, inputs[3].layout,
                 empty_layout, inputs[4].layout, inputs[5].layout},
                cn);
    else
        ret = _infer_output_attrs<MultiHeadAttn>(
                op,
                {inputs[0].layout, inputs[1].layout, inputs[2].layout, inputs[3].layout,
                 inputs[4].layout, inputs[5].layout, inputs[6].layout},
                cn);

    return ret;
}

template <typename Op>
SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    return layout_checker;
}

}  // anonymous namespace

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

CompNode get_rng_handle_compnode(Handle handle) {
    return RNGDnnOpManager::get_comp_node(handle);
}

#define REG_RNG_OP(NAME, Output)                                            \
    namespace {                                                             \
    OP_TRAIT_REG(NAME, NAME, OpMeth<NAME>::OpNode)                          \
            .apply_on_var_node(apply_on_var_node<NAME, Output>)             \
            .apply_on_physical_tensor(apply_on_physical_tensor<NAME>)       \
            .infer_output_attrs_fallible(infer_output_attrs_fallible<NAME>) \
            .get_input_layout_constraint(get_input_layout_constraint<NAME>) \
            .fallback();                                                    \
    }

REG_RNG_OP(UniformRNG, SymbolVar)
REG_RNG_OP(GaussianRNG, SymbolVar)
REG_RNG_OP(GammaRNG, SymbolVar)
REG_RNG_OP(PermutationRNG, SymbolVar)
REG_RNG_OP(PoissonRNG, SymbolVar)
REG_RNG_OP(BetaRNG, SymbolVar)
REG_RNG_OP(ShuffleRNG, SymbolVarArray)
REG_RNG_OP(ExponentialRNG, SymbolVar)
REG_RNG_OP(Dropout, SymbolVarArray)
REG_RNG_OP(MultiHeadAttn, SymbolVarArray)
#undef REG_RNG_OP

}  // namespace mgb::imperative::rng

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
