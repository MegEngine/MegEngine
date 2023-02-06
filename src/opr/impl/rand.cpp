#include "megbrain/opr/rand.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/utility.h"

#include "./internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;
using namespace intl;

template <typename MegDNNOpr>
RNGOprBase<MegDNNOpr>::RNGOprBase(
        const OperatorNodeBaseCtorParam& opr, const Param& param)
        : Super(opr), m_param(param) {}

template <class MegDNNOpr>
UniqPtrWithCN<MegDNNOpr> RNGOprBase<MegDNNOpr>::create_megdnn_opr() {
    auto opr = intl::create_megdnn_opr<MegDNNOpr>(comp_node());
    opr->param() = param();
    return opr;
}

template <typename MegDNNOpr>
void RNGOprBase<MegDNNOpr>::ensure_megdnn_opr() {
    if (!m_dnn_opr || m_dnn_opr.comp_node() != comp_node()) {
        // activate comp_node for curandCreateGenerator in create_megdnn_opr
        comp_node().activate();
        m_dnn_opr = create_megdnn_opr();
    }
}

/* ================= RNG with shape =================  */
#define _INST_RNG_OPR_WITH_SHAPE(RNGOpr, name)                                        \
    MGB_DYN_TYPE_OBJ_FINAL_IMPL(RNGOpr);                                              \
    cg::OperatorNodeBase::NodeProp* RNGOpr::do_make_node_prop() const {               \
        auto prop = Super::do_make_node_prop();                                       \
        prop->add_flag(NodeProp::Flag::IMPURE_FUNC);                                  \
        prop->reset_dep_type(input(), {NodeProp::DepType::HOST_VALUE});               \
        for (auto i : input()) {                                                      \
            prop->add_dep_type_existing_var(i, NodeProp::DepType::VALUE_ALLOW_EMPTY); \
        }                                                                             \
        return prop;                                                                  \
    }                                                                                 \
    RNGOpr::RNGOpr(                                                                   \
            VarNode* shape, const Param& param, const OperatorNodeConfig& config)     \
            : Super({shape->owner_graph(), config, (name), {shape}}, param) {         \
        DType dtype = DType::from_enum(param.dtype);                                  \
        add_input({shape});                                                           \
        add_output(None)->dtype(dtype).add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);    \
        cg::add_workspace_output(this);                                               \
        add_equivalence_component<ScalarHash<void*>>(this);                           \
    }                                                                                 \
    SymbolVar RNGOpr::make(                                                           \
            SymbolVar shape, const Param& param, const OperatorNodeConfig& config) {  \
        return shape.insert_single_output_opr<RNGOpr>(shape.node(), param, config);   \
    }                                                                                 \
    void RNGOpr::init_output_static_infer_desc() {                                    \
        using namespace cg::static_infer;                                             \
        auto&& mgr = owner_graph()->static_infer_manager();                           \
        auto infer_out = [](TensorShape& dest, const InpVal& inp) {                   \
            cg::copy_tensor_value_to_shape(dest, inp.val.at(0).value());              \
            return true;                                                              \
        };                                                                            \
        auto infer_wk = [this](TensorShape& dest, const InpVal& inp) {                \
            ensure_megdnn_opr();                                                      \
            dest.ndim = 1;                                                            \
            dest.shape[0] = m_dnn_opr->get_workspace_in_bytes(                        \
                    {inp.val.at(0).shape(), output(0)->dtype()});                     \
            return true;                                                              \
        };                                                                            \
        mgr.register_shape_infer(                                                     \
                output(0),                                                            \
                {SourceType::DEP, {{input(0), DepType::VALUE}}, infer_out});          \
        mgr.register_shape_infer(                                                     \
                output(1),                                                            \
                {SourceType::DEP, {{output(0), DepType::SHAPE}}, infer_wk});          \
    }                                                                                 \
    void RNGOpr::scn_do_execute() {                                                   \
        auto&& ret = output(0);                                                       \
        if (ret->layout().is_empty()) {                                               \
            mgb_assert(ret->dev_tensor().empty());                                    \
            return;                                                                   \
        }                                                                             \
        m_dnn_opr->exec(                                                              \
                ret->dev_tensor().as_megdnn(),                                        \
                get_megdnn_workspace_from_var(output(1)));                            \
    }

_INST_RNG_OPR_WITH_SHAPE(UniformRNG, "uniform_rng")
_INST_RNG_OPR_WITH_SHAPE(GaussianRNG, "gaussian_rng")
_INST_RNG_OPR_WITH_SHAPE(PermutationRNG, "permutation_rng")
#undef _INST_RNG_OPR_WITH_SHAPE

/* ================= RNG with input =================  */
#define _AS_MEGDNN(idx) input((idx))->dev_tensor().as_megdnn()
#define _INFER_WK_DEPS(idx) \
    { input((idx)), DepType::SHAPE }
#define _INFER_WK_ARGS(idx) \
    { inp.val.at((idx)).shape(), input((idx))->dtype() }

#define _INST_RNG_OPR_WITH_INPUT(RNGOpr, name)                                         \
    MGB_DYN_TYPE_OBJ_FINAL_IMPL(RNGOpr);                                               \
    RNGOpr::RNGOpr(                                                                    \
            _INPUTS(VarNode*, ), const Param& param, const OperatorNodeConfig& config) \
            : Super({i0->owner_graph(), config, (name), {_INPUTS(, )}}, param) {       \
        add_input({_INPUTS(, )});                                                      \
        add_output(None)                                                               \
                ->dtype(i0->dtype())                                                   \
                .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);                           \
        cg::add_workspace_output(this);                                                \
        add_equivalence_component<ScalarHash<void*>>(this);                            \
    }                                                                                  \
    SymbolVar RNGOpr::make(                                                            \
            _INPUTS(SymbolVar, ), const Param& param,                                  \
            const OperatorNodeConfig& config) {                                        \
        return i0.insert_single_output_opr<RNGOpr>(_INPUTS(, .node()), param, config); \
    }                                                                                  \
    void RNGOpr::init_output_static_infer_desc() {                                     \
        using namespace cg::static_infer;                                              \
        auto&& mgr = owner_graph()->static_infer_manager();                            \
        auto infer_wk = [this](TensorShape& dest, const InpVal& inp) {                 \
            ensure_megdnn_opr();                                                       \
            dest.ndim = 1;                                                             \
            dest.shape[0] = m_dnn_opr->get_workspace_in_bytes(                         \
                    _FOR_EACH(_INFER_WK_ARGS),                                         \
                    {output(0)->shape(), output(0)->dtype()});                         \
            return true;                                                               \
        };                                                                             \
        mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(0)));  \
        mgr.register_shape_infer(                                                      \
                output(1), {SourceType::DEP, {_FOR_EACH(_INFER_WK_DEPS)}, infer_wk});  \
    }                                                                                  \
    void RNGOpr::add_input_layout_constraint() {                                       \
        for (auto i : input())                                                         \
            i->add_layout_constraint_contiguous();                                     \
    };                                                                                 \
    void RNGOpr::scn_do_execute() {                                                    \
        auto&& ret = output(0);                                                        \
        if (ret->layout().is_empty()) {                                                \
            mgb_assert(ret->dev_tensor().empty());                                     \
            return;                                                                    \
        }                                                                              \
        m_dnn_opr->exec(                                                               \
                _FOR_EACH(_AS_MEGDNN), output(0)->dev_tensor().as_megdnn(),            \
                get_megdnn_workspace_from_var(output(1)));                             \
    }                                                                                  \
    cg::OperatorNodeBase::NodeProp* RNGOpr::do_make_node_prop() const {                \
        auto prop = Super::do_make_node_prop();                                        \
        prop->add_flag(NodeProp::Flag::IMPURE_FUNC);                                   \
        for (auto i : input()) {                                                       \
            prop->add_dep_type_existing_var(i, NodeProp::DepType::VALUE_ALLOW_EMPTY);  \
        }                                                                              \
        return prop;                                                                   \
    }

/* ================= 1 input =================  */
#define _INPUTS(prefix, subfix) prefix i0 subfix
#define _FOR_EACH(cb)           cb(0)
_INST_RNG_OPR_WITH_INPUT(PoissonRNG, "poisson_rng")
#undef _INPUTS
#undef _FOR_EACH

/* ================= 2 input =================  */
#define _INPUTS(prefix, subfix) prefix i0 subfix, prefix i1 subfix
#define _FOR_EACH(cb)           cb(0), cb(1)
_INST_RNG_OPR_WITH_INPUT(BetaRNG, "beta_rng")
_INST_RNG_OPR_WITH_INPUT(GammaRNG, "gamma_rng")
#undef _INPUTS
#undef _FOR_EACH

#undef _AS_MEGDNN
#undef _INFER_WK_DEPS
#undef _INFER_WK_ARGS
#undef _INST_RNG_OPR_WITH_INPUT

#define IMPL(_cls)                              \
    MGB_IMPL_OPR_GRAD(_cls) {                   \
        MGB_MARK_USED_VAR(out_grad);            \
        return InvalidGrad::make(opr, wrt_idx); \
    }

namespace mgb {
namespace opr {
namespace intl {
template class RNGOprBase<::megdnn::GaussianRNG>;
template class RNGOprBase<::megdnn::UniformRNG>;
template class RNGOprBase<::megdnn::GammaRNG>;
template class RNGOprBase<::megdnn::PermutationRNG>;
template class RNGOprBase<::megdnn::BetaRNG>;
template class RNGOprBase<::megdnn::PoissonRNG>;
template class RNGOprBase<::megdnn::ShuffleRNGForward>;
template class RNGOprBase<::megdnn::ShuffleRNGBackward>;
template class RNGOprBase<::megdnn::DropoutForward>;
template class RNGOprBase<::megdnn::DropoutBackward>;
template class RNGOprBase<::megdnn::MultiHeadAttnForward>;
template class RNGOprBase<::megdnn::MultiHeadAttnBackward>;
#if MGB_ENABLE_GRAD
IMPL(GaussianRNG);
IMPL(UniformRNG);
IMPL(GammaRNG);
IMPL(PoissonRNG);
IMPL(PermutationRNG);
IMPL(BetaRNG);
#endif
}  // namespace intl
}  // namespace opr
}  // namespace mgb

/* ================= ShuffleRNGForward =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ShuffleRNGForward);

ShuffleRNGForward::ShuffleRNGForward(
        VarNode* data, const Param& param, const OperatorNodeConfig& config)
        : Super({data->owner_graph(), config, "shuffle_rng", {data}}, param) {
    add_input({data});
    add_output(None)->dtype(data->dtype()).add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    add_output(None)->dtype(dtype::Int32{}).add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    cg::add_workspace_output(this);
    add_equivalence_component<ScalarHash<void*>>(this);
}

SymbolVarArray ShuffleRNGForward::make(
        SymbolVar in_tensor, const Param& param, const OperatorNodeConfig& config) {
    auto node = in_tensor.node()->owner_graph()->insert_opr(
            std::make_unique<ShuffleRNGForward>(in_tensor.node(), param, config));
    mgb_assert(node->output().size() == 3);
    return {node->output(0), node->output(1)};
}

void ShuffleRNGForward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();

    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(0)));

    auto infer_oshp1 = [this](TensorShape& dest, const InpVal& iv) {
        TensorLayout o0, o1;
        m_dnn_opr->deduce_layout({iv.val[0].shape(), input(0)->dtype()}, o0, o1);
        dest = o1;
        return true;
    };
    mgr.register_shape_infer(
            output(1), {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_oshp1});

    auto infer_wk = [this](TensorShape& dest, const InpVal& inp) {
        ensure_megdnn_opr();
        dest.ndim = 1;
        dest.shape[0] = m_dnn_opr->get_workspace_in_bytes(
                {inp.val[0].shape(), input(0)->dtype()},
                {output(0)->shape(), output(0)->dtype()},
                {output(1)->shape(), output(1)->dtype()});
        return true;
    };
    mgr.register_shape_infer(
            output(2), {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_wk});
}

void ShuffleRNGForward::add_input_layout_constraint() {
    input(0)->add_layout_constraint_contiguous();
};

void ShuffleRNGForward::scn_do_execute() {
    auto&& ret = output(0);
    if (ret->layout().is_empty()) {
        mgb_assert(ret->dev_tensor().empty());
        return;
    }
    m_dnn_opr->exec(
            input(0)->dev_tensor().as_megdnn(), output(0)->dev_tensor().as_megdnn(),
            output(1)->dev_tensor().as_megdnn(),
            get_megdnn_workspace_from_var(output(2)));
}

cg::OperatorNodeBase::NodeProp* ShuffleRNGForward::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    prop->add_flag(NodeProp::Flag::IMPURE_FUNC);
    for (auto i : input()) {
        prop->add_dep_type_existing_var(i, NodeProp::DepType::VALUE_ALLOW_EMPTY);
    }
    return prop;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(ShuffleRNGForward) {
    mgb_assert(out_grad.size() == 3 && wrt_idx == 0 && !out_grad[2]);
    if (!out_grad[0])
        return nullptr;
    return ShuffleRNGBackward::make(out_grad[0], opr.output(1), opr.input(0)).node();
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ShuffleRNGBackward);
MEGDNN_OPR_INIT3(ShuffleRNGBackward, "shuffle_rng_bwd", 2, true)

/* ================= DropoutForward =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(DropoutForward);

DropoutForward::DropoutForward(
        VarNode* inp, const Param& param, const OperatorNodeConfig& config)
        : Super({inp->owner_graph(), config, "dropout", {inp}}, param) {
    add_input({inp});
    add_output(None)->dtype(inp->dtype()).add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    add_output(None)->dtype(dtype::Byte()).add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    cg::add_workspace_output(this);
    add_equivalence_component<ScalarHash<void*>>(this);
}

SymbolVarArray DropoutForward::make(
        SymbolVar inp, const Param& param, const OperatorNodeConfig& config) {
    auto node = inp.node()->owner_graph()->insert_opr(
            std::make_unique<DropoutForward>(inp.node(), param, config));
    mgb_assert(node->output().size() == 3);
    return {node->output(0), node->output(1)};
}

void DropoutForward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(0)));

    auto infer_mask = [this](TensorShape& dest, const InpVal& iv) {
        ensure_megdnn_opr();
        dest.ndim = 1;
        dest.shape[0] = m_dnn_opr->get_mask_size_in_bytes(
                {iv.val[0].shape(), input(0)->dtype()});
        return true;
    };
    mgr.register_shape_infer(
            output(1), {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_mask});

    auto infer_wk = [this](TensorShape& dest, const InpVal& inp) {
        ensure_megdnn_opr();
        dest.ndim = 1;
        dest.shape[0] = m_dnn_opr->get_workspace_in_bytes(
                {inp.val[0].shape(), input(0)->dtype()},
                {output(0)->shape(), output(0)->dtype()},
                {output(1)->shape(), output(1)->dtype()});
        return true;
    };
    mgr.register_shape_infer(
            output(2), {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_wk});
}

void DropoutForward::add_input_layout_constraint() {
    input(0)->add_layout_constraint_contiguous();
};

void DropoutForward::scn_do_execute() {
    auto&& ret = output(0);
    if (ret->layout().is_empty()) {
        mgb_assert(ret->dev_tensor().empty());
        return;
    }
    m_dnn_opr->exec(
            input(0)->dev_tensor().as_megdnn(), output(0)->dev_tensor().as_megdnn(),
            output(1)->dev_tensor().as_megdnn(),
            get_megdnn_workspace_from_var(output(2)));
}

cg::OperatorNodeBase::NodeProp* DropoutForward::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    prop->add_flag(NodeProp::Flag::IMPURE_FUNC);
    for (auto i : input()) {
        prop->add_dep_type_existing_var(i, NodeProp::DepType::VALUE_ALLOW_EMPTY);
    }
    return prop;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(DropoutForward) {
    SymbolVar grad = DropoutBackward::make(out_grad[0], opr.output(1), opr.param());
    VarNodeArray ret;
    ret.push_back(grad.node());
    return ret;
}
#endif

/* ==================== DropoutBackward ==================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(DropoutBackward);

DropoutBackward::DropoutBackward(
        VarNode* doup, VarNode* mask, const Param& param,
        const OperatorNodeConfig& config)
        : Super({doup->owner_graph(), config, "dropout_backward", {doup, mask}}, 0,
                true) {
    init_megdnn_opr(*this, param);
    add_input({doup, mask});
}

SymbolVar DropoutBackward::make(
        SymbolVar doup, SymbolVar mask, const Param& param,
        const OperatorNodeConfig& config) {
    return doup.insert_single_output_opr<DropoutBackward>(
            doup.node(), mask.node(), param, config);
}

void DropoutBackward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(0)));
    this->init_output_static_infer_desc_workspace(false);
}

void DropoutBackward::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
}

size_t DropoutBackward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    return megdnn_opr()->get_workspace_in_bytes(
            {input_shapes[0], input(0)->dtype(), input(0)->format()},
            {input_shapes[1], input(1)->dtype(), input(1)->format()},
            {output_shapes[0], output(0)->dtype(), output(0)->format()});
}

void DropoutBackward::scn_do_execute() {
    megdnn_opr()->exec(
            input(0)->dev_tensor().as_megdnn(), input(1)->dev_tensor().as_megdnn(),
            output(0)->dev_tensor().as_megdnn(), {});
}

/* ==================== MultiHeadAttnForward  ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(MultiHeadAttnForward);

MultiHeadAttnForward::MultiHeadAttnForward(
        VarNode* queries, VarNode* keys, VarNode* values, VarNode* wqkv,
        const Param& param, const OperatorNodeConfig& config)
        : Super{{queries->owner_graph(),
                 config,
                 "multi_head_attn",
                 {queries, keys, values, wqkv}},
                param} {
    add_input({queries, keys, values, wqkv});
    add_output(None)
            ->dtype(queries->dtype())
            .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    add_output(None)->dtype(dtype::Byte()).add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    cg::add_workspace_output(this);
    add_equivalence_component<ScalarHash<void*>>(this);
}

SymbolVarArray MultiHeadAttnForward::make(
        SymbolVar queries, SymbolVar keys, SymbolVar values, SymbolVar wqkv,
        const Param& param, const OperatorNodeConfig& config) {
    auto outs = queries.node()
                        ->owner_graph()
                        ->insert_opr(std::make_unique<MultiHeadAttnForward>(
                                queries.node(), keys.node(), values.node(), wqkv.node(),
                                param, config))
                        ->output();
    mgb_assert(outs.size() == 3);
    return {outs[0], outs[1]};
}

void MultiHeadAttnForward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(0)));

    auto infer_mask = [this](TensorShape& dest, const InpVal& iv) {
        ensure_megdnn_opr();
        dest.ndim = 1;
        dest.shape[0] = m_dnn_opr->get_reservespace_in_bytes(
                {iv.val[0].shape(), input(0)->dtype()},
                {iv.val[1].shape(), input(1)->dtype()},
                {iv.val[2].shape(), input(2)->dtype()},
                {iv.val[3].shape(), input(3)->dtype()}, {}, {});
        return true;
    };
    mgr.register_shape_infer(
            output(1), {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_mask});
}

void MultiHeadAttnForward::add_input_layout_constraint() {
    input(0)->add_layout_constraint_contiguous();
    input(1)->add_layout_constraint_contiguous();
    input(2)->add_layout_constraint_contiguous();
    input(3)->add_layout_constraint_contiguous();
};

void MultiHeadAttnForward::scn_do_execute() {
    auto&& ret = output(0);
    if (ret->layout().is_empty()) {
        mgb_assert(ret->dev_tensor().empty());
        return;
    }
    m_dnn_opr->exec(
            input(0)->dev_tensor().as_megdnn(), input(1)->dev_tensor().as_megdnn(),
            input(2)->dev_tensor().as_megdnn(), input(3)->dev_tensor().as_megdnn(),
            output(0)->dev_tensor().as_megdnn(), output(1)->dev_tensor().as_megdnn(),
            get_megdnn_workspace_from_var(output(2)));
}

cg::OperatorNodeBase::NodeProp* MultiHeadAttnForward::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    prop->add_flag(NodeProp::Flag::IMPURE_FUNC);
    for (auto i : input()) {
        prop->add_dep_type_existing_var(i, NodeProp::DepType::VALUE_ALLOW_EMPTY);
    }
    return prop;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(MultiHeadAttnForward) {
    MGB_MARK_USED_VAR(opr);
    MGB_MARK_USED_VAR(out_grad);
    SymbolVarArray grad;
    VarNodeArray ret;
    mgb_assert(wrt_idx < 5, "wrt_idx %zu is out of range", wrt_idx);
    grad = MultiHeadAttnBackward::make(
            out_grad[0], opr.input(0), opr.input(1), opr.input(2), opr.input(3),
            opr.output(1), opr.param());

    uint32_t nr_ret = 4;
    for (uint32_t i = 0; i < nr_ret; ++i) {
        ret.push_back(grad[i].node());
    }
    return ret;
}
#endif

/* ==================== MultiHeadAttnBackwardData ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(MultiHeadAttnBackward);

MultiHeadAttnBackward::MultiHeadAttnBackward(
        VarNode* diff, VarNode* queries, VarNode* keys, VarNode* values, VarNode* wqkv,
        VarNode* reserveSpace, const Param& param, const OperatorNodeConfig& config)
        : Super({queries->owner_graph(),
                 config,
                 "multi_head_attn_backward",
                 {diff, queries, keys, values, wqkv, reserveSpace}},
                0, true) {
    init_megdnn_opr(*this, param);
    add_input({diff, queries, keys, values, wqkv, reserveSpace});
}

SymbolVarArray MultiHeadAttnBackward::make(
        SymbolVar diff, SymbolVar queries, SymbolVar keys, SymbolVar values,
        SymbolVar wqkv, SymbolVar reserveSpace, const Param& param,
        const OperatorNodeConfig& config) {
    auto outs = queries.node()
                        ->owner_graph()
                        ->insert_opr(std::make_unique<MultiHeadAttnBackward>(
                                diff.node(), queries.node(), keys.node(), values.node(),
                                wqkv.node(), reserveSpace.node(), param, config))
                        ->output();
    mgb_assert(outs.size() == 5);
    return {outs[0], outs[1], outs[2], outs[3]};
}

void MultiHeadAttnBackward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(1)));
    mgr.register_shape_infer(output(1), ShapeInferDesc::make_identity(input(2)));
    mgr.register_shape_infer(output(2), ShapeInferDesc::make_identity(input(3)));
    mgr.register_shape_infer(output(3), ShapeInferDesc::make_identity(input(4)));

    this->init_output_static_infer_desc_workspace(false);
}

void MultiHeadAttnBackward::init_output_dtype() {
    output(0)->dtype(input(1)->dtype());
    output(1)->dtype(input(2)->dtype());
    output(2)->dtype(input(3)->dtype());
    output(3)->dtype(input(4)->dtype());
}

size_t MultiHeadAttnBackward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    MGB_MARK_USED_VAR(input_shapes);
    MGB_MARK_USED_VAR(output_shapes);

    return 0;
}

void MultiHeadAttnBackward::scn_do_execute() {
    megdnn_opr()->exec(
            input(0)->dev_tensor().as_megdnn(), input(1)->dev_tensor().as_megdnn(),
            input(2)->dev_tensor().as_megdnn(), input(3)->dev_tensor().as_megdnn(),
            input(4)->dev_tensor().as_megdnn(), input(5)->dev_tensor().as_megdnn(),
            output(0)->dev_tensor().as_megdnn(), output(1)->dev_tensor().as_megdnn(),
            output(2)->dev_tensor().as_megdnn(), output(3)->dev_tensor().as_megdnn(),
            {});
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
