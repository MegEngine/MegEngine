/**
 * \file src/gopt/impl/inference.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/gopt/inference.h"
#include "megbrain/gopt/gtrans.h"
#include "megbrain/gopt/basic_arith.h"
#include "megbrain/graph/event.h"
#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/utils/shared_set.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/misc.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/nn_int.h"

#include "megdnn/tensor_format.h"

#if MGB_ENABLE_TENSOR_RT
#include "megbrain/tensorrt/tensorrt_opr.h"
#endif

#include "megbrain/gopt/misc.h"

using namespace mgb;
using namespace gopt;

namespace {


template <typename SharedDeviceTensor, typename MultipleDeviceTensorHolder>
void param_merge(OptState& opt_state) {
    auto rewriter = opt_state.graph().make_rewriter();
    ThinHashMap<OperatorNodeBase*, size_t> opr2idx;
    std::vector<OperatorNodeBase*> all_oprs;
    typename MultipleDeviceTensorHolder::ValueArray all_values;

    auto cb_find_opr = [&](cg::OperatorNodeBase* opr) {
        if (opr->same_type<SharedDeviceTensor>()) {
            auto p = &opr->cast_final<SharedDeviceTensor>();
            // ShredD may be manu
            opr2idx[p] = all_values.size();
            all_values.push_back(p->dev_data());
            all_oprs.push_back(p);
        }
    };
    opt_state.graph().iter(cb_find_opr);
    SymbolVarArray new_vars;
    auto cb_replace = [&](cg::OperatorNodeBase* opr) {
        auto iter = opr2idx.find(opr);
        if (iter == opr2idx.end()) {
            rewriter.auto_replace_outputs(opr);
        } else {
            if (new_vars.empty()) {
                // new oprs must be created in iter callback; so we populate
                // new_vars lazily
                new_vars = MultipleDeviceTensorHolder::make(
                        *opt_state.graph().comp_graph(), std::move(all_values),
                        {ssprintf("merged%zu", all_values.size())});
                for (size_t i = 0; i < new_vars.size(); ++i) {
                    auto src = all_oprs[i]->output(0);
                    if (src->has_name_set()) {
                        new_vars[i].rename(src->name());
                    }
                }
            }
            rewriter.replace_var(
                    opr->output(0), new_vars.at(iter->second).node(),
                    mgb_cstr_log("replace multi SharedDeviceTensor(Format) to "
                                 "MultipleDeviceTensorHolder(Format)"));
        }
    };
    opt_state.graph().iter(cb_replace);

    rewriter.apply_inplace();
}

}

/* ================ global functions ================ */

SymbolVarArray gopt::optimize_for_inference(
        const SymbolVarArray& dest_vars,
        const OptimizeForInferenceOptions& opt) {
    return gopt::GraphOptimizer()
            .add_preset_passes(false, &opt,
                               &dest_vars[0].node()->owner_graph()->options())
            .apply({dest_vars})
            .endpoint_vars();
}

namespace {
void modify_conv_policy(opr::mixin::Convolution& conv,
                        megdnn::param::ExecutionPolicy::Strategy strategy) {
    auto policy = conv.execution_policy_transient();
    policy.strategy = strategy;
    conv.set_execution_policy(policy);
}

template <typename Opr>
void inplace_conv_opr_profile_modifier(OperatorNodeBase& opr) {
    modify_conv_policy(
            opr.cast_final_safe<Opr>(),
            opr::mixin::Convolution::ExecutionPolicy::Strategy::PROFILE);
}

template <typename Opr>
void inplace_conv_opr_profile_cache_modifier(OperatorNodeBase& opr) {
    modify_conv_policy(opr.cast_final_safe<Opr>(),
                       opr::mixin::Convolution::ExecutionPolicy::Strategy::
                               PROFILE_HEURISTIC);
}

void modify_conv_policy_workspace_limit(opr::mixin::Convolution& conv,
                                        size_t workspace_limit) {
    auto policy = conv.execution_policy_transient();
    policy.workspace_limit = workspace_limit;
    conv.set_execution_policy(policy);
}

template <typename Opr>
void inplace_conv_opr_workspace_limit_modifier(OperatorNodeBase& opr,
                                               size_t workspace_limit) {
    modify_conv_policy_workspace_limit(opr.cast_final_safe<Opr>(),
                                       workspace_limit);
}

}  // anonymous namespace

#define MGB_FOREACH_FASTRUN_OPR(cb)                                           \
    cb(ConvolutionForward), cb(ConvBiasForward), cb(ConvolutionBackwardData), \
            cb(ConvolutionBackwardFilter), cb(Convolution3DForward),          \
            cb(Convolution3DBackwardData), cb(Convolution3DBackwardFilter),   \
            cb(LocalShareForward), cb(LocalShareBackwardData),                \
            cb(LocalShareBackwardFilter), cb(DeformableConvForward),          \
            cb(DeformableConvBackwardFilter), cb(DeformableConvBackwardData), \
            cb(BatchConvBiasForward),

void gopt::enable_opr_algo_profiling_inplace(
        const VarNodeArrayView& dest_vars) {
#if MGB_ENABLE_FASTRUN
    static const ThinHashMap<Typeinfo*, void (*)(OperatorNodeBase&)> modifiers =
            {
#define CONV(t) {opr::t::typeinfo(), &inplace_conv_opr_profile_modifier<opr::t>}
                    MGB_FOREACH_FASTRUN_OPR(CONV)
#undef CONV
            };

    auto on_opr = [&](OperatorNodeBase* opr) {
        auto iter = modifiers.find(opr->dyn_typeinfo());
        if (iter != modifiers.end()) {
            iter->second(*opr);
        }
    };

    cg::DepOprIter dep_iter{on_opr};
    for (auto i : dest_vars) {
        dep_iter.add(i);
    }
#else
    mgb_throw(MegBrainError, "fastrun is disabled at compile time");
#endif
}

void gopt::enable_opr_use_profiling_cache_inplace(
        const VarNodeArrayView& dest_vars) {
    static const ThinHashMap<Typeinfo*, void (*)(OperatorNodeBase&)> modifiers =
            {
#define CONV(t) \
    {opr::t::typeinfo(), &inplace_conv_opr_profile_cache_modifier<opr::t>}
                    MGB_FOREACH_FASTRUN_OPR(CONV)
#undef CONV
            };

    auto on_opr = [&](OperatorNodeBase* opr) {
        auto iter = modifiers.find(opr->dyn_typeinfo());
        if (iter != modifiers.end()) {
            iter->second(*opr);
        }
    };

    cg::DepOprIter dep_iter{on_opr};
    for (auto i : dest_vars) {
        dep_iter.add(i);
    }
}

void gopt::set_opr_algo_workspace_limit_inplace(
        const VarNodeArrayView& dest_vars, size_t workspace_limit) {
    static const ThinHashMap<Typeinfo*, void (*)(OperatorNodeBase&, size_t)>
            modifiers = {
#define CONV(t) \
    {opr::t::typeinfo(), &inplace_conv_opr_workspace_limit_modifier<opr::t>}
                    MGB_FOREACH_FASTRUN_OPR(CONV)
#undef CONV
            };

    auto on_opr = [&](OperatorNodeBase* opr) {
        auto iter = modifiers.find(opr->dyn_typeinfo());
        if (iter != modifiers.end()) {
            iter->second(*opr, workspace_limit);
        }
    };

    cg::DepOprIter dep_iter{on_opr};
    for (auto i : dest_vars) {
        dep_iter.add(i);
    }
}
#undef MGB_FOREACH_FASTRUN_OPR

/* ================ ParamRedistributePass ================ */
const char* ParamRedistributePass::name() const {
    return mgb_cstr_log("param_redistribute");
}

class ParamRedistributePass::Impl final: public RecursiveSubGraphRewriteHelper {
    ConstVarPropogate m_cvprop;
    UniqReaderCheck m_uniq_reader_check;
    //! oprs already processed in try_distribute_then_reassociate() should be
    //! skipped in on_new_opr_check_should_process()
    ThinHashSet<OperatorNodeBase*> m_opr_blacklist;
    std::string m_distribute_reasso_log_msg;

    //! try applying BinaryTrans20::associtive
    GTransResult try_reassociate(OperatorNodeBase *opr);

    //! try applying BinaryTrans20::distributive_add
    GTransResult try_distribute_add(OperatorNodeBase *opr);

    //! try distribute MUL/DIV over ADD/SUB and then apply
    GTransResult try_distribute_then_reassociate(OperatorNodeBase *opr);

    GTransResult process_opr(VarNode *out_var) override;

    bool on_new_opr_check_should_process(
            OperatorNodeBase*opr, OperatorNodeBase *repl_opr) override {
        m_uniq_reader_check.update_on_opr_auto_replace(opr, repl_opr);
        auto ins = m_cvprop.add_opr(opr);
        return ins.has_const_inp && !ins.all_const_inp &&
            !m_opr_blacklist.count(opr);
    };

    void after_replace_var(VarNode *orig_var, VarNode* new_var) override {
        m_uniq_reader_check.update_on_opr_auto_replace(orig_var->owner_opr(),
                new_var->owner_opr());
    }

    /*!
     * \brief try to reorder opr inputs to a const one and a non-const one
     *
     * return true if it can be reformulated as f(nci, ci), where nci is
     * non-const and ci is const.
     */
    bool reorder_for_normconst(OperatorNodeBase *opr,
            bool &swap_inp, VarNode *&nci, VarNode *&ci);

    public:
        Impl(OptState &state);
};

GTransResult ParamRedistributePass::Impl::process_opr(VarNode *out_var) {
    auto opr = out_var->owner_opr();
    auto trans = try_reassociate(opr);

    if (!trans.valid()) {
        trans = try_distribute_add(opr);
        if (!trans.valid())
            trans = try_distribute_then_reassociate(opr);
    }

    return trans;
}

GTransResult ParamRedistributePass::Impl::try_reassociate(
        OperatorNodeBase *opr) {

    // apply BinaryAssociative0 if opr is the form f(g(a, b), c) and b and c are
    // const

    bool swap_fop_inp = false, swap_gop_inp = false;
    VarNode *a, *b, *c, *ab;
    if (!reorder_for_normconst(opr, swap_fop_inp, ab, c))
        return None;

    if (!m_uniq_reader_check(ab))
        return None;

    if (!reorder_for_normconst(ab->owner_opr(), swap_gop_inp, a, b))
        return None;

    return BinaryTrans20::associtive().apply(opr, swap_fop_inp, swap_gop_inp);
}

GTransResult ParamRedistributePass::Impl::try_distribute_add(
        OperatorNodeBase *opr) {

    if (opr->same_type<opr::Elemwise>() || opr->input().size() != 2)
        return None;

    if (!m_cvprop.is_const(opr->input(1)))
        return None;

    auto ab = as_elem_opr(opr->input(0)->owner_opr(), opr::Elemwise::Mode::ADD);
    if (ab) {
        bool swap;
        VarNode *a, *b;
        if (reorder_for_normconst(ab, swap, a, b)) {
            return BinaryTrans20::distributive_add().apply(
                    opr, false, swap);
        }
    }
    return None;
}

GTransResult ParamRedistributePass::Impl::try_distribute_then_reassociate(
        OperatorNodeBase *opr) {
    if (!opr->same_type<opr::Elemwise>())
        return None;
    using Mode = opr::Elemwise::Mode;
    auto mode = opr->cast_final<opr::Elemwise>().param().mode;
    if (!(mode == Mode::MUL || mode == Mode::TRUE_DIV))
        return None;

    VarNode *a, *b;
    bool swap;
    if (!reorder_for_normconst(opr, swap, a, b))
        return None;

    auto chain_pred = [this](OperatorNodeBase *opr) {
        if (as_elem_opr(opr, Mode::ADD)) {
            auto var = opr->output(0);
            return m_uniq_reader_check(var) || m_cvprop.is_const(var);
        }
        return false;
    };
    auto chain = extract_opr_leaves(a, chain_pred);
    if (chain.size() <= 1)
        return None;
    std::unordered_map<VarNode*, VarNode*> repl_map;
    m_distribute_reasso_log_msg.clear();

    int nr_fail = 0, nr_succ = 0;
    for (auto &&var: chain) {
        {
            auto iter = repl_map.find(var);
            if (iter != repl_map.end()) {
                var = iter->second;
                continue;
            }
        }

        auto vnew = (SymbolVar{var} * b).node();
        m_opr_blacklist.insert(vnew->owner_opr());
        if (!m_cvprop.is_const(var)) {
            auto trans = try_reassociate(vnew->owner_opr());
            if (!trans.valid()) {
                // allow at most one failed redistribution
                if (nr_fail)
                    return None;
                ++ nr_fail;
            } else {
                ++ nr_succ;
                vnew = trans->result;
                if (!m_distribute_reasso_log_msg.empty()) {
                    m_distribute_reasso_log_msg.append(mgb_cstr_log(";"));
                }
                m_distribute_reasso_log_msg.append(trans->msg);
            }
        }

        repl_map[var] = vnew;
        var = vnew;
    }
    if (nr_succ) {
        m_distribute_reasso_log_msg.insert(0,
                mgb_cstr_log("distribute_mul("));
        m_distribute_reasso_log_msg.append(mgb_cstr_log(")"));
        return GTransResultItem{
                elemwise_reduce_var_list(chain, Mode::ADD),
                m_distribute_reasso_log_msg.c_str(),
                {}};
    }
    return None;
}

bool ParamRedistributePass::Impl::reorder_for_normconst(
        OperatorNodeBase *opr, bool &swap_inp, VarNode *&nci, VarNode *&ci) {
    if (opr->input().size() != 2)
        return false;

    nci = opr->input(0);
    ci = opr->input(1);
    if (!m_cvprop.is_const(ci)) {
        if (!is_commutable_binary(opr) || !m_cvprop.is_const(nci))
            return false;
        swap_inp = true;
        std::swap(nci, ci);
    } else {
        if (m_cvprop.is_const(nci))
            return false;
        swap_inp = false;
    }

    return true;
}

ParamRedistributePass::Impl::Impl(OptState &state):
    RecursiveSubGraphRewriteHelper{state},
    m_cvprop{ConstVarType::IMMUTABLE_AND_PARAM},
    m_uniq_reader_check{state.graph()}
{
    auto cg = state.graph().comp_graph();
    auto on_new_opr = [this](const cg::event::OprInserted &ev) {
        if (!ev.is_dedup && !ev.exc) {
            // call add_opr eagerly to avoid deep recursion
            m_cvprop.add_opr(ev.opr);
        }
    };
    auto hdl = cg->event().register_receiver
        <cg::event::OprInserted>(on_new_opr);
    apply();
}

void ParamRedistributePass::apply(OptState &state) const {
    Impl{state};
}

/* ================ ParamFusePass ================ */

class ParamFusePass::ConstVarPropogateWithSizeCheck final:
    public ConstVarPropogateBase
{
    public:
        //! rewrite a var; reader == nullptr means needed by endpoint
        using VarRewriter = std::function<
            void(VarNode *var, OperatorNodeBase *reader)>;

        ConstVarPropogateWithSizeCheck(
                const ParamFusePass &pf, OptState &opt_state,
                const VarRewriter &rewriter):
            ConstVarPropogateBase{ConstVarType::IMMUTABLE_AND_PARAM},
            m_owner{pf}, m_opt_state{opt_state}, m_rewriter{rewriter}
        {
        }

    private:

        const ParamFusePass &m_owner;
        OptState &m_opt_state;
        VarRewriter m_rewriter;

        void on_midconst_opr(
                OperatorNodeBase *opr, size_t max_src_size) override {
            for (auto var: opr->output()) {
                if (var->contain_flag(VarNode::Flag::VOLATILE_CONTENT))
                    continue;

                auto osize = var_mem_size(var);
                if (osize >= max_src_size &&
                        osize - max_src_size > m_owner.m_param_grow_limit) {
                    return;
                }

                // const oprs should be evaluated when output is used by another
                // non-const opr or output is needed by the user
                if (m_opt_state.graph().endpoint_contain(var)) {
                    m_rewriter(var, nullptr);
                }

            }
        }
};

/*!
 * \brief get name for new param
 */
class ParamFusePass::VarNamer {
#if MGB_BUILD_SLIM_SERVING
    public:
        const std::string& name(VarNode*) {
            static std::string ret("fuse");
            return ret;
        }
#else
    using SrcSet = SharedSet<OperatorNodeBase*>;
    //! map from var to source SharedDeviceTensor/MultiSharedDeviceHolder oprs
    //! that it depends on
    ThinHashMap<OperatorNodeBase*, SrcSet> m_opr2srcs;
    std::string m_name_cache;
    std::vector<const char*> m_cur_name;

    SrcSet& get_src_set(OperatorNodeBase* opr) {
        auto opr_typeinfo = opr->dyn_typeinfo();

        auto iter = m_opr2srcs.find(opr);
        if (iter != m_opr2srcs.end()) {
            return iter->second;
        }
        auto &&ret = m_opr2srcs[opr];
        if (opr->input().empty()) {
            if (opr_typeinfo == opr::SharedDeviceTensor::typeinfo() ||
                opr_typeinfo == opr::MultipleDeviceTensorHolder::typeinfo()) {
                ret.insert(opr);
            } else {
                mgb_assert(opr_typeinfo == opr::ImmutableTensor::typeinfo());
            }
            return ret;
        }

        for (auto i: opr->input()) {
            ret.merge_from(get_src_set(i->owner_opr()));
        }
        return ret;
    }

    public:

        const std::string& name(VarNode *var) {
            m_cur_name.clear();
            for (auto i : get_src_set(var->owner_opr())) {
                m_cur_name.push_back(i->cname());
            }

            auto cmp = [](const char *x, const char *y) {
                return strcmp(x, y) < 0;
            };
            std::sort(m_cur_name.begin(), m_cur_name.end(), cmp);
            m_name_cache.clear();
            m_name_cache.append(mgb_cstr_log("fuse("));
            bool first = true;
            for (auto i: m_cur_name) {
                if (first) {
                    first = false;
                } else {
                    m_name_cache.push_back(',');
                }
                m_name_cache.append(i);
            }
            m_name_cache.append(mgb_cstr_log(
                        ssprintf("):%s@%zu", var->cname(), var->id())));
            return m_name_cache;
        }
#endif
};

const char* ParamFusePass::name() const {
    return mgb_cstr_log("param_fuse");
}

void ParamFusePass::apply(OptState &state) const {
    auto rewriter = state.graph().make_rewriter();
    auto cg = state.graph().comp_graph();
    ThinHashSet<VarNode*> processed_var;
    VarNamer var_namer;

    // reader: null if used as endvar
    auto replace_single_var = [&](VarNode *var, OperatorNodeBase *reader) {
        if (!processed_var.insert(var).second)
            return;

        auto inferred_val = std::make_shared<DeviceTensorND>(
                var->comp_node(), var->dtype());
        auto cb = [&](DeviceTensorND& val) {
            // retain format of val
            mgb_assert(val.format() == var->format());
            inferred_val->format(val.format())
                    .resize(val.shape())
                    .copy_from_fixlayout(val);
        };

        {
            auto orig_level = cg->options().log_level;
            cg->options().log_level = 0;
            MGB_TRY {
                cg->compile({{var, cb}})->execute();
            } MGB_FINALLY(cg->options().log_level = orig_level);
        }

        SymbolVar new_var;
        bool is_default_format = var->layout().format.is_default();
        if (cg::is_static_var_value(var) && is_default_format) {
            // use ImmutableTensor for inferable vars
            HostTensorND hv;
            hv.copy_from(*inferred_val).sync();
            new_var = opr::ImmutableTensor::make(
                    *var->owner_graph(), hv, var_namer.name(var));
        } else {
            if (is_default_format) {
                new_var = opr::SharedDeviceTensor::make(
                        *var->owner_graph(), inferred_val, var_namer.name(var));
            } else {
                new_var = opr::SharedDeviceTensorWithFormat::make(
                        *var->owner_graph(), inferred_val, var_namer.name(var));
            }
        }
        std::string log;
        if (reader) {
            log = mgb_ssprintf_log(
                    "due to read by %s{%s}",
                    reader->cname(), reader->dyn_typeinfo()->name);
        } else {
            log = mgb_cstr_log("as endpoint");
        }
        rewriter.replace_var(var, new_var.node(), log.c_str());
    };

    ConstVarPropogateWithSizeCheck cvprop{*this, state, replace_single_var};
    auto on_opr = [&](OperatorNodeBase *opr) {
        auto add_ret = cvprop.add_opr(opr);
        if (!add_ret.all_const_inp && add_ret.has_midconst_inp) {
            for (auto i: opr->input()) {
                if (cvprop.is_midconst(i)) {
                    state.call_with_opr(i->owner_opr(),
                        [&]{replace_single_var(i, opr);});
                }
            }
        }
        rewriter.auto_replace_outputs(opr);
    };

    state.graph().iter(on_opr);
    rewriter.apply_inplace();
}

/* ================ One2OneOprReplacePass ================ */
const char* ConvertF32ToF16Pass::name() const {
    return mgb_cstr_log("convert_f32_to_f16");
}

void ConvertF32ToF16Pass::apply(OptState& state) const {
    state.set_var_replace_check_flag(m_var_replace_check_flag);
    auto rewriter = state.graph().make_rewriter();
    VarNodeArray new_inp_cache;

    auto on_opr = [this, &rewriter, &new_inp_cache,
                   &state](OperatorNodeBase* opr) {
        auto it = m_opr_replace_func.find(opr->dyn_typeinfo());
        if (it != m_opr_replace_func.end()) {
            auto&& new_inp = new_inp_cache;
            new_inp.clear();
            new_inp.reserve(opr->input().size());
            for (auto i: opr->input()) {
                new_inp.push_back(rewriter.get_var(i));
            }
            auto new_opr = (it->second)(opr, new_inp);

            auto &&origin_out = opr->output(), &&cur_out = new_opr->output();
            mgb_assert(origin_out.size() == cur_out.size(),
                       "bad opr replace: src=%s{%s} dst=%s{%s}", opr->cname(),
                       opr->dyn_typeinfo()->name, new_opr->cname(),
                       new_opr->dyn_typeinfo()->name);
            //! change the output type if it's the endpoint
            for (size_t i = 0; i < origin_out.size(); i++) {
                if (state.graph().endpoint_contain(origin_out[i]) &&
                    origin_out[i]->dtype().enumv() !=
                            cur_out[i]->dtype().enumv()) {
                    rewriter.replace_var(
                            origin_out[i],
                            opr::TypeCvt::make(cur_out[i],
                                               origin_out[i]->dtype())
                                    .node(),
                            nullptr);
                } else {
                    rewriter.replace_var(origin_out[i], cur_out[i], nullptr);
                }
            }
        } else {
            auto new_opr = rewriter.auto_replace_outputs(opr);
            auto&& out = opr->output();
            auto&& new_out = new_opr->output();
            for (size_t i = 0; i < out.size(); i++) {
                if (state.graph().endpoint_contain(out[i]) &&
                    new_out[i]->dtype().enumv() != out[i]->dtype().enumv()) {
                    rewriter.replace_var(
                            new_out[i],
                            opr::TypeCvt::make(new_out[i],
                                               out[i]->dtype())
                                    .node(),
                            nullptr);
                }
            }
        }
    };
    state.graph().iter(on_opr);
    rewriter.apply_inplace();
}

std::unique_ptr<ConvertF32ToF16Pass> ConvertF32ToF16Pass::make(
        bool use_f32_comp) {
#if MEGDNN_DISABLE_FLOAT16
    mgb_throw(SystemError, "float16 disabled at compile time.");
#else
    auto replace_h2d_opr = [](OperatorNodeBase* opr,
                              const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& h2d_opr = opr->cast_final_safe<opr::Host2DeviceCopy>();
        if (h2d_opr.output(0)->dtype() == dtype::Float32()) {
            auto cvt_var =
                    opr::TypeCvt::make(h2d_opr.output(0), dtype::Float16(), {});
            return cvt_var.node()->owner_opr();
        }
        return opr;
    };

    auto replace_sdt_opr = [](OperatorNodeBase* opr,
                              const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& sdt_opr = opr->cast_final_safe<opr::SharedDeviceTensor>();
        if (sdt_opr.output(0)->dtype() == dtype::Float32()) {
            auto cvt_var =
                    opr::TypeCvt::make(sdt_opr.output(0), dtype::Float16(), {});
            return cvt_var.node()->owner_opr();
        }
        return opr;
    };

    auto replace_imt_opr = [](OperatorNodeBase* opr,
                              const VarNodeArray& new_inp) {
        mgb_assert(opr->same_type<opr::ImmutableTensor>());
        mgb_assert(opr->input().size() == new_inp.size());
        auto& imt_opr = opr->cast_final_safe<opr::ImmutableTensor>();
        if (imt_opr.output(0)->dtype() == dtype::Float32()) {
            auto cvt_var =
                    opr::TypeCvt::make(imt_opr.output(0), dtype::Float16(), {});
            return cvt_var.node()->owner_opr();
        }
        return opr;
    };

    auto replace_conv_opr = [use_f32_comp](OperatorNodeBase* opr,
                                           const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& conv_opr = opr->cast_final_safe<opr::ConvolutionForward>();
        auto new_param = conv_opr.param();
        if (use_f32_comp) {
            new_param.compute_mode =
                    megdnn::param::Convolution::ComputeMode::FLOAT32;
        }
        mgb_assert(new_inp[0]->dtype() == dtype::Float16(),
                   "inp %s:%s, owner_opr:%s", new_inp[0]->dtype().name(),
                   new_inp[0]->name().c_str(),
                   new_inp[0]->owner_opr()->name().c_str());
        mgb_assert(new_inp[1]->dtype() == dtype::Float16(),
                   "inp %s:%s, owner_opr:%s", new_inp[1]->dtype().name(),
                   new_inp[1]->name().c_str(),
                   new_inp[1]->owner_opr()->name().c_str());
        auto new_conv_opr = opr::Convolution::make(
                new_inp[0], new_inp[1], new_param, conv_opr.execution_policy(),
                conv_opr.config());
        return new_conv_opr.node()->owner_opr();
    };

    auto replace_matmul_opr = [use_f32_comp](OperatorNodeBase* opr,
                                             const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& matmul_opr = opr->cast_final_safe<opr::MatrixMul>();
        auto new_param = matmul_opr.param();
        if (use_f32_comp) {
            new_param.compute_mode =
                    megdnn::param::MatrixMul::ComputeMode::FLOAT32;
        }
        auto new_matmul_opr = opr::MatrixMul::make(
                new_inp[0], new_inp[1], new_param, matmul_opr.config());
        return new_matmul_opr.node()->owner_opr();
    };

    auto replace_reduce_opr = [use_f32_comp](OperatorNodeBase* opr,
                                             const VarNodeArray& new_inp) {
        auto& reduce_opr = opr->cast_final_safe<opr::Reduce>();
        auto new_param = reduce_opr.param();
        if (use_f32_comp) {
            new_param.data_type =
                    megdnn::param::Reduce::DataType::FLOAT_O16xC32;
        }
        if (opr->input().size() == 1) {
            auto new_matmul_opr = opr::Reduce::make(new_inp[0], new_param, {},
                                                    reduce_opr.config());
            return new_matmul_opr.node()->owner_opr();
        } else {
            mgb_assert(opr->input().size() == 2, "invalid input size %zu",
                       opr->input().size());
            auto new_matmul_opr = opr::Reduce::make(
                    new_inp[0], new_param, new_inp[1], reduce_opr.config());
            return new_matmul_opr.node()->owner_opr();
        }
    };

    auto replace_cvt_opr = [](OperatorNodeBase* opr,
                              const VarNodeArray& new_inp) {
        auto& cvt_opr = opr->cast_final_safe<opr::TypeCvt>();
        SymbolVar new_cvt;
        if (cvt_opr.output(0)->dtype() == dtype::Float32()) {
            new_cvt = opr::TypeCvt::make(new_inp[0], dtype::Float16(),
                                              cvt_opr.config());
        } else {
            new_cvt = opr::TypeCvt::make(
                    new_inp[0], cvt_opr.output()[0]->dtype(), cvt_opr.config());
        }
        return new_cvt.node()->owner_opr();
    };

    auto replace_warp_opr = [](OperatorNodeBase* opr,
                               const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size() &&
                   (new_inp.size() == 3 || new_inp.size() == 4));
        auto& warp_opr = opr->cast_final<opr::WarpPerspective>();
        // mat tensor must be float32
        auto new_mat = new_inp[1];
        if (new_inp[1]->dtype() != dtype::Float32()) {
            if (try_cast_as_op<opr::TypeCvt>(new_mat->owner_opr()) &&
                new_mat->owner_opr()->input(0)->dtype() == dtype::Float32())
                new_mat = new_mat->owner_opr()->input(0);
            else
                new_mat =
                        opr::TypeCvt::make(new_inp[1], dtype::Float32(), {}).node();
        }
        SymbolVar new_warp;
        if (new_inp.size() == 3) {
            new_warp = opr::WarpPerspective::make(new_inp[0], new_mat,
                                                  new_inp[2], warp_opr.param(),
                                                  warp_opr.config());
        } else {
            mgb_assert(new_inp.size() == 4);
            new_warp = opr::WarpPerspective::make(
                    new_inp[0], new_mat, new_inp[2], new_inp[3],
                    warp_opr.param(), warp_opr.config());
        }
        return new_warp.node()->owner_opr();
    };

    auto ret = std::make_unique<ConvertF32ToF16Pass>();
    // don't check dtype
    ret->set_var_replace_check_flag(VarReplaceCheckFlag::CHECK_ALL ^
                                    VarReplaceCheckFlag::CHECK_DTYPE);
    auto&& replace_func = ret->m_opr_replace_func;
    replace_func[opr::Host2DeviceCopy::typeinfo()] = replace_h2d_opr;
    replace_func[opr::SharedDeviceTensor::typeinfo()] = replace_sdt_opr;
    replace_func[opr::Convolution::typeinfo()] = replace_conv_opr;
    replace_func[opr::MatrixMul::typeinfo()] = replace_matmul_opr;
    replace_func[opr::Reduce::typeinfo()] = replace_reduce_opr;
    replace_func[opr::ImmutableTensor::typeinfo()] = replace_imt_opr;
    replace_func[opr::TypeCvt::typeinfo()] = replace_cvt_opr;
    replace_func[opr::WarpPerspective::typeinfo()] = replace_warp_opr;
    return ret;
#endif
}

/* ================ ConvertFormatPass ================ */

void ConvertFormatPass::apply(OptState& state) const {
    state.set_var_replace_check_flag(m_var_replace_check_flag);
    auto rewriter = state.graph().make_rewriter();
    VarNodeArray new_inp_cache;
    auto on_opr = [this, &state, &rewriter,
                   &new_inp_cache](OperatorNodeBase* opr) {
        auto it = m_opr_replace_func.find(opr->dyn_typeinfo());
        if (it != m_opr_replace_func.end()) {
            auto&& new_inp = new_inp_cache;
            new_inp.clear();
            new_inp.reserve(opr->input().size());
            for (auto i : opr->input()) {
                new_inp.push_back(rewriter.get_var(i));
            }
            auto new_opr = (it->second)(opr, new_inp);
            auto &&out0 = opr->output(), &&out1 = new_opr->output();
            mgb_assert(out0.size() == out1.size(),
                       "bad opr replace: src=%s{%s} dst=%s{%s}, src.size=%zu "
                       "dst.size=%zu",
                       opr->cname(), opr->dyn_typeinfo()->name,
                       new_opr->cname(), new_opr->dyn_typeinfo()->name,
                       out0.size(), out1.size());
            for (size_t i = 0; i < out0.size(); i++) {
                if (!out0[i]->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                    mgb_assert(!out1[i]->contain_flag(
                            VarNode::Flag::VOLATILE_CONTENT));
                    auto src = out0[i];
                    auto dst = out1[i];
                    auto dst_is_image = dst->format().type() ==
                                        TensorFormat::Type::IMAGE2D_PACK4;
                    if (!dst_is_image &&
                        !src->owner_opr()->same_type<opr::ImmutableTensor>()) {
                        mgb_log_warn(
                                "convert NHWCD4 replaced to non-img format: "
                                "dst_opr=%s{%s} format=%s",
                                dst->owner_opr()->cname(),
                                dst->owner_opr()->dyn_typeinfo()->name,
                                dst->format().to_string().c_str());
                    }
                    if (state.graph().endpoint_contain(src) && dst_is_image) {
                        // relayout back to NCHW for output vars
                        dst = opr::RelayoutFormat::make(
                                      dst, {opr::RelayoutFormat::Param::Mode::
                                                    NHWCD4I_NCHW})
                                      .node();
                    }
                    rewriter.replace_var(src, dst, nullptr);
                }
            }
        } else {
            rewriter.auto_replace_outputs(opr);
        }
    };
    state.graph().iter(on_opr);
    rewriter.apply_inplace();
}

std::unique_ptr<ConvertFormatPass> ConvertFormatPass::make_nhwcd4_converter() {
    auto filter_mode =
            [](const megdnn::param::Convolution::Sparse conv_mode,
               const VarNode* filter) -> megdnn::param::RelayoutFormat::Mode {
        bool use_dot = false;
        if (filter->dtype().enumv() == megdnn::DTypeEnum::QuantizedS8 ||
            filter->dtype().enumv() == megdnn::DTypeEnum::Quantized8Asymm)
            use_dot = true;
        if (conv_mode == megdnn::param::Convolution::Sparse::DENSE) {
            if (use_dot)
                return megdnn::param::RelayoutFormat::Mode::
                        INTER_WEIGHT_DENSEI_DOT;
            return megdnn::param::RelayoutFormat::Mode::INTER_WEIGHT_DENSEI;
        } else {
            mgb_assert(conv_mode == megdnn::param::Convolution::Sparse::GROUP);
            if (filter->shape()[1] == 1 && filter->shape()[2] == 1) {
                return megdnn::param::RelayoutFormat::Mode::INTER_WEIGHT_CHANI;
            } else {
                if (use_dot)
                    return megdnn::param::RelayoutFormat::Mode::
                            INTER_WEIGHT_GROUPI_DOT;
                return megdnn::param::RelayoutFormat::Mode::INTER_WEIGHT_GROUPI;
            }
        }
    };

    auto replace_conv_opr = [&filter_mode](OperatorNodeBase* opr,
                               const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& conv_opr = opr->cast_final_safe<opr::ConvolutionForward>();
        mgb_assert(conv_opr.param().format ==
                           megdnn::param::Convolution::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NHWCD4");
        VarNode *conv_src = nullptr, *conv_weights = nullptr;
        if (new_inp[0]->shape().ndim == 4) {
            // new input src is NCHW
            size_t group, icpg, ocpg;
            if (conv_opr.param().sparse ==
                megdnn::param::Convolution::Sparse::DENSE) {
                group = 1;
                icpg = new_inp[1]->shape()[1];
                ocpg = new_inp[1]->shape()[0];
            } else {
                mgb_assert(conv_opr.param().sparse ==
                           megdnn::param::Convolution::Sparse::GROUP);
                group = new_inp[1]->shape()[0];
                icpg = new_inp[1]->shape()[2];
                ocpg = new_inp[1]->shape()[1];
            }
            if (ocpg % 4 == 0 && (icpg % 4 == 0 || group == 1)) {
                auto param = megdnn::param::RelayoutFormat();
                param.mode = megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I;
                auto rf = opr::RelayoutFormat::make(new_inp[0], param);
                conv_src = rf.node();
            } else {
                // can not convert to hwcd4
                return serialization::copy_opr_shallow(*opr, new_inp,
                                                       opr->config());
            }
        } else {
            size_t ocpg;
            bool is_channel_wise = false;
            if (conv_opr.param().sparse ==
                megdnn::param::Convolution::Sparse::DENSE) {
                ocpg = new_inp[1]->shape()[0];
            } else {
                mgb_assert(conv_opr.param().sparse ==
                           megdnn::param::Convolution::Sparse::GROUP);
                size_t icpg = new_inp[1]->shape()[2];
                ocpg = new_inp[1]->shape()[1];
                if (icpg == 1 && ocpg == 1) {
                   is_channel_wise = true;
                }
            }
            if (ocpg % 4 != 0 && !is_channel_wise) {
                VarNodeArray t_inp = new_inp;
                auto param = megdnn::param::RelayoutFormat();
                param.mode = megdnn::param::RelayoutFormat::Mode::NHWCD4I_NCHW;
                auto rf = opr::RelayoutFormat::make(new_inp[0], param);
                t_inp[0] = rf.node();
                auto new_opr = serialization::copy_opr_shallow(*opr, t_inp,
                                                               opr->config());
                return new_opr;
            }
            // new input src is NHWCD4
            auto&& fmt = new_inp[0]
                                 ->format()
                                 .as_impl<megdnn::Image2DPack4TensorFormat>();
            mgb_assert(new_inp[0]->shape().ndim == 5 && fmt.align_axis() == 2);
            conv_src = new_inp[0];
        }
        mgb_assert(new_inp[1]->format().type() !=
                   TensorFormat::Type::IMAGE2D_PACK4);
        auto param = megdnn::param::RelayoutFormat();
        param.mode = filter_mode(conv_opr.param().sparse, new_inp[1]);
        auto relayout_weight = opr::RelayoutFormat::make(new_inp[1], param);
        conv_weights = relayout_weight.node();
        auto new_param = conv_opr.param();
        new_param.format = megdnn::param::Convolution::Format::NHWCD4;
        mgb_assert(conv_src->shape().ndim == 5 &&
                   conv_src->format().type() ==
                           TensorFormat::Type::IMAGE2D_PACK4);
        auto new_conv_opr = opr::Convolution::make(
                conv_src, conv_weights, new_param, conv_opr.execution_policy(),
                conv_opr.config());
        OperatorNodeBase* ret = new_conv_opr.node()->owner_opr();
        mgb_assert(new_conv_opr.shape().ndim == 5 &&
                   new_conv_opr.format().type() ==
                           TensorFormat::Type::IMAGE2D_PACK4);
        return ret;
    };

    auto replace_conv_bias_opr = [&filter_mode](OperatorNodeBase* opr,
                               const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& conv_bias_opr = opr->cast_final_safe<opr::ConvBiasForward>();
        mgb_assert(conv_bias_opr.param().format ==
                           megdnn::param::ConvBias::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NHWCD4");
        VarNode *conv_bias_src = nullptr, *conv_bias_weights = nullptr,
                *conv_bias_bias = nullptr;
        if (new_inp[0]->shape().ndim == 4) {
            // new input src is NCHW
            size_t group, icpg, ocpg;
            if (conv_bias_opr.param().sparse ==
                megdnn::param::ConvBias::Sparse::DENSE) {
                group = 1;
                icpg = new_inp[1]->shape()[1];
                ocpg = new_inp[1]->shape()[0];
            } else {
                mgb_assert(conv_bias_opr.param().sparse ==
                           megdnn::param::ConvBias::Sparse::GROUP);
                group = new_inp[1]->shape()[0];
                icpg = new_inp[1]->shape()[2];
                ocpg = new_inp[1]->shape()[1];
            }
            if (ocpg % 4 == 0 && (icpg % 4 == 0 || group == 1)) {
                auto param = megdnn::param::RelayoutFormat();
                param.mode = megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I;
                auto rf = opr::RelayoutFormat::make(new_inp[0], param);
                conv_bias_src = rf.node();
            } else {
                // can not convert to hwcd4
                return serialization::copy_opr_shallow(*opr, new_inp,
                                                       opr->config());
            }
        } else {
            size_t ocpg;
            bool is_channel_wise = false;
            if (conv_bias_opr.param().sparse ==
                megdnn::param::ConvBias::Sparse::DENSE) {
                ocpg = new_inp[1]->shape()[0];
            } else {
                mgb_assert(conv_bias_opr.param().sparse ==
                           megdnn::param::ConvBias::Sparse::GROUP);
                size_t icpg = new_inp[1]->shape()[2];
                ocpg = new_inp[1]->shape()[1];
                if (icpg == 1 && ocpg == 1) {
                   is_channel_wise = true;
                }
            }
            if (ocpg % 4 != 0 && !is_channel_wise) {
                VarNodeArray t_inp = new_inp;
                auto param = megdnn::param::RelayoutFormat();
                param.mode = megdnn::param::RelayoutFormat::Mode::NHWCD4I_NCHW;
                auto rf = opr::RelayoutFormat::make(new_inp[0], param);
                t_inp[0] = rf.node();
                auto new_opr = serialization::copy_opr_shallow(*opr, t_inp,
                                                               opr->config());
                return new_opr;
            }
            // new input src is NHWCD4
            auto&& fmt = new_inp[0]
                                 ->format()
                                 .as_impl<megdnn::Image2DPack4TensorFormat>();
            mgb_assert(new_inp[0]->shape().ndim == 5 && fmt.align_axis() == 2);
            conv_bias_src = new_inp[0];
        }
        mgb_assert(new_inp[1]->format().type() !=
                   TensorFormat::Type::IMAGE2D_PACK4);

        auto param = megdnn::param::RelayoutFormat();
        param.mode = filter_mode(conv_bias_opr.param().sparse, new_inp[1]);
        auto relayout_weight = opr::RelayoutFormat::make(new_inp[1], param);
        conv_bias_weights = relayout_weight.node();

        param.mode = megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I;
        auto relayout_bias = opr::RelayoutFormat::make(new_inp[2], param);
        conv_bias_bias = relayout_bias.node();

        auto new_param = conv_bias_opr.param();
        new_param.format = megdnn::param::ConvBias::Format::NHWCD4;
        mgb_assert(conv_bias_src->shape().ndim == 5 &&
                   conv_bias_src->format().type() ==
                           TensorFormat::Type::IMAGE2D_PACK4);
        auto new_conv_bias_opr = opr::ConvBias::make(
                conv_bias_src, conv_bias_weights, conv_bias_bias, new_param,
                conv_bias_opr.execution_policy(), conv_bias_opr.config());
        OperatorNodeBase* ret = new_conv_bias_opr.node()->owner_opr();
        mgb_assert(new_conv_bias_opr.shape().ndim == 5 &&
                   new_conv_bias_opr.format().type() ==
                           TensorFormat::Type::IMAGE2D_PACK4);
        return ret;
    };


    auto replace_deconv_opr = [&filter_mode](OperatorNodeBase* opr,
                               const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& deconv_opr = opr->cast_final_safe<opr::ConvolutionBackwardData>();
        mgb_assert(deconv_opr.param().format ==
                           megdnn::param::Convolution::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NHWCD4");
        VarNode *deconv_src = nullptr, *deconv_weights = nullptr;
        if (new_inp[1]->shape().ndim == 4) {
            // new input src is NCHW
            size_t group, icpg, ocpg;
            if (deconv_opr.param().sparse ==
                megdnn::param::Convolution::Sparse::DENSE) {
                group = 1;
                icpg = new_inp[0]->shape()[0];
                ocpg = new_inp[0]->shape()[1];
            } else {
                mgb_assert(deconv_opr.param().sparse ==
                           megdnn::param::Convolution::Sparse::GROUP);
                group = new_inp[0]->shape()[0];
                icpg = new_inp[0]->shape()[1];
                ocpg = new_inp[0]->shape()[2];
            }
            if (ocpg % 4 == 0 && (icpg % 4 == 0 || group == 1)) {
                auto param = megdnn::param::RelayoutFormat();
                param.mode = megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I;
                auto rf = opr::RelayoutFormat::make(new_inp[1], param);
                deconv_src = rf.node();
            } else {
                // can not convert to hwcd4
                return serialization::copy_opr_shallow(*opr, new_inp,
                                                       opr->config());
            }
        } else {
            //! XXXX, fix me, check filter size
            size_t ocpg;
            if (deconv_opr.param().sparse ==
                megdnn::param::Convolution::Sparse::DENSE) {
                ocpg = new_inp[0]->shape()[1];
            } else {
                mgb_assert(deconv_opr.param().sparse ==
                           megdnn::param::Convolution::Sparse::GROUP);

                ocpg = new_inp[0]->shape()[2];
            }
            if (ocpg % 4 != 0) {
                VarNodeArray t_inp = new_inp;
                auto param = megdnn::param::RelayoutFormat();
                param.mode = megdnn::param::RelayoutFormat::Mode::NHWCD4I_NCHW;
                auto rf = opr::RelayoutFormat::make(new_inp[1], param);
                t_inp[1] = rf.node();
                auto new_opr = serialization::copy_opr_shallow(*opr, t_inp,
                                                               opr->config());
                return new_opr;
            }
            // new input src is NHWCD4
            auto&& fmt = new_inp[1]
                                 ->format()
                                 .as_impl<megdnn::Image2DPack4TensorFormat>();
            mgb_assert(new_inp[1]->shape().ndim == 5 && fmt.align_axis() == 2);
            deconv_src = new_inp[1];
        }
        mgb_assert(new_inp[0]->format().type() !=
                   TensorFormat::Type::IMAGE2D_PACK4);
        auto param = megdnn::param::RelayoutFormat();
        param.mode = filter_mode(deconv_opr.param().sparse, new_inp[0]);
        auto relayout_weight = opr::RelayoutFormat::make(new_inp[0], param);
        deconv_weights = relayout_weight.node();
        auto new_param = deconv_opr.param();
        new_param.format = megdnn::param::Convolution::Format::NHWCD4;
        mgb_assert(deconv_src->shape().ndim == 5 &&
                   deconv_src->format().type() ==
                           TensorFormat::Type::IMAGE2D_PACK4);
        auto new_deconv_opr = opr::ConvolutionBackwardData::make(
                deconv_weights, deconv_src, new_param,
                deconv_opr.execution_policy(), deconv_opr.config());
        OperatorNodeBase* ret = new_deconv_opr.node()->owner_opr();
        mgb_assert(new_deconv_opr.shape().ndim == 5 &&
                   new_deconv_opr.format().type() ==
                           TensorFormat::Type::IMAGE2D_PACK4);
        return ret;
    };

    auto replace_resize_opr = [](OperatorNodeBase* opr,
                                 const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& resize_opr = opr->cast_final_safe<opr::ResizeForward>();
        mgb_assert(resize_opr.param().format ==
                           megdnn::param::Resize::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NHWCD4");
        VarNode* inp = nullptr;
        if (new_inp[0]->shape().ndim == 4) {
            // new input src is NCHW
            auto param = megdnn::param::RelayoutFormat();
            param.mode = megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I;
            auto rf = opr::RelayoutFormat::make(new_inp[0], param);
            inp = rf.node();
        } else {
            // new input src is NHWCD
            auto&& fmt = new_inp[0]
                                 ->format()
                                 .as_impl<megdnn::Image2DPack4TensorFormat>();
            mgb_assert(new_inp[0]->shape().ndim == 5 && fmt.align_axis() == 2);
            inp = new_inp[0];
        }
        auto new_param = resize_opr.param();
        new_param.format = megdnn::param::Resize::Format::NHWCD4;
        auto new_resize_opr = opr::ResizeForward::make(
                inp, new_inp[1], new_param, opr->config());
        return new_resize_opr.node()->owner_opr();
    };

    auto replace_warp_perspective_opr = [](OperatorNodeBase* opr,
                                           const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& warp_opr = opr->cast_final_safe<opr::WarpPerspectiveForward>();
        mgb_assert(warp_opr.param().format ==
                           megdnn::param::WarpPerspective::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NHWCD4");
        VarNode* inp = nullptr;
        if (new_inp[0]->shape().ndim == 4) {
            // new input src is NCHW
            auto param = megdnn::param::RelayoutFormat();
            param.mode = megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I;
            auto rf = opr::RelayoutFormat::make(new_inp[0], param);
            inp = rf.node();
        } else {
            // new input src is NHWCD
            auto&& fmt = new_inp[0]
                                 ->format()
                                 .as_impl<megdnn::Image2DPack4TensorFormat>();
            mgb_assert(new_inp[0]->shape().ndim == 5 && fmt.align_axis() == 2);
            inp = new_inp[0];
        }
        auto new_param = warp_opr.param();
        new_param.format = megdnn::param::WarpPerspective::Format::NHWCD4;
        SymbolVar new_warp_opr;
        if (new_inp.size() == 3) {
            new_warp_opr = opr::WarpPerspectiveForward::make(
                    inp, new_inp[1], nullptr, new_inp[2], new_param,
                    opr->config());
        } else {
            mgb_assert(new_inp.size() == 4);
            new_warp_opr = opr::WarpPerspectiveForward::make(
                    inp, new_inp[1], new_inp[2], new_inp[3], new_param,
                    opr->config());
        }
        return new_warp_opr.node()->owner_opr();
    };

    auto replace_warp_affine_opr = [](OperatorNodeBase* opr,
                                      const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& warp_opr = opr->cast_final_safe<opr::WarpAffineForward>();
        mgb_assert(warp_opr.param().format ==
                           megdnn::param::WarpAffine::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NHWCD4");
        VarNode* inp = nullptr;
        if (new_inp[0]->shape().ndim == 4) {
            // new input src is NCHW
            auto param = megdnn::param::RelayoutFormat();
            param.mode = megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I;
            auto rf = opr::RelayoutFormat::make(new_inp[0], param);
            inp = rf.node();
        } else {
            // new input src is NHWCD
            auto&& fmt = new_inp[0]
                                 ->format()
                                 .as_impl<megdnn::Image2DPack4TensorFormat>();
            mgb_assert(new_inp[0]->shape().ndim == 5 && fmt.align_axis() == 2);
            inp = new_inp[0];
        }
        auto new_param = warp_opr.param();
        new_param.format = megdnn::param::WarpAffine::Format::NHWCD4;
        SymbolVar new_warp_opr;
        new_warp_opr = opr::WarpAffineForward::make(inp, new_inp[1], new_inp[2],
                                                    new_param, opr->config());
        return new_warp_opr.node()->owner_opr();
    };

    auto replace_pooling_opr = [](OperatorNodeBase* opr,
                                  const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& pooling_opr = opr->cast_final_safe<opr::PoolingForward>();
        mgb_assert(pooling_opr.param().format ==
                           megdnn::param::Pooling::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NHWCD4");
        VarNode* inp = nullptr;
        if (new_inp[0]->shape().ndim == 4) {
            // new input src is NCHW
            auto param = megdnn::param::RelayoutFormat();
            param.mode = megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I;
            auto rf = opr::RelayoutFormat::make(new_inp[0], param);
            inp = rf.node();
        } else {
            // new input src is NHWCD
            auto&& fmt = new_inp[0]
                                 ->format()
                                 .as_impl<megdnn::Image2DPack4TensorFormat>();
            mgb_assert(new_inp[0]->shape().ndim == 5 && fmt.align_axis() == 2);
            inp = new_inp[0];
        }
        auto new_param = pooling_opr.param();
        new_param.format = megdnn::param::Pooling::Format::NHWCD4;
        auto new_pooling_opr =
                opr::PoolingForward::make(inp, new_param, opr->config());
        return new_pooling_opr.node()->owner_opr();
    };

    auto relayout_inp_to_chw = [](OperatorNodeBase* opr,
                                  const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        VarNodeArray t_inp = new_inp;
        for (size_t i = 0; i < opr->input().size(); i++) {
            if (!opr->input(i)->shape().eq_shape(new_inp[i]->shape())) {
                mgb_assert(opr->input(i)->shape().ndim == 4 &&
                           opr->input(i)->format().type() !=
                                   TensorFormat::Type::IMAGE2D_PACK4);
                mgb_assert(new_inp[i]->shape().ndim == 5 &&
                           new_inp[i]->format().type() ==
                                   TensorFormat::Type::IMAGE2D_PACK4);
                // Oprs which will change the shape of input like concat,
                // reshape etc. should not be used after cd4 convertion padding,
                // due to the padding info will be lost and we cannot recover
                // the origin unpadded data. For example, concat two tensors of
                // shape {1, 6, 128, 128}, if both tensors convert to cd4 then
                // the channel will be 8, and the result of concat channel will
                // be 16, but there will be 2 padding zeros in the middle of
                // channel axis, which will cause problems in succeding opr.
                if (opr->dyn_typeinfo() == opr::Concat::typeinfo()) {
                    auto concat = try_cast_as_op<opr::Concat>(opr);
                    mgb_assert(
                            !(concat->param().axis == 1 &&
                              concat->input(i)->shape()[1] % 4 != 0),
                            "We cannot concat tensor in channel axis which has "
                            "been padded, as it may lost padding pos if we "
                            "pass "
                            "the output to conv etc.");
                }
                auto param = megdnn::param::RelayoutFormat();
                param.mode = megdnn::param::RelayoutFormat::Mode::NHWCD4I_NCHW;
                auto rf = opr::RelayoutFormat::make(new_inp[i], param);
                t_inp[i] = rf.node();
            }
        }
        auto new_opr =
                serialization::copy_opr_shallow(*opr, t_inp, opr->config());
        return new_opr;
    };

    auto replace_elemwise_opr = [](OperatorNodeBase* opr,
                                   const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        bool has_inp_changed = false;
        for (size_t i = 0; i < opr->input().size(); i++) {
            if (!new_inp[i]->format().is_default()) {
                has_inp_changed = true;
                break;
            }
        }
        if (has_inp_changed) {
            // assumption: all inputs are changed from nchw to nhwcd4
            auto t_inp = new_inp;
            for (size_t i = 0; i < opr->input().size(); i++) {
                if (new_inp[i]->shape().ndim == 4) {
                    auto param = megdnn::param::RelayoutFormat();
                    param.mode =
                            megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I;
                    auto rf = opr::RelayoutFormat::make(new_inp[i], param);
                    t_inp[i] = rf.node();
                } else {
                    mgb_assert((new_inp[i]->shape().ndim == 5 &&
                                new_inp[i]->format().type() ==
                                        TensorFormat::Type::IMAGE2D_PACK4) ||
                               new_inp[i]->shape().is_scalar());
                }
            }
            return serialization::copy_opr_shallow(*opr, t_inp, opr->config());
        } else {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
    };

    auto ret = std::make_unique<ConvertFormatPass>();
    ret->set_var_replace_check_flag(VarReplaceCheckFlag::NOCHECK);
    auto&& replace_func = ret->m_opr_replace_func;
    replace_func[opr::Convolution::typeinfo()] = replace_conv_opr;
    replace_func[opr::ConvBias::typeinfo()] = replace_conv_bias_opr;
    replace_func[opr::ConvolutionBackwardData::typeinfo()] = replace_deconv_opr;
    replace_func[opr::PoolingForward::typeinfo()] = replace_pooling_opr;
    replace_func[opr::Elemwise::typeinfo()] = replace_elemwise_opr;
    replace_func[opr::Concat::typeinfo()] = relayout_inp_to_chw;
    replace_func[opr::Reshape::typeinfo()] = relayout_inp_to_chw;
    replace_func[opr::GetVarShape::typeinfo()] = relayout_inp_to_chw;
    replace_func[opr::Dimshuffle::typeinfo()] = relayout_inp_to_chw;
    replace_func[opr::Reduce::typeinfo()] = relayout_inp_to_chw;
    replace_func[opr::AssertEqual::typeinfo()] = relayout_inp_to_chw;
    replace_func[opr::Subtensor::typeinfo()] = relayout_inp_to_chw;
    replace_func[opr::Broadcast::typeinfo()] = relayout_inp_to_chw;
    replace_func[opr::IncrSubtensor::typeinfo()] = relayout_inp_to_chw;
    replace_func[opr::ResizeForward::typeinfo()] = replace_resize_opr;
    replace_func[opr::WarpPerspectiveForward::typeinfo()] =
            replace_warp_perspective_opr;
    replace_func[opr::WarpAffineForward::typeinfo()] = replace_warp_affine_opr;
    return ret;
}

/* ================ ConvertBatchNormPass ================ */
const char* ConvertBatchNormToElemwisePass::name() const {
    return "convert_batch_norm";
}

void ConvertBatchNormToElemwisePass::apply(OptState& state) const {
    auto rewriter = state.graph().make_rewriter();
    auto on_opr = [&](OperatorNodeBase* opr) {
        if (auto bn = try_cast_as_op<opr::BatchNorm>(opr)) {
            if (bn->input().size() == 5) {
                mgb_assert(bn->param().fwd_mode ==
                           opr::BatchNorm::Param::FwdMode::INFERENCE);
                SymbolVar x = {rewriter.get_var(bn->input(0))};
                SymbolVar scale = {rewriter.get_var(bn->input(1))};
                SymbolVar bias = {rewriter.get_var(bn->input(2))};
                SymbolVar mean = {rewriter.get_var(bn->input(3))};
                SymbolVar variance = {rewriter.get_var(bn->input(4))};
                SymbolVar invsqrt_variance = opr::PowC::make(variance, {-0.5});
                auto res = scale * (x - mean) * invsqrt_variance + bias;
                rewriter.replace_var(
                        opr->output(4), res.node(),
                        mgb_cstr_log(
                                "replace batch_norm(x, scale, bias, mean, "
                                "varience) "
                                "-> (sclae * (x - mean) / sqrt(variance)) + b)"));
                return;
            }
        }
        rewriter.auto_replace_outputs(opr);
    };
    state.graph().iter(on_opr);

    rewriter.apply_inplace();
}

/* ================ FuseConvBiasNonlinPass ================ */
const char* FuseConvBiasNonlinPass::name() const {
    return "combine_conv_bias_and_relu";
}

void FuseConvBiasNonlinPass::apply(OptState& state) const {
    std::unordered_map<VarNode*, std::vector<OperatorNodeBase*>> m_deps;
    state.graph().iter([&m_deps](OperatorNodeBase* opr) {
        for (auto& inp : opr->input()) {
            m_deps[inp].push_back(opr);
        }
    });

    auto rewriter = state.graph().make_rewriter();
    using Mode = opr::Elemwise::Param::Mode;
    using NonlineMode = opr::ConvBiasForward::Param::NonlineMode;

    auto get_nonlinearity_mode = [&](opr::Elemwise* elem) -> NonlineMode {
        if (elem->param().mode == Mode::FUSE_ADD_RELU ||
            elem->param().mode == Mode::RELU) {
            return NonlineMode::RELU;
        } else if (elem->param().mode == Mode::FUSE_ADD_SIGMOID ||
                   elem->param().mode == Mode::SIGMOID) {
            return NonlineMode::SIGMOID;
        } else {
            return NonlineMode::IDENTITY;
        }
    };

    auto try_fuse_bias_nonlinearity = [&](opr::Elemwise* elem) -> bool {

        bool can_be_fused = true;
        can_be_fused &= (elem->input().size() == 2);
        can_be_fused &= (elem->param().mode == Mode::FUSE_ADD_RELU) ||
                        (elem->param().mode == Mode::FUSE_ADD_TANH) ||
                        (elem->param().mode == Mode::FUSE_ADD_SIGMOID);

        return can_be_fused;
    };

    auto try_fuse_bias = [&](opr::Elemwise* elem) -> bool {

        bool can_be_fused = true;
        can_be_fused &= (elem->input().size() == 2);
        can_be_fused &= (elem->param().mode == Mode::ADD);
        return can_be_fused;
    };

    auto try_fuse_nonlinearity = [&](opr::Elemwise* elem) -> bool {

        bool can_be_fused = true;
        can_be_fused &= (elem->input().size() == 1);
        can_be_fused &= (elem->param().mode == Mode::RELU) ||
                        (elem->param().mode == Mode::TANH) ||
                        (elem->param().mode == Mode::SIGMOID);

        return can_be_fused;
    };

    auto convert_to_conv_bias_param = [&](const opr::Convolution::Param& param)
            -> opr::ConvBiasForward::Param {
        using Param = opr::ConvBiasForward::Param;
        return opr::ConvBiasForward::Param{Param::NonlineMode::IDENTITY,
                                           param.mode,
                                           param.sparse,
                                           param.format,
                                           param.pad_h,
                                           param.pad_w,
                                           param.stride_h,
                                           param.stride_w,
                                           param.dilate_h,
                                           param.dilate_w};
    };

    auto check_bias_shape = [&](opr::Convolution* conv, VarNode* bias) -> bool {
        bool valid_bias_shape = true;
        using Format = opr::Convolution::Param::Format;
        using Sparse = opr::Convolution::Param::Sparse;
        auto dst_shape = conv->output(0)->shape();
        auto filter_shape = conv->input(1)->shape();
        auto bias_shape = bias->shape();
        if (dst_shape.eq_shape(bias_shape)) {
            return valid_bias_shape;
        }
        size_t OC = filter_shape[0];
        if (conv->param().sparse == Sparse::GROUP) {
            OC *= filter_shape[1];
        }
        if (conv->param().format == Format::NCHW) {
            valid_bias_shape &=
                    ((bias_shape.ndim == 4) && (bias_shape[0] == 1) &&
                     (bias_shape[1] == OC) && (bias_shape[2] == 1) &&
                     (bias_shape[3] == 1));
        } else if (conv->param().format == Format::NCHW4) {
            valid_bias_shape &=
                    ((bias_shape.ndim == 5) && (bias_shape[0] == 1) &&
                     (bias_shape[1] == OC / 4) && (bias_shape[2] == 1) &&
                     (bias_shape[3] == 1) && bias_shape[4] == 4);
        } else if (conv->param().format == Format::NHWC) {
            valid_bias_shape &= ((bias_shape.ndim == 4) &&
                                 (bias_shape[0] == 1) && (bias_shape[1] == 1) &&
                                 (bias_shape[2] == 1) && (bias_shape[3] == OC));
        } else {
            valid_bias_shape &=
                    ((bias_shape.ndim == 5) && (bias_shape[0] == 1) &&
                     (bias_shape[1] == 1) && (bias_shape[2] == OC) &&
                     (bias_shape[3] == 1) && (bias_shape[4] == 4));
            mgb_assert(conv->param().format == Format::NHWCD4);
        }
        return valid_bias_shape;
    };

    auto try_fuse_typecvt = [&](opr::TypeCvt* typecvt) -> OperatorNodeBase* {
        mgb_assert(typecvt->input().size() == 1);
        auto conv_bias = try_cast_as_op<opr::ConvBias>(
                rewriter.get_var(typecvt->input(0))->owner_opr());
        if (!conv_bias || m_deps.count(typecvt->input(0)) != 1 ||
            typecvt->output(0)->dtype().enumv() !=
                    DTypeTrait<dtype::QuantizedS8>::enumv)
            return nullptr;

        auto config = conv_bias->config();
        config.output_dtype(typecvt->output(0)->dtype());
        if (conv_bias->input().size() == 3) {
            // conv + bias
            return opr::ConvBias::make(conv_bias->input(0), conv_bias->input(1),
                                       conv_bias->input(2), conv_bias->param(),
                                       conv_bias->execution_policy(), config)
                    .node()
                    ->owner_opr();
        } else {
            // conv without bias
            return opr::ConvBias::make(conv_bias->input(0), conv_bias->input(1),
                                       conv_bias->param(),
                                       conv_bias->execution_policy(), config)
                    .node()
                    ->owner_opr();
        }
    };
    auto on_opr = [&](OperatorNodeBase* opr) {
        auto check_conv = [](opr::Convolution* conv) -> bool {
            return conv->param().format ==
                           megdnn::param::Convolution::Format::NHWCD4 ||
                   conv->param().format ==
                           megdnn::param::Convolution::Format::NHWC ||
                   conv->param().format ==
                           megdnn::param::Convolution::Format::NCHW ||
                   conv->param().format ==
                           megdnn::param::Convolution::Format::NCHW4
                   ;
        };
        if (auto elem = try_cast_as_op<opr::Elemwise>(opr)) {
            if (try_fuse_bias_nonlinearity(elem) || try_fuse_bias(elem)) {
                auto inp1 = rewriter.get_var(elem->input(0));
                auto inp2 = rewriter.get_var(elem->input(1));
                opr::Convolution* conv = nullptr;
                size_t bias_idx = 0;
                if (inp1->owner_opr()->same_type<opr::Convolution>() &&
                    m_deps[elem->input(0)].size() == 1) {
                    conv = try_cast_as_op<opr::Convolution>(inp1->owner_opr());
                    bias_idx = 1;
                } else if (inp2->owner_opr()->same_type<opr::Convolution>() &&
                           m_deps[elem->input(1)].size() == 1) {
                    conv = try_cast_as_op<opr::Convolution>(inp2->owner_opr());
                    bias_idx = 0;
                }
                auto bias_inp = rewriter.get_var(elem->input(bias_idx));
                if (conv && check_conv(conv) &&
                    check_bias_shape(conv, bias_inp)) {
                    opr::ConvBiasForward::Param param =
                            convert_to_conv_bias_param(conv->param());
                    param.nonlineMode = get_nonlinearity_mode(elem);
                    auto new_var =
                            opr::ConvBiasForward::make(
                                    conv->input(0), conv->input(1), bias_inp,
                                    param, conv->execution_policy(),
                                    conv->config())
                                    .node();
                    rewriter.replace_var(
                            opr->output(0), new_var,
                            mgb_cstr_log("replace nonlinearity(conv(x, w) + b) "
                                         "-> conv_bias(x, w, b)"));
                    return;
                }
            } else if (try_fuse_nonlinearity(elem)) {
                auto inp = rewriter.get_var(elem->input(0));
                {
                    auto conv =
                            try_cast_as_op<opr::Convolution>(inp->owner_opr());
                    if (conv && check_conv(conv) &&
                        m_deps[elem->input(0)].size() == 1) {
                        opr::ConvBiasForward::Param param =
                                convert_to_conv_bias_param(conv->param());
                        param.nonlineMode = get_nonlinearity_mode(elem);
                        auto new_var = opr::ConvBiasForward::make(
                                               conv->input(0), conv->input(1),
                                               param, conv->execution_policy(),
                                               conv->config())
                                               .node();
                        rewriter.replace_var(
                                opr->output(0), new_var,
                                mgb_cstr_log("replace nonlinearity(conv(x, w)) "
                                             "-> conv_bias(x, w)"));
                        return;
                    }
                }
                {
                    auto conv = try_cast_as_op<opr::ConvBias>(inp->owner_opr());
                    auto check_conv_bias = [&](opr::ConvBias* opr) {
                        return opr->param().format ==
                                       opr::ConvBias::Param::Format::NHWC ||
                               opr->param().format ==
                                       opr::ConvBias::Param::Format::NCHW ||
                               opr->param().format ==
                                       opr::ConvBias::Param::Format::NCHW4
                               ;
                    };
                    if (conv && check_conv_bias(conv) &&
                        m_deps[elem->input(0)].size() == 1) {
                        auto param = conv->param();
                        param.nonlineMode = get_nonlinearity_mode(elem);
                        auto new_var = opr::ConvBiasForward::make(
                                               conv->input(0), conv->input(1),
                                               conv->input(2), param,
                                               conv->execution_policy(),
                                               conv->config())
                                               .node();
                        rewriter.replace_var(
                                opr->output(0), new_var,
                                mgb_cstr_log("replace nonlinearity(conv(x, w)) "
                                             "-> conv_bias(x, w)"));
                        return;
                    }
                }
            }
        } else if (auto typecvt = try_cast_as_op<opr::TypeCvt>(opr)) {
            auto new_opr = try_fuse_typecvt(typecvt);
            if (new_opr) {
                rewriter.replace_var(
                        opr->output(0), new_opr->output(0),
                        mgb_cstr_log("replace typecvt(conv_bias(x, w, b)) -> "
                                     "conv_bias(x, w, b)"));
                return;
            }
        }
        rewriter.auto_replace_outputs(opr);

    };
    state.graph().iter(on_opr);

    rewriter.apply_inplace();
}

/* ================ FuseConvBiasZPass ================ */
const char* FuseConvBiasZPass::name() const {
    return "combine_conv_bias_and_z";
}

void FuseConvBiasZPass::apply(OptState& state) const {
    UniqReaderCheck uniq_reader_check{state.graph()};

    auto rewriter = state.graph().make_rewriter();
    using Mode = opr::Elemwise::Param::Mode;
    using MultiMode = opr::ElemwiseMultiType::Param::Mode;
    using NonlineMode = opr::ConvBiasForward::Param::NonlineMode;

    auto check_conv_bias = [](opr::ConvBias* conv_bias) -> bool {
        return conv_bias->param().format ==
                       megdnn::param::ConvBias::Format::NHWC ||
               conv_bias->param().format ==
                       megdnn::param::ConvBias::Format::NCHW ||
               conv_bias->param().format ==
                       megdnn::param::ConvBias::Format::NCHW4
               ;
    };
    auto check_fuse_shape = [&](opr::ConvBias* conv_bias, VarNode* z) -> bool {
        bool valid_fuse_shape = true;
        auto z_shape = z->shape();
        auto bias_shape = conv_bias->input(2)->shape();
        auto conv_bias_shape = conv_bias->output(0)->shape();

        valid_fuse_shape &= (!conv_bias_shape.eq_shape(bias_shape));
        valid_fuse_shape &= conv_bias_shape.eq_shape(z_shape);

        return valid_fuse_shape;
    };
    auto check_fuse_dtype = [&](opr::ConvBias* conv_bias, VarNode* z) -> bool {
        return conv_bias->output(0)->dtype().enumv() == z->dtype().enumv();
    };
    auto get_convbias_nonline_mode = [&](OperatorNodeBase* opr) -> NonlineMode {
        if (opr->same_type<opr::Elemwise>()) {
            auto elem = try_cast_as_op<opr::Elemwise>(opr);
            if (elem->param().mode == Mode::FUSE_ADD_RELU)
                return NonlineMode::RELU;
        }

        if (opr->same_type<opr::ElemwiseMultiType>()) {
            auto elem = try_cast_as_op<opr::ElemwiseMultiType>(opr);
            if (elem->param().mode == MultiMode::QFUSE_ADD_RELU)
                return NonlineMode::RELU;
        }
        return NonlineMode::IDENTITY;
    };
    auto try_replace_var_node = [&](OperatorNodeBase* opr) {
        opr::ConvBias* conv_bias = nullptr;
        size_t z_idx = 0;
        size_t nr_inps = opr->input().size();
        for (size_t i = 0; i < nr_inps; i++) {
            auto inp = rewriter.get_var(opr->input(i));
            if (inp->owner_opr()->same_type<opr::ConvBias>()) {
                auto cb = try_cast_as_op<opr::ConvBias>(inp->owner_opr());
                if (cb->input().size() == 3 &&
                    cb->param().nonlineMode ==
                            opr::ConvBias::Param::NonlineMode::IDENTITY &&
                    uniq_reader_check(opr->input(i))) {
                    conv_bias = cb;
                    z_idx = nr_inps - i - 1;
                    break;
                }
            }
        }
        auto z_inp = rewriter.get_var(opr->input(z_idx));

        if (conv_bias && check_conv_bias(conv_bias) &&
            check_fuse_shape(conv_bias, z_inp) &&
            check_fuse_dtype(conv_bias, z_inp)) {
            auto param = conv_bias->param();
            param.nonlineMode = get_convbias_nonline_mode(opr);
            auto config = conv_bias->config();

            auto new_var = opr::ConvBiasForward::make(
                                   conv_bias->input(0), conv_bias->input(1),
                                   conv_bias->input(2), z_inp, param,
                                   conv_bias->execution_policy(),
                                   config.output_dtype(opr->output(0)->dtype()))
                                   .node();
            rewriter.replace_var(
                    opr->output(0), new_var,
                    mgb_cstr_log("replace "
                                 "nonlinearity(conv_bias(x,w,b) + z) "
                                 "-> conv_bias(x, w, b, z)"));
            uniq_reader_check.update_on_opr_auto_replace(opr,
                                                         new_var->owner_opr());
            return true;
        }
        return false;
    };
    auto try_fuse_elemwise = [&](OperatorNodeBase* opr) {
        if (!opr->same_type<opr::Elemwise>())
            return false;
        auto elem = try_cast_as_op<opr::Elemwise>(opr);
        if (elem->input().size() != 2)
            return false;
        if (elem->param().mode != Mode::ADD &&
            elem->param().mode != Mode::FUSE_ADD_RELU)
            return false;
        return try_replace_var_node(opr);
    };

    auto try_fuse_elemwise_multi_type = [&](OperatorNodeBase* opr) {
        if (!opr->same_type<opr::ElemwiseMultiType>())
            return false;
        auto elem = try_cast_as_op<opr::ElemwiseMultiType>(opr);
        if (elem->input().size() != 2)
            return false;
        if (elem->param().mode != MultiMode::QADD &&
            elem->param().mode != MultiMode::QFUSE_ADD_RELU)
            return false;
        return try_replace_var_node(opr);
    };

    auto on_opr = [&](OperatorNodeBase* opr) {
        if (try_fuse_elemwise(opr))
            return;
        if (try_fuse_elemwise_multi_type(opr))
            return;
        auto new_opr = rewriter.auto_replace_outputs(opr);
        uniq_reader_check.update_on_opr_auto_replace(opr, new_opr);
    };
    state.graph().iter(on_opr);

    rewriter.apply_inplace();
}

/* ================ FuseDeconvCvtPass ================ */
const char* FuseDeconvCvtPass::name() const {
    return "combine_deconv_and_typecvt";
}


void FuseDeconvCvtPass::apply(OptState& state) const {
    std::unordered_map<VarNode*, std::vector<OperatorNodeBase*>> m_deps;
    state.graph().iter([&m_deps](OperatorNodeBase* opr) {
        for (auto& inp : opr->input()) {
            m_deps[inp].push_back(opr);
        }
    });

    UniqReaderCheck uniq_reader_check{state.graph()};
    auto rewriter = state.graph().make_rewriter();
    auto try_fuse_deconv_typecvt =
            [&](opr::TypeCvt* typecvt) -> OperatorNodeBase* {
        mgb_assert(typecvt->input().size() == 1);
        auto deconv = try_cast_as_op<opr::ConvolutionBackwardData>(
                rewriter.get_var(typecvt->input(0))->owner_opr());
        if (!deconv
                || m_deps.count(typecvt->input(0)) != 1 ||
            typecvt->output(0)->dtype().enumv() !=
                    DTypeTrait<dtype::QuantizedS8>::enumv) {
            return nullptr;
        }
        if (!uniq_reader_check(deconv->output(0)))
            return nullptr;

        auto config = deconv->config();
        config.output_dtype(typecvt->output(0)->dtype());
        return opr::ConvolutionBackwardData::make(
                       deconv->input(0), deconv->input(1), deconv->param(),
                       deconv->execution_policy(), config)
                .node()
                ->owner_opr();
    };

    auto on_opr = [&](OperatorNodeBase* opr) {
        if (auto typecvt = try_cast_as_op<opr::TypeCvt>(opr)) {
            if (auto deconv_new = try_fuse_deconv_typecvt(typecvt)) {
                rewriter.replace_var(
                        opr->output(0), deconv_new->output(0),
                        mgb_cstr_log("replace typecvt(deconv(x, w)) -> "
                                     "deconv(x, w)"));
                uniq_reader_check.update_on_opr_auto_replace(opr, deconv_new);
                return;
            }
        }
        auto new_opr = rewriter.auto_replace_outputs(opr);
        uniq_reader_check.update_on_opr_auto_replace(
                opr, new_opr);
    };
    state.graph().iter(on_opr);

    rewriter.apply_inplace();
}

/* ================ ParamMergePass ================ */
const char* ParamMergePass::name() const {
    return mgb_cstr_log("param_merge");
}

void ParamMergePass::apply(OptState& opt_state) const {
    param_merge<opr::SharedDeviceTensor, opr::MultipleDeviceTensorHolder>(
            opt_state);
    param_merge<opr::SharedDeviceTensorWithFormat,
                opr::MultipleDeviceTensorWithFormatHolder>(opt_state);
}

/* ================ TensorReformatPass =============== */
/*!
 * \brief relayout placeholder opr
 *
 * RelayoutPlaceholder oprs act as the placeholders of the ComputingGraph
 * during graph opt pass `TensorReformatPass`. These oprs are introduced
 * into a ComputingGraph for conveniently discovering further optimize
 * opportunities (such as fuse consecutive relayouts, translate into
 * optimized implementations). They are canonized to have a shape infer, so
 * the ouput's shape can be correctly deduced during the opt pass.
 *
 * Note that the oprs in the ComputingGraph are only used as intermediate
 * representations before being translated to MegBrain oprs, so the
 * oprs should not get involved in any actual computing.
 */
MGB_DEFINE_OPR_CLASS(TensorReformatPass::RelayoutPlaceholder,
                           cg::SingleCNOperatorNodeBase) // {
public:
    //! relayout type of this opr
    enum class LayoutType {
        NCHW4_TO_NCHW32,              //!< from nchw4 layout to nchw32 layout
        NCHW32_TO_NCHW4,              //!< from nchw32 layout to nchw4 layout
        NCHW4_TO_CHWN4,               //!< from nchw4 layout to chwn4 layout
        CHWN4_TO_NCHW4,               //!< from chwn4 layout to nchw4 layout
        NCHW_TO_NCHW88,               //!< from nchw layout to nchw88 layout
        NCHW88_TO_NCHW,               //!< from nchw88 layout to nchw layout
        WEIGHT_NCHW_TO_NCHW88_DENSE,  //!< weight from nchw layout to nchw88
                                      //!< layout
        WEIGHT_NCHW_TO_NCHW88_GROUP,  //!< group weight from nchw layout to
                                      //!< nchw88 layout
        WEIGHT_NCHW_TO_NCHW88_CHAN,   //!< channel wise weight from nchw layout
                                      //!< to nchw88 layout
        //!< the weight layout of input is nchw output is nchw88, special for
        //!< shape weight in nchw like {64, 2, 3, 3} to {8, 3, 3, 2, 8}
        WEIGHT_HYBIRD_NCHW_NCHW88,
    };

    RelayoutPlaceholder(VarNode* src_var, LayoutType layout_type);

    /*!
     * \param src_var the input var
     * \param layout_type tensor layout transform type of this relayout
     * placeholder as described in LayoutType
     */
    static SymbolVar make(VarNode* src_var, LayoutType layout_type);

    LayoutType layout_type() const { return m_layout_type; }

private:
    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    void init_output_comp_node() override;
    const LayoutType m_layout_type;
};

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TensorReformatPass::RelayoutPlaceholder);

TensorReformatPass::RelayoutPlaceholder::RelayoutPlaceholder(
        VarNode* src_var, LayoutType layout_type)
        : Super(src_var->owner_graph(), {}, "RelayoutPlaceholder", {src_var}),
          m_layout_type{layout_type} {
    add_input({src_var});
    add_equivalence_component<ScalarHash<LayoutType>>(m_layout_type);
    add_output(None)->dtype(src_var->dtype());
}

void TensorReformatPass::RelayoutPlaceholder::scn_do_execute() {
    mgb_throw(InternalError, "RelayoutPlaceholder opr can not be executed");
}

void TensorReformatPass::RelayoutPlaceholder::init_output_comp_node() {
    output(0)->comp_node(input(0)->comp_node());
}

void TensorReformatPass::RelayoutPlaceholder::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    DepVal deps;
    for (auto i : input())
        deps.push_back({i, DepType::SHAPE});
    auto infer_shape = [this](TensorShape& dst, const InpVal& inp) {
        TensorShape inp_shape = inp.val[0].shape();
        dst = inp_shape;
        if (layout_type() == RelayoutPlaceholder::LayoutType::NCHW4_TO_NCHW32) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[4] == 4);
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1] / 8;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = inp_shape[4] * 8;
        } else if (layout_type() ==
                   RelayoutPlaceholder::LayoutType::NCHW32_TO_NCHW4) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[4] == 32);
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1] * 8;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = inp_shape[4] / 8;
        } else if (layout_type() ==
                   RelayoutPlaceholder::LayoutType::NCHW4_TO_CHWN4) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[4] == 4);
            dst[0] = inp_shape[1];
            dst[1] = inp_shape[2];
            dst[2] = inp_shape[3];
            dst[3] = inp_shape[0];
            dst[4] = inp_shape[4];
        } else if (layout_type() ==
                   RelayoutPlaceholder::LayoutType::CHWN4_TO_NCHW4) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[4] == 4);
            dst[0] = inp_shape[3];
            dst[1] = inp_shape[0];
            dst[2] = inp_shape[1];
            dst[3] = inp_shape[2];
            dst[4] = inp_shape[4];
        } else if (layout_type() ==
                   RelayoutPlaceholder::LayoutType::NCHW_TO_NCHW88) {
            mgb_assert(inp_shape.ndim == 4 && inp_shape[1] % 8 == 0);
            dst.ndim = 5;
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1] / 8;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = 8;
        } else if (layout_type() ==
                   RelayoutPlaceholder::LayoutType::NCHW88_TO_NCHW) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[4] == 8);
            dst.ndim = 4;
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1] * 8;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
        } else if (layout_type() == RelayoutPlaceholder::LayoutType::
                                            WEIGHT_NCHW_TO_NCHW88_DENSE) {
            mgb_assert(inp_shape.ndim == 4 && inp_shape[0] % 8 == 0 &&
                       inp_shape[1] % 8 == 0);
            dst.ndim = 6;
            dst[0] = inp_shape[0] / 8;
            dst[1] = inp_shape[1] / 8;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = 8;
            dst[5] = 8;
        } else if (layout_type() == RelayoutPlaceholder::LayoutType::
                                            WEIGHT_NCHW_TO_NCHW88_GROUP) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[1] % 8 == 0 &&
                       inp_shape[2] % 8 == 0);
            dst.ndim = 7;
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1] / 8;
            dst[2] = inp_shape[2] / 8;
            dst[3] = inp_shape[3];
            dst[4] = inp_shape[4];
            dst[5] = 8;
            dst[6] = 8;
        } else if (layout_type() == RelayoutPlaceholder::LayoutType::
                                            WEIGHT_NCHW_TO_NCHW88_CHAN) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[1] == 1 &&
                       inp_shape[2] == 1 && inp_shape[0] % 8 == 0);
            dst.ndim = 6;
            dst[0] = inp_shape[0] / 8;
            dst[1] = inp_shape[1];
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = inp_shape[4];
            dst[5] = 8;
        } else {
            mgb_assert(
                    layout_type() ==
                    RelayoutPlaceholder::LayoutType::WEIGHT_HYBIRD_NCHW_NCHW88);
            mgb_assert(inp_shape.ndim == 4 && inp_shape[0] % 8 == 0);
            dst.ndim = 5;
            dst[0] = inp_shape[0] / 8;
            dst[1] = inp_shape[2];
            dst[2] = inp_shape[3];
            dst[3] = inp_shape[1];
            dst[4] = 8;
        }
        return true;
    };
    mgr.register_shape_infer(output(0), {SourceType::DEP, deps, infer_shape});
}

SymbolVar TensorReformatPass::RelayoutPlaceholder::make(
        VarNode* src_var, LayoutType layout_type) {
    return src_var->owner_graph()
            ->insert_opr(
                    std::make_unique<RelayoutPlaceholder>(src_var, layout_type))
            ->output(0);
}

void TensorReformatPass::insert_pass(OptState& opt) const {
    opt.set_var_replace_check_flag(m_var_replace_check_flag);
    auto rewriter = opt.graph().make_rewriter();
    VarNodeArray new_inp_cache;
    auto on_opr = [this, &opt, &rewriter,
                   &new_inp_cache](OperatorNodeBase* opr) {
        auto it = m_opr_replace_func.find(opr->dyn_typeinfo());
        if (it != m_opr_replace_func.end()) {
            auto& new_inp = new_inp_cache;
            new_inp.clear();
            new_inp.reserve(opr->input().size());
            for (auto&& inp : opr->input()) {
                new_inp.push_back(rewriter.get_var(inp));
            }
            auto new_opr = (it->second)(opr, new_inp);
            auto &&out0 = opr->output(), &&out1 = new_opr->output();
            mgb_assert(out0.size() == out1.size(),
                       "bad opr replace: src=%s{%s} dst=%s{%s}, src.size=%zu "
                       "dst.size=%zu",
                       opr->cname(), opr->dyn_typeinfo()->name,
                       new_opr->cname(), new_opr->dyn_typeinfo()->name,
                       out0.size(), out1.size());
            for (size_t i = 0; i < out0.size(); ++i) {
                if (!out0[i]->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                    mgb_assert(!out1[i]->contain_flag(
                            VarNode::Flag::VOLATILE_CONTENT));
                    auto src = out0[i];
                    auto dst = out1[i];
                    if (opt.graph().endpoint_contain(src)) {
                        // additional process on endpoint var node
                        dst = on_graph_endpoint_var(dst, src);
                    }
                    rewriter.replace_var(src, dst, nullptr);
                }
            }
        } else {
            rewriter.auto_replace_outputs(opr);
        }
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
}

void TensorReformatPass::translate_pass(OptState& opt) const {
    ThinHashMap<RelayoutPlaceholder::LayoutType,
                thin_function<VarNode*(VarNode*)>>
            reformat;
    using LayoutType = RelayoutPlaceholder::LayoutType;
    reformat[LayoutType::NCHW4_TO_CHWN4] = [](VarNode* inp) -> VarNode* {
        megdnn::param::RelayoutFormat param;
        param.mode = megdnn::param::RelayoutFormat::Mode::NCHW4_CHWN4;
        auto reformat = opr::RelayoutFormat::make(inp, param);
        return reformat.node();
    };
    reformat[LayoutType::CHWN4_TO_NCHW4] = [](VarNode* inp) -> VarNode* {
        megdnn::param::RelayoutFormat param;
        param.mode = megdnn::param::RelayoutFormat::Mode::CHWN4_NCHW4;
        auto reformat = opr::RelayoutFormat::make(inp, param);
        return reformat.node();
    };
    reformat[LayoutType::NCHW4_TO_NCHW32] = [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0), sub(1) / 8, cv(8), sub(2), sub(3), sub(4)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0), sub(1) / 8, sub(2), sub(3), sub(4) * 8}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2, 5});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::NCHW32_TO_NCHW4] = [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0), sub(1), sub(2), sub(3), cv(8), sub(4) / 8}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0), sub(1) * 8, sub(2), sub(3), sub(4) / 8}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 4, 2, 3, 5});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::NCHW_TO_NCHW88] = [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0), sub(1) / 8, cv(8), sub(2), sub(3)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0), sub(1) / 8, sub(2), sub(3), cv(8)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::NCHW88_TO_NCHW] = [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make({sub(0), sub(1) * 8, sub(2), sub(3)}, 0);
        auto y0 = opr::Dimshuffle::make(x, {0, 1, 4, 2, 3});
        auto y1 = opr::Reshape::make(y0, tshp0);
        return y1.node();
    };
    reformat[LayoutType::WEIGHT_NCHW_TO_NCHW88_DENSE] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0) / 8, cv(8), sub(1) / 8, cv(8), sub(2), sub(3)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0) / 8, sub(1) / 8, sub(2), sub(3), cv(8), cv(8)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 2, 4, 5, 3, 1});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::WEIGHT_NCHW_TO_NCHW88_GROUP] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make({sub(0), sub(1) / 8, cv(8), sub(2) / 8,
                                        cv(8), sub(3), sub(4)},
                                       0),
             tshp1 = opr::Concat::make({sub(0), sub(1) / 8, sub(2) / 8, sub(3),
                                        sub(4), cv(8), cv(8)},
                                       0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 5, 6, 4, 2});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::WEIGHT_NCHW_TO_NCHW88_CHAN] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0) / 8, cv(8), sub(1), sub(2), sub(3), sub(4)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0) / 8, sub(1), sub(2), sub(3), sub(4), cv(8)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 2, 3, 4, 5, 1});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::WEIGHT_HYBIRD_NCHW_NCHW88] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0) / 8, cv(8), sub(1), sub(2), sub(3)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0) / 8, sub(2), sub(3), sub(1), cv(8)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 3, 4, 2, 1});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };

    auto rewriter = opt.graph().make_rewriter();
    auto on_opr = [&reformat, &rewriter](OperatorNodeBase* opr) {
        if (opr->same_type<RelayoutPlaceholder>()) {
            auto ph = try_cast_as_op<RelayoutPlaceholder>(opr);
            auto new_inp = rewriter.get_var(opr->input(0));
            mgb_assert(reformat.count(ph->layout_type()),
                       "no replace rule can be found for layout_type(%u)",
                       static_cast<uint32_t>(ph->layout_type()));
            auto new_var = reformat[ph->layout_type()](new_inp);
            rewriter.replace_var(opr->output(0), new_var,
                                 mgb_cstr_log("replace relayout placeholder"));
            return;
        }
        rewriter.auto_replace_outputs(opr);
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
}

void TensorReformatPass::apply(OptState& opt) const {
    insert_pass(opt);
    translate_pass(opt);
}

/* ================ EnableTensorCorePass =============== */
VarNode* EnableTensorCorePass::on_graph_endpoint_var(VarNode* new_var,
                                                     VarNode* orig_var) const {
    if (!orig_var->shape().eq_shape(new_var->shape())) {
        return RelayoutPlaceholder::make(
                       new_var,
                       RelayoutPlaceholder::LayoutType::NCHW32_TO_NCHW4)
                .node();
    }
    return new_var;
}

std::unique_ptr<EnableTensorCorePass>
EnableTensorCorePass::make_tensorcore_converter() {
    // replace rule for conv bias opr
    auto replace_conv_bias_opr = [](OperatorNodeBase* opr,
                                    const VarNodeArray& new_inp) {
        using Param = megdnn::param::ConvBias;
        using Format = Param::Format;
        using Sparse = Param::Sparse;
        mgb_assert(opr->input().size() == new_inp.size());
        auto& conv_bias = opr->cast_final_safe<opr::ConvBiasForward>();
        if (conv_bias.param().format != Format::NCHW4 ||
            conv_bias.output(0)->dtype().enumv() != DTypeEnum::QuantizedS8) {
            size_t nr_inps = opr->input().size();
            bool shape_has_changed = false;
            for (size_t i = 0; i < nr_inps; ++i) {
                if (!opr->input(i)->shape().eq_shape(new_inp[i]->shape())) {
                    shape_has_changed = true;
                }
            }
            MGB_MARK_USED_VAR(shape_has_changed);
            mgb_assert(
                    !shape_has_changed,
                    "EnableTensorCorePass assumes that the shape of inputs of"
                    "ConvBias operators whose output dtype is not QuantizedS8 "
                    "can not be changed in this opt pass");
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
        mgb_assert(opr->input(1)->shape().eq_shape(new_inp[1]->shape()),
                   "EnableTensorCorePass assumes that filter tensor of "
                   "conv_bias operator can not be changed by other operators");
        VarNode* orig_filter = opr->input(1);
        auto is_nchw4 = [](TensorShape shape) -> bool {
            return shape.ndim == 5 && shape[4] == 4;
        };
        auto is_nchw32 = [](TensorShape shape) -> bool {
            return shape.ndim == 5 && shape[4] == 32;
        };
        bool can_replace_nchw32 = false;
        VarNode *src = nullptr, *weight = nullptr, *bias = nullptr,
                *z_inp = nullptr;
        // process src tensor
        if (is_nchw4(new_inp[0]->shape())) {  // new input is NCHW4 layout
            size_t group = 1, icpg, ocpg;
            if (conv_bias.param().sparse == Sparse::DENSE) {
                icpg = orig_filter->shape()[1] * 4;
                ocpg = orig_filter->shape()[0];
            } else {
                mgb_assert(conv_bias.param().sparse == Sparse::GROUP);
                group = orig_filter->shape()[0];
                icpg = orig_filter->shape()[2];
                ocpg = orig_filter->shape()[1];
                if (icpg == 1 && ocpg == 1) {  // channel wise conv
                    group *= 4;
                } else {
                    icpg *= 4;
                }
            }
            // nchw32 layout need that input width and height are larger than 3
            size_t ih = new_inp[0]->shape()[2], iw = new_inp[0]->shape()[3];
            if (group == 1 && ocpg % 32 == 0 && icpg % 32 == 0 && ih >= 3 &&
                iw >= 3) {
                auto symvar = RelayoutPlaceholder::make(
                        new_inp[0],
                        RelayoutPlaceholder::LayoutType::NCHW4_TO_NCHW32);
                src = symvar.node();
                can_replace_nchw32 = true;
            } else {
                src = new_inp[0];
            }
        } else {  // new input is NCHW32 layout
            mgb_assert(is_nchw32(new_inp[0]->shape()));
            size_t group = 1, ocpg;
            if (conv_bias.param().sparse == Sparse::DENSE) {
                ocpg = orig_filter->shape()[0];
            } else {
                mgb_assert(conv_bias.param().sparse == Sparse::GROUP);
                size_t icpg = orig_filter->shape()[2];
                ocpg = orig_filter->shape()[1];
                if (icpg == 1 && ocpg == 1) {
                    group *= 4;
                } else {
                    icpg *= 4;
                }
            }
            size_t ih = new_inp[0]->shape()[2], iw = new_inp[0]->shape()[3];
            if (group == 1 && ocpg % 32 == 0 && ih >= 3 && iw >= 3) {
                can_replace_nchw32 = true;
                src = new_inp[0];
            } else {
                auto symvar = RelayoutPlaceholder::make(
                        new_inp[0],
                        RelayoutPlaceholder::LayoutType::NCHW32_TO_NCHW4);
                src = symvar.node();
            }
        }
        // process filter tensor
        if (can_replace_nchw32) {
            auto symvar = RelayoutPlaceholder::make(
                    new_inp[1],
                    RelayoutPlaceholder::LayoutType::NCHW4_TO_NCHW32);
            weight = symvar.node();
        } else {
            weight = new_inp[1];
        }
        if (new_inp.size() == 2) {
            if (can_replace_nchw32) {
                auto param = conv_bias.param();
                param.format = Format::NCHW32;
                auto new_opr = opr::ConvBiasForward::make(
                        src, weight, param, conv_bias.execution_policy(),
                        conv_bias.config());
                return new_opr.node()->owner_opr();
            } else {
                VarNodeArray inps{src, weight};
                auto new_opr = serialization::copy_opr_shallow(*opr, inps,
                                                               opr->config());
                return new_opr;
            }
        }
        auto process_inp = [&](VarNode* inp) -> VarNode* {
            if (can_replace_nchw32) {
                if (is_nchw4(inp->shape())) {
                    auto symvar = RelayoutPlaceholder::make(
                            inp,
                            RelayoutPlaceholder::LayoutType::NCHW4_TO_NCHW32);
                    return symvar.node();
                } else {
                    mgb_assert(is_nchw32(inp->shape()));
                    return inp;
                }
            } else {
                if (is_nchw4(inp->shape())) {
                    return inp;
                } else {
                    mgb_assert(is_nchw32(inp->shape()));
                    auto symvar = RelayoutPlaceholder::make(
                            inp,
                            RelayoutPlaceholder::LayoutType::NCHW32_TO_NCHW4);
                    return symvar.node();
                }
            }
        };
        // process bias tensor
        bias = process_inp(new_inp[2]);
        if (new_inp.size() == 3) {
            if (can_replace_nchw32) {
                auto param = conv_bias.param();
                param.format = Format::NCHW32;
                auto new_opr = opr::ConvBiasForward::make(
                        src, weight, bias, param, conv_bias.execution_policy(),
                        conv_bias.config());
                return new_opr.node()->owner_opr();
            } else {
                VarNodeArray inps{src, weight, bias};
                auto new_opr = serialization::copy_opr_shallow(*opr, inps,
                                                               opr->config());
                return new_opr;
            }
        }
        // process z_inp tensor
        z_inp = process_inp(new_inp[3]);
        if (can_replace_nchw32) {
            auto param = conv_bias.param();
            param.format = Format::NCHW32;
            auto new_opr = opr::ConvBiasForward::make(
                    src, weight, bias, z_inp, param,
                    conv_bias.execution_policy(), conv_bias.config());
            return new_opr.node()->owner_opr();
        }
        VarNodeArray inps{src, weight, bias, z_inp};
        auto new_opr =
                serialization::copy_opr_shallow(*opr, inps, opr->config());
        return new_opr;
    };
    // replace rule for elemwise like opr
    // for oprs support NCHW4 and NCHW32 layout
    auto replace_elemwise_like_opr = [](OperatorNodeBase* opr,
                                        const VarNodeArray new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        size_t nr_inps = new_inp.size();
        size_t nr_shape_changed = 0;
        for (size_t i = 0; i < nr_inps; ++i) {
            if (!opr->input(i)->shape().eq_shape(new_inp[i]->shape())) {
                nr_shape_changed++;
            }
        }
        if (nr_shape_changed) {
            auto inps = new_inp;
            if (nr_shape_changed >=
                nr_inps / 2) {  // NCHW32 > NCHW4 -> use NCHW32
                for (size_t i = 0; i < nr_inps; ++i) {
                    if (opr->input(i)->shape().eq_shape(new_inp[i]->shape())) {
                        auto symvar = RelayoutPlaceholder::make(
                                new_inp[i], RelayoutPlaceholder::LayoutType::
                                                    NCHW4_TO_NCHW32);
                        inps[i] = symvar.node();
                    }
                }
            } else {  // NCHW32 < NCHW4 -> use NCHW4
                for (size_t i = 0; i < nr_inps; ++i) {
                    if (!opr->input(i)->shape().eq_shape(new_inp[i]->shape())) {
                        auto symvar = RelayoutPlaceholder::make(
                                new_inp[i], RelayoutPlaceholder::LayoutType::
                                                    NCHW32_TO_NCHW4);
                        inps[i] = symvar.node();
                    }
                }
            }
            return serialization::copy_opr_shallow(*opr, inps, opr->config());
        }
        return serialization::copy_opr_shallow(*opr, new_inp, opr->config());
    };
    // for oprs only supports NCHW4 layout
    auto replace_inps_to_nchw4 = [](OperatorNodeBase* opr,
                                    const VarNodeArray new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        VarNodeArray inps = new_inp;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (!opr->input(i)->shape().eq_shape(new_inp[i]->shape())) {
                mgb_assert(opr->input(i)->shape().ndim == 5 &&
                           opr->input(i)->shape()[4] == 4);
                mgb_assert(new_inp[i]->shape().ndim == 5 &&
                           new_inp[i]->shape()[4] == 32);
                auto symvar = RelayoutPlaceholder::make(
                        new_inp[i],
                        RelayoutPlaceholder::LayoutType::NCHW32_TO_NCHW4);
                inps[i] = symvar.node();
            }
        }
        auto new_opr =
                serialization::copy_opr_shallow(*opr, inps, opr->config());
        return new_opr;
    };
    auto replace_non_nchw4_opr = [](OperatorNodeBase* opr,
                                    const VarNodeArray new_inp) {
        size_t nr_inps = opr->input().size();
        bool shape_has_changed = false;
        for (size_t i = 0; i < nr_inps; ++i) {
            if (!opr->input(i)->shape().eq_shape(new_inp[i]->shape())) {
                shape_has_changed = true;
            }
        }
        mgb_assert(!shape_has_changed,
                   "EnableTensorCorePass assumes that inputs' shape of "
                   "non-nchw4 operators "
                   "can not be changed in this opt "
                   "pass");
        return serialization::copy_opr_shallow(*opr, new_inp, opr->config());

    };
    auto replace_warp_affine_opr =
            [replace_inps_to_nchw4, replace_non_nchw4_opr](
                    OperatorNodeBase* opr, const VarNodeArray new_inp) {
                using Param = opr::WarpAffineForward::Param;
                using Format = Param::Format;
                mgb_assert(opr->input().size() == new_inp.size());
                auto& warp = opr->cast_final_safe<opr::WarpAffineForward>();
                if (warp.param().format != Format::NCHW4) {
                    return replace_non_nchw4_opr(opr, new_inp);
                }
                return replace_inps_to_nchw4(opr, new_inp);
            };
    auto replace_warp_perspective_opr =
            [replace_inps_to_nchw4, replace_non_nchw4_opr](
                    OperatorNodeBase* opr, const VarNodeArray new_inp) {
                using Param = opr::WarpPerspectiveForward::Param;
                using Format = Param::Format;
                mgb_assert(opr->input().size() == new_inp.size());
                auto& warp =
                        opr->cast_final_safe<opr::WarpPerspectiveForward>();
                if (warp.param().format != Format::NCHW4) {
                    return replace_non_nchw4_opr(opr, new_inp);
                }
                return replace_inps_to_nchw4(opr, new_inp);
            };
    auto replace_resize_opr = [replace_inps_to_nchw4, replace_non_nchw4_opr](
                                      OperatorNodeBase* opr,
                                      const VarNodeArray new_inp) {
        using Param = opr::ResizeForward::Param;
        using Format = Param::Format;
        mgb_assert(opr->input().size() == new_inp.size());
        auto& resize = opr->cast_final_safe<opr::ResizeForward>();
        if (resize.param().format != Format::NCHW4) {
            return replace_non_nchw4_opr(opr, new_inp);
        }
        return replace_inps_to_nchw4(opr, new_inp);
    };
    auto replace_pooling_opr = [replace_non_nchw4_opr](
                                       OperatorNodeBase* opr,
                                       const VarNodeArray new_inp) {
        using Param = opr::PoolingForward::Param;
        using Format = Param::Format;
        mgb_assert(opr->input().size() == new_inp.size());
        auto& pooling = opr->cast_final_safe<opr::PoolingForward>();
        if (pooling.param().format != Format::NCHW4) {
            return replace_non_nchw4_opr(opr, new_inp);
        }
        size_t nr_inps = opr->input().size();
        MGB_MARK_USED_VAR(nr_inps);
        mgb_assert(nr_inps == 1);
        if (!opr->input(0)->shape().eq_shape(new_inp[0]->shape())) {
            mgb_assert(opr->input(0)->shape().ndim == 5 &&
                       opr->input(0)->shape()[4] == 4);
            mgb_assert(new_inp[0]->shape().ndim == 5 &&
                       new_inp[0]->shape()[4] == 32);
            auto new_param = pooling.param();
            new_param.format = Format::NCHW32;
            auto new_pooling = opr::PoolingForward::make(new_inp[0], new_param,
                                                         opr->config());
            return new_pooling.node()->owner_opr();
        }
        return serialization::copy_opr_shallow(*opr, new_inp, opr->config());
    };
    auto ret = std::make_unique<EnableTensorCorePass>();
    ret->set_var_replace_check_flag(VarReplaceCheckFlag::NOCHECK);
    auto&& replace_func = ret->m_opr_replace_func;
    replace_func[opr::ConvBiasForward::typeinfo()] = replace_conv_bias_opr;

    // elemwise like
    replace_func[opr::Elemwise::typeinfo()] = replace_elemwise_like_opr;
    replace_func[opr::TypeCvt::typeinfo()] = replace_elemwise_like_opr;
    replace_func[opr::ElemwiseMultiType::typeinfo()] =
            replace_elemwise_like_opr;
    replace_func[opr::PowC::typeinfo()] = replace_elemwise_like_opr;

    // format aware
    replace_func[opr::PoolingForward::typeinfo()] = replace_pooling_opr;
    replace_func[opr::WarpAffineForward::typeinfo()] = replace_warp_affine_opr;
    replace_func[opr::WarpPerspectiveForward::typeinfo()] =
            replace_warp_perspective_opr;
    replace_func[opr::ResizeForward::typeinfo()] = replace_resize_opr;

    // to nchw4
    replace_func[opr::Reduce::typeinfo()] = replace_inps_to_nchw4;
    replace_func[opr::Concat::typeinfo()] = replace_inps_to_nchw4;
    replace_func[opr::Reshape::typeinfo()] = replace_inps_to_nchw4;
    replace_func[opr::GetVarShape::typeinfo()] = replace_inps_to_nchw4;
    replace_func[opr::Dimshuffle::typeinfo()] = replace_inps_to_nchw4;
    return ret;
}

/* ================ EnableCHWN4Pass =============== */
VarNode* EnableCHWN4Pass::on_graph_endpoint_var(VarNode* new_var,
                                                VarNode* /* orig_var */) const {
    if (m_varshape_changed.count(new_var)) {
        return RelayoutPlaceholder::make(
                       new_var, RelayoutPlaceholder::LayoutType::CHWN4_TO_NCHW4)
                .node();
    }
    return new_var;
}

std::unique_ptr<EnableCHWN4Pass> EnableCHWN4Pass::make_chwn4_converter() {
    auto ret = std::make_unique<EnableCHWN4Pass>();
    ret->set_var_replace_check_flag(VarReplaceCheckFlag::NOCHECK);
    auto&& replace_func = ret->m_opr_replace_func;
    auto&& varshape_changed = ret->m_varshape_changed;
    // replace rule for conv bias opr
    auto replace_conv_bias_opr = [&varshape_changed](
                                         OperatorNodeBase* opr,
                                         const VarNodeArray& new_inp) {
        using Param = megdnn::param::ConvBias;
        using Format = Param::Format;
        mgb_assert(opr->input().size() == new_inp.size());
        auto& conv_bias = opr->cast_final_safe<opr::ConvBiasForward>();
        if (conv_bias.param().format != Format::NCHW4 ||
            conv_bias.output(0)->dtype().enumv() != DTypeEnum::QuantizedS8) {
            size_t nr_inps = new_inp.size();
            bool shape_has_changed = false;
            for (size_t i = 0; i < nr_inps; ++i) {
                if (varshape_changed.count(new_inp[i])) {
                    shape_has_changed = true;
                    break;
                }
            }
            mgb_assert(
                    !shape_has_changed,
                    "EnableCHWN4Pass assumes that the shape of inputs of"
                    "ConvBias operators whose output dtype is not QuantizedS8 "
                    "can not be changed in this opt pass");
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
        mgb_assert(varshape_changed.count(new_inp[1]) == 0,
                   "EnableCHWN4Pass assumes that filter tensor of "
                   "conv_bias operator can not be changed by other operators");
        VarNode *src = nullptr, *weight = nullptr, *bias = nullptr,
                *z_inp = nullptr;
        // process src tensor
        if (varshape_changed.count(new_inp[0]) ==
            0) {  // new input is NCHW4 layout
            // currently not support group conv
            auto symvar = RelayoutPlaceholder::make(
                    new_inp[0],
                    RelayoutPlaceholder::LayoutType::NCHW4_TO_CHWN4);
            src = symvar.node();
        } else {  // new input is NCHW32 layout
            src = new_inp[0];
        }
        // process weight tensor
        {
            auto symvar = RelayoutPlaceholder::make(
                    new_inp[1],
                    RelayoutPlaceholder::LayoutType::NCHW4_TO_CHWN4);
            weight = symvar.node();
        }
        if (new_inp.size() == 2) {
            auto param = conv_bias.param();
            param.format = Format::CHWN4;
            auto new_opr = opr::ConvBiasForward::make(
                    src, weight, param, conv_bias.execution_policy(),
                    conv_bias.config());
            varshape_changed.insert(new_opr.node());
            return new_opr.node()->owner_opr();
        }
        auto process_inp = [&](VarNode* inp) -> VarNode* {
            if (varshape_changed.count(inp) == 0) {
                auto symvar = RelayoutPlaceholder::make(
                        inp, RelayoutPlaceholder::LayoutType::NCHW4_TO_CHWN4);
                return symvar.node();
            } else {
                return inp;
            }
        };
        // process bias tensor
        bias = process_inp(new_inp[2]);
        if (new_inp.size() == 3) {
            auto param = conv_bias.param();
            param.format = Format::CHWN4;
            auto new_opr = opr::ConvBiasForward::make(
                    src, weight, bias, param, conv_bias.execution_policy(),
                    conv_bias.config());
            varshape_changed.insert(new_opr.node());
            return new_opr.node()->owner_opr();
        }
        // process z_inp tensor
        z_inp = process_inp(new_inp[3]);
        auto param = conv_bias.param();
        param.format = Format::CHWN4;
        auto new_opr = opr::ConvBiasForward::make(
                src, weight, bias, z_inp, param, conv_bias.execution_policy(),
                conv_bias.config());
        varshape_changed.insert(new_opr.node());
        return new_opr.node()->owner_opr();
    };
    // replace rule for elemwise like opr
    // for oprs support NCHW4 and CHWN4 layout
    auto replace_elemwise_like_opr = [&varshape_changed](
                                             OperatorNodeBase* opr,
                                             const VarNodeArray new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        size_t nr_inps = new_inp.size();
        size_t nr_shape_changed = 0;
        for (size_t i = 0; i < nr_inps; ++i) {
            if (varshape_changed.count(new_inp[i])) {
                nr_shape_changed++;
            }
        }
        if (nr_shape_changed) {
            auto inps = new_inp;
            if (nr_shape_changed >= nr_inps / 2) {  // CHWN4 > NCHW4 -> use CHWN4
                for (size_t i = 0; i < nr_inps; ++i) {
                    if (varshape_changed.count(new_inp[i]) == 0) {
                        auto symvar = RelayoutPlaceholder::make(
                                new_inp[i], RelayoutPlaceholder::LayoutType::
                                                    NCHW4_TO_CHWN4);
                        inps[i] = symvar.node();
                    }
                }
                auto new_opr = serialization::copy_opr_shallow(*opr, inps,
                                                               opr->config());
                varshape_changed.insert(new_opr->output(0));
                return new_opr;
            } else {  // CHWN4 < NCHW4 -> use NCHW4
                for (size_t i = 0; i < nr_inps; ++i) {
                    if (varshape_changed.count(new_inp[i])) {
                        auto symvar = RelayoutPlaceholder::make(
                                new_inp[i], RelayoutPlaceholder::LayoutType::
                                                    CHWN4_TO_NCHW4);
                        inps[i] = symvar.node();
                    }
                }
                return serialization::copy_opr_shallow(*opr, inps,
                                                       opr->config());
            }
        }
        return serialization::copy_opr_shallow(*opr, new_inp, opr->config());
    };
    // for oprs only supports NCHW4 layout
    auto replace_inps_to_nchw4 = [&varshape_changed](
                                         OperatorNodeBase* opr,
                                         const VarNodeArray new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        VarNodeArray inps = new_inp;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (varshape_changed.count(new_inp[i])) {
                auto symvar = RelayoutPlaceholder::make(
                        new_inp[i],
                        RelayoutPlaceholder::LayoutType::CHWN4_TO_NCHW4);
                inps[i] = symvar.node();
            }
        }
        auto new_opr =
                serialization::copy_opr_shallow(*opr, inps, opr->config());
        return new_opr;
    };
    auto replace_non_nchw4_opr = [&varshape_changed](
                                         OperatorNodeBase* opr,
                                         const VarNodeArray new_inp) {
        size_t nr_inps = opr->input().size();
        bool shape_has_changed = false;
        for (size_t i = 0; i < nr_inps; ++i) {
            if (varshape_changed.count(new_inp[i])) {
                shape_has_changed = true;
            }
        }
        mgb_assert(!shape_has_changed,
                   "EnableCHWN4Pass assumes that inputs' shape of "
                   "non-nchw4 operators "
                   "can not be changed in this opt "
                   "pass");
        return serialization::copy_opr_shallow(*opr, new_inp, opr->config());

    };
    // capture by copy to avoid use after return
    auto replace_warp_affine_opr =
            [replace_inps_to_nchw4, replace_non_nchw4_opr](
                    OperatorNodeBase* opr, const VarNodeArray new_inp) {
                using Param = opr::WarpAffineForward::Param;
                using Format = Param::Format;
                mgb_assert(opr->input().size() == new_inp.size());
                auto& warp = opr->cast_final_safe<opr::WarpAffineForward>();
                if (warp.param().format != Format::NCHW4) {
                    return replace_non_nchw4_opr(opr, new_inp);
                }
                return replace_inps_to_nchw4(opr, new_inp);
            };
    auto replace_warp_perspective_opr =
            [replace_inps_to_nchw4, replace_non_nchw4_opr](
                    OperatorNodeBase* opr, const VarNodeArray new_inp) {
                using Param = opr::WarpPerspectiveForward::Param;
                using Format = Param::Format;
                mgb_assert(opr->input().size() == new_inp.size());
                auto& warp =
                        opr->cast_final_safe<opr::WarpPerspectiveForward>();
                if (warp.param().format != Format::NCHW4) {
                    return replace_non_nchw4_opr(opr, new_inp);
                }
                return replace_inps_to_nchw4(opr, new_inp);
            };
    auto replace_resize_opr = [replace_inps_to_nchw4, replace_non_nchw4_opr](
                                      OperatorNodeBase* opr,
                                      const VarNodeArray new_inp) {
        using Param = opr::ResizeForward::Param;
        using Format = Param::Format;
        mgb_assert(opr->input().size() == new_inp.size());
        auto& resize = opr->cast_final_safe<opr::ResizeForward>();
        if (resize.param().format != Format::NCHW4) {
            return replace_non_nchw4_opr(opr, new_inp);
        }
        return replace_inps_to_nchw4(opr, new_inp);
    };
    auto replace_pooling_opr = [&varshape_changed, replace_non_nchw4_opr](
                                       OperatorNodeBase* opr,
                                       const VarNodeArray new_inp) {
        using Param = opr::PoolingForward::Param;
        using Format = Param::Format;
        mgb_assert(opr->input().size() == new_inp.size());
        auto& pooling = opr->cast_final_safe<opr::PoolingForward>();
        if (pooling.param().format != Format::NCHW4) {
            return replace_non_nchw4_opr(opr, new_inp);
        }
        size_t nr_inps = opr->input().size();
        MGB_MARK_USED_VAR(nr_inps);
        mgb_assert(nr_inps == 1);
        if (varshape_changed.count(new_inp[0])) {
            auto new_param = pooling.param();
            new_param.format = Format::CHWN4;
            auto new_pooling = opr::PoolingForward::make(new_inp[0], new_param,
                                                         opr->config());
            varshape_changed.insert(new_pooling.node());
            return new_pooling.node()->owner_opr();
        }
        return serialization::copy_opr_shallow(*opr, new_inp, opr->config());
    };
    replace_func[opr::ConvBiasForward::typeinfo()] = replace_conv_bias_opr;

    // elemwise like
    replace_func[opr::Elemwise::typeinfo()] = replace_elemwise_like_opr;
    replace_func[opr::TypeCvt::typeinfo()] = replace_elemwise_like_opr;
    replace_func[opr::ElemwiseMultiType::typeinfo()] =
            replace_elemwise_like_opr;
    replace_func[opr::PowC::typeinfo()] = replace_elemwise_like_opr;

    // format aware
    replace_func[opr::PoolingForward::typeinfo()] = replace_pooling_opr;
    replace_func[opr::WarpAffineForward::typeinfo()] = replace_warp_affine_opr;
    replace_func[opr::WarpPerspectiveForward::typeinfo()] =
            replace_warp_perspective_opr;
    replace_func[opr::ResizeForward::typeinfo()] = replace_resize_opr;

    // to nchw4
    replace_func[opr::Reduce::typeinfo()] = replace_inps_to_nchw4;
    replace_func[opr::Concat::typeinfo()] = replace_inps_to_nchw4;
    replace_func[opr::Reshape::typeinfo()] = replace_inps_to_nchw4;
    replace_func[opr::GetVarShape::typeinfo()] = replace_inps_to_nchw4;
    replace_func[opr::Dimshuffle::typeinfo()] = replace_inps_to_nchw4;
    replace_func[opr::BatchConvBias::typeinfo()] = replace_inps_to_nchw4;
    return ret;
}

/* ================ EnableNchwxxPass =============== */
VarNode* EnableNchwxxPass::on_graph_endpoint_var(VarNode* new_var,
                                                 VarNode* orig_var) const {
    if (!orig_var->shape().eq_shape(new_var->shape())) {
        return RelayoutPlaceholder::make(
                       new_var, RelayoutPlaceholder::LayoutType::NCHW88_TO_NCHW)
                .node();
    }
    return new_var;
}

std::unique_ptr<EnableNchwxxPass> EnableNchwxxPass::make_nchwxx_converter(
        size_t pack_c_size) {
    auto ret = std::make_unique<EnableNchwxxPass>();
    ret->set_var_replace_check_flag(VarReplaceCheckFlag::NOCHECK);
    //! First is whether the conv can trans to nchwxx, second is the filter
    //! trans mode
    using RelayoutMode = RelayoutPlaceholder::LayoutType;
    using TestFilterResult = std::pair<TransType, RelayoutMode>;
    RelayoutMode weight_to_nchwxx_mode_dense =
            RelayoutMode::WEIGHT_NCHW_TO_NCHW88_DENSE;
    RelayoutMode weight_to_nchwxx_mode_group =
            RelayoutMode::WEIGHT_NCHW_TO_NCHW88_GROUP;
    RelayoutMode weight_to_nchwxx_mode_chan =
            RelayoutMode::WEIGHT_NCHW_TO_NCHW88_CHAN;
    RelayoutMode hybrid_nchw_nchwxx = RelayoutMode::WEIGHT_HYBIRD_NCHW_NCHW88;
    RelayoutMode src_to_nchwxx_mode = RelayoutMode::NCHW_TO_NCHW88;
    RelayoutMode src_to_nchw_mode = RelayoutMode::NCHW88_TO_NCHW;
    megdnn::param::ConvBias::Format conv_bias_format =
            megdnn::param::ConvBias::Format::NCHW88;
    megdnn::param::Convolution::Format conv_format =
            megdnn::param::ConvolutionV0::Format::NCHW88;
    megdnn::param::Pooling::Format pooling_format =
            megdnn::param::Pooling::Format::NCHW88;
    std::string convter_pass_name = "conv_format_nchw88";
    mgb_assert(pack_c_size == static_cast<size_t>(8),
               "The ConvertFormatPass to nchwxx only support NCHW88 now !");
    auto test_trans_nchwxx =
            [pack_c_size, weight_to_nchwxx_mode_dense,
             weight_to_nchwxx_mode_group, weight_to_nchwxx_mode_chan,
             hybrid_nchw_nchwxx](
                    const megdnn::param::Convolution::Sparse conv_mode,
                    const VarNode* filter) -> TestFilterResult {
        TestFilterResult ret{TransType::TRANS_NONE, {}};
        if (conv_mode == megdnn::param::Convolution::Sparse::DENSE) {
            size_t IC = filter->shape()[1];
            size_t OC = filter->shape()[0];
            if ((IC % pack_c_size == 0) && (OC % pack_c_size == 0)) {
                ret.first = TransType::TRANS_PURE_NCHWXX;
                ret.second = weight_to_nchwxx_mode_dense;
            } else if (IC < pack_c_size && OC % pack_c_size == 0) {
                ret.first = TransType::TRANS_HYBIRD_NCHWXX;
                ret.second = hybrid_nchw_nchwxx;
            }
        } else {
            mgb_assert(conv_mode == megdnn::param::Convolution::Sparse::GROUP);
            size_t group = filter->shape()[0];
            size_t ocpg = filter->shape()[1];
            size_t icpg = filter->shape()[2];
            if (icpg == 1 && ocpg == 1 && (group % pack_c_size == 0)) {
                ret.first = TransType::TRANS_PURE_NCHWXX;
                ret.second = weight_to_nchwxx_mode_chan;
            } else if ((icpg % pack_c_size == 0) && (ocpg % pack_c_size == 0)) {
                ret.first = TransType::TRANS_PURE_NCHWXX;
                ret.second = weight_to_nchwxx_mode_group;
            }
        }
        return ret;
    };
    auto replace_conv_opr = [test_trans_nchwxx, conv_format, src_to_nchwxx_mode,
                             src_to_nchw_mode](OperatorNodeBase* opr,
                                               const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& conv_opr = opr->cast_final_safe<opr::ConvolutionForward>();
        mgb_assert(conv_opr.param().format ==
                           megdnn::param::Convolution::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NCHWXX");
        auto is_trans = test_trans_nchwxx(conv_opr.param().sparse, new_inp[1]);
        //! can not trans to nchwxx
        if (is_trans.first == TransType::TRANS_NONE) {
            mgb_assert(new_inp[1]->shape().ndim == 4 ||
                               new_inp[1]->shape().ndim == 5,
                       "The origin filter is not NCHW mode");
            VarNodeArray temp_inp = new_inp;
            //! if src is nchwxx, should RelayoutPlaceholder to nchw
            if (temp_inp[0]->shape().ndim == 5) {
                auto new_src =
                        RelayoutPlaceholder::make(new_inp[0], src_to_nchw_mode);
                temp_inp[0] = new_src.node();
            }
            auto new_opr = serialization::copy_opr_shallow(*opr, temp_inp,
                                                           opr->config());
            return new_opr;
        } else if (is_trans.first == TransType::TRANS_PURE_NCHWXX) {
            //! filter trans to nchwxx mode
            mgb_assert(new_inp[1]->shape().ndim == 4 ||
                               new_inp[1]->shape().ndim == 5,
                       "The origin filter is not NCHW mode");
            VarNode *conv_src = new_inp[0], *conv_filter = new_inp[1];
            auto new_filter =
                    RelayoutPlaceholder::make(new_inp[1], is_trans.second);
            conv_filter = new_filter.node();
            //! src trans to nchwxx mode
            if (new_inp[0]->shape().ndim != 5) {
                mgb_assert(new_inp[0]->shape().ndim == 4);
                auto new_src = RelayoutPlaceholder::make(new_inp[0],
                                                         src_to_nchwxx_mode);
                conv_src = new_src.node();
            }
            auto new_param = conv_opr.param();
            new_param.format = conv_format;
            mgb_assert(conv_src->shape().ndim == 5 &&
                               conv_filter->shape().ndim >= 6,
                       "The conv src dim is not trans to nchwxx");
            auto new_conv_opr = opr::Convolution::make(
                    conv_src, conv_filter, new_param,
                    conv_opr.execution_policy(), conv_opr.config());
            OperatorNodeBase* new_opr = new_conv_opr.node()->owner_opr();
            mgb_assert(new_conv_opr.shape().ndim == 5,
                       "The conv dst dim is not trans to nchwxx");
            return new_opr;
        } else {
            mgb_assert(is_trans.first == TransType::TRANS_HYBIRD_NCHWXX);
            VarNode *conv_src = new_inp[0], *conv_filter = new_inp[1];
            auto new_filter =
                    RelayoutPlaceholder::make(new_inp[1], is_trans.second);
            conv_filter = new_filter.node();
            mgb_assert(conv_src->shape().ndim == 4 &&
                               conv_filter->shape().ndim == 5,
                       "The src and filter is OK");
            auto new_param = conv_opr.param();
            new_param.format = conv_format;
            auto new_conv_opr = opr::Convolution::make(
                    conv_src, conv_filter, new_param,
                    conv_opr.execution_policy(), conv_opr.config());
            OperatorNodeBase* new_opr = new_conv_opr.node()->owner_opr();
            mgb_assert(new_conv_opr.shape().ndim == 5,
                       "The conv dst dim is not trans to nchwxx");
            return new_opr;
        }
    };

    auto replace_conv_bias_opr = [test_trans_nchwxx, conv_bias_format,
                                  src_to_nchwxx_mode, src_to_nchw_mode](
                                         OperatorNodeBase* opr,
                                         const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& conv_bias_opr = opr->cast_final_safe<opr::ConvBiasForward>();
        mgb_assert(conv_bias_opr.param().format ==
                           megdnn::param::ConvBias::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NCHWXX");
        auto is_trans =
                test_trans_nchwxx(conv_bias_opr.param().sparse, new_inp[1]);
        //! can not trans to nchwxx
        if (is_trans.first == TransType::TRANS_NONE) {
            mgb_assert(new_inp[1]->shape().ndim == 4 ||
                               new_inp[1]->shape().ndim == 5,
                       "The origin filter is not NCHW mode");
            VarNodeArray temp_inp = new_inp;
            //! if src is nchwxx, should RelayoutPlaceholder to nchw
            if (temp_inp[0]->shape().ndim == 5) {
                auto new_src =
                        RelayoutPlaceholder::make(new_inp[0], src_to_nchw_mode);
                temp_inp[0] = new_src.node();
            }
            //! the bias is nchwxx
            if (temp_inp[2]->shape().ndim == 5) {
                auto new_bias =
                        RelayoutPlaceholder::make(new_inp[2], src_to_nchw_mode);
                temp_inp[2] = new_bias.node();
            }
            auto new_opr = serialization::copy_opr_shallow(*opr, temp_inp,
                                                           opr->config());
            return new_opr;
        } else if (is_trans.first == TransType::TRANS_PURE_NCHWXX) {
            VarNode *conv_bias_src = new_inp[0], *conv_bias_filter = new_inp[1],
                    *conv_bias_bias = new_inp[2];
            //! filter trans to nchwxx mode
            mgb_assert(new_inp[1]->shape().ndim == 4 ||
                               new_inp[1]->shape().ndim == 5,
                       "The origin filter is not NCHW mode");
            auto new_filter =
                    RelayoutPlaceholder::make(new_inp[1], is_trans.second);
            conv_bias_filter = new_filter.node();
            //! src trans to nchwxx mode
            if (new_inp[0]->shape().ndim != 5) {
                mgb_assert(new_inp[0]->shape().ndim == 4);
                auto new_src = RelayoutPlaceholder::make(new_inp[0],
                                                         src_to_nchwxx_mode);
                conv_bias_src = new_src.node();
            }
            //! bias trans to nchwxx mode, bias may be scale
            if (new_inp[2]->shape().ndim == 4) {
                auto new_bias = RelayoutPlaceholder::make(new_inp[2],
                                                          src_to_nchwxx_mode);
                conv_bias_bias = new_bias.node();
            }

            auto new_param = conv_bias_opr.param();
            new_param.format = conv_bias_format;
            mgb_assert(conv_bias_src->shape().ndim == 5 &&
                               conv_bias_filter->shape().ndim >= 6,
                       "The conv_bias src dim is not trans to nchwxx");
            auto new_conv_bias_opr = opr::ConvBias::make(
                    conv_bias_src, conv_bias_filter, conv_bias_bias, new_param,
                    conv_bias_opr.execution_policy(), conv_bias_opr.config());
            OperatorNodeBase* new_opr = new_conv_bias_opr.node()->owner_opr();
            mgb_assert(new_conv_bias_opr.shape().ndim == 5,
                       "The conv_bias dst dim is not trans to nchwxx");
            return new_opr;
        } else {
            mgb_assert(is_trans.first == TransType::TRANS_HYBIRD_NCHWXX);
            VarNode *conv_bias_src = new_inp[0], *conv_bias_filter = new_inp[1],
                    *conv_bias_bias = new_inp[2];
            auto new_filter =
                    RelayoutPlaceholder::make(new_inp[1], is_trans.second);
            conv_bias_filter = new_filter.node();
            //! bias trans to nchwxx mode, bias may be scale
            if (new_inp[2]->shape().ndim == 4) {
                auto new_bias = RelayoutPlaceholder::make(new_inp[2],
                                                          src_to_nchwxx_mode);
                conv_bias_bias = new_bias.node();
            }
            mgb_assert(conv_bias_src->shape().ndim == 4 &&
                       conv_bias_filter->shape().ndim == 5);
            mgb_assert((conv_bias_bias->shape().ndim == 5) ||
                       conv_bias_bias->shape().is_scalar());
            auto new_param = conv_bias_opr.param();
            new_param.format = conv_bias_format;
            auto new_conv_bias_opr = opr::ConvBias::make(
                    conv_bias_src, conv_bias_filter, new_param,
                    conv_bias_opr.execution_policy(), conv_bias_opr.config());
            OperatorNodeBase* new_opr = new_conv_bias_opr.node()->owner_opr();
            mgb_assert(new_conv_bias_opr.shape().ndim == 5,
                       "The conv dst dim is not trans to nchwxx");
            return new_opr;
        }
    };

    auto replace_pooling_opr = [=](OperatorNodeBase* opr,
                                  const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& pooling_opr = opr->cast_final_safe<opr::PoolingForward>();
        mgb_assert(pooling_opr.param().format ==
                           megdnn::param::Pooling::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NCHWxx");
        VarNode* inp = new_inp[0];
        //! if input is nchwxx
        if (inp->shape().ndim == 5) {
            auto new_param = pooling_opr.param();
            new_param.format = pooling_format;
            auto new_pooling_opr =
                    opr::PoolingForward::make(inp, new_param, opr->config());
            mgb_assert(new_pooling_opr.shape().ndim == 5,
                       "The pooling dst dim is not trans to nchwxx");
            return new_pooling_opr.node()->owner_opr();
        } else {
            auto new_opr = serialization::copy_opr_shallow(*opr, new_inp,
                                                           opr->config());
            return new_opr;
        }
    };

    auto replace_elemwise_opr = [=](OperatorNodeBase* opr,
                                    const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        bool has_inp_changed = false;
        for (size_t i = 0; i < opr->input().size(); i++) {
            if (new_inp[i]->shape().ndim == 5) {
                has_inp_changed = true;
                break;
            }
        }
        if (has_inp_changed) {
            auto temp_inp = new_inp;
            for (size_t i = 0; i < opr->input().size(); i++) {
                if (new_inp[i]->shape().ndim == 4) {
                    auto new_var = RelayoutPlaceholder::make(
                            new_inp[i], src_to_nchwxx_mode);
                    temp_inp[i] = new_var.node();
                } else {
                    mgb_assert((new_inp[i]->shape().ndim == 5) ||
                               new_inp[i]->shape().is_scalar());
                }
            }
            return serialization::copy_opr_shallow(*opr, temp_inp,
                                                   opr->config());
        } else {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
    };

    auto relayout_inp_to_nchw = [=](OperatorNodeBase* opr,
                                  const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        VarNodeArray temp_inp = new_inp;
        for (size_t i = 0; i < opr->input().size(); i++) {
            if (!opr->input(i)->shape().eq_shape(new_inp[i]->shape())) {
                mgb_assert(opr->input(i)->shape().ndim == 4);
                mgb_assert(new_inp[i]->shape().ndim == 5);
                auto new_var =
                        RelayoutPlaceholder::make(new_inp[i], src_to_nchw_mode);
                temp_inp[i] = new_var.node();
            }
        }
        return serialization::copy_opr_shallow(*opr, temp_inp, opr->config());
    };

    ret->set_name(convter_pass_name);
    auto&& replace_func = ret->m_opr_replace_func;
    //! supportted nchwxx
    replace_func[opr::Convolution::typeinfo()] = replace_conv_opr;
    replace_func[opr::ConvBias::typeinfo()] = replace_conv_bias_opr;
    replace_func[opr::PoolingForward::typeinfo()] = replace_pooling_opr;
    replace_func[opr::Elemwise::typeinfo()] = replace_elemwise_opr;
    replace_func[opr::TypeCvt::typeinfo()] = replace_elemwise_opr;
    replace_func[opr::ElemwiseMultiType::typeinfo()] = replace_elemwise_opr;
    replace_func[opr::PowC::typeinfo()] = replace_elemwise_opr;
    //! not support yet
    replace_func[opr::ConvolutionBackwardData::typeinfo()] =
            relayout_inp_to_nchw;
    replace_func[opr::Subtensor::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Concat::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Reshape::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::GetVarShape::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Dimshuffle::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Reduce::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::AssertEqual::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Broadcast::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::IncrSubtensor::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::ResizeForward::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::WarpPerspectiveForward::typeinfo()] =
            relayout_inp_to_nchw;
    replace_func[opr::WarpAffineForward::typeinfo()] = relayout_inp_to_nchw;
    return ret;
}

/* ==================== ShuffleShuffleRemovePass ================= */
class ShuffleShuffleRemovePass::Impl {
    using TensorFormat = opr::ConvBias::Param::Format;

    OptState& m_opt_state;
    ThinHashMap<std::pair<TensorFormat, TensorFormat>,
                thin_function<VarNode*(VarNode*)>>
            m_reformat;

    class AbstractShuffleOpr;

    void detect_shuffle_operations();
    void do_replace();

public:
    Impl(OptState& opt_state) : m_opt_state{opt_state} {
        m_reformat[std::make_pair(TensorFormat::NCHW, TensorFormat::NCHW4)] =
                [](VarNode* inp) -> VarNode* {
            auto x = SymbolVar(inp);
            auto xshp = opr::GetVarShape::make(x);

            auto cv = [&x](int v) { return x.make_scalar(v); };
            auto sub = [&xshp, &cv](int idx) {
                return opr::IndexAt::make(xshp, {{0, cv(idx)}});
            };
            auto tshp = opr::Concat::make(
                    {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
            auto y0 = opr::Reshape::make(x, tshp);
            auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});
            return y1.node();
        };

        m_reformat[std::make_pair(TensorFormat::NCHW, TensorFormat::NCHW32)] =
                [](VarNode* inp) -> VarNode* {
            auto x = SymbolVar(inp);
            auto xshp = opr::GetVarShape::make(x);

            auto cv = [&x](int v) { return x.make_scalar(v); };
            auto sub = [&xshp, &cv](int idx) {
                return opr::IndexAt::make(xshp, {{0, cv(idx)}});
            };
            auto tshp = opr::Concat::make(
                    {sub(0), sub(1) / 32, cv(32), sub(2), sub(3)}, 0);
            auto y0 = opr::Reshape::make(x, tshp);
            auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});
            return y1.node();
        };

        m_reformat[std::make_pair(TensorFormat::NCHW4, TensorFormat::NCHW)] =
                [](VarNode* inp) -> VarNode* {
            mgb_assert(inp->shape().ndim == 5 && inp->shape()[4] == 4);
            auto x = SymbolVar(inp);
            auto xshp = opr::GetVarShape::make(x);

            auto cv = [&x](int v) { return x.make_scalar(v); };
            auto sub = [&xshp, &cv](int idx) {
                return opr::IndexAt::make(xshp, {{0, cv(idx)}});
            };
            auto tshp =
                    opr::Concat::make({sub(0), sub(1) * 4, sub(2), sub(3)}, 0);
            auto y0 = opr::Dimshuffle::make(x, {0, 1, 4, 2, 3});
            auto y1 = opr::Reshape::make(y0, tshp);
            return y1.node();
        };

        m_reformat[std::make_pair(TensorFormat::NCHW32, TensorFormat::NCHW)] =
                [](VarNode* inp) -> VarNode* {
            mgb_assert(inp->shape().ndim == 5 && inp->shape()[4] == 32);
            auto x = SymbolVar(inp);
            auto xshp = opr::GetVarShape::make(x);

            auto cv = [&x](int v) { return x.make_scalar(v); };
            auto sub = [&xshp, &cv](int idx) {
                return opr::IndexAt::make(xshp, {{0, cv(idx)}});
            };
            auto tshp =
                    opr::Concat::make({sub(0), sub(1) * 32, sub(2), sub(3)}, 0);
            auto y0 = opr::Dimshuffle::make(x, {0, 1, 4, 2, 3});
            auto y1 = opr::Reshape::make(y0, tshp);
            return y1.node();
        };

        m_reformat[std::make_pair(TensorFormat::NCHW4, TensorFormat::NCHW32)] =
                [](VarNode* inp) -> VarNode* {
            mgb_assert(inp->shape().ndim == 5 && inp->shape()[4] == 4);
            auto x = SymbolVar(inp);
            auto xshp = opr::GetVarShape::make(x);

            auto cv = [&x](int v) { return x.make_scalar(v); };
            auto sub = [&xshp, &cv](int idx) {
                return opr::IndexAt::make(xshp, {{0, cv(idx)}});
            };
            auto tshp0 = opr::Concat::make(
                         {sub(0), sub(1) / 8, cv(8), sub(2), sub(3), sub(4)},
                         0),
                 tshp1 = opr::Concat::make(
                         {sub(0), sub(1) / 8, sub(2), sub(3), sub(4) * 8}, 0);
            auto y0 = opr::Reshape::make(x, tshp0);
            auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2, 5});
            auto y2 = opr::Reshape::make(y1, tshp1);
            return y2.node();
        };

        m_reformat[std::make_pair(TensorFormat::NCHW32, TensorFormat::NCHW4)] =
                [](VarNode* inp) -> VarNode* {
            mgb_assert(inp->shape().ndim == 5 && inp->shape()[4] == 32);
            auto x = SymbolVar(inp);
            auto xshp = opr::GetVarShape::make(x);

            auto cv = [&x](int v) { return x.make_scalar(v); };
            auto sub = [&xshp, &cv](int idx) {
                return opr::IndexAt::make(xshp, {{0, cv(idx)}});
            };
            auto tshp0 = opr::Concat::make(
                         {sub(0), sub(1), sub(2), sub(3), cv(8), sub(4) / 8},
                         0),
                 tshp1 = opr::Concat::make(
                         {sub(0), sub(1) * 8, sub(2), sub(3), sub(4) / 8}, 0);
            auto y0 = opr::Reshape::make(x, tshp0);
            auto y1 = opr::Dimshuffle::make(y0, {0, 1, 4, 2, 3, 5});
            auto y2 = opr::Reshape::make(y1, tshp1);
            return y2.node();
        };

        m_reformat[std::make_pair(TensorFormat::NCHW4, TensorFormat::CHWN4)] =
                [](VarNode* inp) -> VarNode* {
            megdnn::param::RelayoutFormat param;
            param.mode = megdnn::param::RelayoutFormat::Mode::NCHW4_CHWN4;
            auto reformat = opr::RelayoutFormat::make(inp, param);
            return reformat.node();

        };
        
        m_reformat[std::make_pair(TensorFormat::CHWN4, TensorFormat::NCHW4)] =
                [](VarNode* inp) -> VarNode* {
            megdnn::param::RelayoutFormat param;
            param.mode = megdnn::param::RelayoutFormat::Mode::CHWN4_NCHW4;
            auto reformat = opr::RelayoutFormat::make(inp, param);
            return reformat.node();
        };

        m_reformat[std::make_pair(TensorFormat::NCHW, TensorFormat::CHWN4)] =
                [](VarNode* inp) -> VarNode* {
            auto x = SymbolVar(inp);
            auto xshp = opr::GetVarShape::make(x);

            auto cv = [&x](int v) { return x.make_scalar(v); };
            auto sub = [&xshp, &cv](int idx) {
                return opr::IndexAt::make(xshp, {{0, cv(idx)}});
            };
            auto tshp = opr::Concat::make(
                    {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
            auto y0 = opr::Reshape::make(x, tshp);
            auto y1 = opr::Dimshuffle::make(y0, {1, 3, 4, 0, 2});
            return y1.node();

        };

        m_reformat[std::make_pair(TensorFormat::CHWN4, TensorFormat::NCHW)] =
                [](VarNode* inp) -> VarNode* {
            mgb_assert(inp->shape().ndim == 5 && inp->shape()[4] == 4);
            auto x = SymbolVar(inp);
            auto xshp = opr::GetVarShape::make(x);

            auto cv = [&x](int v) { return x.make_scalar(v); };
            auto sub = [&xshp, &cv](int idx) {
                return opr::IndexAt::make(xshp, {{0, cv(idx)}});
            };
            auto tshp =
                    opr::Concat::make({sub(3), sub(0) * 4, sub(1), sub(2)}, 0);
            auto y0 = opr::Dimshuffle::make(x, {3, 0, 4, 1, 2});
            auto y1 = opr::Reshape::make(y0, tshp);
            return y1.node();
        };
        detect_shuffle_operations();
        do_replace();
    }
};

/*!
 * \brief abstract operator representation of shuffle operation
 */
MGB_DEFINE_OPR_CLASS(ShuffleShuffleRemovePass::Impl::AbstractShuffleOpr,
                           cg::SingleCNOperatorNodeBase) // {
public:
    AbstractShuffleOpr(VarNode* inpvar, TensorFormat inp_format,
                       TensorFormat out_format);

    static SymbolVar make(VarNode* inpvar, TensorFormat inp_format,
                          TensorFormat out_format);

    TensorFormat inp_format() const { return m_inp_format; }

    TensorFormat out_format() const { return m_out_format; }

private:
    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    const TensorFormat m_inp_format;
    const TensorFormat m_out_format;
};

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ShuffleShuffleRemovePass::Impl::AbstractShuffleOpr);

void ShuffleShuffleRemovePass::Impl::AbstractShuffleOpr::scn_do_execute() {
    mgb_throw(InternalError, "AbstractShuffleOpr cannot be executed");
}

void ShuffleShuffleRemovePass::Impl::AbstractShuffleOpr::
        init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    DepVal deps;
    for (auto i : input())
        deps.push_back({i, DepType::SHAPE});
    auto infer_shape = [this](TensorShape& dst, const InpVal& inp) {
        TensorShape inp_shape = inp.val[0].shape();
        if (m_inp_format == TensorFormat::NCHW4 &&
            m_out_format == TensorFormat::NCHW32) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[4] == 4);
            dst = inp_shape;
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1] / 8;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = inp_shape[4] * 8;
        } else if (m_inp_format == TensorFormat::NCHW32 &&
                   m_out_format == TensorFormat::NCHW4) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[4] == 32);
            dst = inp_shape;
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1] * 8;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = inp_shape[4] / 8;
        } else if (m_inp_format == TensorFormat::NCHW &&
                   m_out_format == TensorFormat::NCHW4) {
            mgb_assert(inp_shape.ndim == 4);
            dst.ndim = 5;
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1] / 4;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = 4;
        } else if (m_inp_format == TensorFormat::NCHW4 &&
                   m_out_format == TensorFormat::NCHW) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[4] == 4);
            dst.ndim = 4;
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1] * 4;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
        } else if (m_inp_format == TensorFormat::NCHW4 &&
                   m_out_format == TensorFormat::CHWN4) {
            dst.ndim = 5;
            dst[0] = inp_shape[1];
            dst[1] = inp_shape[2];
            dst[2] = inp_shape[3];
            dst[3] = inp_shape[0];
            dst[4] = inp_shape[4];
        } else if (m_inp_format == TensorFormat::CHWN4 &&
                   m_out_format == TensorFormat::NCHW4) {
            dst.ndim = 5;
            dst[0] = inp_shape[3];
            dst[1] = inp_shape[0];
            dst[2] = inp_shape[1];
            dst[3] = inp_shape[2];
            dst[4] = inp_shape[4];
        } else {
            mgb_throw(InternalError,
                      "Unsupported input format and output format.");
        }
        return true;
    };
    mgr.register_shape_infer(output(0), {SourceType::DEP, deps, infer_shape});
}

ShuffleShuffleRemovePass::Impl::AbstractShuffleOpr::AbstractShuffleOpr(
        VarNode* inpvar, TensorFormat inp_format, TensorFormat out_format)
        : Super(inpvar->owner_graph(), {}, "AbstractShuffleOpr", {inpvar}),
          m_inp_format{inp_format},
          m_out_format{out_format} {
    add_input({inpvar});
    add_equivalence_component<ScalarHash<TensorFormat>>(m_inp_format);
    add_equivalence_component<ScalarHash<TensorFormat>>(m_out_format);
    add_output(None)->dtype(inpvar->dtype());
}

SymbolVar ShuffleShuffleRemovePass::Impl::AbstractShuffleOpr::make(
        VarNode* inpvar, TensorFormat inp_format, TensorFormat out_format) {
    return inpvar->owner_graph()
            ->insert_opr(std::make_unique<AbstractShuffleOpr>(
                    inpvar, inp_format, out_format))
            ->output(0);
}

void ShuffleShuffleRemovePass::Impl::detect_shuffle_operations() {
    auto rewriter = m_opt_state.graph().make_rewriter();
    auto uniq_reader_check = UniqReaderCheck{m_opt_state.graph()};
    auto try_reshape_shuffle = [&rewriter,
                                &uniq_reader_check](OperatorNodeBase* opr) {
        // check shuffle
        auto shuffle = try_cast_as_op<opr::Dimshuffle>(opr);
        if (shuffle == nullptr)
            return false;
        auto&& param = shuffle->param();
        if (param.pattern_len != 5)
            return false;
        bool is_nchw2nchw4 = param.pattern[0] == 0 && param.pattern[1] == 1 &&
                             param.pattern[2] == 3 && param.pattern[3] == 4 &&
                             param.pattern[4] == 2 &&
                             opr->output(0)->shape()[4] == 4;
        if (!is_nchw2nchw4)
            return false;
        if (!uniq_reader_check(shuffle->input(0)))
            return false;

        // check reshape
        auto reshape = try_cast_as_op<opr::Reshape>(opr->input(0)->owner_opr());
        if (reshape == nullptr)
            return false;
        auto inp_var = rewriter.get_var(reshape->input(0));
        auto abstract_shuffle = AbstractShuffleOpr::make(
                inp_var, TensorFormat::NCHW, TensorFormat::NCHW4);
        rewriter.replace_var(
                opr->output(0), abstract_shuffle.node(),
                mgb_cstr_log("replace reformat(nchw -> nchw4) to "
                             "AbstractShuffleOpr(nchw -> nchw4)."));
        return true;
    };

    auto try_reshape_shuffle_reshape = [&rewriter, &uniq_reader_check](
                                               OperatorNodeBase* opr) {
        // check reshape
        auto reshape1 = try_cast_as_op<opr::Reshape>(opr);
        if (reshape1 == nullptr)
            return false;
        if (!uniq_reader_check(reshape1->input(0)))
            return false;

        // check shuffle
        auto shuffle =
                try_cast_as_op<opr::Dimshuffle>(opr->input(0)->owner_opr());
        if (shuffle == nullptr)
            return false;
        auto&& param = shuffle->param();
        if (param.pattern_len != 6)
            return false;
        bool is_nchw42nchw32 = param.pattern[0] == 0 && param.pattern[1] == 1 &&
                               param.pattern[2] == 3 && param.pattern[3] == 4 &&
                               param.pattern[4] == 2 && param.pattern[5] == 5 &&
                               shuffle->input(0)->shape()[5] == 4 &&
                               shuffle->input(0)->shape()[2] == 8;
        bool is_nchw322nchw4 = param.pattern[0] == 0 && param.pattern[1] == 1 &&
                               param.pattern[2] == 4 && param.pattern[3] == 2 &&
                               param.pattern[4] == 3 && param.pattern[5] == 5 &&
                               shuffle->input(0)->shape()[4] == 8 &&
                               shuffle->input(0)->shape()[5] == 4;
        if (!is_nchw42nchw32 && !is_nchw322nchw4)
            return false;
        if (!uniq_reader_check(shuffle->input(0)))
            return false;

        // check reshape
        auto reshape2 =
                try_cast_as_op<opr::Reshape>(shuffle->input(0)->owner_opr());
        if (reshape2 == nullptr)
            return false;
        auto inp_var = rewriter.get_var(reshape2->input(0));
        TensorFormat inp_format = is_nchw42nchw32 ? TensorFormat::NCHW4
                                                  : TensorFormat::NCHW32,
                     out_format = is_nchw42nchw32 ? TensorFormat::NCHW32
                                                  : TensorFormat::NCHW4;
        auto abstract_shuffle =
                AbstractShuffleOpr::make(inp_var, inp_format, out_format);
        std::string reformat_type =
                is_nchw42nchw32 ? "nchw4 -> nchw32" : "nchw32 -> nchw4";
        rewriter.replace_var(opr->output(0), abstract_shuffle.node(),
                             mgb_cstr_log(ssprintf("replace reformat(%s) to "
                                                   "AbstractShuffleOpr(%s).",
                                                   reformat_type.c_str(),
                                                   reformat_type.c_str())
                                                  .c_str()));
        return true;
    };

    auto try_shuffle_reshape = [&rewriter,
                                &uniq_reader_check](OperatorNodeBase* opr) {
        // check reshape
        auto reshape = try_cast_as_op<opr::Reshape>(opr);
        if (reshape == nullptr)
            return false;
        if (!uniq_reader_check(reshape->input(0)))
            return false;

        // check shuffle
        auto shuffle =
                try_cast_as_op<opr::Dimshuffle>(opr->input(0)->owner_opr());
        if (shuffle == nullptr)
            return false;
        auto&& param = shuffle->param();
        if (param.pattern_len != 5)
            return false;
        bool is_nchw42nchw = param.pattern[0] == 0 && param.pattern[1] == 1 &&
                             param.pattern[2] == 4 && param.pattern[3] == 2 &&
                             param.pattern[4] == 3 &&
                             shuffle->input(0)->shape()[4] == 4;
        if (!is_nchw42nchw)
            return false;
        auto inp_var = rewriter.get_var(shuffle->input(0));
        auto abstract_shuffle = AbstractShuffleOpr::make(
                inp_var, TensorFormat::NCHW4, TensorFormat::NCHW);
        rewriter.replace_var(
                opr->output(0), abstract_shuffle.node(),
                mgb_cstr_log("replace reformat(nchw4 -> nchw) to "
                             "AbstractShuffleOpr(nchw4 -> nchw)."));
        return true;
    };

    auto try_relayout_format = [&rewriter](OperatorNodeBase* opr) {
        // check relayout format
        auto reformat = try_cast_as_op<opr::RelayoutFormat>(opr);
        if (reformat == nullptr)
            return false;
        auto&& param = reformat->param();
        if (param.mode != opr::RelayoutFormat::Param::Mode::CHWN4_NCHW4 &&
            param.mode != opr::RelayoutFormat::Param::Mode::NCHW4_CHWN4)
            return false;
        auto inp_var = rewriter.get_var(reformat->input(0));
        cg::SymbolVar abstract_shuffle;
        if (param.mode == opr::RelayoutFormat::Param::Mode::NCHW4_CHWN4) {
            abstract_shuffle = AbstractShuffleOpr::make(
                    inp_var, TensorFormat::NCHW4, TensorFormat::CHWN4);
        } else {
            abstract_shuffle = AbstractShuffleOpr::make(
                    inp_var, TensorFormat::CHWN4, TensorFormat::NCHW4);
        }
        rewriter.replace_var(
                opr->output(0), abstract_shuffle.node(),
                mgb_cstr_log("replace reformat(nchw4 -> nchw) to "
                             "AbstractShuffleOpr(nchw4 -> nchw)."));
        return true;
    };

    auto on_opr = [&try_reshape_shuffle, &try_shuffle_reshape,
                   &try_reshape_shuffle_reshape, &try_relayout_format,
                   &rewriter, &uniq_reader_check](OperatorNodeBase* opr) {
        if (!try_reshape_shuffle_reshape(opr) && !try_reshape_shuffle(opr) &&
            !try_shuffle_reshape(opr) && !try_relayout_format(opr)) {
            auto new_opr = rewriter.auto_replace_outputs(opr);
            uniq_reader_check.update_on_opr_auto_replace(opr, new_opr);
        }
    };
    m_opt_state.graph().iter(on_opr);
    rewriter.apply_inplace();
}

void ShuffleShuffleRemovePass::Impl::do_replace() {
    auto rewriter = m_opt_state.graph().make_rewriter();
    auto uniq_reader_check = UniqReaderCheck{m_opt_state.graph()};
    ThinHashMap<VarNode*, VarNode*> var2endpoint;
    ThinHashSet<VarNode*> trt_opr_inps;
    SmallVector<OperatorNodeBase*> topo_order;

    auto cb = [&topo_order, &trt_opr_inps](OperatorNodeBase* opr) {
        topo_order.push_back(opr);
        MGB_MARK_USED_VAR(trt_opr_inps);
#if MGB_ENABLE_TENSOR_RT
        if (opr->same_type<opr::TensorRTOpr>()) {
            for (auto&& inp : opr->input())
                trt_opr_inps.insert(inp);
        }
#endif
    };
    m_opt_state.graph().iter(cb);

    for (auto&& opr : reverse_adaptor(topo_order)) {
        if (opr->same_type<opr::TypeCvt>() ||
            opr->same_type<AbstractShuffleOpr>()) {
            auto find = var2endpoint.find(opr->output(0));
            if (find != var2endpoint.end()) {
                if (uniq_reader_check(opr->output(0))) {
                    var2endpoint[opr->input(0)] = find->second;
                } else {
                    var2endpoint[opr->input(0)] = opr->output(0);
                }
            } else {
                var2endpoint[opr->input(0)] = opr->output(0);
            }
        }
    }

    auto on_opr = [this, &rewriter, &uniq_reader_check, &trt_opr_inps,
                   &var2endpoint](OperatorNodeBase* opr) {
        MGB_MARK_USED_VAR(trt_opr_inps);
        bool cond_opr = opr->same_type<opr::TypeCvt>() ||
                        opr->same_type<AbstractShuffleOpr>();
        if (cond_opr) {
            bool cond_endpoint = var2endpoint[opr->input(0)] == opr->output(0);
            if (!cond_endpoint)
                return;
            auto cur = opr;
            auto var = opr->output(0), inp_var = opr->input(0);
            bool force_folding_typecvt = false;
            bool first_shuffle = false;
            // initialize inp_format and out_format
            TensorFormat out_format = TensorFormat::NCHW, inp_format = out_format;
            megdnn::DType inp_dtype = cur->input(0)->dtype(),
                          out_dtype = cur->output(0)->dtype();
            SmallVector<megdnn::DType> out_dtype_vec;
            while (cond_opr) {
                if (cur->same_type<AbstractShuffleOpr>()) {
                    auto shuffle = try_cast_as_op<AbstractShuffleOpr>(cur);
                    inp_format = shuffle->inp_format();
                    if (!first_shuffle) {
                        out_format = shuffle->out_format();
                        first_shuffle = true;
                    }
                } else {
                    mgb_assert(cur->same_type<opr::TypeCvt>());
                    out_dtype_vec.push_back(cur->output(0)->dtype());
                }
                inp_var = cur->input(0);
                bool cond_reader = uniq_reader_check(inp_var);
                if (!cond_reader)
                    break;
                cur = cur->input(0)->owner_opr();
                cond_opr = cur->same_type<opr::TypeCvt>() ||
                           cur->same_type<AbstractShuffleOpr>();
            }
            std::reverse(out_dtype_vec.begin(), out_dtype_vec.end());
#if MGB_ENABLE_TENSOR_RT
            force_folding_typecvt =
                    inp_var->owner_opr()->same_type<opr::TensorRTOpr>() ||
                    trt_opr_inps.count(var);
#endif
            auto new_var = rewriter.get_var(inp_var);
            if (inp_format != out_format) {
                new_var = m_reformat[std::make_pair(inp_format, out_format)](
                        new_var);
            }
            if (force_folding_typecvt) {
                inp_dtype = inp_var->dtype();
                if (inp_dtype != out_dtype) {
                    auto type_cvt = opr::TypeCvt::make(new_var, out_dtype);
                    new_var = type_cvt.node();
                }
            } else {
                if (out_dtype_vec.back() != var->dtype())
                    out_dtype_vec.push_back(var->dtype());
                for (auto&& dtype : out_dtype_vec) {
                    auto type_cvt = opr::TypeCvt::make(new_var, dtype);
                    new_var = type_cvt.node();
                }
            }
            rewriter.replace_var(
                    var, new_var,
                    mgb_cstr_log("replace Dimshuffle and TypeCvt chain"));
        } else {
            auto new_opr = rewriter.auto_replace_outputs(opr);
            uniq_reader_check.update_on_opr_auto_replace(opr, new_opr);
        }
    };
    m_opt_state.graph().iter(on_opr);
    rewriter.apply_inplace();
}

const char* ShuffleShuffleRemovePass::name() const {
    return mgb_cstr_log("shuffle shuffle remove pass");
}

void ShuffleShuffleRemovePass::apply(OptState& opt) const {
    opt.set_var_replace_check_flag(VarReplaceCheckFlag::CHECK_SHAPE |
                                   VarReplaceCheckFlag::CHECK_DTYPE);
    Impl{opt};
}

void gopt::reformat_to_chwn4_transform_dest_vars_inplace(
        mgb::cg::VarNodeArray& dest_vars) {
    gopt::GraphOptimizer optimizer;
    optimizer.add_pass<FuseConvBiasNonlinPass>();
    optimizer.add_pass<FuseConvBiasZPass>();
    optimizer.add_pass(EnableCHWN4Pass::make_chwn4_converter());
    optimizer.add_pass<ShuffleShuffleRemovePass>();
    optimizer.add_pass<RemoveRedundantTypeCvtPass>();
    optimizer.add_pass<ParamFusePass>();
    optimizer.apply_inplace(dest_vars);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
