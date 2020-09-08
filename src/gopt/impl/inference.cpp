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
#include "megbrain/opr/dnn/local.h"
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
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/utils/hash_ct.h"

#include "megdnn/tensor_format.h"

#if MGB_ENABLE_TENSOR_RT
#include "megbrain/tensorrt/tensorrt_opr.h"
#endif

#include "megbrain/gopt/misc.h"

#include "megbrain/utils/hash_ct.h"
#include "midout.h"

MIDOUT_DECL(megbrain_inference)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_inference, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

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
void modify_conv_strategy(
        opr::mixin::Convolution& conv,
        opr::mixin::Convolution::ExecutionPolicy::Strategy strategy) {
    auto policy = conv.execution_policy_transient();
    policy.strategy = strategy;
    conv.set_execution_policy(policy);
}

template <typename Opr>
void inplace_conv_opr_modifier(
        OperatorNodeBase& opr,
        opr::mixin::Convolution::ExecutionPolicy::Strategy strategy) {
    modify_conv_strategy(
            opr.cast_final_safe<Opr>(),
            strategy);
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

void gopt::modify_opr_algo_strategy_inplace(
        const VarNodeArrayView& dest_vars,
        opr::mixin::Convolution::ExecutionPolicy::Strategy strategy) {
#if !MGB_ENABLE_FASTRUN
    using S = opr::mixin::Convolution::ExecutionPolicy::Strategy;
    if (strategy == S::PROFILE || strategy == S::PROFILE_REPRODUCIBLE) {
        mgb_throw(MegBrainError, "fastrun is disabled at compile time");
    }
#endif
    const ThinHashMap<Typeinfo*, std::function<void(OperatorNodeBase&)>>
            modifiers = {
#define CONV(t)                                                       \
    {opr::t::typeinfo(), std::bind(inplace_conv_opr_modifier<opr::t>, \
                                   std::placeholders::_1, strategy)}
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

void gopt::enable_opr_algo_profiling_inplace(
        const VarNodeArrayView& dest_vars) {
    modify_opr_algo_strategy_inplace(dest_vars,
                                     opr::mixin::Convolution::ExecutionPolicy::
                                             Strategy::PROFILE);
}

void gopt::enable_opr_use_profiling_cache_inplace(
        const VarNodeArrayView& dest_vars) {
    modify_opr_algo_strategy_inplace(dest_vars,
                                     opr::mixin::Convolution::ExecutionPolicy::
                                             Strategy::PROFILE_HEURISTIC);
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
    MIDOUT_B("ParamRedistributePass::apply")
    Impl{state};
    MIDOUT_E
}

/* ================ ParamFusePass ================ */

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
    MIDOUT_B("ParamFusePass::apply")
    auto rewriter = state.graph().make_rewriter();
    auto cg = state.graph().comp_graph();

    ConstVarPropogate cvprop{ConstVarType::IMMUTABLE_AND_PARAM};
    state.graph().iter([&cvprop](OperatorNodeBase *opr) {
        cvprop.add_opr(opr);
    });


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
        bool is_default_format = var->format().is_default();
        if (cg::is_static_var_value(var) && is_default_format) {
            // use ImmutableTensor for inferable vars
            HostTensorND hv;
            hv.copy_from(*inferred_val).sync();
            new_var = opr::ImmutableTensor::make(
                    *var->owner_graph(), hv, var_namer.name(var));
        } else {
            if (is_default_format) {
                new_var = opr::SharedDeviceTensor::make_const(
                        *var->owner_graph(), inferred_val, var_namer.name(var));
            } else {
                new_var = opr::SharedDeviceTensorWithFormat::make_const(
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

    auto replace_opr = [&](OperatorNodeBase* opr) {
        auto add_ret = cvprop.opr_rst(opr);
        if (!add_ret.all_const_inp && add_ret.has_midconst_inp) {
            for (auto i: opr->input()) {
                if (cvprop.is_midconst(i)) {
                    state.call_with_opr(i->owner_opr(),
                        [&]{replace_single_var(i, opr);});
                }
            }
        }
        rewriter.auto_replace_outputs(opr);

        //! we should deal with midconst var after auto_replace_outputs, as
        //! on_midconst_opr will replace the endpoint output which may cause
        //! double replace.
        if (add_ret.all_const_inp) {
            for (auto var : opr->output()) {
                if (var->contain_flag(VarNode::Flag::VOLATILE_CONTENT))
                    continue;

                auto osize = ConstVarPropogate::var_mem_size(var);
                if (osize >= cvprop.max_size(opr) &&
                    osize - cvprop.max_size(opr) > m_param_grow_limit) {
                    return;
                }

                // const oprs should be evaluated when output is used by another
                // non-const opr or output is needed by the user
                if (state.graph().endpoint_contain(var)) {
                    replace_single_var(var, nullptr);
                }

            }

        }
    };

    state.graph().iter(replace_opr);
    rewriter.apply_inplace();
    MIDOUT_E
}

/* ================ One2OneOprReplacePass ================ */
const char* ConvertF32ToF16Pass::name() const {
    return mgb_cstr_log("convert_f32_to_f16");
}

void ConvertF32ToF16Pass::apply(OptState& state) const {
    MIDOUT_B("ConvertF32ToF16Pass::apply")
    state.set_var_replace_check_flag(m_var_replace_check_flag);
    auto rewriter = state.graph().make_rewriter();
    VarNodeArray new_inp_cache;

    // record original output dtype
    const SymbolVarArray& vars = state.graph().endpoint_vars();
    std::vector<DType> dtypes;
    for (size_t i = 0; i < vars.size(); i++) {
        dtypes.push_back(vars[i].node()->dtype());
    }

    auto on_opr = [this, &rewriter, &new_inp_cache](OperatorNodeBase* opr) {
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
            for (size_t i = 0; i < origin_out.size(); i++) {
                rewriter.replace_var(origin_out[i], cur_out[i], nullptr);
            }
        } else {
            rewriter.auto_replace_outputs(opr);
        }
    };
    state.graph().iter(on_opr);
    rewriter.apply_inplace();

    // recover output dtype
    rewriter = state.graph().make_rewriter();
    const SymbolVarArray& endpoints = state.graph().endpoint_vars();
    auto replace_output = [&]() {
        for (size_t i = 0; i < endpoints.size(); i++) {
            VarNode* var = endpoints[i].node();
            if (var->dtype().enumv() != dtypes[i].enumv()) {
                auto new_var = opr::TypeCvt::make(var, dtypes[i]).node();
                rewriter.replace_var(var, new_var, nullptr);
            }
        }
    };
    mgb_assert(endpoints.size() > 0);
    auto opr = endpoints[0].node()->owner_opr();
    state.call_with_opr(opr, replace_output, OprPropertyFlag::NONE);
    rewriter.apply_inplace();
    MIDOUT_E
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

    auto replace_lsp_opr = [](OperatorNodeBase* opr,
                              const VarNodeArray& new_inp) {
        mgb_assert(opr->same_type<opr::Linspace>());
        mgb_assert(opr->input().size() == new_inp.size());
        auto& lsp_opr = opr->cast_final_safe<opr::Linspace>();
        if (lsp_opr.output(0)->dtype() != dtype::Float16()) {
            auto cvt_var =
                    opr::TypeCvt::make(lsp_opr.output(0), dtype::Float16(), {});
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

    auto replace_deconv_opr = [use_f32_comp](OperatorNodeBase* opr,
                                           const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& deconv_opr = opr->cast_final_safe<opr::ConvolutionBackwardData>();
        auto new_param = deconv_opr.param();
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
        auto new_deconv_opr = opr::ConvolutionBackwardData::make(
                new_inp[0], new_inp[1], new_param, deconv_opr.execution_policy(),
                deconv_opr.config());
        return new_deconv_opr.node()->owner_opr();
    };

    auto replace_convbias_opr = [use_f32_comp](OperatorNodeBase* opr,
                                               const VarNodeArray& new_inp) {
        auto& convbias_opr = opr->cast_final_safe<opr::ConvBiasForward>();
        auto new_param = convbias_opr.param();
        if (use_f32_comp) {
            new_param.compute_mode =
                    megdnn::param::ConvBias::ComputeMode::FLOAT32;
        }
        mgb_assert(new_inp[0]->dtype() == dtype::Float16(),
                   "inp %s:%s, owner_opr:%s", new_inp[0]->dtype().name(),
                   new_inp[0]->name().c_str(),
                   new_inp[0]->owner_opr()->name().c_str());
        mgb_assert(new_inp[1]->dtype() == dtype::Float16(),
                   "inp %s:%s, owner_opr:%s", new_inp[1]->dtype().name(),
                   new_inp[1]->name().c_str(),
                   new_inp[1]->owner_opr()->name().c_str());
        if(opr->input().size() == 2) {
            auto new_conv_opr = opr::ConvBias::make(
                    new_inp[0], new_inp[1], new_param, convbias_opr.execution_policy(),
                    convbias_opr.config());
            return new_conv_opr.node()->owner_opr();
        } else if(opr->input().size() == 3) {
            auto new_conv_opr = opr::ConvBias::make(
                    new_inp[0], new_inp[1], new_inp[2], new_param, convbias_opr.execution_policy(),
                    convbias_opr.config());
            return new_conv_opr.node()->owner_opr();
        } else {
            mgb_assert(opr->input().size() == 4, "invalid input size %zu",
                       opr->input().size());
            auto new_conv_opr = opr::ConvBias::make(
                    new_inp[0], new_inp[1], new_inp[2], new_inp[3], new_param, convbias_opr.execution_policy(),
                    convbias_opr.config());
            return new_conv_opr.node()->owner_opr();
        }
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

    auto replace_batched_matmul_opr = [use_f32_comp](
                                              OperatorNodeBase* opr,
                                              const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& matmul_opr = opr->cast_final_safe<opr::BatchedMatrixMul>();
        auto new_param = matmul_opr.param();
        if (use_f32_comp) {
            new_param.compute_mode =
                    megdnn::param::MatrixMul::ComputeMode::FLOAT32;
        }
        mgb_assert(new_inp[0]->dtype() == dtype::Float16(),
                   "inp %s:%s, owner_opr:%s", new_inp[0]->dtype().name(),
                   new_inp[0]->name().c_str(),
                   new_inp[0]->owner_opr()->name().c_str());
        mgb_assert(new_inp[1]->dtype() == dtype::Float16(),
                   "inp %s:%s, owner_opr:%s", new_inp[1]->dtype().name(),
                   new_inp[1]->name().c_str(),
                   new_inp[1]->owner_opr()->name().c_str());
        auto new_matmul_opr = opr::BatchedMatrixMul::make(
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

    auto replace_remap_opr = [](OperatorNodeBase* opr,
                                const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size() &&
                   (new_inp.size() == 2));
        auto& remap_opr = opr->cast_final<opr::Remap>();
        // map tensor must be float32
        auto new_map = new_inp[1];
        if (new_inp[1]->dtype() != dtype::Float32()) {
            if (try_cast_as_op<opr::TypeCvt>(new_map->owner_opr()) &&
                new_map->owner_opr()->input(0)->dtype() == dtype::Float32())
                new_map = new_map->owner_opr()->input(0);
            else
                new_map =
                        opr::TypeCvt::make(new_inp[1], dtype::Float32(), {}).node();
        }
        SymbolVar new_remap;

        new_remap = opr::Remap::make(new_inp[0], new_map,
                                               remap_opr.param(),
                                               remap_opr.config());
        return new_remap.node()->owner_opr();
    };


    auto ret = std::make_unique<ConvertF32ToF16Pass>();
    // don't check dtype
    ret->set_var_replace_check_flag(VarReplaceCheckFlag::CHECK_ALL ^
                                    VarReplaceCheckFlag::CHECK_DTYPE);
    auto&& replace_func = ret->m_opr_replace_func;
    replace_func[opr::Linspace::typeinfo()] = replace_lsp_opr;
    replace_func[opr::Host2DeviceCopy::typeinfo()] = replace_h2d_opr;
    replace_func[opr::SharedDeviceTensor::typeinfo()] = replace_sdt_opr;
    replace_func[opr::Convolution::typeinfo()] = replace_conv_opr;
    replace_func[opr::ConvolutionBackwardData::typeinfo()] = replace_deconv_opr;
    replace_func[opr::ConvBias::typeinfo()] = replace_convbias_opr;
    replace_func[opr::MatrixMul::typeinfo()] = replace_matmul_opr;
    replace_func[opr::Reduce::typeinfo()] = replace_reduce_opr;
    replace_func[opr::ImmutableTensor::typeinfo()] = replace_imt_opr;
    replace_func[opr::TypeCvt::typeinfo()] = replace_cvt_opr;
    replace_func[opr::WarpPerspective::typeinfo()] = replace_warp_opr;
    replace_func[opr::Remap::typeinfo()] = replace_remap_opr;
    replace_func[opr::BatchedMatrixMul::typeinfo()] =
            replace_batched_matmul_opr;
    return ret;
#endif
}

/* ================ ConvertFormatPass ================ */

void ConvertFormatPass::apply(OptState& state) const {
    MIDOUT_B("ConvertFormatPass::apply")
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
    MIDOUT_E
}

std::unique_ptr<ConvertFormatPass> ConvertFormatPass::make_nhwcd4_converter() {
    MIDOUT_B("ConvertFormatPass::make")
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

        mgb_assert(new_inp.size() < 4,
                   "ConvertFormat pass does not support fuse Z");
        bool has_bias = new_inp.size() > 2;
        if (has_bias &&
            new_inp[2]->format().type() == TensorFormat::Type::DEFAULT) {
            param.mode = megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I;
            auto relayout_bias = opr::RelayoutFormat::make(new_inp[2], param);
            conv_bias_bias = relayout_bias.node();
        } else if (has_bias) {
            conv_bias_bias = new_inp[2];
        }

        auto new_param = conv_bias_opr.param();
        new_param.format = megdnn::param::ConvBias::Format::NHWCD4;
        mgb_assert(conv_bias_src->shape().ndim == 5 &&
                   conv_bias_src->format().type() ==
                           TensorFormat::Type::IMAGE2D_PACK4);
        SymbolVar new_conv_bias_opr;
        if (has_bias) {
            new_conv_bias_opr = opr::ConvBias::make(
                    conv_bias_src, conv_bias_weights, conv_bias_bias, new_param,
                    conv_bias_opr.execution_policy(), conv_bias_opr.config());
        } else {
            new_conv_bias_opr = opr::ConvBias::make(
                    conv_bias_src, conv_bias_weights, new_param,
                    conv_bias_opr.execution_policy(), conv_bias_opr.config());
        }
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
    /* This helper function guarantees the format convert pass won't change
     * output var's channel. Changing output's channel will cause channel
     * mismatch problem for replacing conv/conv_bias operator.
     */
    auto replace_helper = [](OperatorNodeBase* opr,
                             const VarNodeArray& new_inp) -> OperatorNodeBase* {
        auto&& new_shp = new_inp[0]->shape();
        size_t inp_channel = new_shp[1];
        if (new_shp.eq_shape(opr->input(0)->shape())&& inp_channel % 4 != 0) {
            auto new_opr = serialization::copy_opr_shallow(*opr, new_inp,
                                                           opr->config());
            return new_opr;
        }
        return nullptr;
    };
    auto replace_resize_opr = [replace_helper](OperatorNodeBase* opr,
                                 const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        if (auto opr_shallow_copy = replace_helper(opr, new_inp)) {
            return opr_shallow_copy;
        }
        auto& resize_opr = opr->cast_final_safe<opr::ResizeForward>();
        mgb_assert(resize_opr.param().format ==
                           megdnn::param::Resize::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NHWCD4");
        VarNode* inp = nullptr;
        if (new_inp[0]->shape().ndim == 4) {
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

    auto replace_warp_perspective_opr = [replace_helper](
                                                OperatorNodeBase* opr,
                                                const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        if (auto opr_shallow_copy = replace_helper(opr, new_inp)) {
            return opr_shallow_copy;
        }
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

    auto replace_warp_affine_opr = [replace_helper](OperatorNodeBase* opr,
                                      const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        if (auto opr_shallow_copy = replace_helper(opr, new_inp)) {
            return opr_shallow_copy;
        }
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

    auto replace_pooling_opr = [replace_helper](OperatorNodeBase* opr,
                                  const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        if (auto opr_shallow_copy = replace_helper(opr, new_inp)) {
            return opr_shallow_copy;
        }
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

    auto var_to_chw = [](VarNode* inp, VarNode* new_inp) {
        if (!inp->shape().eq_shape(new_inp->shape())) {
            mgb_assert(inp->shape().ndim == 4 &&
                       inp->format().type() !=
                               TensorFormat::Type::IMAGE2D_PACK4);
            mgb_assert(new_inp->shape().ndim == 5 &&
                       new_inp->format().type() ==
                               TensorFormat::Type::IMAGE2D_PACK4);
            auto param = megdnn::param::RelayoutFormat();
            param.mode = megdnn::param::RelayoutFormat::Mode::NHWCD4I_NCHW;
            auto rf = opr::RelayoutFormat::make(new_inp, param);
            return rf.node();
        }
        return new_inp;
    };

    auto relayout_inp_to_chw = [var_to_chw](OperatorNodeBase* opr,
                                  const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        VarNodeArray t_inp = new_inp;
        for (size_t i = 0; i < opr->input().size(); i++) {
            t_inp[i] = var_to_chw(opr->input(i), new_inp[i]);
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

    /* This helper function converts the first input to the NCHW format to
     * handle operations that do not support NHWCD4 format
     */
    auto relayout_first_inp_to_chw =
            [var_to_chw](OperatorNodeBase* opr,
               const VarNodeArray& new_inp) -> OperatorNodeBase* {
        mgb_assert(opr->input().size() == new_inp.size());
        VarNodeArray t_inp = new_inp;
        t_inp[0] = var_to_chw(opr->input(0), new_inp[0]);
        return serialization::copy_opr_shallow(*opr, t_inp, opr->config());
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
    replace_func[opr::LocalForward::typeinfo()] = relayout_first_inp_to_chw;
    replace_func[opr::GroupLocalForward::typeinfo()] =
            relayout_first_inp_to_chw;
    return ret;
    MIDOUT_E
}

/* ================ ConvertBatchNormPass ================ */
const char* ConvertBatchNormToElemwisePass::name() const {
    return "convert_batch_norm";
}

void ConvertBatchNormToElemwisePass::apply(OptState& state) const {
    MIDOUT_B("ConvertBatchNormToElemwisePass::apply")
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
                SymbolVar invsqrt_variance = opr::PowC::make(variance
                        + variance.make_scalar_dt(float(bn->param().epsilon)), {-0.5});
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
    MIDOUT_E
}

/* ================ FuseConvBiasNonlinPass ================ */
const char* FuseConvBiasNonlinPass::name() const {
    return "combine_conv_bias_and_relu";
}

void FuseConvBiasNonlinPass::apply(OptState& state) const {
    MIDOUT_B("FuseConvBiasNonlinPass::apply")
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
                                           param.dilate_w,
                                           0,
                                           param.compute_mode};
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
                    DTypeTrait<dtype::QuantizedS8>::enumv ||
            typecvt->input(0)->dtype().enumv() !=
                    DTypeTrait<dtype::QuantizedS32>::enumv)
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
    MIDOUT_E
}

/* ================ FuseConvBiasZPass ================ */
const char* FuseConvBiasZPass::name() const {
    return "combine_conv_bias_and_z";
}

void FuseConvBiasZPass::apply(OptState& state) const {
    MIDOUT_B("FuseConvBiasZPass::apply")
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
            else if (elem->param().mode == MultiMode::QFUSE_ADD_H_SWISH)
                return NonlineMode::H_SWISH;
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
            elem->param().mode != MultiMode::QFUSE_ADD_RELU &&
            elem->param().mode != MultiMode::QFUSE_ADD_H_SWISH)
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
    MIDOUT_E
}

/* ================ FuseDeconvCvtPass ================ */
const char* FuseDeconvCvtPass::name() const {
    return "combine_deconv_and_typecvt";
}


void FuseDeconvCvtPass::apply(OptState& state) const {
    MIDOUT_B("FuseDeconvCvtPass::apply")
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
    MIDOUT_E
}

/* ================ ParamMergePass ================ */
const char* ParamMergePass::name() const {
    return mgb_cstr_log("param_merge");
}

void ParamMergePass::apply(OptState& opt_state) const {
    MIDOUT_B("ParamMergePass::apply")
    param_merge<opr::SharedDeviceTensor, opr::MultipleDeviceTensorHolder>(
            opt_state);
    param_merge<opr::SharedDeviceTensorWithFormat,
                opr::MultipleDeviceTensorWithFormatHolder>(opt_state);
    MIDOUT_E
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
