/**
 * \file src/gopt/impl/layout_transform_pass.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/gopt/layout_transform_pass.h"
#include "./opr_format_modifier.h"
#include "./utils.h"
#include "megbrain/gopt/layout_transform_context.h"
#include "megbrain/gopt/profiler.h"
#include "megbrain/gopt/solver.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/serialization/sereg.h"

#include "megbrain/utils/hash_ct.h"
#include "midout.h"

using namespace mgb;
using namespace gopt;
using namespace cg;

MIDOUT_DECL(megbrain_global_layout_transform)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_global_layout_transform, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

/* =================== LayoutTransformPass ======================*/
void LayoutTransformPass::apply(OptState& opt) const {
    MIDOUT_B("apply")
    opt.set_var_replace_check_flag(
            VarReplaceCheckFlag::CHECK_ALL ^ VarReplaceCheckFlag::CHECK_SHAPE);
    SubGraphExtractor extractor(m_ctx->opr_list());
    auto partitions = extractor.extract(opt.graph().endpoint_vars());

    using Solution = SolverBase::Solution;
    using OprFormat = SolverBase::OprFormat;
    Solution solution;
    ThinHashSet<VarNode*> endpoint_vars;
    for (auto&& partition : partitions) {
        if (solution.empty()) {
            solution = m_solver->solve(Problem(partition, *m_ctx));
        } else {
            auto s = m_solver->solve(Problem(partition, *m_ctx));
            for (auto&& kv : s)
                solution.insert({kv.first, kv.second});
        }
        for (auto&& o : partition.output()) {
            endpoint_vars.insert(o);
        }
    }

    auto&& opr_configs = m_ctx->opr_configs();
    auto&& base_fmt = m_ctx->attribute().base_tensor_formats;
    auto&& base_cfg_id = m_ctx->attribute().base_config_id;
    auto&& reformat_attribute = m_ctx->attribute().reformat_attribute;
    ThinHashMap<VarNode*, TensorFormats> var2fmts;
    static ThinHashSet<Typeinfo*> format_aware_oprs = {
#define cb(_Opr) opr::_Opr::typeinfo(),
            FOREACH_FORMAT_AWARE_OPR(cb)
#undef cb
    };
    auto rewriter = opt.graph().make_rewriter();
    auto on_opr = [&opr_configs, &base_fmt, &base_cfg_id, &reformat_attribute,
                   &rewriter, &solution, &var2fmts,
                   &endpoint_vars](OperatorNodeBase* opr) {
        auto it = solution.find(opr);
        if (it != solution.end()) {
            auto cfg_id = it->second;
            auto find = opr_configs.find(opr->dyn_typeinfo());
            Maybe<OprTensorFormatsConfiguration> fmtcfg = None;
            Maybe<OprTensorFormatsConfiguration> basecfg = None;
            Maybe<OprFormat> opr_fmt = None;
            if (find != opr_configs.end()) {
                fmtcfg = (*find->second.at(cfg_id))(opr);
                auto _ = OprTensorFormatsConfiguration::find_dispatcher_by_type_format(
                        opr->dyn_typeinfo(), base_cfg_id);
                basecfg = (*_)(opr);
                opr_fmt = fmtcfg.val().opr_format;
            } else {
                opr_fmt =
                        OprTensorFormatsConfiguration::safe_cast_to_opr_format(cfg_id);
            }
            VarNodeArray new_inp;
            size_t nr_inps = opr->input().size();
            TensorFormats out_fmt;
            if (fmtcfg.valid()) {
                nr_inps = std::min(fmtcfg.val().input_tensor_formats.size(), nr_inps);
                out_fmt = fmtcfg.val().output_tensor_formats[0];
            } else {
                out_fmt = opr_format_to_tensor_formats(opr_fmt.val());
            }
            new_inp.resize(nr_inps);
            for (size_t i = 0; i < nr_inps; ++i) {
                auto&& var = opr->input(i);
                auto&& new_var = rewriter.get_var(var);
                auto find = var2fmts.find(new_var);
                TensorFormats from;
                if (find == var2fmts.end()) {
                    from = base_fmt;
                } else {
                    from = find->second;
                }
                auto to = fmtcfg.valid() ? fmtcfg.val().input_tensor_formats[i]
                                         : opr_format_to_tensor_formats(opr_fmt.val());
                bool is_parameter =
                        fmtcfg.valid() &&
                        fmtcfg.val().input_tensor_types[i] == TensorType::WEIGHT;
                if (is_parameter) {
                    mgb_assert(basecfg.valid());
                    from = basecfg.val().input_tensor_formats[i];
                }
                // need relayout
                if (from != to && !new_var->shape().is_scalar()) {
                    ReformatManager::ReformatImpl reformat;
                    ReformatManager::ReformatKey key{
                            from, to, reformat_attribute, var->dtype().enumv(),
                            var->dtype().enumv()};
                    if (is_parameter) {
                        auto aligned_desc =
                                ReformatManager::make_aligned_desc(from, out_fmt);
                        reformat = ReformatManager::instance()
                                           .auto_aligned_reformat_weight(
                                                   var, key, aligned_desc);
                    } else {
                        reformat = ReformatManager::instance()
                                           .auto_aligned_reformat_featrue(
                                                   var, base_fmt, key);
                    }
                    new_var = reformat({new_var});
                }
                new_inp[i] = new_var;
            }
            VarNode* new_out;
            if (format_aware_oprs.count(opr->dyn_typeinfo()) > 0) {
                new_out = intl::modify_opr_format(opr_fmt.val(), new_inp, opr);
            } else {
                new_out = serialization::copy_opr_shallow(*opr, new_inp, opr->config())
                                  ->output(0);
            }
            auto &&out0 = opr->output(), &&out1 = new_out->owner_opr()->output();
            mgb_assert(
                    opr->usable_output().size() ==
                            new_out->owner_opr()->usable_output().size(),
                    "bad opr replace: src=%s{%s} dst=%s{%s}, "
                    "src.size=%zu "
                    "dst.size=%zu",
                    opr->cname(), opr->dyn_typeinfo()->name,
                    new_out->owner_opr()->cname(),
                    new_out->owner_opr()->dyn_typeinfo()->name, out0.size(),
                    out1.size());
            size_t nr_outs = opr->usable_output().size();
            for (size_t i = 0; i < nr_outs; ++i) {
                const auto& ovar = out0[i];
                auto new_ovar = out1[i];
                if (endpoint_vars.count(ovar) && out_fmt != base_fmt) {
                    ReformatManager::ReformatKey key{
                            out_fmt, base_fmt, reformat_attribute,
                            ovar->dtype().enumv(), ovar->dtype().enumv()};
                    auto reformat =
                            ReformatManager::instance().auto_aligned_reformat_featrue(
                                    ovar, base_fmt, key);
                    new_ovar = reformat({new_ovar});
                    var2fmts[new_ovar] = base_fmt;
                } else {
                    var2fmts[new_ovar] = out_fmt;
                }
                rewriter.replace_var(
                        ovar, new_ovar,
                        mgb_cstr_log(ssprintf(
                                             "replace opr(%s) to new opr "
                                             "format config(%s)",
                                             opr->cname(), config_id_to_string(cfg_id))
                                             .c_str()));
            }
        } else {
            auto new_opr = rewriter.auto_replace_outputs(opr);
            for (auto&& ov : new_opr->usable_output()) {
                var2fmts[ov] = base_fmt;
            }
        }
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
    MIDOUT_E
}

std::unique_ptr<LayoutTransformPass> LayoutTransformPass::make(
        GraphTuningOptions::Target target) {
    MIDOUT_B("make")
    auto profiler = ProfilerBase::make_profiler();
    std::unique_ptr<SolverBase> solver{
            new DynamicProgrammingSolver(std::move(profiler))};
    auto ctx = LayoutTransformContext::make(target);
    return std::make_unique<LayoutTransformPass>(std::move(ctx), std::move(solver));
    MIDOUT_E
}

// vim: syntax=cpp.doxygen
