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

#include "./opr_format_modifier.h"
#include "./utils.h"
#include "megbrain/gopt/global_layout_transform.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/serialization/sereg.h"

using namespace mgb;
using namespace gopt;
using namespace cg;

/* =================== LayoutTransformPass ======================*/
void LayoutTransformPass::apply(OptState& opt) const {
    opt.set_var_replace_check_flag(VarReplaceCheckFlag::CHECK_ALL ^
                                   VarReplaceCheckFlag::CHECK_SHAPE);
    SubGraphExtractor extractor(m_ctx->opr_list());
    auto partitions = extractor.extract(opt.graph().endpoint_vars());

    using Solution = SolverBase::Solution;
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
    auto&& reformat_attribute = m_ctx->attribute().reformat_attribute;
    ThinHashMap<VarNode*, TensorFormats> var2fmts;
    static ThinHashSet<Typeinfo*> format_aware_oprs = {
#define cb(_Opr) opr::_Opr::typeinfo(),
            FOREACH_FORMAT_AWARE_OPR(cb)
#undef cb
    };
    auto rewriter = opt.graph().make_rewriter();
    auto on_opr = [this, &opr_configs, &base_fmt, &reformat_attribute,
                   &rewriter, &solution, &var2fmts,
                   &endpoint_vars](OperatorNodeBase* opr) {
        auto it = solution.find(opr);
        if (it != solution.end()) {
            auto opr_fmt = it->second;
            auto find = opr_configs.find(opr->dyn_typeinfo());
            Maybe<OprTensorFormatsConfiguration> fmtcfg = None;
            if (find != opr_configs.end()) {
                fmtcfg = (*find->second.at(opr_fmt))(opr);
            }
            VarNodeArray new_inp;
            size_t nr_inps = opr->input().size();
            TensorFormats out_fmt;
            if (fmtcfg.valid()) {
                nr_inps = std::min(fmtcfg.val().input_tensor_formats.size(),
                                   nr_inps);
                out_fmt = fmtcfg.val().output_tensor_formats[0];
            } else {
                out_fmt = opr_format_to_tensor_formats(opr_fmt);
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
                auto to = fmtcfg.valid()
                                  ? fmtcfg.val().input_tensor_formats[i]
                                  : opr_format_to_tensor_formats(opr_fmt);
                bool is_parameter =
                        fmtcfg.valid() && fmtcfg.val().input_tensor_types[i] ==
                                                  TensorType::WEIGHT;
                // need relayout
                if (from != to && !new_var->shape().is_scalar()) {
                    ReformatManager::ReformatImpl reformat;
                    ReformatManager::ReformatKey key{
                            from, to, reformat_attribute, var->dtype().enumv(),
                            var->dtype().enumv()};
                    if (is_parameter) {
                        auto aligned_desc = ReformatManager::make_aligned_desc(
                                base_fmt, out_fmt);
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
                if (from != to && !new_var->shape().is_scalar())
                    new_var = reformat({new_var});
                new_inp[i] = new_var;
            }
            VarNode* new_out;
            if (format_aware_oprs.count(opr->dyn_typeinfo()) > 0) {
                new_out = intl::modify_opr_format(opr_fmt, new_inp, opr);
            } else {
                new_out = serialization::copy_opr_shallow(*opr, new_inp,
                                                          opr->config())
                                  ->output(0);
            }
            if (endpoint_vars.count(opr->output(0)) && out_fmt != base_fmt) {
                ReformatManager::ReformatKey key{
                        out_fmt, base_fmt, reformat_attribute,
                        opr->output(0)->dtype().enumv(),
                        opr->output(0)->dtype().enumv()};
                auto reformat = ReformatManager::instance()
                                        .auto_aligned_reformat_featrue(
                                                opr->output(0), base_fmt, key);
                new_out = reformat({new_out});
                var2fmts[new_out] = base_fmt;
            } else {
                var2fmts[new_out] = out_fmt;
            }
            auto &&out0 = opr->output(),
                 &&out1 = new_out->owner_opr()->output();
            mgb_assert(opr->usable_output().size() ==
                               new_out->owner_opr()->usable_output().size(),
                       "bad opr replace: src=%s{%s} dst=%s{%s}, "
                       "src.size=%zu "
                       "dst.size=%zu",
                       opr->cname(), opr->dyn_typeinfo()->name,
                       new_out->owner_opr()->cname(),
                       new_out->owner_opr()->dyn_typeinfo()->name, out0.size(),
                       out1.size());
            for (size_t i = 0; i < out0.size(); ++i) {
                if (!out0[i]->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                    mgb_assert(!out1[i]->contain_flag(
                            VarNode::Flag::VOLATILE_CONTENT));
                    auto src = out0[i];
                    auto dst = out1[i];
                    rewriter.replace_var(
                            src, dst,
                            mgb_cstr_log(ssprintf("replace opr(%s) to new opr "
                                                  "format(%s)",
                                                  opr->cname(),
                                                  opr_format_to_string(opr_fmt))
                                                 .c_str()));
                }
            }
        } else {
            auto new_opr = rewriter.auto_replace_outputs(opr);
            var2fmts[new_opr->output(0)] = base_fmt;
        }
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
}

// vim: syntax=cpp.doxygen
