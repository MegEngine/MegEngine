/**
 * \file src/jit/impl/fusion_pass.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/jit/fusion_pass.h"
#include "megbrain/common.h"
#include "megbrain/gopt/gtrans.h"
#include "megbrain/jit/ast_c.h"
#include "megbrain/jit/compiler.h"
#include "megbrain/jit/internal_graph.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/serialization/serializer.h"

#if MGB_JIT

#if MGB_JIT_MLIR
#include "./mlir/ir/each_mode.h"
#endif

using namespace mgb;
using namespace gopt;
using namespace jit;

class JITFusionPass::Impl final {
    using Mode = opr::Elemwise::Mode;
    using DepType = OperatorNodeBase::NodeProp::DepType;
    const bool m_after_grad;
    JITFeatureBits m_feature_bits;
    OptState& m_opt_state;
    CompNode::UnorderedMap<size_t> m_cn2max_nr_input;

    SubGraph::Rewriter m_rewriter;
    SmallVector<std::unique_ptr<InternalGraphGenerator>> m_igraph_gen_storage;
    ThinHashMap<VarNode*, InternalGraphGenerator*> m_var2igraph_gen;

    //! map from var to its reader oprs and the corresponding dependency types
    ThinHashMap<VarNode*, SmallVector<std::pair<OperatorNodeBase*, DepType>>>
            m_var_readers;
    ThinHashSet<VarNode*> m_endpoint_set;

    //! create a new InternalGraphGenerator rooted at given opr
    InternalGraphGenerator* create_new_igraph_gen(OperatorNodeBase* opr);

    //! process a single operator, maintaining m_var2igraph_gen
    void process_opr(OperatorNodeBase* opr);

    size_t max_nr_input(CompNode cn);

    //! check whether all oprs which depend on the var are in i_graph
    bool test_all_readers_in_the_graph(VarNode* var,
                                       InternalGraphGenerator* i_graph);

    //! check shape to determine whether the opr should be added to the internal
    //! graph
    bool check_shape(cg::OperatorNodeBase* opr, InternalGraphGenerator* i_graph);

    //! use m_rewriter to update graph
    void update_graph();

    //! find the subgraph which can be fused
    void detect_fusion();

    //! check whether an opr can be fused
    bool can_be_fused(cg::OperatorNodeBase* opr) const;

    static size_t nr_non_const_vars(const VarNodeArray& vars) {
        size_t num = 0;
        for (auto i : vars) {
            num += !SymbolVar{i}.as_immutable_scalar().valid();
        }
        return num;
    }

public:
    Impl(bool after_grad, JITFeatureBits feature_bits, OptState& opt_state)
            : m_after_grad{after_grad},
              m_feature_bits{feature_bits},
              m_opt_state{opt_state},
              m_rewriter{opt_state.graph().make_rewriter()} {
        detect_fusion();
        update_graph();
    }
};

void JITFusionPass::Impl::detect_fusion() {
    std::vector<OperatorNodeBase*> topo_order;
    m_opt_state.graph().iter([this, &topo_order](OperatorNodeBase* opr) {
        topo_order.push_back(opr);
        for (auto&& i : opr->node_prop().dep_map()) {
            m_var_readers[i.first].emplace_back(opr, i.second);
        }
    });

    for (auto opr : reverse_adaptor(topo_order)) {
        if (can_be_fused(opr)) {
            process_opr(opr);
        }
    }
}

void JITFusionPass::Impl::update_graph() {
    auto process = [this](OperatorNodeBase* opr) {
        if (!Compiler::is_supported_device(
                    opr->output(0)->comp_node().device_type()))
            return;

        auto fuse_varnode = [this](VarNode* var) {
            auto ig_gen_iter = m_var2igraph_gen.find(var);
            if (ig_gen_iter == m_var2igraph_gen.end()) {
                return;
            }
            auto ig_gen = ig_gen_iter->second;
            if (m_endpoint_set.count(var) != 0 &&
                ig_gen->opr_set().size() >= 2) {
                auto igraph = ig_gen->generate();
                auto&& inputs = ig_gen->orig_inps();
                if (m_after_grad || nr_non_const_vars(inputs) == 1) {
                    // in the forward pass, only fuse oprs with one non-const
                    // inp
                    VarNodeArray rewritten_inputs;
                    for (auto&& input : inputs) {
                        auto new_input = m_rewriter.get_var(input);
                        rewritten_inputs.push_back(new_input);
                    }
                    auto fusion_op =
                            JITExecutor::make(igraph, rewritten_inputs);
                    m_rewriter.replace_var(
                            var, fusion_op.node(),
                            mgb_ssprintf_log("fuse endpoint: %s",
                                             var->owner_opr()->cname())
                                    .c_str());
                }
            }
        };

        for (auto i : opr->input()) {
            if (!m_rewriter.has_manual_replace(i)) {
                // if input i is a endpoint, and number of oprs in this subgraph
                // is greater than 2
                m_opt_state.call_with_opr(i->owner_opr(),
                                          [&] { fuse_varnode(i); });
            }
        }
        m_rewriter.auto_replace_outputs(opr);
        if (m_opt_state.graph().endpoint_contain(opr->output(0))) {
            // process final endpoint
            fuse_varnode(opr->output(0));
        }
    };
    m_opt_state.graph().iter(process);
    m_rewriter.apply_inplace();
}

bool JITFusionPass::Impl::test_all_readers_in_the_graph(
        VarNode* var, InternalGraphGenerator* ig_gen) {
    for (auto&& reader : m_var_readers.at(var)) {
        if (reader.second & DepType::DEV_VALUE) {
            if (ig_gen->opr_set().count(reader.first) == 0) {
                return false;
            }
        }
    }
    return true;
}

bool JITFusionPass::Impl::check_shape(cg::OperatorNodeBase* opr,
                                      InternalGraphGenerator* ig_gen) {
    if (!cg::is_static_var_shape(opr->output(0))) {
        // currently we do not handle dynamic shape in JIT
        return false;
    }
    if (!(m_feature_bits & JITFeatureBits::REDUCE)) {
        // By requiring opr output shape to be the same as final output shape,
        // we permit only one broadcast. If multiple broadcasts are fused,
        // together, execution would be actually slower.
        if ((m_feature_bits & JITFeatureBits::DIMSHUFFLE) &&
            ig_gen->has_dimshuffle() &&
            ig_gen->oprs_depended_by_dimshuffe().count(opr)) {
            return opr->output(0)->shape().eq_shape(
                    ig_gen->oprs_depended_by_dimshuffe()
                            .at(opr)
                            ->input(0)
                            ->shape());
        } else {
            return opr->output(0)->shape().eq_shape(ig_gen->output()->shape());
        }
    }

    bool before_reduce = false;
    for (auto&& op_set : ig_gen->reduce_out_var_deps()) {
        if (op_set.second.count(opr)) {
            before_reduce = true;
            break;
        }
    }

    if (opr->same_type<JITExecutor>()) {
        auto jit = &opr->cast_final<JITExecutor>();
        bool jit_has_reduce = jit->has_reduce();
        auto jit_inp_shp = jit->broadcasted_input_shape();
        if (jit_has_reduce) {
            if (before_reduce)
                return jit_inp_shp.eq_shape(jit->output(0)->shape()) &&
                       jit_inp_shp.eq_shape(ig_gen->before_reduce_shape());
            else {
                bool ret = true;
                if (ig_gen->has_reduce()) {
                    ret &= jit_inp_shp.eq_shape(ig_gen->before_reduce_shape());
                }
                ret &= jit->output(0)->shape().eq_shape(
                        ig_gen->output()->shape());
                return ret;
            }
        }
    }

    if (opr->same_type<opr::Reduce>()) {
        // TODO: handle reduce target shape in sub graph (especially considering
        // placeholder has constant shape)
        //
        // The best way is to have a dedicated AST for the internal graph; but
        // we want to reuse the deduplication and gradient mechanisms from the
        // mgb cg
        auto reduce = &opr->cast_final<opr::Reduce>();
        if (before_reduce) {
            return reduce->input(0)->shape().eq_shape(
                           ig_gen->before_reduce_shape()) &&
                   reduce->output(0)->shape().eq_shape(
                           ig_gen->before_reduce_shape());
        } else {
            bool ret = true;
            if (ig_gen->has_reduce()) {
                ret &= reduce->input(0)->shape().eq_shape(
                        ig_gen->before_reduce_shape());
            }
            ret &= reduce->output(0)->shape().eq_shape(
                    ig_gen->output()->shape());
            return ret;
        }
    }

    if (before_reduce) {
        return opr->output(0)->shape().eq_shape(ig_gen->before_reduce_shape());
    } else {
        return opr->output(0)->shape().eq_shape(ig_gen->output()->shape());
    }
}

InternalGraphGenerator* JITFusionPass::Impl::create_new_igraph_gen(
        OperatorNodeBase* opr) {
    auto uptr = std::make_unique<InternalGraphGenerator>(opr);
    auto ptr = uptr.get();
    m_igraph_gen_storage.emplace_back(std::move(uptr));
    m_var2igraph_gen[opr->output(0)] = ptr;
    m_endpoint_set.insert(opr->output(0));
    return ptr;
}

void JITFusionPass::Impl::process_opr(OperatorNodeBase* opr) {
    auto max_nr_input = this->max_nr_input(opr->output(0)->comp_node());
    if (nr_non_const_vars(opr->input()) > max_nr_input ||
        !cg::is_static_var_shape(opr->output(0))) {
        return;
    }
    // dimshuffle should not be an endpoint, because megbrain has lazy
    // dimshuffle machanism
    InternalGraphGenerator* ig_gen = nullptr;
    if (m_var2igraph_gen.count(opr->output(0)) == 0) {
        // because of the reverse traversal, when an operator is being
        // processed but not in m_var2igraph_gen, means it is a endpoint of a
        // JIT subgraph.
        if (opr->same_type<opr::Dimshuffle>()) {
            return;
        }
        ig_gen = create_new_igraph_gen(opr);
    } else {
        ig_gen = m_var2igraph_gen[opr->output(0)];
        // if all oprs which depend on this elemwise opr's output were already
        // in the subgraph and the opr's comp_node is same with the subgraph's,
        // then this opr can be fused to this graph as an internal node rather
        // than a leaf.
        bool cond_readers =
                     test_all_readers_in_the_graph(opr->output(0), ig_gen),
             cond_cn = opr->output(0)->comp_node() ==
                       ig_gen->output()->comp_node(),
             cond_shp = check_shape(opr, ig_gen),
             cond_nr_inp = ig_gen->get_cnt_input_if_add(opr) <= max_nr_input,
             cond_mlir_specific = true;

        if (cond_readers && cond_cn && cond_shp && cond_nr_inp &&
            cond_mlir_specific) {
            ig_gen->add_opr(opr);
        } else {
            if (opr->same_type<opr::Dimshuffle>()) {
                return;
            }
            // create a new sub graph starting from this opr
            mgb_log_debug(
                    "JIT graph stopped at opr %s{%s}: cond: readers=%d cn=%d "
                    "shp=%d nr_inp=%d",
                    opr->cname(), opr->dyn_typeinfo()->name, cond_readers,
                    cond_cn, cond_shp, cond_nr_inp);
            ig_gen = create_new_igraph_gen(opr);
        }
    }

    // handle const inputs
    for (auto&& i : opr->node_prop().dep_map()) {
        if (i.second & cg::OperatorNodeBase::NodeProp::DepType::DEV_VALUE) {
            if (SymbolVar{i.first}
                        .as_immutable_scalar_require_shape()
                        .valid()) {
                auto opr = i.first->owner_opr();
                mgb_assert(opr->same_type<opr::ImmutableTensor>(),
                           "got imm scalar from non ImmutableTensor: %s{%s}",
                           opr->cname(), opr->dyn_typeinfo()->name);
                ig_gen->add_opr(opr);
                continue;
            }
        }
        m_var2igraph_gen[i.first] = ig_gen;
    }
}

size_t JITFusionPass::Impl::max_nr_input(CompNode cn) {
    auto&& ret = m_cn2max_nr_input[cn];
    if (!ret) {
        ret = Compiler::get(*m_opt_state.graph().comp_graph(), cn)
                      ->property()
                      .max_nr_input;
        mgb_assert(ret);
    }
    return ret;
}

bool JITFusionPass::Impl::can_be_fused(cg::OperatorNodeBase* opr) const {
    if (!Compiler::is_supported_device(
                opr->output(0)->comp_node().device_type())) {
        return false;
    }

    //! As MLIR backend has some contraints
    const char* backend = MGB_GETENV("MGB_JIT_BACKEND");
    if (!backend) {
        backend = "DEFAULT";
    }
    // float elemwise
    if (auto elem = gopt::try_cast_as_op<opr::Elemwise>(opr)) {
        bool ret = true;
#if MGB_JIT_MLIR
        if (!strcmp(backend, "MLIR")) {
            switch (elem->param().mode) {
#define cb(_, _mode)                 \
    case opr::Elemwise::Mode::_mode: \
        ret = true;                  \
        break;

                MLIR_MGB_FOREACH_ELEMWISE_MODE_UNARY(cb)
                MLIR_MGB_FOREACH_ELEMWISE_MODE_BINARY(cb)
                MLIR_MGB_FOREACH_ELEMWISE_MODE_TERNARY(cb)
                default:
                    ret = false;
#undef cb
            }
#define FOREACH_ELEMWISE_SKIP_MODE(cb) cb(SIN)

            //! FIXME mlir on cuda does't support sin currently.
            if (opr->output(0)->comp_node().device_type() ==
                CompNode::DeviceType::CUDA) {
                switch (elem->param().mode) {
#define cb(_mode)                    \
    case opr::Elemwise::Mode::_mode: \
        ret = false;                 \
        break;

                    FOREACH_ELEMWISE_SKIP_MODE(cb)
                    default:
                        break;
#undef cb
                }
            }

#undef FOREACH_ELEMWISE_SKIP_MODE
        }
#endif  // MGB_JIT_MLIR
        return ret && ast_c::check_elem_mode(elem->param().mode) &&
               elem->output(0)->dtype().category() == DTypeCategory::FLOAT;
    }

    if (strcmp(backend, "MLIR")) {
        if (opr->same_type<opr::PowC>()) {
            return true;
        }

        // float typecvt (e.g. used in f16 training)
        if (opr->same_type<opr::TypeCvt>()) {
            auto category = opr->input(0)->dtype().category();
            if (category != opr->output(0)->dtype().category())
                return false;
            return category == DTypeCategory::FLOAT;
        }

        // float reduce
        if ((m_feature_bits & JITFeatureBits::REDUCE) &&
            opr->same_type<opr::Reduce>()) {
            return opr->output(0)->dtype().category() == DTypeCategory::FLOAT;
        }

        // dimshuffle
        if ((m_feature_bits & JITFeatureBits::DIMSHUFFLE) &&
            opr->same_type<opr::Dimshuffle>()) {
            auto param = opr->cast_final_safe<opr::Dimshuffle>().param();
            return param.pattern_len <= 4;
        }
    }

    // existing JITExecutor
    if (opr->same_type<JITExecutor>())
        return true;

    return false;
}

JITFusionPass::JITFusionPass(bool after_grad, int8_t jit_opt_level)
        : m_after_grad{after_grad}, m_feature_bits{JITFeatureBits::NONE} {
    // TODO reduce and dimshuffle can not coexsit now.
    if (jit_opt_level >= 2) {
        m_feature_bits |= JITFeatureBits::REDUCE;
    } else {
        m_feature_bits |= JITFeatureBits::DIMSHUFFLE;
    }
}

const char* JITFusionPass::name() const {
    return mgb_cstr_log("fusion_pass");
}

void JITFusionPass::apply(OptState& opt) const {
    Impl{m_after_grad, m_feature_bits, opt};
}

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
