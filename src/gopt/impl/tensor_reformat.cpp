/**
 * \file src/gopt/impl/tensor_reformat.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/gopt/basic_arith.h"
#include "megbrain/gopt/gtrans.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/graph/event.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/local.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/misc.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/utils/shared_set.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/tensor_format.h"

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#if MGB_ENABLE_TENSOR_RT
#include "megbrain/tensorrt/tensorrt_opr.h"
#endif

#include "megbrain/gopt/misc.h"
#include "megbrain/utils/hash_ct.h"

#include "midout.h"

MIDOUT_DECL(megbrain_tensor_reformat)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_tensor_reformat, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace gopt;

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
                     cg::SingleCNOperatorNodeBase)  // {
public:
//! relayout type of this opr
enum class LayoutType {
    NCHW4_TO_NCHW32,              //!< from nchw4 layout to nchw32 layout
    NCHW32_TO_NCHW4,              //!< from nchw32 layout to nchw4 layout
    NCHW4_TO_CHWN4,               //!< from nchw4 layout to chwn4 layout
    CHWN4_TO_NCHW4,               //!< from chwn4 layout to nchw4 layout
    NCHW_TO_NCHW4,                //!< from nchw layout to nchw4 layout
    NCHW_TO_NCHW4_IC_SMALL_CONV,  ///< from nchw layout to nchw4 whose
                                  ///< channel size less than 4
    NCHW4_TO_NCHW,                //!< from nchw4 layout to nchw layout
    NCHW_TO_NCHW88,               //!< from nchw layout to nchw88 layout
    NCHW88_TO_NCHW,               //!< from nchw88 layout to nchw layout

    WEIGHT_NCHW_TO_NCHW4_DENSE,  //!< weight from nchw layout to nchw4
                                 //!< layout
    WEIGHT_NCHW_TO_NCHW4_GROUP,  //!< group weight from nchw layout to
                                 //!< nchw4 layout
    WEIGHT_NCHW_TO_NCHW4_DENSE_IC_SMALL_CONV,  //!< weight from nchw layout
                                               //!< to nchw4 layout whose
                                               //! channel size less than 4

    WEIGHT_NCHW_TO_NCHW88_DENSE,  //!< weight from nchw layout to nchw88
                                  //!< layout
    WEIGHT_NCHW_TO_NCHW88_GROUP,  //!< group weight from nchw layout to
                                  //!< nchw88 layout
    WEIGHT_NCHW_TO_NCHW88_CHAN,   //!< channel wise weight from nchw layout
                                  //!< to nchw88 layout
    //!< the weight layout of input is nchw output is nchw88, special for
    //!< shape weight in nchw like {64, 2, 3, 3} to {8, 3, 3, 2, 8}
    WEIGHT_HYBIRD_NCHW_NCHW88,

    WEIGHT_NCHW_TO_NCHW44_DENSE,  //!< weight from nchw layout to nchw44
                                  //!< layout
    WEIGHT_NCHW_TO_NCHW44_GROUP,  //!< group weight from nchw layout to
                                  //!< nchw44 layout
    WEIGHT_NCHW_TO_NCHW44_CHAN,   //!< channel wise weight from nchw layout
                                  //!< to nchw44 layout
    //!< the weight layout of input is nchw output is nchw44, special for
    //!< shape weight in nchw like {64, 2, 3, 3} to {16, 3, 3, 2, 4}
    WEIGHT_HYBIRD_NCHW_NCHW44,
    WEIGHT_NCHW_TO_NCHW44_DOT_DENSE,  //!< weight from NCHW44 layout to
                                      //!< NCHW44_DOT layout dense
    WEIGHT_NCHW_TO_NCHW44_DOT_GROUP,  //!< weight from NCHW44 layout to
                                      //!< NCHW44_DOT layout group
};

RelayoutPlaceholder(VarNode* src_var, LayoutType layout_type);

/*!
 * \param src_var the input var
 * \param layout_type tensor layout transform type of this relayout
 * placeholder as described in LayoutType
 */
static SymbolVar make(VarNode* src_var, LayoutType layout_type);

LayoutType layout_type() const {
    return m_layout_type;
}

private:
void init_output_static_infer_desc() override;
void scn_do_execute() override;
void init_output_comp_node() override;
const LayoutType m_layout_type;
}
;

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
                           RelayoutPlaceholder::LayoutType::NCHW_TO_NCHW4 ||
                   layout_type() == RelayoutPlaceholder::LayoutType::
                                            NCHW_TO_NCHW4_IC_SMALL_CONV) {
            if (layout_type() ==
                RelayoutPlaceholder::LayoutType::NCHW_TO_NCHW4) {
                mgb_assert(inp_shape.ndim == 4 && inp_shape[1] % 4 == 0);
            } else {
                mgb_assert(layout_type() ==
                           RelayoutPlaceholder::LayoutType::
                                   NCHW_TO_NCHW4_IC_SMALL_CONV);
                mgb_assert(inp_shape.ndim == 4 && inp_shape[1] < 4);
            }
            dst.ndim = 5;
            dst[0] = inp_shape[0];
            dst[1] = (inp_shape[1] + 4 - 1) / 4;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = 4;
        } else if (layout_type() ==
                   RelayoutPlaceholder::LayoutType::NCHW4_TO_NCHW) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[4] == 4);
            dst.ndim = 4;
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1] * 4;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
        } else if (layout_type() == RelayoutPlaceholder::LayoutType::
                                            WEIGHT_NCHW_TO_NCHW4_DENSE ||
                   layout_type() ==
                           RelayoutPlaceholder::LayoutType::
                                   WEIGHT_NCHW_TO_NCHW4_DENSE_IC_SMALL_CONV) {
            if (layout_type() ==
                RelayoutPlaceholder::LayoutType::WEIGHT_NCHW_TO_NCHW4_DENSE) {
                mgb_assert(inp_shape.ndim == 4 && inp_shape[1] % 4 == 0);
            } else {
                mgb_assert(layout_type() ==
                           RelayoutPlaceholder::LayoutType::
                                   WEIGHT_NCHW_TO_NCHW4_DENSE_IC_SMALL_CONV);
                mgb_assert(inp_shape.ndim == 4 && inp_shape[1] < 4);
            }

            dst.ndim = 5;
            dst[0] = inp_shape[0];
            dst[1] = (inp_shape[1] + 4 - 1) / 4;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = 4;
        } else if (layout_type() == RelayoutPlaceholder::LayoutType::
                                            WEIGHT_NCHW_TO_NCHW4_GROUP) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[2] % 4 == 0);
            dst.ndim = 6;
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1];
            dst[2] = inp_shape[2] / 4;
            dst[3] = inp_shape[3];
            dst[4] = inp_shape[4];
            dst[5] = 4;
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
        } else if (layout_type() ==
                   RelayoutPlaceholder::LayoutType::WEIGHT_HYBIRD_NCHW_NCHW88) {
            mgb_assert(inp_shape.ndim == 4 && inp_shape[0] % 8 == 0);
            dst.ndim = 5;
            dst[0] = inp_shape[0] / 8;
            dst[1] = inp_shape[2];
            dst[2] = inp_shape[3];
            dst[3] = inp_shape[1];
            dst[4] = 8;
        } else if (layout_type() == RelayoutPlaceholder::LayoutType::
                                            WEIGHT_NCHW_TO_NCHW44_DENSE ||
                   layout_type() == RelayoutPlaceholder::LayoutType::
                                            WEIGHT_NCHW_TO_NCHW44_DOT_DENSE) {
            mgb_assert(inp_shape.ndim == 4 && inp_shape[0] % 4 == 0 &&
                       inp_shape[1] % 4 == 0);
            dst.ndim = 6;
            dst[0] = inp_shape[0] / 4;
            dst[1] = inp_shape[1] / 4;
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = 4;
            dst[5] = 4;
        } else if (layout_type() == RelayoutPlaceholder::LayoutType::
                                            WEIGHT_NCHW_TO_NCHW44_GROUP ||
                   layout_type() == RelayoutPlaceholder::LayoutType::
                                            WEIGHT_NCHW_TO_NCHW44_DOT_GROUP) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[1] % 4 == 0 &&
                       inp_shape[2] % 4 == 0);
            dst.ndim = 7;
            dst[0] = inp_shape[0];
            dst[1] = inp_shape[1] / 4;
            dst[2] = inp_shape[2] / 4;
            dst[3] = inp_shape[3];
            dst[4] = inp_shape[4];
            dst[5] = 4;
            dst[6] = 4;
        } else if (layout_type() == RelayoutPlaceholder::LayoutType::
                                            WEIGHT_NCHW_TO_NCHW44_CHAN) {
            mgb_assert(inp_shape.ndim == 5 && inp_shape[1] == 1 &&
                       inp_shape[2] == 1 && inp_shape[0] % 4 == 0);
            dst.ndim = 6;
            dst[0] = inp_shape[0] / 4;
            dst[1] = inp_shape[1];
            dst[2] = inp_shape[2];
            dst[3] = inp_shape[3];
            dst[4] = inp_shape[4];
            dst[5] = 4;
        } else {
            mgb_assert(
                    layout_type() ==
                    RelayoutPlaceholder::LayoutType::WEIGHT_HYBIRD_NCHW_NCHW44);
            mgb_assert(inp_shape.ndim == 4 && inp_shape[0] % 4 == 0);
            dst.ndim = 5;
            dst[0] = inp_shape[0] / 4;
            dst[1] = inp_shape[2];
            dst[2] = inp_shape[3];
            dst[3] = inp_shape[1];
            dst[4] = 4;
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
                       "bad opr replace: src=%s{%s} dst=%s{%s}, "
                       "src.size=%zu "
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

    reformat[LayoutType::NCHW_TO_NCHW4_IC_SMALL_CONV] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto y = opr::RelayoutFormat::make(
                x, megdnn::param::RelayoutFormat::Mode::NCHW_NCHW4_IC_SMALL);
        return y.node();
    };
    reformat[LayoutType::WEIGHT_NCHW_TO_NCHW4_DENSE_IC_SMALL_CONV] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto y = opr::RelayoutFormat::make(
                x, megdnn::param::RelayoutFormat::Mode::
                           NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT);
        return y.node();
    };

    reformat[LayoutType::NCHW_TO_NCHW4] = [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});
        return y1.node();
    };
    reformat[LayoutType::NCHW4_TO_NCHW] = [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make({sub(0), sub(1) * 4, sub(2), sub(3)}, 0);
        auto y0 = opr::Dimshuffle::make(x, {0, 1, 4, 2, 3});
        auto y1 = opr::Reshape::make(y0, tshp0);
        return y1.node();
    };
    reformat[LayoutType::WEIGHT_NCHW_TO_NCHW4_DENSE] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0), sub(1) / 4, sub(2), sub(3), cv(4)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::WEIGHT_NCHW_TO_NCHW4_GROUP] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0), sub(1), sub(2) / 4, cv(4), sub(3), sub(4)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0), sub(1), sub(2) / 4, sub(3), sub(4), cv(4)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 2, 4, 5, 3});
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
    reformat[LayoutType::WEIGHT_NCHW_TO_NCHW44_DENSE] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0) / 4, cv(4), sub(1) / 4, cv(4), sub(2), sub(3)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0) / 4, sub(1) / 4, sub(2), sub(3), cv(4), cv(4)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 2, 4, 5, 3, 1});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::WEIGHT_NCHW_TO_NCHW44_GROUP] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make({sub(0), sub(1) / 4, cv(4), sub(2) / 4,
                                        cv(4), sub(3), sub(4)},
                                       0),
             tshp1 = opr::Concat::make({sub(0), sub(1) / 4, sub(2) / 4, sub(3),
                                        sub(4), cv(4), cv(4)},
                                       0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 5, 6, 4, 2});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::WEIGHT_NCHW_TO_NCHW44_CHAN] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0) / 4, cv(4), sub(1), sub(2), sub(3), sub(4)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0) / 4, sub(1), sub(2), sub(3), sub(4), cv(4)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 2, 3, 4, 5, 1});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::WEIGHT_HYBIRD_NCHW_NCHW44] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0) / 4, cv(4), sub(1), sub(2), sub(3)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0) / 4, sub(2), sub(3), sub(1), cv(4)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 3, 4, 2, 1});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::WEIGHT_NCHW_TO_NCHW44_DOT_DENSE] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0) / 4, cv(4), sub(1) / 4, cv(4), sub(2), sub(3)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0) / 4, sub(1) / 4, sub(2), sub(3), cv(4), cv(4)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 2, 4, 5, 1, 3});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2.node();
    };
    reformat[LayoutType::WEIGHT_NCHW_TO_NCHW44_DOT_GROUP] =
            [](VarNode* inp) -> VarNode* {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make({sub(0), sub(1) / 4, cv(4), sub(2) / 4,
                                        cv(4), sub(3), sub(4)},
                                       0),
             tshp1 = opr::Concat::make({sub(0), sub(1) / 4, sub(2) / 4, sub(3),
                                        sub(4), cv(4), cv(4)},
                                       0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 5, 6, 2, 4});
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
    MIDOUT_B("TensorReformatPass::apply")
    insert_pass(opt);
    translate_pass(opt);
    MIDOUT_E
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
    MIDOUT_B("EnableTensorCorePass::make")
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
        size_t nr_channels = opr->input(0)->shape()[1] * 4;
        if (nr_channels % 32 == 0) {  // use nchw32 format
            VarNode* new_inp_var = new_inp[0];
            if (opr->input(0)->shape().eq_shape(new_inp[0]->shape())) {
                new_inp_var =
                        RelayoutPlaceholder::make(
                                new_inp[0], RelayoutPlaceholder::LayoutType::
                                                    NCHW4_TO_NCHW32)
                                .node();
            } else {
                mgb_assert(opr->input(0)->shape().ndim == 5 &&
                           opr->input(0)->shape()[4] == 4);
                mgb_assert(new_inp[0]->shape().ndim == 5 &&
                           new_inp[0]->shape()[4] == 32);
            }
            auto new_param = pooling.param();
            new_param.format = Format::NCHW32;
            auto new_pooling = opr::PoolingForward::make(new_inp_var, new_param,
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
    MIDOUT_E
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
    MIDOUT_B("EnableCHWN4Pass::make")
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
            if (nr_shape_changed >=
                nr_inps / 2) {  // CHWN4 > NCHW4 -> use CHWN4
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
    MIDOUT_E
}

/* ================ EnableNCHW4Pass ================ */
VarNode* EnableNCHW4Pass::on_graph_endpoint_var(VarNode* new_var,
                                                VarNode* orig_var) const {
    if (!orig_var->shape().eq_shape(new_var->shape())) {
        return RelayoutPlaceholder::make(
                       new_var, RelayoutPlaceholder::LayoutType::NCHW4_TO_NCHW)
                .node();
    }
    return new_var;
}

//! FIXME: All float oprs do not support NCHW4. Supports it in the future plz.
std::unique_ptr<EnableNCHW4Pass> EnableNCHW4Pass::make_nchw4_converter() {
    MIDOUT_B("EnableNCHW4Pass::make")
    auto ret = std::make_unique<EnableNCHW4Pass>();
    ret->set_var_replace_check_flag(VarReplaceCheckFlag::NOCHECK);
    using RelayoutMode = RelayoutPlaceholder::LayoutType;
    megdnn::param::Convolution::Format conv_format =
            megdnn::param::Convolution::Format::NCHW4;
    megdnn::param::ConvBias::Format conv_bias_format =
            megdnn::param::ConvBias::Format::NCHW4;
    megdnn::param::BatchConvBias::Format batch_conv_bias_format =
            megdnn::param::BatchConvBias::Format::NCHW4;
    RelayoutMode src_to_nchw4_mode = RelayoutMode::NCHW_TO_NCHW4;
    RelayoutMode src_to_nchw_mode = RelayoutMode::NCHW4_TO_NCHW;
    RelayoutMode weight_to_nchw4_mode_dense =
            RelayoutMode::WEIGHT_NCHW_TO_NCHW4_DENSE;
    RelayoutMode weight_to_nchw4_mode_group =
            RelayoutMode::WEIGHT_NCHW_TO_NCHW4_GROUP;

    struct ConvMode {
        RelayoutMode weight;
        RelayoutMode src;
    };

    auto trans_nchw4 =
            [weight_to_nchw4_mode_dense, weight_to_nchw4_mode_group,
             src_to_nchw4_mode](
                    const megdnn::param::Convolution::Sparse conv_mode,
                    const VarNode* filter) -> ConvMode {
        if (conv_mode == megdnn::param::Convolution::Sparse::DENSE) {
            mgb_assert(filter->shape().ndim == 4,
                       "The origin filter is not NCHW mode");
            size_t IC = filter->shape()[1];
            if (IC < 4) {
                return {RelayoutMode::WEIGHT_NCHW_TO_NCHW4_DENSE_IC_SMALL_CONV,
                        RelayoutMode::NCHW_TO_NCHW4_IC_SMALL_CONV};
            } else {
                return {weight_to_nchw4_mode_dense, src_to_nchw4_mode};
            }
        } else {
            mgb_assert(conv_mode == megdnn::param::Convolution::Sparse::GROUP);
            mgb_assert(filter->shape().ndim == 5,
                       "The origin filter if not NCHW mode");
            size_t IC = filter->shape()[2];
            mgb_assert(IC % 4 == 0,
                       "The input channel should be divisible by 4 for group "
                       "conv");
            return {weight_to_nchw4_mode_group, src_to_nchw4_mode};
        }
    };
    auto replace_conv_opr = [trans_nchw4, conv_format](
                                    OperatorNodeBase* opr,
                                    const VarNodeArray& new_inp) {
        if (new_inp[0]->dtype().enumv() == DTypeEnum::Float32) {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
        mgb_assert(opr->input().size() == new_inp.size());
        auto& conv_opr = opr->cast_final_safe<opr::ConvolutionForward>();
        if (conv_opr.param().format !=
            megdnn::param::Convolution::Format::NCHW) {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
        auto conv_mode = trans_nchw4(conv_opr.param().sparse, new_inp[1]);
        VarNode *conv_src = new_inp[0], *conv_filter = new_inp[1];
        // src: NCHW --> NCWH4
        if (new_inp[0]->shape().ndim != 5) {
            mgb_assert(new_inp[0]->shape().ndim == 4);
            auto new_src = RelayoutPlaceholder::make(new_inp[0], conv_mode.src);
            conv_src = new_src.node();
        }
        // weight: NCHW --> NCHW4
        auto new_filter =
                RelayoutPlaceholder::make(new_inp[1], conv_mode.weight);
        conv_filter = new_filter.node();
        // format: NCHW --> NCHW4
        auto new_param = conv_opr.param();
        new_param.format = conv_format;
        // dst
        auto new_conv_opr = opr::Convolution::make(
                conv_src, conv_filter, new_param, conv_opr.execution_policy(),
                conv_opr.config());
        OperatorNodeBase* new_opr = new_conv_opr.node()->owner_opr();
        mgb_assert(new_conv_opr.shape().ndim == 5,
                   "The conv dst dim is not trans to nchw4");
        return new_opr;
    };

    auto replace_batch_conv_bias_opr = [batch_conv_bias_format,
                                        src_to_nchw4_mode](
                                               OperatorNodeBase* opr,
                                               const VarNodeArray& new_inp) {
        if (new_inp[0]->dtype().enumv() == DTypeEnum::Float32) {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
        mgb_assert(opr->input().size() == new_inp.size());
        auto& batch_conv_bias_opr =
                opr->cast_final_safe<opr::BatchConvBiasForward>();
        if (batch_conv_bias_opr.param().format !=
            megdnn::param::BatchConvBias::Format::NCHW) {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }

        mgb_assert(batch_conv_bias_opr.param().format ==
                           megdnn::param::BatchConvBias::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NCHW4");
        // what should be converted: src, weight
        VarNode *src = new_inp[0], *filter = new_inp[1];
        // src: NCHW --> NCHW4
        if (new_inp[0]->shape().ndim != 5) {
            mgb_assert(new_inp[0]->shape().ndim == 4);
            auto new_src =
                    RelayoutPlaceholder::make(new_inp[0], src_to_nchw4_mode);
            src = new_src.node();
        }
        // weight: BNCHW --> BNCHW4
        // only support dense mode, which is similar with conv->group.
        auto weight_mode =
                RelayoutPlaceholder::LayoutType::WEIGHT_NCHW_TO_NCHW4_GROUP;
        auto new_filter = RelayoutPlaceholder::make(new_inp[1], weight_mode);
        filter = new_filter.node();
        // format: NCHW --> NCHW4
        auto new_param = batch_conv_bias_opr.param();
        new_param.format = batch_conv_bias_format;
        if (new_inp.size() == 2) {
            auto dst = opr::BatchConvBias::make(
                    src, filter, new_param,
                    batch_conv_bias_opr.execution_policy(),
                    batch_conv_bias_opr.config());
            OperatorNodeBase* new_opr = dst.node()->owner_opr();
            mgb_assert(dst.shape().ndim == 5,
                       "The conv_bias dst dim is not trans to nchw4");
            return new_opr;
        }
        // bias: NCHW --> NCHW4
        VarNode* bias = new_inp[2];
        if (new_inp[2]->shape().ndim == 4) {
            auto new_bias =
                    RelayoutPlaceholder::make(new_inp[2], src_to_nchw4_mode);
            bias = new_bias.node();
        }
        if (new_inp.size() == 3) {
            auto dst = opr::BatchConvBias::make(
                    src, filter, bias, new_param,
                    batch_conv_bias_opr.execution_policy(),
                    batch_conv_bias_opr.config());
            OperatorNodeBase* new_opr = dst.node()->owner_opr();
            mgb_assert(dst.shape().ndim == 5,
                       "The conv_bias dst dim is not trans to nchw4");
            return new_opr;
        }
        // z_inp: NCHW --> NCHW4
        VarNode* z_inp = new_inp[3];
        if (new_inp[3]->shape().ndim == 4) {
            auto new_z =
                    RelayoutPlaceholder::make(new_inp[3], src_to_nchw4_mode);
            z_inp = new_z.node();
        }
        auto dst =
                opr::BatchConvBias::make(src, filter, bias, z_inp, new_param,
                                         batch_conv_bias_opr.execution_policy(),
                                         batch_conv_bias_opr.config());
        OperatorNodeBase* new_opr = dst.node()->owner_opr();
        mgb_assert(dst.shape().ndim == 5,
                   "The conv_bias dst dim is not trans to nchw4");
        return new_opr;
    };
    auto replace_conv_bias_opr = [trans_nchw4, conv_bias_format,
                                  src_to_nchw4_mode](
                                         OperatorNodeBase* opr,
                                         const VarNodeArray& new_inp) {
        if (new_inp[0]->dtype().enumv() == DTypeEnum::Float32) {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
        mgb_assert(opr->input().size() == new_inp.size());
        auto& conv_bias_opr = opr->cast_final_safe<opr::ConvBiasForward>();
        if (conv_bias_opr.param().format !=
            megdnn::param::Convolution::Format::NCHW) {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }

        // what should be converted: src, weight
        VarNode *conv_bias_src = new_inp[0], *conv_bias_filter = new_inp[1];
        auto conv_mode = trans_nchw4(conv_bias_opr.param().sparse, new_inp[1]);
        // src: NCHW --> NCHW4
        if (new_inp[0]->shape().ndim != 5) {
            mgb_assert(new_inp[0]->shape().ndim == 4);
            auto new_src = RelayoutPlaceholder::make(new_inp[0], conv_mode.src);
            conv_bias_src = new_src.node();
        }
        // weight: NCHW --> NCHW4 or GNCHW --> GNCHW4
        auto new_filter =
                RelayoutPlaceholder::make(new_inp[1], conv_mode.weight);
        conv_bias_filter = new_filter.node();
        // format: NCHW --> NCHW4
        auto new_param = conv_bias_opr.param();
        new_param.format = conv_bias_format;
        if (new_inp.size() == 2) {
            auto new_conv_bias_opr = opr::ConvBias::make(
                    conv_bias_src, conv_bias_filter, new_param,
                    conv_bias_opr.execution_policy(), conv_bias_opr.config());
            OperatorNodeBase* new_opr = new_conv_bias_opr.node()->owner_opr();
            mgb_assert(new_conv_bias_opr.shape().ndim == 5,
                       "The conv_bias dst dim is not trans to nchw4");
            return new_opr;
        }
        // bias: NCHW --> NCHW4
        VarNode* conv_bias_bias = new_inp[2];
        if (new_inp[2]->shape().ndim == 4) {
            auto new_bias =
                    RelayoutPlaceholder::make(new_inp[2], src_to_nchw4_mode);
            conv_bias_bias = new_bias.node();
        }
        if (new_inp.size() == 3) {
            auto new_conv_bias_opr = opr::ConvBias::make(
                    conv_bias_src, conv_bias_filter, conv_bias_bias, new_param,
                    conv_bias_opr.execution_policy(), conv_bias_opr.config());
            OperatorNodeBase* new_opr = new_conv_bias_opr.node()->owner_opr();
            mgb_assert(new_conv_bias_opr.shape().ndim == 5,
                       "The conv_bias dst dim is not trans to nchw4");
            return new_opr;
        }
        // z_inp: NCHW --> NCHW4
        VarNode* z_inp = new_inp[3];
        if (new_inp[3]->shape().ndim == 4) {
            auto new_z =
                    RelayoutPlaceholder::make(new_inp[3], src_to_nchw4_mode);
            z_inp = new_z.node();
        }
        auto new_conv_bias_opr = opr::ConvBias::make(
                conv_bias_src, conv_bias_filter, conv_bias_bias, z_inp,
                new_param, conv_bias_opr.execution_policy(),
                conv_bias_opr.config());
        OperatorNodeBase* new_opr = new_conv_bias_opr.node()->owner_opr();
        mgb_assert(new_conv_bias_opr.shape().ndim == 5,
                   "The conv_bias dst dim is not trans to nchw4");
        return new_opr;
    };
    auto replace_elemwise_opr = [=](OperatorNodeBase* opr,
                                    const VarNodeArray& new_inp) {
        if (new_inp[0]->dtype().enumv() == DTypeEnum::Float32) {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
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
                    auto new_var = RelayoutPlaceholder::make(new_inp[i],
                                                             src_to_nchw4_mode);
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
    auto replace_pooling_opr = [](OperatorNodeBase* opr,
                                  const VarNodeArray& new_inp) {
        if (new_inp[0]->dtype().enumv() == DTypeEnum::Float32) {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
        using Param = opr::PoolingForward::Param;
        using Format = Param::Format;
        mgb_assert(opr->input().size() == new_inp.size());
        auto& pooling = opr->cast_final_safe<opr::PoolingForward>();
        if (pooling.param().format != Format::NCHW) {
            return opr;
        }
        if (new_inp[0]->shape().ndim == 5) {
            mgb_assert(new_inp[0]->dtype().enumv() == DTypeEnum::QuantizedS8);
            auto new_param = pooling.param();
            new_param.format = Format::NCHW4;
            auto new_pooling = opr::PoolingForward::make(new_inp[0], new_param,
                                                         opr->config());
            mgb_assert(new_pooling.shape().ndim == 5,
                       "out var of Pooling opr after transform must be 5 (got: "
                       "%zu).",
                       new_pooling.shape().ndim);
            return new_pooling.node()->owner_opr();
        }
        auto new_opr =
                serialization::copy_opr_shallow(*opr, new_inp, opr->config());
        return new_opr;
    };
    auto replace_resize_opr = [](OperatorNodeBase* opr,
                                 const VarNodeArray& new_inp) {
        if (new_inp[0]->dtype().enumv() == DTypeEnum::Float32) {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
        using Param = opr::ResizeForward::Param;
        using Format = Param::Format;
        mgb_assert(opr->input().size() == new_inp.size());
        auto& resize = opr->cast_final_safe<opr::ResizeForward>();
        if (new_inp[0]->shape().ndim == 5) {
            mgb_assert(new_inp[0]->dtype().enumv() == DTypeEnum::QuantizedS8);
            auto new_param = resize.param();
            new_param.format = Format::NCHW4;
            auto new_resize = opr::ResizeForward::make(
                    new_inp[0], new_inp[1], new_param, opr->config());
            mgb_assert(new_resize.shape().ndim == 5,
                       "out var of Resize opr after transform must be 5 (got: "
                       "%zu).",
                       new_resize.shape().ndim);
            return new_resize.node()->owner_opr();
        }
        auto new_opr =
                serialization::copy_opr_shallow(*opr, new_inp, opr->config());
        return new_opr;
    };
    auto replace_warp_perspective_opr = [](OperatorNodeBase* opr,
                                           const VarNodeArray& new_inp) {
        if (new_inp[0]->dtype().enumv() == DTypeEnum::Float32) {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
        using Param = opr::WarpPerspective::Param;
        using Format = Param::Format;
        mgb_assert(opr->input().size() == new_inp.size());
        auto& warp = opr->cast_final_safe<opr::WarpPerspectiveForward>();
        if (new_inp[0]->shape().ndim == 5) {
            mgb_assert(new_inp[0]->dtype().enumv() == DTypeEnum::QuantizedS8);
            auto new_param = warp.param();
            new_param.format = Format::NCHW4;
            SymbolVar new_warp;
            if (new_inp.size() == 3) {
                new_warp = opr::WarpPerspectiveForward::make(
                        new_inp[0], new_inp[1], nullptr, new_inp[2], new_param,
                        opr->config());
            } else {
                mgb_assert(new_inp.size() == 4);
                new_warp = opr::WarpPerspectiveForward::make(
                        new_inp[0], new_inp[1], new_inp[2], new_inp[3],
                        new_param, opr->config());
            }
            mgb_assert(new_warp.shape().ndim == 5,
                       "out var of WarpPerspective opr after transform must be "
                       "5 (got: "
                       "%zu).",
                       new_warp.shape().ndim);
            return new_warp.node()->owner_opr();
        }
        auto new_opr =
                serialization::copy_opr_shallow(*opr, new_inp, opr->config());
        return new_opr;
    };
    auto&& replace_func = ret->m_opr_replace_func;
    //! supportted nchw4
    replace_func[opr::Convolution::typeinfo()] = replace_conv_opr;
    replace_func[opr::ConvBias::typeinfo()] = replace_conv_bias_opr;
    replace_func[opr::BatchConvBias::typeinfo()] = replace_batch_conv_bias_opr;
    replace_func[opr::PoolingForward::typeinfo()] = replace_pooling_opr;
    replace_func[opr::ResizeForward::typeinfo()] = replace_resize_opr;
    replace_func[opr::WarpPerspectiveForward::typeinfo()] =
            replace_warp_perspective_opr;
    replace_func[opr::Elemwise::typeinfo()] = replace_elemwise_opr;
    replace_func[opr::TypeCvt::typeinfo()] = replace_elemwise_opr;
    replace_func[opr::ElemwiseMultiType::typeinfo()] = replace_elemwise_opr;
    replace_func[opr::PowC::typeinfo()] = replace_elemwise_opr;
    //! not supported nchw4
    replace_func[opr::Concat::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::ConvolutionBackwardData::typeinfo()] =
            relayout_inp_to_nchw;
    replace_func[opr::Subtensor::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::GetVarShape::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Dimshuffle::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Reduce::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::AssertEqual::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::IncrSubtensor::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::WarpAffineForward::typeinfo()] = relayout_inp_to_nchw;
    return ret;
    MIDOUT_E
}

/* ================ EnableNchwxxPass =============== */
VarNode* EnableNchwxxPass::on_graph_endpoint_var(VarNode* new_var,
                                                 VarNode* orig_var) const {
    if (!orig_var->shape().eq_shape(new_var->shape())) {
        if (m_pack_c_size == 8) {
            return RelayoutPlaceholder::make(
                           new_var,
                           RelayoutPlaceholder::LayoutType::NCHW88_TO_NCHW)
                    .node();
        } else if (m_pack_c_size == 4) {
            return RelayoutPlaceholder::make(
                           new_var,
                           RelayoutPlaceholder::LayoutType::NCHW4_TO_NCHW)
                    .node();
        }
    }
    return new_var;
}

static inline TensorShape nchwxx_shape_2_nchw_shape(
        const TensorShape& origin_shape) {
    mgb_assert(origin_shape.ndim == 5);
    TensorShape result = origin_shape;
    result[1] *= result[4];
    result.ndim = 4;
    return result;
}

template <typename OprType>
static inline bool nchw_nchwxx_valid(
        const OprType& opr, const VarNodeArray& new_inp, const size_t pack_size,
        megdnn::param::ConvBias::NonlineMode nonline_mode =
                megdnn::param::ConvBias::NonlineMode::IDENTITY,
        bool is_dot = false) {
    auto& src_node = new_inp[0];
    auto& filter_node = new_inp[1];
    auto dst_node = opr.output(0);
    //! already transformed or have fuse Z
    if (filter_node->shape().ndim != 4 || new_inp.size() == 4) {
        return false;
    }
    megdnn::ConvolutionBase<megdnn::param::Convolution>::CanonizedFilterMeta fm;
    fm.format = megdnn::param::Convolution::Format::NCHW;
    fm.should_flip =
            opr.param().mode == megdnn::ConvBiasForward::Mode::CONVOLUTION;
    fm.group = 1;
    fm.spatial_ndim = 2;
    fm.ocpg = filter_node->shape()[0];
    fm.icpg = filter_node->shape()[1];
    fm.spatial[0] = filter_node->shape()[2];
    fm.spatial[1] = filter_node->shape()[3];
    fm.stride[0] = opr.param().stride_h;
    fm.stride[1] = opr.param().stride_w;
    fm.padding[0] = opr.param().pad_h;
    fm.padding[1] = opr.param().pad_w;
    fm.dilation[0] = opr.param().dilate_h;
    fm.dilation[1] = opr.param().dilate_w;

    megdnn::ConvBiasForward::BiasMode bias_mode =
            megdnn::ConvBiasForward::BiasMode::NO_BIAS;
    if (std::is_same<OprType, opr::ConvBiasForward>::value &&
        new_inp.size() > 2) {
        TensorShape bias_shape = new_inp[2]->shape();
        if (bias_shape.ndim == 5) {
            bias_shape = nchwxx_shape_2_nchw_shape(bias_shape);
        }
        if (bias_shape.ndim == 0) {
            bias_mode = megdnn::ConvBiasForward::BiasMode::NO_BIAS;
        } else if (bias_shape.eq_shape(dst_node->shape())) {
            bias_mode = megdnn::ConvBiasForward::BiasMode::BIAS;
        } else {
            //! just check the ndim, the detail shape check is in check_exec
            mgb_assert(bias_shape.ndim == dst_node->shape().ndim);
            bias_mode =
                    megdnn::ConvBiasForward::BiasMode::BROADCAST_CHANNEL_BIAS;
        }
    }

    if (pack_size == 4) {
        if (is_dot && filter_node->dtype().enumv() == DTypeEnum::QuantizedS8) {
            fm.format = megdnn::param::Convolution::Format::NCHW44_DOT;
        } else {
            fm.format = megdnn::param::Convolution::Format::NCHW44;
        }
    } else if (pack_size == 8) {
        fm.format = megdnn::param::Convolution::Format::NCHW88;
    } else {
        mgb_assert(0, "only support nchw44 nchw88");
    }

    return megdnn::ConvBiasForward::is_nchw_nchwxx_optimized(
            src_node->dtype().enumv(), filter_node->dtype().enumv(),
            dst_node->dtype().enumv(), fm, bias_mode, nonline_mode);
}

void EnableNchwxxPass::fill_opr_convert_fun(size_t pack_c_size) {
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

    if (pack_c_size == 4) {
        weight_to_nchwxx_mode_dense = RelayoutMode::WEIGHT_NCHW_TO_NCHW44_DENSE;
        weight_to_nchwxx_mode_group = RelayoutMode::WEIGHT_NCHW_TO_NCHW44_GROUP;
        weight_to_nchwxx_mode_chan = RelayoutMode::WEIGHT_NCHW_TO_NCHW44_CHAN;
        hybrid_nchw_nchwxx = RelayoutMode::WEIGHT_HYBIRD_NCHW_NCHW44;
        src_to_nchwxx_mode = RelayoutMode::NCHW_TO_NCHW4;
        src_to_nchw_mode = RelayoutMode::NCHW4_TO_NCHW;
        conv_bias_format = megdnn::param::ConvBias::Format::NCHW44;
        conv_format = megdnn::param::ConvolutionV0::Format::NCHW44;
        pooling_format = megdnn::param::Pooling::Format::NCHW44;
        convter_pass_name = "conv_format_nchw44";
    }
    auto test_trans_nchwxx =
            [pack_c_size, weight_to_nchwxx_mode_dense,
             weight_to_nchwxx_mode_group, weight_to_nchwxx_mode_chan,
             hybrid_nchw_nchwxx](
                    const megdnn::param::Convolution::Sparse conv_mode,
                    const VarNode* filter, const size_t stride_h,
                    const size_t stride_w,
                    bool valid_nchw_nchw44) -> TestFilterResult {
        TestFilterResult ret{TransType::TRANS_NONE, {}};
        if (conv_mode == megdnn::param::Convolution::Sparse::DENSE) {
            size_t OC = filter->shape()[0];
            size_t IC = filter->shape()[1];
            if ((IC % pack_c_size == 0) && (OC % pack_c_size == 0)) {
                ret.first = TransType::TRANS_PURE_NCHWXX;
                ret.second = weight_to_nchwxx_mode_dense;
            } else if (valid_nchw_nchw44) {
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
                             src_to_nchw_mode,
                             pack_c_size](OperatorNodeBase* opr,
                                          const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& conv_opr = opr->cast_final_safe<opr::ConvolutionForward>();
        mgb_assert(conv_opr.param().format ==
                           megdnn::param::Convolution::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NCHWXX");
        bool valid_nchw_nchw44 =
                nchw_nchwxx_valid(conv_opr, new_inp, pack_c_size);
        auto is_trans = test_trans_nchwxx(
                conv_opr.param().sparse, new_inp[1], conv_opr.param().stride_h,
                conv_opr.param().stride_w, valid_nchw_nchw44);
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
                                  src_to_nchwxx_mode, src_to_nchw_mode,
                                  pack_c_size](OperatorNodeBase* opr,
                                               const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        mgb_assert(opr->input().size() <= 3,
                   "nchwxx does not support conv_bias fuse Z right now");
        auto& conv_bias_opr = opr->cast_final_safe<opr::ConvBiasForward>();
        mgb_assert(conv_bias_opr.param().format ==
                           megdnn::param::ConvBias::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NCHWXX");
        bool valid_nchw_nchw44 =
                nchw_nchwxx_valid(conv_bias_opr, new_inp, pack_c_size,
                                  conv_bias_opr.param().nonlineMode);
        auto is_trans = test_trans_nchwxx(
                conv_bias_opr.param().sparse, new_inp[1],
                conv_bias_opr.param().stride_h, conv_bias_opr.param().stride_w,
                valid_nchw_nchw44);

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
            if (new_inp.size() > 2 && temp_inp[2]->shape().ndim == 5) {
                auto new_bias =
                        RelayoutPlaceholder::make(new_inp[2], src_to_nchw_mode);
                temp_inp[2] = new_bias.node();
            }
            auto new_opr = serialization::copy_opr_shallow(*opr, temp_inp,
                                                           opr->config());
            return new_opr;
        } else if (is_trans.first == TransType::TRANS_PURE_NCHWXX) {
            VarNode *conv_bias_src = new_inp[0], *conv_bias_filter = new_inp[1],
                    *conv_bias_bias = nullptr;
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
            //! bias trans to nchwxx mode
            if (new_inp.size() > 2) {
                if (new_inp[2]->shape().ndim == 4) {
                    auto new_bias = RelayoutPlaceholder::make(
                            new_inp[2], src_to_nchwxx_mode);
                    conv_bias_bias = new_bias.node();
                } else {
                    mgb_assert(new_inp[2]->shape().ndim == 5);
                    conv_bias_bias = new_inp[2];
                }
            }
            auto new_param = conv_bias_opr.param();
            new_param.format = conv_bias_format;
            mgb_assert(conv_bias_src->shape().ndim == 5 &&
                               conv_bias_filter->shape().ndim >= 6,
                       "The conv_bias src dim is not trans to nchwxx");
            SymbolVar new_conv_bias_opr;
            if (conv_bias_bias) {
                new_conv_bias_opr = opr::ConvBias::make(
                        conv_bias_src, conv_bias_filter, conv_bias_bias,
                        new_param, conv_bias_opr.execution_policy(),
                        conv_bias_opr.config());
            } else {
                new_conv_bias_opr = opr::ConvBias::make(
                        conv_bias_src, conv_bias_filter, new_param,
                        conv_bias_opr.execution_policy(),
                        conv_bias_opr.config());
            }
            OperatorNodeBase* new_opr = new_conv_bias_opr.node()->owner_opr();
            mgb_assert(new_conv_bias_opr.shape().ndim == 5,
                       "The conv_bias dst dim is not trans to nchwxx");
            return new_opr;
        } else {
            mgb_assert(is_trans.first == TransType::TRANS_HYBIRD_NCHWXX);
            VarNode *conv_bias_src = new_inp[0], *conv_bias_filter = new_inp[1],
                    *conv_bias_bias = nullptr;
            auto new_filter =
                    RelayoutPlaceholder::make(new_inp[1], is_trans.second);
            conv_bias_filter = new_filter.node();
            //! bias trans to nchwxx mode, bias may be scale
            if (new_inp.size() > 2) {
                if (new_inp[2]->shape().ndim == 4) {
                    auto new_bias = RelayoutPlaceholder::make(
                            new_inp[2], src_to_nchwxx_mode);
                    conv_bias_bias = new_bias.node();
                } else {
                    mgb_assert(new_inp[2]->shape().ndim == 5);
                    conv_bias_bias = new_inp[2];
                }
            }
            mgb_assert(conv_bias_src->shape().ndim == 4 &&
                       conv_bias_filter->shape().ndim == 5);
            auto new_param = conv_bias_opr.param();
            new_param.format = conv_bias_format;
            SymbolVar new_conv_bias_opr;
            if (conv_bias_bias) {
                new_conv_bias_opr = opr::ConvBias::make(
                        conv_bias_src, conv_bias_filter, conv_bias_bias,
                        new_param, conv_bias_opr.execution_policy(),
                        conv_bias_opr.config());
            } else {
                new_conv_bias_opr = opr::ConvBias::make(
                        conv_bias_src, conv_bias_filter, new_param,
                        conv_bias_opr.execution_policy(),
                        conv_bias_opr.config());
            }
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
    //! When input change and all input can convert to nchwxx, this opr will run
    //! in nchwxx mode, else it will run in nchw mode, for example concat and
    //! elemwise opr
    auto replace_multi_inp_opr = [=](OperatorNodeBase* opr,
                                     const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        bool has_inp_changed = false;
        bool can_exec_ncwxx = true;
        for (size_t i = 0; i < opr->input().size(); i++) {
            if (new_inp[i]->shape().ndim == 5) {
                has_inp_changed = true;
            } else if (new_inp[i]->shape().ndim == 4) {
                if (new_inp[i]->shape()[1] % pack_c_size != 0) {
                    can_exec_ncwxx = false;
                }
            } else if (!new_inp[i]->shape().is_scalar()) {
                can_exec_ncwxx = false;
            }
        }
        if (has_inp_changed) {
            auto temp_inp = new_inp;
            if (can_exec_ncwxx) {
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
            } else {
                for (size_t i = 0; i < opr->input().size(); i++) {
                    if (new_inp[i]->shape().ndim == 5) {
                        auto new_var = RelayoutPlaceholder::make(
                                new_inp[i], src_to_nchw_mode);
                        temp_inp[i] = new_var.node();
                    }
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

    auto&& replace_func = m_opr_replace_func;
    //! supportted nchwxx
    replace_func[opr::Convolution::typeinfo()] = replace_conv_opr;
    replace_func[opr::ConvBias::typeinfo()] = replace_conv_bias_opr;
    replace_func[opr::PoolingForward::typeinfo()] = replace_pooling_opr;
    replace_func[opr::Concat::typeinfo()] = replace_multi_inp_opr;
    replace_func[opr::Elemwise::typeinfo()] = replace_multi_inp_opr;
    replace_func[opr::TypeCvt::typeinfo()] = replace_multi_inp_opr;
    replace_func[opr::ElemwiseMultiType::typeinfo()] = replace_multi_inp_opr;
    replace_func[opr::PowC::typeinfo()] = replace_multi_inp_opr;
    //! not support yet
    replace_func[opr::ConvolutionBackwardData::typeinfo()] =
            relayout_inp_to_nchw;
    replace_func[opr::Subtensor::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::GetVarShape::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Dimshuffle::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Reduce::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::AssertEqual::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::IncrSubtensor::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::ResizeForward::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::WarpPerspectiveForward::typeinfo()] =
            relayout_inp_to_nchw;
    replace_func[opr::WarpAffineForward::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Reshape::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::AxisAddRemove::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Argmax::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::Broadcast::typeinfo()] = relayout_inp_to_nchw;
    replace_func[opr::ImmutableTensor::typeinfo()] = relayout_inp_to_nchw;
}

std::unique_ptr<EnableNchwxxPass> EnableNchwxxPass::make_nchwxx_converter(
        size_t pack_c_size) {
    MIDOUT_B("EnableNchwxxPass::make")
    auto ret = std::make_unique<EnableNchwxxPass>(pack_c_size);
    ret->set_var_replace_check_flag(VarReplaceCheckFlag::NOCHECK);
    std::string convter_pass_name = "conv_format_nchw88";
    if (pack_c_size == 4) {
        convter_pass_name = "conv_format_nchw44";
    }
    ret->fill_opr_convert_fun(pack_c_size);
    ret->set_name(convter_pass_name);
    return ret;
    MIDOUT_E
}

/* ================ EnableNchw44DotPass =============== */
VarNode* EnableNchw44DotPass::on_graph_endpoint_var(VarNode* new_var,
                                                    VarNode* orig_var) const {
    if (!orig_var->shape().eq_shape(new_var->shape())) {
        return RelayoutPlaceholder::make(
                       new_var, RelayoutPlaceholder::LayoutType::NCHW4_TO_NCHW)
                .node();
    }
    return new_var;
}

std::unique_ptr<EnableNchw44DotPass>
EnableNchw44DotPass::make_nchw44_dot_converter() {
    MIDOUT_B("EnableNchw44DotPass::make")
    auto ret = std::make_unique<EnableNchw44DotPass>();
    ret->set_var_replace_check_flag(VarReplaceCheckFlag::NOCHECK);
    //! First is whether the conv can trans to nchwxx, second is the filter
    //! trans mode

    using RelayoutMode = RelayoutPlaceholder::LayoutType;
    struct TestTransResult {
        TransType trans_type;
        RelayoutMode relayout_mod;
        megdnn::param::ConvolutionV0::Format conv_format;
    };
    constexpr size_t pack_c_size = 4_z;
    auto test_trans_nchw44_dot =
            [](const megdnn::param::Convolution::Sparse conv_mode,
               const VarNode* filter, const size_t stride_h,
               const size_t stride_w,
               const bool valid_nchw_nchw44) -> TestTransResult {
        TestTransResult ret{TransType::TRANS_NONE, {}, {}};
        bool is_int8 = filter->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                       filter->dtype().enumv() == DTypeEnum::Int8;
        if (conv_mode == megdnn::param::Convolution::Sparse::DENSE) {
            size_t OC = filter->shape()[0];
            size_t IC = filter->shape()[1];
            if ((IC % pack_c_size == 0) && (OC % pack_c_size == 0)) {
                ret.trans_type = TransType::TRANS_PURE_NCHWXX;
                if (is_int8) {
                    ret.relayout_mod =
                            RelayoutMode::WEIGHT_NCHW_TO_NCHW44_DOT_DENSE;
                    ret.conv_format =
                            megdnn::param::ConvBias::Format::NCHW44_DOT;
                } else {
                    ret.relayout_mod =
                            RelayoutMode::WEIGHT_NCHW_TO_NCHW44_DENSE;
                    ret.conv_format = megdnn::param::ConvBias::Format::NCHW44;
                }
            } else if (valid_nchw_nchw44) {
                ret.trans_type = TransType::TRANS_HYBIRD_NCHWXX;
                ret.relayout_mod = RelayoutMode::WEIGHT_HYBIRD_NCHW_NCHW44;
                if (is_int8) {
                    ret.conv_format =
                            megdnn::param::ConvBias::Format::NCHW44_DOT;
                } else {
                    ret.conv_format = megdnn::param::ConvBias::Format::NCHW44;
                }
            }
        } else {
            mgb_assert(conv_mode == megdnn::param::Convolution::Sparse::GROUP);
            size_t group = filter->shape()[0];
            size_t ocpg = filter->shape()[1];
            size_t icpg = filter->shape()[2];
            if (icpg == 1 && ocpg == 1 && (group % pack_c_size == 0)) {
                ret.trans_type = TransType::TRANS_PURE_NCHWXX;
                ret.relayout_mod = RelayoutMode::WEIGHT_NCHW_TO_NCHW44_CHAN;
                ret.conv_format = megdnn::param::ConvBias::Format::NCHW44;
            } else if ((icpg % pack_c_size == 0) && (ocpg % pack_c_size == 0)) {
                ret.trans_type = TransType::TRANS_PURE_NCHWXX;
                if (is_int8) {
                    ret.relayout_mod =
                            RelayoutMode::WEIGHT_NCHW_TO_NCHW44_DOT_GROUP;
                    ret.conv_format =
                            megdnn::param::ConvBias::Format::NCHW44_DOT;
                } else {
                    ret.relayout_mod =
                            RelayoutMode::WEIGHT_NCHW_TO_NCHW44_GROUP;
                    ret.conv_format = megdnn::param::ConvBias::Format::NCHW44;
                }
            }
        }
        return ret;
    };
    auto replace_conv_opr = [test_trans_nchw44_dot](
                                    OperatorNodeBase* opr,
                                    const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto& conv_opr = opr->cast_final_safe<opr::ConvolutionForward>();
        mgb_assert(conv_opr.param().format ==
                           megdnn::param::Convolution::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to "
                   "NCHW44_DOT");
        bool valid_nchw_nchw44 = nchw_nchwxx_valid(
                conv_opr, new_inp, pack_c_size,
                megdnn::param::ConvBias::NonlineMode::IDENTITY, true);
        auto is_trans = test_trans_nchw44_dot(
                conv_opr.param().sparse, new_inp[1], conv_opr.param().stride_h,
                conv_opr.param().stride_w, valid_nchw_nchw44);
        //! can not trans to nchwxx
        if (is_trans.trans_type == TransType::TRANS_NONE) {
            mgb_assert(new_inp[1]->shape().ndim == 4 ||
                               new_inp[1]->shape().ndim == 5,
                       "The origin filter is not NCHW mode");
            VarNodeArray temp_inp = new_inp;
            //! if src is nchwxx, should RelayoutPlaceholder to nchw
            if (temp_inp[0]->shape().ndim == 5) {
                auto new_src = RelayoutPlaceholder::make(
                        new_inp[0], RelayoutMode::NCHW4_TO_NCHW);
                temp_inp[0] = new_src.node();
            }
            auto new_opr = serialization::copy_opr_shallow(*opr, temp_inp,
                                                           opr->config());
            return new_opr;
        } else if (is_trans.trans_type == TransType::TRANS_PURE_NCHWXX) {
            //! filter trans to nchwxx mode
            mgb_assert(new_inp[1]->shape().ndim == 4 ||
                               new_inp[1]->shape().ndim == 5,
                       "The origin filter is not NCHW mode");
            VarNode *conv_src = new_inp[0], *conv_filter = new_inp[1];
            auto new_filter = RelayoutPlaceholder::make(new_inp[1],
                                                        is_trans.relayout_mod);
            conv_filter = new_filter.node();
            //! src trans to nchwxx mode
            if (new_inp[0]->shape().ndim != 5) {
                mgb_assert(new_inp[0]->shape().ndim == 4);
                auto new_src = RelayoutPlaceholder::make(
                        new_inp[0], RelayoutMode::NCHW_TO_NCHW4);
                conv_src = new_src.node();
            }
            auto new_param = conv_opr.param();
            new_param.format = is_trans.conv_format;
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
            mgb_assert(is_trans.trans_type == TransType::TRANS_HYBIRD_NCHWXX);
            VarNode *conv_src = new_inp[0], *conv_filter = new_inp[1];
            auto new_filter = RelayoutPlaceholder::make(new_inp[1],
                                                        is_trans.relayout_mod);
            conv_filter = new_filter.node();
            mgb_assert(conv_src->shape().ndim == 4 &&
                               conv_filter->shape().ndim == 5,
                       "The src and filter is OK");
            auto new_param = conv_opr.param();
            new_param.format = is_trans.conv_format;
            auto new_conv_opr = opr::Convolution::make(
                    conv_src, conv_filter, new_param,
                    conv_opr.execution_policy(), conv_opr.config());
            OperatorNodeBase* new_opr = new_conv_opr.node()->owner_opr();
            mgb_assert(new_conv_opr.shape().ndim == 5,
                       "The conv dst dim is not trans to nchwxx");
            return new_opr;
        }
    };

    auto replace_conv_bias_opr = [test_trans_nchw44_dot](
                                         OperatorNodeBase* opr,
                                         const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        mgb_assert(opr->input().size() <= 3,
                   "nchwxx-dot does not support conv_bias fuse Z right now");
        auto& conv_bias_opr = opr->cast_final_safe<opr::ConvBiasForward>();
        mgb_assert(conv_bias_opr.param().format ==
                           megdnn::param::ConvBias::Format::NCHW,
                   "ConvertFormat Pass only support converting NCHW to NCHWXX");
        bool valid_nchw_nchw44 =
                nchw_nchwxx_valid(conv_bias_opr, new_inp, pack_c_size,
                                  conv_bias_opr.param().nonlineMode, true);
        auto is_trans = test_trans_nchw44_dot(
                conv_bias_opr.param().sparse, new_inp[1],
                conv_bias_opr.param().stride_h, conv_bias_opr.param().stride_w,
                valid_nchw_nchw44);
        auto megdnn_conv =
                opr::intl::get_megdnn_handle(conv_bias_opr.comp_node())
                        ->create_operator<megdnn::ConvBiasForward>();
        SmallVector<TensorLayout> layouts;

        //! can not trans to nchwxx
        if (is_trans.trans_type == TransType::TRANS_NONE) {
            mgb_assert(new_inp[1]->shape().ndim == 4 ||
                               new_inp[1]->shape().ndim == 5,
                       "The origin filter is not NCHW mode");
            VarNodeArray temp_inp = new_inp;
            //! if src is nchwxx, should RelayoutPlaceholder to nchw
            if (temp_inp[0]->shape().ndim == 5) {
                auto new_src = RelayoutPlaceholder::make(
                        new_inp[0], RelayoutMode::NCHW4_TO_NCHW);
                temp_inp[0] = new_src.node();
            }

            //! the bias is nchwxx
            if (new_inp.size() > 2 && temp_inp[2]->shape().ndim == 5) {
                auto new_bias = RelayoutPlaceholder::make(
                        new_inp[2], RelayoutMode::NCHW4_TO_NCHW);
                temp_inp[2] = new_bias.node();
            }
            auto new_opr = serialization::copy_opr_shallow(*opr, temp_inp,
                                                           opr->config());
            return new_opr;
        } else if (is_trans.trans_type == TransType::TRANS_PURE_NCHWXX) {
            VarNode *conv_bias_src = new_inp[0], *conv_bias_filter = new_inp[1],
                    *conv_bias_bias = nullptr;
            //! filter trans to nchwxx mode
            mgb_assert(new_inp[1]->shape().ndim == 4 ||
                               new_inp[1]->shape().ndim == 5,
                       "The origin filter is not NCHW mode");
            auto new_filter = RelayoutPlaceholder::make(new_inp[1],
                                                        is_trans.relayout_mod);
            conv_bias_filter = new_filter.node();
            //! src trans to nchwxx mode
            if (new_inp[0]->shape().ndim != 5) {
                mgb_assert(new_inp[0]->shape().ndim == 4);
                auto new_src = RelayoutPlaceholder::make(
                        new_inp[0], RelayoutMode::NCHW_TO_NCHW4);
                conv_bias_src = new_src.node();
            }
            //! bias trans to nchwxx mode
            if (new_inp.size() > 2) {
                if (new_inp[2]->shape().ndim == 4) {
                    auto new_bias = RelayoutPlaceholder::make(
                            new_inp[2], RelayoutMode::NCHW_TO_NCHW4);
                    conv_bias_bias = new_bias.node();
                } else {
                    mgb_assert(new_inp[2]->shape().ndim == 5);
                    conv_bias_bias = new_inp[2];
                }
            }
            auto new_param = conv_bias_opr.param();
            new_param.format = is_trans.conv_format;
            mgb_assert(conv_bias_src->shape().ndim == 5 &&
                               conv_bias_filter->shape().ndim >= 6,
                       "The conv_bias src dim is not trans to nchwxx");
            SymbolVar new_conv_bias_opr;
            if (conv_bias_bias) {
                new_conv_bias_opr = opr::ConvBias::make(
                        conv_bias_src, conv_bias_filter, conv_bias_bias,
                        new_param, conv_bias_opr.execution_policy(),
                        conv_bias_opr.config());
            } else {
                new_conv_bias_opr = opr::ConvBias::make(
                        conv_bias_src, conv_bias_filter, new_param,
                        conv_bias_opr.execution_policy(),
                        conv_bias_opr.config());
            }
            OperatorNodeBase* new_opr = new_conv_bias_opr.node()->owner_opr();
            mgb_assert(new_conv_bias_opr.shape().ndim == 5,
                       "The conv_bias dst dim is not trans to nchwxx");
            return new_opr;
        } else {
            mgb_assert(is_trans.trans_type == TransType::TRANS_HYBIRD_NCHWXX);
            VarNode *conv_bias_src = new_inp[0], *conv_bias_filter = new_inp[1],
                    *conv_bias_bias = nullptr;
            auto new_filter = RelayoutPlaceholder::make(new_inp[1],
                                                        is_trans.relayout_mod);
            conv_bias_filter = new_filter.node();
            //! bias trans to nchwxx mode, bias may be scale
            if (new_inp.size() > 2) {
                if (new_inp[2]->shape().ndim == 4) {
                    auto new_bias = RelayoutPlaceholder::make(
                            new_inp[2], RelayoutMode::NCHW_TO_NCHW4);
                    conv_bias_bias = new_bias.node();
                } else {
                    mgb_assert(new_inp[2]->shape().ndim == 5);
                    conv_bias_bias = new_inp[2];
                }
            }
            mgb_assert(conv_bias_src->shape().ndim == 4 &&
                       conv_bias_filter->shape().ndim == 5);
            auto new_param = conv_bias_opr.param();
            new_param.format = is_trans.conv_format;
            SymbolVar new_conv_bias_opr;
            if (conv_bias_bias) {
                new_conv_bias_opr = opr::ConvBias::make(
                        conv_bias_src, conv_bias_filter, conv_bias_bias,
                        new_param, conv_bias_opr.execution_policy(),
                        conv_bias_opr.config());
            } else {
                new_conv_bias_opr = opr::ConvBias::make(
                        conv_bias_src, conv_bias_filter, new_param,
                        conv_bias_opr.execution_policy(),
                        conv_bias_opr.config());
            }
            OperatorNodeBase* new_opr = new_conv_bias_opr.node()->owner_opr();
            mgb_assert(new_conv_bias_opr.shape().ndim == 5,
                       "The conv dst dim is not trans to nchwxx");
            return new_opr;
        }
    };
    ret->fill_opr_convert_fun(4);
    auto&& replace_func = ret->m_opr_replace_func;
    //! supportted nchwxx
    replace_func[opr::Convolution::typeinfo()] = replace_conv_opr;
    replace_func[opr::ConvBias::typeinfo()] = replace_conv_bias_opr;
    return ret;
    MIDOUT_E
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
                     cg::SingleCNOperatorNodeBase)  // {
public:
    AbstractShuffleOpr(VarNode* inpvar, TensorFormat inp_format,
                       TensorFormat out_format);
    
    static SymbolVar make(VarNode* inpvar, TensorFormat inp_format,
                          TensorFormat out_format);
    
    TensorFormat inp_format() const {
        return m_inp_format;
    }
    
    TensorFormat out_format() const {
        return m_out_format;
    }
    
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
            TensorFormat out_format = TensorFormat::NCHW,
                         inp_format = out_format;
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
                mgb_assert(m_reformat.find(std::make_pair(
                                   inp_format, out_format)) != m_reformat.end(),
                           "Unsupported shuffle shuffle remove pass");
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
    MIDOUT_B("ShuffleShuffleRemovePass::apply")
    opt.set_var_replace_check_flag(VarReplaceCheckFlag::CHECK_SHAPE |
                                   VarReplaceCheckFlag::CHECK_DTYPE);
    Impl{opt};
    MIDOUT_E
}

/* ==================== FoldingConvBiasDimshufflePass ================= */
const char* FoldingConvBiasDimshufflePass::name() const {
    return mgb_cstr_log("folding conv bias dimshuffle pass");
}

void FoldingConvBiasDimshufflePass::apply(OptState& opt) const {
    MIDOUT_B("FoldingConvBiasDimshufflePass::apply");
    using DepType = cg::OperatorNodeProp::DepType;
    ThinHashMap<OperatorNodeBase*,
                SmallVector<std::pair<OperatorNodeBase*, DepType>>>
            readers;
    static const ThinHashSet<Typeinfo*> opr_type_list = {
            opr::TypeCvt::typeinfo(), opr::Dimshuffle::typeinfo(),
            opr::Reshape::typeinfo(), opr::ConvBias::typeinfo()};
    opt.graph().iter([&readers](OperatorNodeBase* opr) {
        for (auto&& i : opr->node_prop().dep_map()) {
            if (opr_type_list.count(i.first->owner_opr()->dyn_typeinfo())) {
                readers[i.first->owner_opr()].emplace_back(opr, i.second);
            }
        }
    });

    auto rewriter = opt.graph().make_rewriter();
    auto nchw42nchw = [](VarNode* inp) -> VarNode* {
        mgb_assert(inp->shape().ndim == 5 && inp->shape()[4] == 4);
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make({sub(0), sub(1) * 4, sub(2), sub(3)}, 0);
        auto y0 = opr::Dimshuffle::make(x, {0, 1, 4, 2, 3});
        auto y1 = opr::Reshape::make(y0, tshp);
        auto y2 = opr::TypeCvt::make(y1, dtype::Float32());  
        return y2.node();
    };

    auto nchw42nchw32 = [](VarNode* inp) -> VarNode* {
        mgb_assert(inp->shape().ndim == 5 && inp->shape()[4] == 4);
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

    auto nchw322nchw4 = [](VarNode* inp) -> VarNode* {
        mgb_assert(inp->shape().ndim == 5 && inp->shape()[4] == 32);
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

    auto try_conv_dimshuffle_reshape_typecvt = [&rewriter, &readers,
                                                &nchw42nchw](
                                                       OperatorNodeBase* opr) {
        ThinHashSet<OperatorNodeBase*> opr_set;
        ThinHashSet<OperatorNodeBase*> reader_set;
        // check typecvt
        auto typecvt = try_cast_as_op<opr::TypeCvt>(opr);
        if (typecvt == nullptr)
            return false;
        auto inp_dtype = typecvt->input(0)->dtype(),
             out_dtype = typecvt->output(0)->dtype();
        bool is_s82f32 = inp_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                         out_dtype.enumv() == DTypeEnum::Float32;
        if (!is_s82f32)
            return false;
        opr_set.insert(opr);

        // check reshape
        auto reshape =
                try_cast_as_op<opr::Reshape>(typecvt->input(0)->owner_opr());
        if (reshape == nullptr)
            return false;
        opr_set.insert(reshape);
        for (auto&& i : readers[reshape]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }

        // check shuffle
        auto shuffle =
                try_cast_as_op<opr::Dimshuffle>(reshape->input(0)->owner_opr());
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
        opr_set.insert(shuffle);
        for (auto&& i : readers[shuffle]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }

        // check conv bias
        auto conv_bias =
                try_cast_as_op<opr::ConvBias>(shuffle->input(0)->owner_opr());
        if (conv_bias == nullptr)
            return false;
        inp_dtype = conv_bias->input(0)->dtype();
        bool is_s8nchw4 = inp_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                          conv_bias->param().format ==
                                  megdnn::param::ConvBias::Format::NCHW4;
        if (!is_s8nchw4)
            return false;
        if (conv_bias->input().size() != 3)
            return false;
        opr_set.insert(conv_bias);
        for (auto&& i : readers[conv_bias]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        for (auto reader : reader_set) {
            if (opr_set.count(reader) <= 0) {
                return false;
            }
        }
        auto src = rewriter.get_var(conv_bias->input(0)),
             filter = rewriter.get_var(conv_bias->input(1)),
             bias = rewriter.get_var(conv_bias->input(2));
        auto new_bias = nchw42nchw(bias);
        auto new_param = conv_bias->param();
        new_param.format = megdnn::param::ConvBias::Format::NCHW4_NCHW;
        auto conv_bias_shuffle = opr::ConvBias::make(
                src, filter, new_bias, new_param, conv_bias->execution_policy(),
                OperatorNodeConfig{dtype::Float32()});
        rewriter.replace_var(opr->output(0), conv_bias_shuffle.node(),
                             mgb_cstr_log("replace conv_bias + typecvt + "
                                          "dimshuffle + "
                                          "reshape to conv_bias(NCHW4_NCHW)"));
        return true;
    };

    auto try_conv_reformat_nchw42nchw32 = [&rewriter, &nchw42nchw32,
                                           &readers](OperatorNodeBase* opr) {
        ThinHashSet<OperatorNodeBase*> opr_set;
        ThinHashSet<OperatorNodeBase*> reader_set;
        // check reshape
        auto reshape1 = try_cast_as_op<opr::Reshape>(opr);
        if (reshape1 == nullptr)
            return false;
        opr_set.insert(opr);
        // check dimshuffle
        auto shuffle = try_cast_as_op<opr::Dimshuffle>(
                reshape1->input(0)->owner_opr());
        if (shuffle == nullptr)
            return false;
        auto&& param = shuffle->param();
        if (param.pattern_len != 6)
            return false;
        bool is_nchw42nchw32 = param.pattern[0] == 0 && param.pattern[1] == 1 &&
                               param.pattern[2] == 3 && param.pattern[3] == 4 &&
                               param.pattern[4] == 2 && param.pattern[5] == 5 &&
                               shuffle->output(0)->shape()[5] == 4 &&
                               shuffle->output(0)->shape()[4] == 8;
        if (!is_nchw42nchw32)
            return false;
        opr_set.insert(shuffle);
        for (auto&& i : readers[shuffle]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        // check reshape
        auto reshape2 =
                try_cast_as_op<opr::Reshape>(shuffle->input(0)->owner_opr());
        if (reshape2 == nullptr)
            return false;
        opr_set.insert(reshape2);
        for (auto&& i : readers[reshape2]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        // check conv bias
        auto conv_bias =
                try_cast_as_op<opr::ConvBias>(reshape2->input(0)->owner_opr());
        if (conv_bias == nullptr)
            return false;
        auto inp_dtype = conv_bias->input(0)->dtype();
        bool is_s8nchw4 = inp_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                          conv_bias->param().format ==
                                  megdnn::param::ConvBias::Format::NCHW4;
        if (!is_s8nchw4)
            return false;
        if (conv_bias->input().size() != 3)
            return false;
        opr_set.insert(conv_bias);
        for (auto&& i : readers[conv_bias]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        for (auto reader : reader_set) {
            if (opr_set.count(reader) <= 0) {
                return false;
            }
        }
        auto src = rewriter.get_var(conv_bias->input(0)),
             filter = rewriter.get_var(conv_bias->input(1)),
             bias = rewriter.get_var(conv_bias->input(2));
        auto new_bias = nchw42nchw32(bias);
        auto new_param = conv_bias->param();
        new_param.format = megdnn::param::ConvBias::Format::NCHW4_NCHW32;
        auto conv_bias_shuffle = opr::ConvBias::make(
                src, filter, new_bias, new_param, conv_bias->execution_policy(),
                conv_bias->config());
        rewriter.replace_var(
                opr->output(0), conv_bias_shuffle.node(),
                mgb_cstr_log("replace conv_bias + "
                             "reformat to conv_bias(NCHW4_NCHW32)"));
        return true;
    };

    auto try_conv_reformat_nchw322nchw4 = [&rewriter, &readers, &nchw322nchw4](
                                                  OperatorNodeBase* opr) {
        ThinHashSet<OperatorNodeBase*> opr_set;
        ThinHashSet<OperatorNodeBase*> reader_set;
        // check reshape
        auto reshape1 = try_cast_as_op<opr::Reshape>(opr);
        if (reshape1 == nullptr)
            return false;
        opr_set.insert(opr);
        // check dimshuffle
        auto shuffle = try_cast_as_op<opr::Dimshuffle>(
                reshape1->input(0)->owner_opr());
        if (shuffle == nullptr)
            return false;
        auto&& param = shuffle->param();
        if (param.pattern_len != 6)
            return false;
        bool is_nchw322nchw4 = param.pattern[0] == 0 && param.pattern[1] == 1 &&
                               param.pattern[2] == 4 && param.pattern[3] == 2 &&
                               param.pattern[4] == 3 && param.pattern[5] == 5 &&
                               shuffle->input(0)->shape()[5] == 4 &&
                               shuffle->input(0)->shape()[4] == 8;
        if (!is_nchw322nchw4)
            return false;
        opr_set.insert(shuffle);
        for (auto&& i : readers[shuffle]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        // check reshape
        auto reshape2 =
                try_cast_as_op<opr::Reshape>(shuffle->input(0)->owner_opr());
        if (reshape2 == nullptr)
            return false;
        opr_set.insert(reshape2);
        for (auto&& i : readers[reshape2]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        // check conv bias
        auto conv_bias =
                try_cast_as_op<opr::ConvBias>(reshape2->input(0)->owner_opr());
        if (conv_bias == nullptr)
            return false;
        auto inp_dtype = conv_bias->input(0)->dtype();
        bool is_s8nchw32 = inp_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                          conv_bias->param().format ==
                                  megdnn::param::ConvBias::Format::NCHW32;
        if (!is_s8nchw32)
            return false;
        if (conv_bias->input().size() != 3)
            return false;
        opr_set.insert(conv_bias);
        for (auto&& i : readers[conv_bias]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        for (auto reader : reader_set) {
            if (opr_set.count(reader) <= 0) {
                return false;
            }
        }
        auto src = rewriter.get_var(conv_bias->input(0)),
             filter = rewriter.get_var(conv_bias->input(1)),
             bias = rewriter.get_var(conv_bias->input(2));
        auto new_bias = nchw322nchw4(bias);
        auto new_param = conv_bias->param();
        new_param.format = megdnn::param::ConvBias::Format::NCHW32_NCHW4;
        auto conv_bias_shuffle = opr::ConvBias::make(
                src, filter, new_bias, new_param, conv_bias->execution_policy(),
                conv_bias->config());
        rewriter.replace_var(
                opr->output(0), conv_bias_shuffle.node(),
                mgb_cstr_log("replace conv_bias + "
                             "reformat to conv_bias(NCHW32_NCHW4)"));
        return true;
    };
    MGB_MARK_USED_VAR(try_conv_reformat_nchw322nchw4);

    auto on_opr = [&try_conv_dimshuffle_reshape_typecvt,
                   &try_conv_reformat_nchw42nchw32,
#if CUDA_VERSION >= 10020
                   &try_conv_reformat_nchw322nchw4,
#endif
                   &rewriter](OperatorNodeBase* opr) {
        if (!try_conv_dimshuffle_reshape_typecvt(opr) &&
            !try_conv_reformat_nchw42nchw32(opr)
#if CUDA_VERSION >= 10020
            && !try_conv_reformat_nchw322nchw4(opr)
#endif
        ) {
            rewriter.auto_replace_outputs(opr);
        }
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();

    MIDOUT_E
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
