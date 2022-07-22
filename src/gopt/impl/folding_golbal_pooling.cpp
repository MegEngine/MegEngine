#include "megbrain/gopt/inference.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/adaptive_pooling.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/opr_shallow_copy.h"

#include "megdnn/opr_param_defs.h"

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "../../core/impl/graph/cg_impl.h"
#include "./gopt_helper.h"

#include "megbrain/utils/hash_ct.h"

#include "midout.h"

MIDOUT_DECL(megbrain_folding_global_pooling)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_folding_global_pooling, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace gopt;

/* ==================== FoldingGlobalPoolingPass ================= */
const char* FoldingGlobalPoolingPass::name() const {
    return mgb_cstr_log("folding reduce mean pass");
}

void FoldingGlobalPoolingPass::apply(OptState& opt) const {
    MIDOUT_B("FoldingGlobalPoolingPass::apply");

    FindNext find_tool(opt);

    auto rewriter = opt.graph().make_rewriter();
    /**
     *
     *   reshape+------>reduce(mean or max)+--->axis_add_remove*n
     *                          ||
     *                          ||
     *                          ||
     *                          \/
     *                  adaptive_pooling(1,1)
     */
    auto try_fuse_global_pooling_axis_add = [&rewriter,
                                             &find_tool](OperatorNodeBase* opr) {
        ThinHashSet<OperatorNodeBase*> opr_set;
        ThinHashSet<OperatorNodeBase*> reader_set;
        MGB_MARK_USED_VAR(rewriter);
        MGB_MARK_USED_VAR(find_tool);

        auto axis_modi = try_cast_as_op<opr::AxisAddRemove>(opr);
        CHECK_OR_RETURN(axis_modi);
        CHECK_OR_RETURN(find_tool.used_count(axis_modi) <= 1);
        auto output_shape = axis_modi->output(0)->shape();
        CHECK_OR_RETURN(output_shape.ndim == 4);
        CHECK_OR_RETURN(output_shape[2] == output_shape[3] && output_shape[2] == 1);

        auto axis_input = axis_modi->input(0)->owner_opr();
        auto axis_modi_x = axis_input->try_cast_final<opr::AxisAddRemove>();
        auto reduce = axis_input->try_cast_final<opr::Reduce>();
        while (axis_modi_x) {
            CHECK_OR_RETURN(find_tool.used_count(axis_modi_x) == 1);
            auto axis_input_x = axis_modi_x->input(0)->owner_opr();
            reduce = axis_input_x->try_cast_final<opr::Reduce>();
            axis_modi_x = axis_input_x->try_cast_final<opr::AxisAddRemove>();
        }

        CHECK_OR_RETURN(reduce);
        auto reduce_mode = reduce->param().mode;
        CHECK_OR_RETURN(
                reduce_mode == opr::Reduce::Param::Mode::MAX ||
                reduce_mode == opr::Reduce::Param::Mode::MEAN);
        auto reduce_axis = reduce->param().axis;
        CHECK_OR_RETURN(reduce_axis == 2)

        auto reshape = reduce->input(0)->owner_opr()->try_cast_final<opr::Reshape>();
        CHECK_OR_RETURN(reshape);

        auto reshape_in_shape = reshape->input(0)->shape();
        auto reshape_out_shape = reshape->output(0)->shape();
        bool merge_hw =
                reshape_out_shape.ndim == 3 && reshape_in_shape.ndim == 4 &&
                reshape_in_shape[2] * reshape_in_shape[3] == reshape_out_shape[2];
        CHECK_OR_RETURN(merge_hw);
        opr::AdaptivePooling::Param param;
        if (reduce_mode == opr::Reduce::Param::Mode::MAX) {
            param.mode = opr::AdaptivePooling::Param::Mode::MAX;
        } else {
            mgb_assert(reduce_mode == opr::Reduce::Param::Mode::MEAN);
            param.mode = opr::AdaptivePooling::Param::Mode::AVERAGE;
        }

        auto new_node = opr::AdaptivePooling::make(
                rewriter.get_var(reshape->input(0)), {1, 1}, param);
        rewriter.replace_var(
                axis_modi->output(0), new_node.node(),
                mgb_cstr_log("replace reshape+reduce+add_axis -> adaptive pooling"));
        return true;
    };
    /**
     *
     *   reshape+------>reduce(mean or max)+--->dimshuffle(0,1,-1,-1)
     *                          ||
     *                          ||
     *                          ||
     *                          \/
     *                  adaptive_pooling(1,1)
     */
    auto try_fuse_global_pooling_dimshuffle = [&rewriter,
                                               &find_tool](OperatorNodeBase* opr) {
        ThinHashSet<OperatorNodeBase*> opr_set;
        ThinHashSet<OperatorNodeBase*> reader_set;
        MGB_MARK_USED_VAR(rewriter);
        MGB_MARK_USED_VAR(find_tool);

        auto dimshuffle = try_cast_as_op<opr::Dimshuffle>(opr);
        CHECK_OR_RETURN(dimshuffle);
        auto patten_param = dimshuffle->param();

        CHECK_OR_RETURN(patten_param.pattern_len == 4);
        auto patten = patten_param.pattern;
        CHECK_OR_RETURN(
                patten[0] == 0 && patten[1] == 1 && patten[2] == -1 && patten[3] == -1);
        auto axis_remove =
                dimshuffle->input(0)->owner_opr()->try_cast_final<opr::AxisAddRemove>();
        CHECK_OR_RETURN(axis_remove);

        auto reduce = axis_remove->input(0)->owner_opr()->try_cast_final<opr::Reduce>();
        CHECK_OR_RETURN(reduce);
        auto reduce_mode = reduce->param().mode;
        CHECK_OR_RETURN(
                reduce_mode == opr::Reduce::Param::Mode::MAX ||
                reduce_mode == opr::Reduce::Param::Mode::MEAN);
        auto reduce_axis = reduce->param().axis;
        CHECK_OR_RETURN(reduce_axis == 2)

        auto reshape = reduce->input(0)->owner_opr()->try_cast_final<opr::Reshape>();
        CHECK_OR_RETURN(reshape);

        auto reshape_in_shape = reshape->input(0)->shape();
        auto reshape_out_shape = reshape->output(0)->shape();
        bool merge_hw =
                reshape_out_shape.ndim == 3 && reshape_in_shape.ndim == 4 &&
                reshape_in_shape[2] * reshape_in_shape[3] == reshape_out_shape[2];
        CHECK_OR_RETURN(merge_hw);
        opr::AdaptivePooling::Param param;
        if (reduce_mode == opr::Reduce::Param::Mode::MAX) {
            param.mode = opr::AdaptivePooling::Param::Mode::MAX;
        } else {
            mgb_assert(reduce_mode == opr::Reduce::Param::Mode::MEAN);
            param.mode = opr::AdaptivePooling::Param::Mode::AVERAGE;
        }
        auto new_node = opr::AdaptivePooling::make(
                rewriter.get_var(reshape->input(0)), {1, 1}, param);
        rewriter.replace_var(
                dimshuffle->output(0), new_node.node(),
                mgb_cstr_log("replace reshape+reduce+dimshuffle -> adaptive pooling"));
        return true;
    };

    auto on_opr = [&try_fuse_global_pooling_axis_add,
                   &try_fuse_global_pooling_dimshuffle,
                   &rewriter](OperatorNodeBase* opr) {
        if (!try_fuse_global_pooling_axis_add(opr) &&
            !try_fuse_global_pooling_dimshuffle(opr)) {
            rewriter.auto_replace_outputs(opr);
        }
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();

    MIDOUT_E
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
