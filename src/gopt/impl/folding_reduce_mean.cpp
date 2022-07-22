#include "megbrain/gopt/inference.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/opr_shallow_copy.h"

#include "megdnn/opr_param_defs.h"

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "../../core/impl/graph/cg_impl.h"
#include "./gopt_helper.h"

#include "megbrain/utils/hash_ct.h"

#include "midout.h"

MIDOUT_DECL(megbrain_folding_reduce_mean)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_folding_reduce_mean, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace gopt;

/* ==================== FoldingReduceMeanPass ================= */
const char* FoldingReduceMeanPass::name() const {
    return mgb_cstr_log("folding reduce mean pass");
}

void FoldingReduceMeanPass::apply(OptState& opt) const {
    MIDOUT_B("FoldingReduceMeanPass::apply");
    FindNext find_tool(opt);

    auto rewriter = opt.graph().make_rewriter();

    /**
     *   reshape+---------->reduce(axis, sum)+--------->axis_remove+----------->true_div
     *   |                                                                       ^
     *   |                                                                       |
     *   +--------------> get_var_shape(axis)+------------>type_cvt(fp32)+-------+
     *                                   ||
     *                                   ||
     *                                   \/
     *   reshape+-------->reduce(axis, mean)+--------->axis_remove
     *
     *
     **/
    auto try_fuse_reduce_mean = [&rewriter, &find_tool](OperatorNodeBase* opr) {
        ThinHashSet<OperatorNodeBase*> opr_set;
        ThinHashSet<OperatorNodeBase*> reader_set;
        MGB_MARK_USED_VAR(rewriter);
        // check true_div
        auto elemwise = try_cast_as_op<opr::Elemwise>(opr);
        CHECK_OR_RETURN(elemwise);
        auto mode_ok = elemwise->param().mode == opr::Elemwise::Mode::TRUE_DIV;
        CHECK_OR_RETURN(mode_ok);

        auto input0 = elemwise->input(0)->owner_opr();
        auto remove_axis = input0->try_cast_final<opr::AxisAddRemove>();
        auto reduce = input0->try_cast_final<opr::Reduce>();
        if (remove_axis) {
            reduce = remove_axis->input(0)->owner_opr()->try_cast_final<opr::Reduce>();
        }
        CHECK_OR_RETURN(reduce);

        bool reduce_sum = reduce->param().mode == opr::Reduce::Param::Mode::SUM;
        CHECK_OR_RETURN(reduce_sum);

        auto input1 = elemwise->input(1)->owner_opr();
        auto typecvt = input1->try_cast_final<opr::TypeCvt>();
        CHECK_OR_RETURN(typecvt);
        auto is_typecvt_f32 = typecvt->param().enumv() == DTypeEnum::Float32;
        CHECK_OR_RETURN(is_typecvt_f32);

        auto get_var_shape =
                typecvt->input(0)->owner_opr()->try_cast_final<opr::GetVarShape>();
        CHECK_OR_RETURN(get_var_shape);

        bool same_parent =
                get_var_shape->input(0)->owner_opr() == reduce->input(0)->owner_opr();
        CHECK_OR_RETURN(same_parent);

        CHECK_OR_RETURN(
                find_tool.used_count(get_var_shape->input(0)->owner_opr()) == 2);

        bool same_axis = get_var_shape->param().axis == reduce->param().axis;
        CHECK_OR_RETURN(same_axis);

        auto new_reduce_param = reduce->param();
        new_reduce_param.mode = opr::Reduce::Mode::MEAN;
        auto new_node =
                opr::Reduce::make(rewriter.get_var(reduce->input(0)), new_reduce_param);
        if (remove_axis) {
            new_node = opr::AxisAddRemove::make(
                    new_node, remove_axis->param(), remove_axis->config());
        }
        rewriter.replace_var(
                opr->output(0), new_node.node(),
                mgb_cstr_log("replace reduce_sum+div_axis -> reduce_mean"));
        return true;
    };

    auto on_opr = [&try_fuse_reduce_mean, &rewriter](OperatorNodeBase* opr) {
        if (!try_fuse_reduce_mean(opr)) {
            rewriter.auto_replace_outputs(opr);
        }
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
    MIDOUT_E
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
