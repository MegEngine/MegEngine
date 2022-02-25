#include "megbrain/gopt/inference.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/opr_shallow_copy.h"

#include "megdnn/opr_param_defs.h"

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "megbrain/utils/hash_ct.h"

#include "midout.h"

#include "megbrain/gopt/reformat_manager.h"

#if CUDA_VERSION >= 10020
MIDOUT_DECL(megbrain_folding_conv_typecvt)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_folding_conv_typecvt, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace gopt;
using ReformatKey = ReformatManager::ReformatKey;

/* ==================== FoldingConvBiasTypecvtPass ================= */
const char* FoldingConvBiasTypecvtPass::name() const {
    return mgb_cstr_log("folding conv bias typecvt pass");
}

void FoldingConvBiasTypecvtPass::apply(OptState& opt) const {
    MIDOUT_B("FoldingConvBiasTypecvtPass::apply");
    using DepType = cg::OperatorNodeProp::DepType;
    ThinHashMap<OperatorNodeBase*, SmallVector<std::pair<OperatorNodeBase*, DepType>>>
            readers;
    static const ThinHashSet<Typeinfo*> opr_type_list = {
            opr::TypeCvt::typeinfo(), opr::ConvBias::typeinfo()};
    opt.graph().iter([&readers](OperatorNodeBase* opr) {
        for (auto&& i : opr->node_prop().dep_map()) {
            if (opr_type_list.count(i.first->owner_opr()->dyn_typeinfo())) {
                readers[i.first->owner_opr()].emplace_back(opr, i.second);
            }
        }
    });

    auto rewriter = opt.graph().make_rewriter();

    auto try_conv_typecvt = [&rewriter, &readers](OperatorNodeBase* opr) {
        ThinHashSet<OperatorNodeBase*> opr_set;
        ThinHashSet<OperatorNodeBase*> reader_set;
        // check typecvt
        auto typecvt = try_cast_as_op<opr::TypeCvt>(opr);
        if (typecvt == nullptr)
            return false;
        auto inp_dtype_typecvt = typecvt->input(0)->dtype(),
             out_dtype_typecvt = typecvt->output(0)->dtype();
        bool is_s82f32 = inp_dtype_typecvt.enumv() == DTypeEnum::QuantizedS8 &&
                         out_dtype_typecvt.enumv() == DTypeEnum::Float32;
        bool is_s82s4 = inp_dtype_typecvt.enumv() == DTypeEnum::QuantizedS8 &&
                        (out_dtype_typecvt.enumv() == DTypeEnum::QuantizedS4 ||
                         out_dtype_typecvt.enumv() == DTypeEnum::Quantized4Asymm);
        bool is_s42s8 = (inp_dtype_typecvt.enumv() == DTypeEnum::QuantizedS4 ||
                         inp_dtype_typecvt.enumv() == DTypeEnum::Quantized4Asymm) &&
                        out_dtype_typecvt.enumv() == DTypeEnum::QuantizedS8;

        if (!(is_s82f32 || is_s82s4 || is_s42s8))
            return false;
        opr_set.insert(opr);

        // check conv bias
        auto conv_bias = try_cast_as_op<opr::ConvBias>(typecvt->input(0)->owner_opr());
        if (conv_bias == nullptr)
            return false;
        auto inp_dtype_conv = conv_bias->input(0)->dtype(),
             out_dtype_conv = conv_bias->input(0)->dtype();
        bool is_s8nhwc =
                inp_dtype_conv.enumv() == DTypeEnum::QuantizedS8 &&
                out_dtype_conv.enumv() == inp_dtype_conv.enumv() &&
                conv_bias->param().format == megdnn::param::ConvBias::Format::NHWC;
        bool is_s4nhwc =
                (inp_dtype_conv.enumv() == DTypeEnum::QuantizedS4 ||
                 inp_dtype_conv.enumv() == DTypeEnum::Quantized4Asymm) &&
                out_dtype_conv.enumv() == inp_dtype_conv.enumv() &&
                conv_bias->param().format == megdnn::param::ConvBias::Format::NHWC;
        if (!(is_s8nhwc || is_s4nhwc))
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
        auto new_bias = (out_dtype_typecvt.enumv() == DTypeEnum::Float32)
                              ? opr::TypeCvt::make(bias, dtype::Float32()).node()
                              : bias;
        auto new_param = conv_bias->param();
        new_param.format = megdnn::param::ConvBias::Format::NHWC;
        auto conv_bias_typecvt = opr::ConvBias::make(
                src, filter, new_bias, new_param, conv_bias->execution_policy(),
                OperatorNodeConfig{out_dtype_typecvt});
        rewriter.replace_var(
                opr->output(0), conv_bias_typecvt.node(),
                mgb_cstr_log("replace conv_bias(NHWC) + typecvt "
                             "to conv_bias(NHWC)"));
        return true;
    };

    auto on_opr = [&try_conv_typecvt, &rewriter](OperatorNodeBase* opr) {
        if (!try_conv_typecvt(opr)) {
            rewriter.auto_replace_outputs(opr);
        }
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();

    MIDOUT_E
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
