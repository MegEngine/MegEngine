#include "megbrain/imperative/transformations/dtype_promote.h"
#include "megbrain/imperative/ops/autogen.h"

namespace mgb::imperative {

bool DTypePromoteCfg::convert_input_enabled = true;
bool DTypePromoteCfg::amp_dtype_autocast_enabled = false;
DType DTypePromoteCfg::amp_high_prec_dtype = dtype::Float32();
DType DTypePromoteCfg::amp_low_prec_dtype = dtype::Float16();

namespace {
// TODO: ScalarRule and DTypePromoteRule should be unified
using DTypePromoteRule = std::function<ValueRefList(const OpDef&, Span<ValueRef>)>;
static std::unordered_map<Typeinfo*, DTypePromoteRule> dtype_promotion_rules;

template <typename T>
void register_dtype_promote_rule(const DTypePromoteRule& rule) {
    dtype_promotion_rules[T::typeinfo()] = [rule](const OpDef& def,
                                                  Span<ValueRef> inputs) {
        return rule(def.cast_final_safe<T>(), inputs);
    };
}

bool is_quantized_dtype(const DType& dtype) {
    return dtype.category() == DTypeCategory::QUANTIZED;
}

bool is_all_integer(const SmallVector<DType>& dtypes) {
    for (size_t i = 0; i < dtypes.size(); ++i) {
        if (dtypes[i].category() != DTypeCategory::INT) {
            return false;
        }
    }
    return true;
}

SmallVector<DType> get_value_dtypes(const Span<ValueRef> inputs) {
    SmallVector<DType> dtypes(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        dtypes[i] = *(inputs[i].dtype());
    }
    return dtypes;
}

mgb::DType get_promoted_dtype(const SmallVector<DType>& dtypes) {
    if (dtypes.size() == 0) {
        mgb_assert(false, "there is no input for operator, dtype promote failed");
    }
    mgb::DType ret = dtypes[0];
    for (size_t i = 1; i < dtypes.size(); ++i) {
        ret = mgb::dtype_promotion(ret, dtypes[i]);
    }
    return ret;
}

ValueRefList elemwise_rule(const OpDef& op, Span<ValueRef> inputs) {
    auto&& elem_op = op.cast_final_safe<Elemwise>();

    SmallVector<DType> dtypes(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        dtypes[i] = *(inputs[i].dtype());
    }

    ValueRefList converted(inputs.size());
    mgb::DType target_dtype = get_promoted_dtype(dtypes);

    // TODO: we can save the dtypes of inputs here and perform TypeCvt at the end of
    // this function, rather than perform TypeCvt eagerly. But for the compatibility, we
    // implement this function with the similar process as the python version and
    // perform TypeCvt here, so we maybe do TypeCvt several times in these function

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!is_quantized_dtype(dtypes[i]) && dtypes[i] != target_dtype &&
            DTypePromoteCfg::convert_input_enabled) {
            converted[i] = imperative::apply(
                    ApplyOp(*TypeCvt::make(target_dtype)), inputs[i])[0];
            dtypes[i] = target_dtype;
        } else {
            converted[i] = inputs[i];
        }
    }

    static std::unordered_set<Elemwise::Mode> cast_case1 = {
            Elemwise::Mode::TRUE_DIV, Elemwise::Mode::EXP,
            Elemwise::Mode::POW,      Elemwise::Mode::LOG,
            Elemwise::Mode::EXPM1,    Elemwise::Mode::LOG1P,
            Elemwise::Mode::ACOS,     Elemwise::Mode::ASIN,
            Elemwise::Mode::ATAN2,    Elemwise::Mode::COS,
            Elemwise::Mode::SIN,      Elemwise::Mode::LOG_SUM_EXP,
    };

    static std::unordered_set<Elemwise::Mode> cast_case2 = {
            Elemwise::Mode::TANH,
    };

    auto cast_to_high_prec = [&]() {
        for (size_t i = 0; i < dtypes.size(); ++i) {
            if (dtypes[i] != DTypePromoteCfg::amp_high_prec_dtype) {
                converted[i] = imperative::apply(
                        ApplyOp(*TypeCvt::make(DTypePromoteCfg::amp_high_prec_dtype)),
                        converted[i])[0];
                dtypes[i] = DTypePromoteCfg::amp_high_prec_dtype;
            }
        }
    };

    if (cast_case1.find(elem_op.mode) != cast_case1.end()) {
        if (DTypePromoteCfg::amp_dtype_autocast_enabled || is_all_integer(dtypes)) {
            cast_to_high_prec();
        }
    }

    if (cast_case2.find(elem_op.mode) != cast_case2.end()) {
        if (is_all_integer(dtypes)) {
            cast_to_high_prec();
        }
    }

    static std::unordered_set<Elemwise::Mode> cast_case3 = {
            Elemwise::Mode::CEIL, Elemwise::Mode::FLOOR, Elemwise::Mode::ROUND};

    if (cast_case3.find(elem_op.mode) != cast_case3.end()) {
        if (dtypes[0].category() == DTypeCategory::INT) {
            return converted;
        }
    }

    return imperative::apply(op, converted);
}

ValueRefList reduce_rule(const OpDef& op, Span<ValueRef> inputs) {
    auto&& reduce_op = op.cast_final_safe<Reduce>();
    DType org_dtype = *(inputs[0].dtype());
    DType target_dtype = org_dtype;

    ValueRefList converted(inputs.begin(), inputs.end());

    if (reduce_op.mode == Reduce::Mode::MEAN) {
        target_dtype = dtype::Float32();
    } else if (org_dtype.category() == DTypeCategory::BOOL) {
        target_dtype = dtype::Int32();
    }

    if (target_dtype != org_dtype) {
        converted[0] =
                imperative::apply(ApplyOp(*TypeCvt::make(target_dtype)), inputs[0])[0];
    }

    ValueRefList ret = imperative::apply(op, converted);

    if (org_dtype.category() == DTypeCategory::BOOL) {
        if (reduce_op.mode == Reduce::Mode::MIN ||
            reduce_op.mode == Reduce::Mode::MAX) {
            ret[0] = imperative::apply(
                    ApplyOp(*TypeCvt::make(dtype::Bool())), ret[0])[0];
        }
    }
    return ret;
}

ValueRefList convolution_rule(const OpDef& op, Span<ValueRef> inputs) {
    auto&& conv_op = const_cast<Convolution&>(op.cast_final_safe<Convolution>());
    SmallVector<DType> dtypes = get_value_dtypes(inputs);
    mgb::DType target_dtype;

    if (DTypePromoteCfg::amp_dtype_autocast_enabled) {
        conv_op.compute_mode = Convolution::ComputeMode::FLOAT32;
        target_dtype = DTypePromoteCfg::amp_low_prec_dtype;
    } else {
        target_dtype = get_promoted_dtype(dtypes);
    }

    ValueRefList converted(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (dtypes[i] != target_dtype) {
            converted[i] = imperative::apply(
                    ApplyOp(*TypeCvt::make(target_dtype)), inputs[i])[0];
        } else {
            converted[i] = inputs[i];
        }
    }

    return imperative::apply(op, converted);
}

ValueRefList matmul_rule(const OpDef& op, Span<ValueRef> inputs) {
    auto&& conv_op = const_cast<MatrixMul&>(op.cast_final_safe<MatrixMul>());
    SmallVector<DType> dtypes = get_value_dtypes(inputs);
    mgb::DType target_dtype;

    if (DTypePromoteCfg::amp_dtype_autocast_enabled) {
        conv_op.compute_mode = MatrixMul::ComputeMode::FLOAT32;
        target_dtype = DTypePromoteCfg::amp_low_prec_dtype;
    } else {
        target_dtype = get_promoted_dtype(dtypes);
    }

    ValueRefList converted(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (dtypes[i] != target_dtype) {
            converted[i] = imperative::apply(
                    ApplyOp(*TypeCvt::make(target_dtype)), inputs[i])[0];
        } else {
            converted[i] = inputs[i];
        }
    }

    return imperative::apply(op, converted);
}

ValueRefList batch_matmul_rule(const OpDef& op, Span<ValueRef> inputs) {
    auto&& conv_op =
            const_cast<BatchedMatrixMul&>(op.cast_final_safe<BatchedMatrixMul>());
    SmallVector<DType> dtypes = get_value_dtypes(inputs);
    mgb::DType target_dtype;

    if (DTypePromoteCfg::amp_dtype_autocast_enabled) {
        conv_op.compute_mode = BatchedMatrixMul::ComputeMode::FLOAT32;
        target_dtype = DTypePromoteCfg::amp_low_prec_dtype;
    } else {
        target_dtype = get_promoted_dtype(dtypes);
    }

    ValueRefList converted(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (dtypes[i] != target_dtype) {
            converted[i] = imperative::apply(
                    ApplyOp(*TypeCvt::make(target_dtype)), inputs[i])[0];
        } else {
            converted[i] = inputs[i];
        }
    }

    return imperative::apply(op, converted);
}

// differ from Convolution, ConvolutionBackwardData is used in both
// functional.conv_transpose2d and quantize.conv_transpose2d
ValueRefList convolution_backward_rule(const OpDef& op, Span<ValueRef> inputs) {
    auto&& conv_op = const_cast<ConvolutionBackwardData&>(
            op.cast_final_safe<ConvolutionBackwardData>());
    SmallVector<DType> dtypes = get_value_dtypes(inputs);

    if (is_quantized_dtype(dtypes[0]) && is_quantized_dtype(dtypes[1])) {
        return imperative::apply(op, inputs);
    }

    mgb::DType target_dtype;
    if (DTypePromoteCfg::amp_dtype_autocast_enabled) {
        conv_op.compute_mode = ConvolutionBackwardData::ComputeMode::FLOAT32;
        target_dtype = DTypePromoteCfg::amp_low_prec_dtype;
    } else {
        target_dtype = get_promoted_dtype(dtypes);
    }

    ValueRefList converted(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (dtypes[i] != target_dtype) {
            converted[i] = imperative::apply(
                    ApplyOp(*TypeCvt::make(target_dtype)), inputs[i])[0];
        } else {
            converted[i] = inputs[i];
        }
    }

    return imperative::apply(op, converted);
}

ValueRefList batch_norm_rule(const OpDef& op, Span<ValueRef> inputs) {
    if (DTypePromoteCfg::amp_dtype_autocast_enabled) {
        mgb_assert(inputs.size() > 0);
        SmallVector<DType> dtypes = get_value_dtypes(inputs);
        ValueRefList converted(inputs.size());

        for (size_t i = 0; i < inputs.size(); ++i) {
            mgb::DType target_dtype = i == 0 ? DTypePromoteCfg::amp_low_prec_dtype
                                             : DTypePromoteCfg::amp_high_prec_dtype;
            if (dtypes[i] != target_dtype) {
                converted[i] = imperative::apply(
                        ApplyOp(*TypeCvt::make(target_dtype)), inputs[i])[0];
            } else {
                converted[i] = inputs[i];
            }
        }

        return imperative::apply(op, converted);
    }

    return imperative::apply(op, inputs);
}

ValueRefList naive_promote_rule(const OpDef& op, Span<ValueRef> inputs) {
    SmallVector<DType> dtypes = get_value_dtypes(inputs);
    mgb::DType target_dtype = get_promoted_dtype(dtypes);

    ValueRefList converted(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (dtypes[i] != target_dtype) {
            converted[i] = imperative::apply(
                    ApplyOp(*TypeCvt::make(target_dtype)), inputs[i])[0];
        } else {
            converted[i] = inputs[i];
        }
    }

    return imperative::apply(op, converted);
}

struct DTypePromoteRuleRegistry {
    DTypePromoteRuleRegistry() {
        register_dtype_promote_rule<Elemwise>(elemwise_rule);
        register_dtype_promote_rule<Concat>(naive_promote_rule);
        register_dtype_promote_rule<GroupLocal>(naive_promote_rule);
        register_dtype_promote_rule<Reduce>(reduce_rule);
        register_dtype_promote_rule<Convolution>(convolution_rule);
        register_dtype_promote_rule<MatrixMul>(matmul_rule);
        register_dtype_promote_rule<BatchedMatrixMul>(batch_matmul_rule);
        register_dtype_promote_rule<ConvolutionBackwardData>(convolution_backward_rule);
        register_dtype_promote_rule<BatchNorm>(batch_norm_rule);
        register_dtype_promote_rule<Convolution3D>(naive_promote_rule);
        register_dtype_promote_rule<Convolution3DBackwardData>(naive_promote_rule);
    }
} register_helper;

}  // namespace

ValueRefList DTypePromoteTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    if (auto apply_op = op.as<ApplyOp>()) {
        auto iter = dtype_promotion_rules.find(apply_op->op().dyn_typeinfo());
        if (iter != dtype_promotion_rules.end()) {
            return iter->second(apply_op->op(), inputs);
        } else {
            return imperative::apply(op, inputs);
        }
    }
    return imperative::apply(op, inputs);
}

ValueRef DTypePromoteTransformation::unwrap(ValueRef value) {
    return value;
}

std::string DTypePromoteTransformation::name() const {
    return "DTypePromoteTransformation";
}

void DTypePromoteTransformation::on_register() {
    // printf("DTypePromoteTransformation has been registered\n");
}

void DTypePromoteTransformation::on_unregister() noexcept {
    // printf("DTypePromoteTransformation has been unregistered\n");
}

}  // namespace mgb::imperative