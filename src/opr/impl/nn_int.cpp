#include "megbrain/opr/nn_int.h"
#include "./internal/megdnn_opr_wrapper.inl"
#include "megbrain/opr/utility.h"
#include "megdnn/oprs/general.h"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(AffineInt);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ElemwiseMultiType);

ElemwiseMultiType::ElemwiseMultiType(
        const VarNodeArrayView& inputs, Param param, const OperatorNodeConfig& config)
        : Super{inputs.at(0)->owner_graph(), config,
                ModeTrait::from_mode(param.mode).name, inputs} {
    Super::init_megdnn_opr(*this, param);
    for (auto i : inputs) {
        add_input({i});
    }
    output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
}

SymbolVar ElemwiseMultiType::make(
        const VarNodeArrayView& inputs, Param param, const OperatorNodeConfig& config) {
    mgb_assert(!inputs.empty());
    return SymbolVar{inputs[0]}.insert_single_output_opr<ElemwiseMultiType>(
            inputs, param, config);
}

void ElemwiseMultiType::init_output_dtype() {
    auto trait = ModeTrait::from_mode(param().mode);
    mgb_throw_if(
            trait.arity != input().size(), MegBrainError,
            "%s requires %u inputs, but %zu are given", trait.name, trait.arity,
            input().size());
    for (size_t i = 0; i < trait.arity; ++i) {
        auto dtype = input()[i]->dtype();
        trait.check_inp[i](dtype);
    }
    if (trait.need_specify_out_dtype) {
        auto dtype = config().output_dtype();
        mgb_assert(dtype.valid());
        output(0)->dtype(dtype);
        trait.check_out(dtype, true);
    } else {
        DType dtype;
        trait.check_out(dtype, false);
        output(0)->dtype(dtype);
    }
}

void ElemwiseMultiType::scn_do_execute() {
    megdnn::TensorNDArray inp_arr(input().size());
    for (size_t i = 0; i < input().size(); ++i) {
        if (input()[i]->dev_tensor().empty()) {
            mgb_assert(output(0)->dev_tensor().empty());
            return;
        }
        inp_arr[i] = input()[i]->dev_tensor().as_megdnn();
    }
    mgb_assert(!output(0)->dev_tensor().empty());
    megdnn_opr()->exec(inp_arr, output(0)->dev_tensor().as_megdnn());
}

void ElemwiseMultiType::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    mgb_assert(out_shape.size() == 1);
    megdnn::Elemwise::deduce_shape(inp_shape, out_shape[0]);
}

void ElemwiseMultiType::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
}

void ElemwiseMultiType::add_input_layout_constraint() {
#if (MEGDNN_AARCH64 || MEGDNN_ARMV7) && !MGB_OPENCL && !MGB_CUDA
    for (auto i : input()) {
        i->add_layout_constraint_contiguous();
    }
#endif
}

ElemwiseMultiType::NodeProp* ElemwiseMultiType::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    for (auto& inp : input()) {
        ret->add_dep_type_existing_var(inp, NodeProp::DepType::VALUE_ALLOW_EMPTY);
    }
    return ret;
}

void ElemwiseMultiType::init_output_static_infer_desc() {
    Super::init_output_static_infer_desc();
    static StaticInferOpr<megdnn::ElemwiseMultiType> static_infer_opr;

    using namespace cg::static_infer;

    auto infer_value = [this](DeviceTensorND& dest, const InpVal& inp) {
        SmallVector<DeviceTensorND> inp_vals(inp.val.size());
        for (size_t i = 0; i < inp_vals.size(); ++i)
            inp_vals[i] = inp.val[i].value();

        DType out_dt;
        auto trait = ModeTrait::from_mode(param().mode);
        if (trait.need_specify_out_dtype) {
            auto dtype = config().output_dtype();
            mgb_assert(dtype.valid());
            out_dt = dtype;
        } else {
            DType dtype;
            trait.check_out(dtype, false);
            out_dt = dtype;
        }
        auto sopr = static_infer_opr.lock();
        perform(param().mode, out_dt, dest, inp_vals, sopr());
        return true;
    };
    DepVal deps(input().size());
    for (size_t i = 0; i < input().size(); ++i)
        deps[i] = {input(i), DepType::VALUE};
    owner_graph()->static_infer_manager().register_value_infer(
            output(0), {SourceType::DEP, deps, infer_value});
}

TensorShape ElemwiseMultiType::get_output_var_shape(
        Mode mode, const TensorShapeArray& input_shapes) {
    mgb_assert(input_shapes.size() == ModeTrait::from_mode(mode).arity);
    TensorShape ret;
    megdnn::Elemwise::deduce_shape(input_shapes, ret);
    return ret;
}

void ElemwiseMultiType::call_megdnn_opr_exec(
        CompNode comp_node, megdnn::TensorNDArray& inp, const megdnn::TensorND& out,
        megdnn::ElemwiseMultiType* opr, ElemwiseMultiType* caller) {
    // All Elemwise operations on QuantizedS32/QuantizedS8 are not related to
    // scale. MegDNN does not support computing Elemwise for
    // QuantizedS32/QuantizedS8, we translate the data type to Int32/Int8 before
    // passing to MegDNN.
    if (inp.size() && inp[0].layout.dtype.category() == DTypeCategory::QUANTIZED) {
        auto inp_dtype = inp[0].layout.dtype;
        DType compute_dtype;
        if (inp_dtype.enumv() == DTypeEnum::QuantizedS32) {
            compute_dtype = dtype::Int32();
        } else if (inp_dtype.enumv() == DTypeEnum::QuantizedS8) {
            compute_dtype = dtype::Int8();
        } else {
            mgb_throw(
                    MegBrainError, "Unsupported Quantized Elemwise Mode %s: %d on %s",
                    inp[0].layout.dtype.name(), int(opr->param().mode),
                    comp_node.to_string().c_str());
        }

        megdnn::TensorNDArray run_inp(inp);
        for (size_t i = 0; i < inp.size(); i++) {
            run_inp[i].layout.dtype = compute_dtype;
        }
        megdnn::TensorND run_out = out;
        run_out.layout.dtype = compute_dtype;
        opr->exec(run_inp, run_out);
        return;
    }

    opr->exec(inp, out);
}

void ElemwiseMultiType::perform(
        Mode mode, DType out_dt, DeviceTensorND& dest,
        const SmallVector<DeviceTensorND>& inputs,
        intl::UniqPtrWithCN<megdnn::ElemwiseMultiType>& opr) {
    megdnn::TensorNDArray dnn_inputs(inputs.size());
    TensorShapeArray inp_shapes(inputs.size());
    CompNode out_cn;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto&& t = inputs[i];
        if (!i) {
            out_cn = t.comp_node();
        } else {
            mgb_assert(t.comp_node() == out_cn);
        }
        if (t.shape().is_empty()) {
            mgb_assert(dest.empty());
            return;
        }
        inp_shapes[i] = t.shape();
    }
    if (!opr) {
        opr = intl::create_megdnn_opr<megdnn::ElemwiseMultiType>(out_cn);
    } else {
        mgb_assert(out_cn == opr.comp_node());
    }
    out_cn.activate();
    for (size_t i = 0; i < inputs.size(); ++i)
        dnn_inputs[i] = inputs[i].as_megdnn();
    dest.comp_node(out_cn).dtype(out_dt).resize(get_output_var_shape(mode, inp_shapes));
    opr->param() = {mode};
    call_megdnn_opr_exec(out_cn, dnn_inputs, dest.as_megdnn(), opr.get(), nullptr);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
