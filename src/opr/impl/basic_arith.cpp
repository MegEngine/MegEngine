/**
 * \file src/opr/impl/basic_arith.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/basic_arith.h"
#include "megbrain/gopt/basic_arith.h"
#include "megbrain/gopt/gtrans.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/cond.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/utils/arith_helper.h"

#include "./internal/megdnn_opr_wrapper.inl"

#include <cmath>

using namespace mgb;
using namespace opr;

namespace {

//! global operator instance for static inference
template <class Opr>
class StaticInferOpr {
    intl::UniqPtrWithCN<Opr> m_opr;
    MGB_MUTEX m_mtx;

public:
    class Lock {
        friend class StaticInferOpr;
        StaticInferOpr* m_owner;

        explicit Lock(StaticInferOpr* owner) : m_owner{owner} {
#if !__DEPLOY_ON_XP_SP2__
            m_owner->m_mtx.lock();
#endif
        }

    public:
        Lock(Lock&& rhs) : m_owner{rhs.m_owner} { rhs.m_owner = nullptr; }

        ~Lock() {
#if !__DEPLOY_ON_XP_SP2__
            if (m_owner)
                m_owner->m_mtx.unlock();
#endif
        }

        Lock& operator=(const Lock&) = delete;
        Lock& operator=(Lock&&) = delete;

        intl::UniqPtrWithCN<Opr>& operator()() { return m_owner->m_opr; }
    };

    //! lock and acquire the operator
    Lock lock() {
        Lock ret{this};
        if (!m_opr) {
            m_opr = intl::create_megdnn_opr<Opr>(CompNode::default_cpu());
        }
        return ret;
    }
};
}  // anonymous namespace

/* ========================= BatchedDTypePromotion ========================= */
intl::BatchedDTypePromotion::BatchedDTypePromotion(const VarNodeArrayView& vars)
        : m_orig_vars{vars} {
    mgb_assert(!vars.empty());
    DType final_dtype;
    bool changed = false;
    for (size_t i = 0; i < vars.size(); ++i) {
        auto cur = vars[i]->dtype();
        if (!i) {
            final_dtype = cur;
        } else {
            auto promoted = dtype_promotion(final_dtype, cur);
            changed |= promoted != final_dtype || promoted != cur;
            final_dtype = promoted;
        }
    }
    m_changed = changed;
    m_final_dtype = final_dtype;
}

void intl::BatchedDTypePromotion::set_dtype(DType dtype) {
    mgb_assert(!m_finalized);
    if (m_final_dtype != dtype) {
        m_final_dtype = dtype;
        m_changed = true;
    }
}

const VarNodeArrayView& intl::BatchedDTypePromotion::get_vars() {
    m_finalized = true;
    if (!m_changed) {
        return m_orig_vars;
    }
    if (!m_cvt_vars_view.valid()) {
        m_cvt_vars.resize(m_orig_vars.size());
        auto dtype = m_final_dtype;
        for (size_t i = 0; i < m_cvt_vars.size(); ++i) {
            m_cvt_vars[i] = TypeCvt::make(m_orig_vars[i], dtype).node();
        }
        m_cvt_vars_view.emplace(m_cvt_vars);
    }
    return m_cvt_vars_view.val();
}

/* =========================== Elemwise =========================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Elemwise);
Elemwise::Elemwise(
        const ModeTrait& mode_trait, const VarNodeArrayView& inputs, Param param,
        const OperatorNodeConfig& config)
        : Super{inputs.at(0)->owner_graph(), config, mode_trait.name, inputs} {
    init_megdnn_opr(*this, param);
    output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    if (mode_trait.commutable) {
        mgb_assert(inputs.size() == 2);
        add_input({inputs[0], inputs[1]}, AddInputSortType::CUR_ADDED);
    } else {
        if (param.mode == Mode::FUSE_MUL_ADD3) {
            add_input({inputs[0], inputs[1]}, AddInputSortType::CUR_ADDED);
            add_input({inputs[2]});
        } else if (param.mode == Mode::FUSE_MUL_ADD4) {
            auto i0 = inputs[0], i1 = inputs[1], i2 = inputs[2], i3 = inputs[3];
            if (i0->id() > i1->id())
                std::swap(i0, i1);
            if (i2->id() > i3->id())
                std::swap(i2, i3);
            if (i0->id() > i2->id()) {
                std::swap(i0, i2);
                std::swap(i1, i3);
            }
            add_input({i0, i1, i2, i3});
        } else {
            for (auto i : inputs)
                add_input({i});
        }
    }

    mgb_assert(m_input_broadcastable.size() >= inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (input()[i]->owner_opr()->same_type<opr::MarkNoBroadcastElemwise>()) {
            m_input_broadcastable[i] = false;
        } else {
            m_input_broadcastable[i] = true;
        }
    }
    if (inputs.size() == 1) {
        m_input_broadcastable[0] = false;
    } else {
        Maybe<size_t> non_scalar;
        using namespace cg::static_infer;
        auto&& mgr = owner_graph()->static_infer_manager();
        for (size_t i = 0; i < input().size(); ++i) {
            auto it = mgr.get_infer_type(input(i));
            if (!((it.shape & InferType::CONST) &&
                  mgr.infer_shape(input(i)).is_scalar())) {
                if (non_scalar.valid()) {
                    non_scalar.invalidate();
                    break;
                }
                non_scalar = i;
            }
        }
        if (non_scalar.valid()) {
            // exactly one input is non-scalar
            m_input_broadcastable[non_scalar.val()] = false;
        }
    }

    if (inputs.size() && inputs[0]->dtype().category() == DTypeCategory::QUANTIZED) {
        mgb_assert(
                param.mode == Param::Mode::ADD || param.mode == Param::Mode::SUB ||
                        param.mode == Param::Mode::NEGATE ||
                        param.mode == Param::Mode::RELU ||
                        param.mode == Param::Mode::MAX ||
                        param.mode == Param::Mode::MIN,
                "Only ADD, SUB, NEGATE, RELU, MAX and MIN is guaranteed "
                "to be supported on Elemwise for quantized DType, no support %d",
                (int)param.mode);
    }
}

SymbolVar Elemwise::make(
        const VarNodeArrayView& inputs, Param param, const OperatorNodeConfig& config) {
    auto trait = ModeTrait::from_mode(param.mode);
    mgb_assert(
            inputs.size() == trait.arity, "%s expects %u inputs; got %zu actually",
            trait.name, trait.arity, inputs.size());
    intl::BatchedDTypePromotion dtp{inputs};
    if (dtp.get_dtype().category() == DTypeCategory::INT && !trait.allow_int) {
        dtp.set_dtype(dtype::Float32());
    }

    mgb_throw_if(
            dtp.get_dtype().category() == DTypeCategory::FLOAT && !trait.allow_float,
            ConversionError,
            "elemwise mode %s does not allow float input; "
            "got inputs: %s",
            trait.name, cg::dump_var_info(inputs).c_str());

#if !MGB_BUILD_SLIM_SERVING
    auto&& options = inputs[0]->owner_graph()->options();
    if (options.graph_opt_level && !(options.disable_inplace_arith_opt)) {
        auto repl = gopt::optimize_elemwise_expr_inplace(dtp.get_vars(), param, config);
        if (repl)
            return repl;
    }
#endif

    return SymbolVar{inputs[0]}.insert_single_output_opr<Elemwise>(
            trait, dtp.get_vars(), param, config);
}

TensorShape Elemwise::get_output_var_shape(
        Mode mode, const TensorShapeArray& input_shapes) {
    mgb_assert(input_shapes.size() == ModeTrait::from_mode(mode).arity);
    TensorShape ret;
    megdnn::Elemwise::deduce_shape(input_shapes, ret);
    return ret;
}

void Elemwise::perform(
        Mode mode, DeviceTensorND& dest, const SmallVector<DeviceTensorND>& inputs,
        intl::UniqPtrWithCN<megdnn::Elemwise>& opr) {
    megdnn::TensorNDArray dnn_inputs(inputs.size());
    TensorShapeArray inp_shapes(inputs.size());
    DType out_dt;
    CompNode out_cn;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto&& t = inputs[i];
        if (!i) {
            out_cn = t.comp_node();
            out_dt = t.dtype();
        } else {
            mgb_assert(t.comp_node() == out_cn);
            mgb_assert(t.dtype() == out_dt);
        }
        if (t.shape().is_empty()) {
            mgb_assert(dest.empty());
            return;
        }
        inp_shapes[i] = t.shape();
    }
    if (!opr) {
        opr = intl::create_megdnn_opr<megdnn::Elemwise>(out_cn);
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

TensorLayoutArray Elemwise::collective_collapse(const TensorLayoutArray& layouts) {
    TensorLayoutPtrArray inp(layouts.size());
    TensorLayoutArray result(inp.size());
    for (size_t i = 0; i < layouts.size(); ++i) {
        result[i] = layouts[i];
        inp[i] = &result[i];
    }
    collective_collapse_inplace(inp);
    return result;
}

void Elemwise::collective_collapse_inplace(const TensorLayoutPtrArray& layouts) {
    mgb_assert(layouts.size());
    size_t ndim = layouts[0]->ndim;
    for (auto i : layouts) {
        if (i->ndim != ndim)
            mgb_throw(MegBrainError, "ndims must be same");
    }

    auto update_all = [&layouts](size_t axis) {
        for (auto i : layouts) {
            i->shape[axis] *= i->shape[axis + 1];
            i->stride[axis] = i->stride[axis + 1];
            i->remove_axis_inplace(axis + 1);
        }
    };

    auto check = [&layouts](size_t axis) -> bool {
        auto std_p =
                std::make_pair(layouts[0]->shape[axis], layouts[0]->shape[axis + 1]);
        for (auto i : layouts) {
            auto cur_p = std::make_pair(i->shape[axis], i->shape[axis + 1]);
            if (std_p != cur_p)
                return false;
            if (i->stride[axis] !=
                i->stride[axis + 1] * static_cast<ptrdiff_t>(i->shape[axis + 1]))
                return false;
        }
        return true;
    };

    for (int i = static_cast<int>(ndim) - 2; i >= 0; i--) {
        if (check(i)) {
            update_all(i);
        }
    }
}

void Elemwise::broadcast_collective_collapse(
        const TensorLayoutPtrArray& inp_layouts, TensorLayout* target_layout) {
    for (auto&& p : inp_layouts) {
        *p = p->broadcast(*target_layout);
    }
    TensorLayoutPtrArray buf(inp_layouts.size() + 1);
    buf[0] = target_layout;
    for (size_t i = 0; i < inp_layouts.size(); i++) {
        buf[i + 1] = inp_layouts[i];
    }
    collective_collapse_inplace(buf);
}

void Elemwise::mem_plan_fwd_in2out_writable() {
    mixin_mem_plan_fwd_in2out_writable(*this);
}

void Elemwise::scn_do_execute() {
    auto&& inp = input();
    megdnn::TensorNDArray dnn_inp;
    mgb_assert(dnn_inp.capacity() >= inp.size(), "heap allocation in elemwise exec");
    dnn_inp.resize(inp.size());
    for (size_t i = 0; i < inp.size(); ++i) {
        if (inp[i]->dev_tensor().empty()) {
            mgb_assert(output(0)->dev_tensor().empty());
            return;
        }
        dnn_inp[i] = (inp[i]->dev_tensor().as_megdnn());
    }
    mgb_assert(!output(0)->dev_tensor().empty());

    megdnn_opr()->param() = param();
    call_megdnn_opr_exec(
            comp_node(), dnn_inp, output(0)->dev_tensor().as_megdnn(), megdnn_opr(),
            this);
}

void Elemwise::init_output_static_infer_desc() {
    Super::init_output_static_infer_desc();
    static StaticInferOpr<megdnn::Elemwise> static_infer_opr;

    using namespace cg::static_infer;

    auto infer_value = [this](DeviceTensorND& dest, const InpVal& inp) {
        SmallVector<DeviceTensorND> inp_vals(inp.val.size());
        for (size_t i = 0; i < inp_vals.size(); ++i)
            inp_vals[i] = inp.val[i].value();
        auto sopr = static_infer_opr.lock();
        perform(param().mode, dest, inp_vals, sopr());
        return true;
    };

    DepVal deps(input().size());
    for (size_t i = 0; i < input().size(); ++i)
        deps[i] = {input(i), DepType::VALUE};
    owner_graph()->static_infer_manager().register_value_infer(
            output(0), {SourceType::DEP, deps, infer_value});
}

void Elemwise::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    out_shape.at(0) = get_output_var_shape(param().mode, inp_shape);
    for (size_t i = 0; i < input().size(); ++i) {
        mgb_throw_if(
                !m_input_broadcastable[i] && !out_shape[0].eq_shape(inp_shape[i]),
                GraphError,
                "input %zu declared to be non-broadcastable but broacast "
                "actually happened",
                i);
    }
}

void Elemwise::add_input_layout_constraint() {
    for (auto i : input()) {
        i->add_layout_constraint_monotone();
    }
}

void Elemwise::call_megdnn_opr_exec(
        CompNode comp_node, megdnn::TensorNDArray& inp, const megdnn::TensorND& out,
        megdnn::Elemwise* opr, Elemwise* caller) {
    if (opr->param().mode == Mode::FUSE_MUL_ADD3 &&
        !(inp[2].layout.eq_layout(inp[0].layout) ||
          inp[2].layout.eq_layout(inp[1].layout) || inp[2].layout.is_scalar())) {
        if (caller && !caller->fuse_badlayout_warn_printed()) {
            mgb_log_debug(
                    "%s: FUSE_MUL_ADD3 input layouts mismatch: %s %s %s; "
                    "fallback to normal computing",
                    caller->cname(), inp[0].layout.to_string().c_str(),
                    inp[1].layout.to_string().c_str(),
                    inp[2].layout.to_string().c_str());
            caller->m_fuse_badlayout_warn_printed = true;
        }

        for (auto&& i : inp) {
            i.layout = i.layout.broadcast(out.layout);
        }

        megdnn::TensorNDArray run_inp(2);
        auto run = [&](Mode mode, const megdnn::TensorND& i0,
                       const megdnn::TensorND& i1, const megdnn::TensorND& out) {
            run_inp[0] = i0;
            run_inp[1] = i1;
            opr->param() = {mode};
            opr->exec(run_inp, out);
        };

        auto tmp = intl::get_temp_tensor(
                caller ? caller->owner_graph() : nullptr, comp_node, out.layout);
        auto tmpv = tmp.as_megdnn();

        MGB_TRY {
            run(Mode::MUL, inp[0], inp[1], tmpv);
            run(Mode::ADD, inp[2], tmpv, out);
        }
        MGB_FINALLY(opr->param() = {Mode::FUSE_MUL_ADD3});
        return;
    }

    if (opr->param().mode == Mode::FUSE_MUL_ADD4 &&
        !(inp[0].layout.eq_layout(inp[2].layout) &&
          inp[1].layout.eq_layout(inp[3].layout)) &&
        !(inp[0].layout.eq_layout(inp[3].layout) &&
          inp[1].layout.eq_layout(inp[2].layout))) {
        if (caller && !caller->fuse_badlayout_warn_printed()) {
            mgb_log_debug(
                    "%s: FUSE_MUL_ADD4 input layouts mismatch: %s %s %s %s; "
                    "fallback to normal computing",
                    caller->cname(), inp[0].layout.to_string().c_str(),
                    inp[1].layout.to_string().c_str(),
                    inp[2].layout.to_string().c_str(),
                    inp[3].layout.to_string().c_str());
            caller->m_fuse_badlayout_warn_printed = true;
        }

        for (auto&& i : inp) {
            i.layout = i.layout.broadcast(out.layout);
        }

        megdnn::TensorNDArray run_inp(2);
        auto run = [&](Mode mode, const megdnn::TensorND& i0,
                       const megdnn::TensorND& i1, const megdnn::TensorND& out) {
            run_inp[0] = i0;
            run_inp[1] = i1;
            opr->param() = {mode};
            opr->exec(run_inp, out);
        };

        auto tmp = intl::get_temp_tensor(
                caller ? caller->owner_graph() : nullptr, comp_node, out.layout);
        auto tmpv = tmp.as_megdnn();

        MGB_TRY {
            run(Mode::MUL, inp[0], inp[1], tmpv);
            run(Mode::MUL, inp[2], inp[3], out);
            run(Mode::ADD, out, tmpv, out);
        }
        MGB_FINALLY(opr->param() = {Mode::FUSE_MUL_ADD4});
        return;
    }

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

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Elemwise) {
    SymbolVar i[5];
    SymbolVar i0(opr.input(0)), i1, i2, out(opr.output(0)), og{out_grad.at(0)}, result;
    for (size_t t = 0; t < opr.input().size(); ++t)
        i[t] = opr.input()[t];
    if (opr.input().size() >= 2)
        i1 = opr.input(1);
    if (opr.input().size() >= 3)
        i2 = opr.input(2);

    // negate after reduce, for better performance
    bool negate_result = false;
#define RET(_v)    \
    result = (_v); \
    break
#define EL1(_mode, _a)         Elemwise::make({_a}, Mode::_mode)
#define EL2(_mode, _a, _b)     Elemwise::make({_a, _b}, Mode::_mode)
#define EL3(_mode, _a, _b, _c) Elemwise::make({_a, _b, _c}, Mode::_mode)
#define RET_INVALID()          return InvalidGrad::make(opr, wrt_idx)

    using Mode = Elemwise::Mode;

    switch (opr.param().mode) {
        // unary
        case Mode::RELU:
        case Mode::FUSE_ADD_RELU:
            RET(EL2(SWITCH_GT0, out, og));
        case Mode::ABS:
            RET(EL2(ABS_GRAD, i0, og));
        case Mode::ACOS:
            negate_result = true;
            RET(og / EL1(SIN, out));
        case Mode::ASIN:
            RET(og / EL1(COS, out));
        case Mode::ATAN2:
            if (wrt_idx) {
                negate_result = true;
            }
            RET(og * i[!wrt_idx] / (i0 * i0 + i1 * i1));
        case Mode::CEIL:
            return nullptr;
        case Mode::COS:
            negate_result = true;
            RET(EL1(SIN, i0) * og);
        case Mode::EXP:
            RET(og * out);
        case Mode::EXPM1:
            RET(og * EL1(EXP, i0));
        case Mode::FLOOR:
            return nullptr;
        case Mode::LOG:
            RET(og / i0);
        case Mode::LOG1P:
            RET(og / (i0 + 1));
        case Mode::NEGATE:
            negate_result = true;
            RET(og);
        case Mode::SIGMOID:
        case Mode::FUSE_ADD_SIGMOID:
            RET(EL2(SIGMOID_GRAD, out, og));
        case Mode::SIN:
            RET(EL1(COS, i0) * og);
        case Mode::TANH:
        case Mode::FUSE_ADD_TANH:
            RET(EL2(TANH_GRAD, out, og));
        case Mode::FAST_TANH:
            RET(EL2(FAST_TANH_GRAD, i0, og));
        case Mode::ROUND:
            return nullptr;
        case Mode::ERF:
            RET(EL1(EXP, -i0 * i0) * 2 / static_cast<float>(sqrt(M_PI)) * og);
        case Mode::ERFINV:
            RET(EL1(EXP, out * out) * static_cast<float>(sqrt(M_PI)) / 2 * og);
        case Mode::ERFC:
            RET(-EL1(EXP, -i0 * i0) * 2 / static_cast<float>(sqrt(M_PI)) * og);
        case Mode::H_SWISH:
            RET(EL2(H_SWISH_GRAD, i0, og));
        case Mode::FUSE_ADD_H_SWISH:
            RET(EL2(H_SWISH_GRAD, (i0 + i1), og));
        case Mode::NOT:
            return nullptr;
        case Mode::SILU:
            RET(EL2(SILU_GRAD, i0, og));
        case Mode::GELU:
            RET(EL2(GELU_GRAD, i0, og));

        // binary
        case Mode::ABS_GRAD:
            if (wrt_idx == 0) {
                return nullptr;
            }
            RET(EL2(ABS_GRAD, i0, og));
        case Mode::ADD:
            RET(og);
        case Mode::FLOOR_DIV:
            return nullptr;
        case Mode::MAX:
            RET(EL3(COND_LEQ_MOV, i[!wrt_idx], i[wrt_idx], og));
        case Mode::MIN:
            RET(EL3(COND_LEQ_MOV, i[wrt_idx], i[!wrt_idx], og));
        case Mode::MOD:
            if (wrt_idx == 0) {
                RET(og);
            }
            RET_INVALID();
        case Mode::MUL:
            RET(og * i[!wrt_idx]);
        case Mode::POW:
            if (wrt_idx) {
                RET(out * EL1(LOG, i0) * og);
            }
            RET(og * i1 * EL2(POW, i0, i1 - 1));
        case Mode::SIGMOID_GRAD:
            if (wrt_idx == 0) {
                auto one = i0.make_scalar_dt(1), two = i0.make_scalar_dt(2);
                RET((one - i0 * two) * i1 * og);
            }
            RET(EL2(SIGMOID_GRAD, i0, og));
        case Mode::SUB:
            negate_result = wrt_idx;
            RET(og);
        case Mode::SWITCH_GT0:
            if (!wrt_idx)
                return nullptr;
            RET(EL2(SWITCH_GT0, i0, og));
        case Mode::TANH_GRAD:
            if (wrt_idx == 0) {
                auto mtwo = i0.make_scalar_dt(-2);
                RET(mtwo * i0 * i1 * og);
            }
            RET(EL2(TANH_GRAD, i0, og));
        case Mode::TRUE_DIV:
            if (wrt_idx == 0) {
                RET(og / i1);
            }
            negate_result = true;
            RET((og * i0) * EL2(POW, i1, i1.make_scalar(-2)));
        case Mode::LOG_SUM_EXP:
            if (wrt_idx == 0) {
                RET(og * EL1(SIGMOID, i0 - i1));
            }
            RET(og * EL1(SIGMOID, i1 - i0));
        case Mode::LT:
        case Mode::LEQ:
            return nullptr;
        case Mode::EQ:
            RET_INVALID();
        case Mode::OR:
        case Mode::XOR:
        case Mode::AND:
            return nullptr;

        // ternary
        case Mode::COND_LEQ_MOV:
            if (wrt_idx <= 1)
                return nullptr;
            RET(EL3(COND_LEQ_MOV, i0, i1, og));

        // fuse oprs
        case Mode::FUSE_MUL_ADD3:
            if (wrt_idx < 2) {
                RET(og * i[wrt_idx ^ 1]);
            } else {
                RET(og);
            }
        case Mode::FUSE_MUL_ADD4:
            RET(og * i[wrt_idx ^ 1]);
        default:
            mgb_throw(
                    GraphError, "grad for elemwise mode %s unimplemented",
                    megdnn::Elemwise::ModeTrait::from_mode(opr.param().mode).name);
    }
#undef EL3
#undef EL2
#undef EL1
#undef RET

    if (opr.input_broadcastable()[wrt_idx]) {
        result = reduce_sum(result, opr::GetVarShape::make(opr.input(wrt_idx)));
    } else if (result.node()->owner_opr()->same_type<Broadcast>()) {
        // forward broadcast for optimizer to work
        result = opr::Broadcast::make(
                result.node()->owner_opr()->input(0),
                opr::GetVarShape::make(i[wrt_idx]));
    }
    if (negate_result)
        result = -result;
    return result.node();
}
#endif

VarNode* Elemwise::sum_grad_list(VarNode* wrt, VarNodeArray& grads) {
    mgb_assert(!grads.empty());
    if (grads.size() == 1)
        return grads[0];
#if MGB_ENABLE_COND_EXEC
    CondExecMerge::modify_grad_sum_list(wrt, grads);
#endif
    VarNodeArray mid_results;
    VarNode* ret;
    if (wrt->owner_graph()->options().graph_opt_level) {
        ret = gopt::GradSumListOptimizer{wrt, grads, mid_results}.get_sum();
    } else {
        ret = gopt::elemwise_reduce_var_list(grads, Elemwise::Mode::ADD, &mid_results);
    }
    mid_results.swap(grads);
    return ret;
}

void Elemwise::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
}

Elemwise::NodeProp* Elemwise::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    for (auto& inp : input()) {
        ret->add_dep_type_existing_var(inp, NodeProp::DepType::VALUE_ALLOW_EMPTY);
    }
    return ret;
}

/* =========================== TypeCvt =========================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TypeCvt);

TypeCvt::TypeCvt(VarNode* inp, DType dest_type, const OperatorNodeConfig& config)
        : Super{inp->owner_graph(),
                config,
                std::string("as") + dest_type.name(),
                {inp}} {
    init_megdnn_opr(*this, {});
    mgb_assert(dest_type.valid());
    add_input({inp});
    add_equivalence_component<ScalarHash<const void*>>(dest_type.handle());
    output(0)->dtype(dest_type).add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
}

SymbolVar TypeCvt::make(
        SymbolVar input, DType dest_type, const OperatorNodeConfig& config) {
    if (input.dtype() == dest_type)
        return input;
    return input.insert_single_output_opr<TypeCvt>(input.node(), dest_type, config);
}

void TypeCvt::perform(
        DeviceTensorND& dest, DType dest_type, const DeviceTensorND& src,
        intl::UniqPtrWithCN<megdnn::TypeCvt>& opr) {
    mgb_assert(src.comp_node() == opr.comp_node());
    mgb_assert(dest_type.valid());
    if (src.empty()) {
        mgb_assert(dest.empty());
        return;
    }
    if (src.dtype() == dest_type) {
        dest.copy_from(src);
        return;
    }
    src.comp_node().activate();
    dest.comp_node(src.comp_node()).dtype(dest_type).resize(src.shape());
    opr->exec(src.as_megdnn(), dest.as_megdnn());
}

void TypeCvt::add_input_layout_constraint() {
    //! Because the implementation of typecvt on arm/x86/cuda/opencl support
    //! non-contiguous memory. So we change constraint of typecvt to monotone
    for (auto i : input()) {
        i->add_layout_constraint_monotone();
    }
}

TypeCvt::NodeProp* TypeCvt::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(0), NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(TypeCvt) {
    MGB_MARK_USED_VAR(wrt_idx);
    auto itype = opr.input(0)->dtype(), otype = opr.output(0)->dtype();
    if (itype.category() == DTypeCategory::FLOAT &&
        otype.category() == DTypeCategory::INT) {
        return nullptr;
    }
    if (itype.category() != DTypeCategory::FLOAT) {
        return InvalidGrad::make(opr, 0);
    }
    return TypeCvt::make(out_grad[0], opr.input(0)->dtype()).node();
}
#endif

void TypeCvt::mem_plan_fwd_in2out_writable() {
    bool cond_low_bit = input(0)->dtype().is_low_bit() &&
                        output(0)->dtype().is_low_bit() &&
                        input(0)->dtype().low_bit() == output(0)->dtype().low_bit();
    bool cond_normal = !input(0)->dtype().is_low_bit() &&
                       !output(0)->dtype().is_low_bit() &&
                       input(0)->dtype().size() == output(0)->dtype().size();
    if ((cond_low_bit || cond_normal) && input(0)->layout().is_contiguous()) {
        output(0)->set_fwd_in2out_writable(input(0));
    }
}

void TypeCvt::scn_do_execute() {
    auto ovar = output(0)->dev_tensor().as_megdnn();
    for (size_t i = 0; i < ovar.layout.ndim; ++i) {
        if (!ovar.layout[i]) {
            // skip execution for empty var
            return;
        }
    }
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(), ovar);
}

void TypeCvt::init_output_static_infer_desc() {
    static StaticInferOpr<megdnn::TypeCvt> static_infer_opr;
    Super::init_output_static_infer_desc();

    using namespace cg::static_infer;

    auto infer_value = [this](DeviceTensorND& dest, const InpVal& inp) {
        auto sopr = static_infer_opr.lock();
        perform(dest, output(0)->dtype(), inp.val.at(0).value(), sopr());
        return true;
    };
    owner_graph()->static_infer_manager().register_value_infer(
            output(0), {SourceType::DEP, {{input(0), DepType::VALUE}}, infer_value});
}

void TypeCvt::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
}

/* =========================== AddUpdate =========================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(AddUpdate);

AddUpdate::AddUpdate(
        VarNode* dest, VarNode* delta, const Param& param,
        const OperatorNodeConfig& config)
        : Super{dest->owner_graph(), config, "inplace_add", {dest, delta}},
          m_param{param} {
    auto dest_opr = dest->owner_opr();
    mgb_throw_if(
            dest_opr->same_type<ImmutableTensor>(), GraphError,
            "AddUpdate cannot be applied on ImmutableTensor; ");
    add_input({dest, delta});

    /*
     * here we tell the system that output(0) would force-update input(0); the
     * topo-sorting system would ensure that all the readers finish before
     * executing this AddUpdate operation
     */
    add_output(None)->set_fwd_in2out_writable_force(input(0)).add_flag(
            VarNode::Flag::NO_MEM_RECLAIM);

    mgb_assert(
            m_param.disable->dtype() == dtype::Int32{},
            "dtype of disable flag on AddUpdate must be Int32, got %s actually.",
            m_param.disable->dtype().name());

    add_equivalence_component<ScalarHash<void*>>(m_param.alpha.get());
    add_equivalence_component<ScalarHash<void*>>(m_param.beta.get());
    add_equivalence_component<ScalarHash<void*>>(m_param.bias.get());
    add_equivalence_component<ScalarHash<void*>>(m_param.disable.get());
}

SymbolVar AddUpdate::make(
        SymbolVar dest, SymbolVar delta, const Param& param,
        const OperatorNodeConfig& config) {
    delta = opr::TypeCvt::make(delta, dest.dtype());
    return dest.insert_single_output_opr<AddUpdate>(
            dest.node(), delta.node(), param, config);
}

cg::OperatorNodeBase::NodeProp* AddUpdate::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_flag(NodeProp::Flag::FORCE_UPDATE_INPUT_VAR);
    return ret;
}

void AddUpdate::create_megdnn_opr() {
    set_megdnn_opr(
            intl::get_megdnn_handle(comp_node())->create_operator<megdnn::AddUpdate>());
}

void AddUpdate::scn_do_execute() {
    mgb_assert(
            m_param.disable->dtype() == dtype::Int32{},
            "dtype of disable flag on AddUpdate must be Int32, got %s actually.",
            m_param.disable->dtype().name());
    auto disable = m_param.disable->get_cast<int>();
    if (disable == 1)
        return;
    mgb_assert(
            disable == 0,
            "disable flag on AddUpdate can only be 0 or 1,"
            " got %d actually.",
            disable);

    auto&& dest = output(0)->dev_tensor();
    auto&& delta_nobrd = input(1)->dev_tensor();
    auto delta = delta_nobrd.sub(SubTensorSpec::make_from_offset_elem(
            delta_nobrd.layout().broadcast(dest.shape()), 0));
    mgb_assert(input(0)->dev_tensor().raw_ptr() == dest.raw_ptr());
    auto beta = m_param.beta->get_cast<float>();
    if (!m_param.alpha->get_cast<bool>() && beta == 1 &&
        !m_param.bias->get_cast<bool>()) {
        dest.copy_from_fixlayout(delta);
    } else {
        auto opr = static_cast<megdnn::AddUpdate*>(megdnn_opr());
        opr->param() = {
                m_param.alpha->get_cast<float>(), beta,
                m_param.bias->get_cast<float>()};
        opr->exec(dest.as_megdnn(), delta.as_megdnn());
    }
}

void AddUpdate::init_output_static_infer_desc() {
    using namespace cg::static_infer;

    owner_graph()->static_infer_manager().register_shape_infer(
            output(0), ShapeInferDesc::make_identity(input(0)));
}

void AddUpdate::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(AddUpdate) {
    // actually valid, just not implemented
    return InvalidGrad::make(opr, wrt_idx);
}
#endif

/* =========================== Reduce =========================== */

class Reduce::KernScheduler {
    class ValueDep final : public ExecDependency {
        DeviceTensorStorage m_val;

    public:
        explicit ValueDep(DeviceTensorStorage val) : m_val(std::move(val)) {}
    };

public:
    bool has_actual_computing() const {
        mgb_assert(m_shape_computed);
        return !m_kern_param.empty() || m_apply_side_effect;
    }

    size_t workspace_size() const { return m_workspace_spec[2].end(); }

    bool shape_computed() const { return m_shape_computed; }

    //! init shapes in kern param
    void init_shapes(
            megdnn::Reduce* opr, CompNode comp_node, DType dtype, Mode mode,
            TensorShape ishp, TensorShape oshp, const Param::DataType data_type);

    void setup_kern_params_layout_and_mode(
            Mode mode, DType inp_dtype, TensorShape& inp_shp, const Param::DataType);

    void check_shapes(const TensorShape& ishp, const TensorShape& oshp) {
        mgb_assert(m_prev_ishp.eq_shape(ishp) && m_prev_oshp.eq_shape(oshp));
    }

    //! update pointers in kern param; the tensors must have been allocated
    void update_ptr(
            const DeviceTensorND& input, const DeviceTensorND& dest,
            const DeviceTensorND& workspace);

    void execute(
            megdnn::Reduce* opr, const DeviceTensorND& input,
            const DeviceTensorND& dest);

    void record_execute_deps(ExecDependencyArray& deps) {
        if (m_elemwise_trans_opr) {
            deps.emplace_back(std::make_unique<intl::MegDNNGraphDep>(
                    std::move(m_elemwise_trans_opr)));
        }
        if (m_typecvt_opr) {
            deps.emplace_back(
                    std::make_unique<intl::MegDNNGraphDep>(std::move(m_typecvt_opr)));
        }
        deps.emplace_back(std::make_unique<ValueDep>(m_side_affect_wkspc.storage()));
    }

private:
    struct KernParam {
        megdnn::TensorND input, output;

        //! param passed to megdnn
        megdnn::param::Reduce kparam;

        megdnn::Workspace workspace;

        KernParam(Mode mode, int32_t ra) : kparam{mode, ra} {}
    };

    struct SubWorkspace {
        size_t size, offset;
        size_t end() const { return size + offset; }
    };

    void update_kparam_for_elemwise_side_effect(
            CompNode comp_node, Mode mode, const Param::DataType data_type);

    bool m_shape_computed = false;
    std::vector<KernParam> m_kern_param;
    TensorShape m_prev_ishp, m_prev_oshp;
    SubWorkspace m_workspace_spec[3];  //! tmp output[2], kern workspce

    /*!
     * some reduce mode (like SUM_SQR) has side effect of element-wise
     * trans. If this is the case and there is no kernel param,
     * m_apply_side_effect would be non-null
     */
    thin_function<void(const DeviceTensorND& in, const DeviceTensorND& out)>
            m_apply_side_effect;
    std::unique_ptr<megdnn::Elemwise> m_elemwise_trans_opr;
    std::unique_ptr<megdnn::TypeCvt> m_typecvt_opr;
    std::unique_ptr<megdnn::Fill> m_fill_opr;
    DeviceTensorND m_side_affect_wkspc;
};

void Reduce::KernScheduler::setup_kern_params_layout_and_mode(
        Mode mode, DType inp_dtype, TensorShape& ishp,
        const Param::DataType data_type) {
    auto prev_dtype = inp_dtype;
    for (size_t idx = 0; idx < m_kern_param.size(); ++idx) {
        auto&& i = m_kern_param[idx];

#if !MEGDNN_DISABLE_FLOAT16
        if (idx == 0 && data_type == Param::DataType::FLOAT_O32xC32) {
            i.input.layout.dtype = inp_dtype;
            i.output.layout.dtype = dtype::Float32();
            i.kparam.data_type = data_type;
        } else if (data_type == Param::DataType::FLOAT_O16xC32) {
            i.input.layout.dtype = prev_dtype;
            if (idx + 1 == m_kern_param.size()) {
                i.output.layout.dtype = dtype::Float16();
                i.kparam.data_type = data_type;
            } else {
                i.output.layout.dtype = dtype::Float32();
                i.kparam.data_type = Param::DataType::FLOAT_O32xC32;
            }
        } else
#endif
        {
            mgb_assert(
                    data_type == Param::DataType::DEFAULT ||
                    (data_type == Param::DataType::FLOAT_O32xC32 && idx));
            i.input.layout.dtype = prev_dtype;
            i.output.layout.dtype = prev_dtype;
            i.kparam.data_type = Param::DataType::DEFAULT;
        }
        prev_dtype = i.output.layout.dtype;

        i.input.layout.init_contiguous_stride(ishp);
        ishp.shape[i.kparam.axis] = 1;
        i.output.layout.init_contiguous_stride(ishp);
    }
    if (mode == Mode::SUM_SQR) {
        for (size_t i = 1; i < m_kern_param.size(); ++i)
            m_kern_param[i].kparam.mode = Mode::SUM;
    }
}

void Reduce::KernScheduler::init_shapes(
        megdnn::Reduce* opr, CompNode comp_node, DType inp_dtype, Mode mode,
        TensorShape ishp, TensorShape oshp, const Param::DataType data_type) {
    mgb_assert(ishp.ndim && oshp.ndim);

    if (ishp.eq_shape(m_prev_ishp) && oshp.eq_shape(m_prev_oshp))
        return;

    m_prev_ishp = ishp;
    m_prev_oshp = oshp;

    m_kern_param.clear();

    if (oshp.is_scalar()) {
        // if ishp is non-contiguous, add_layout_constraint_contiguous would be
        // added; so we do not have to worry about this
        ishp.shape[0] = ishp.total_nr_elems();
        ishp.ndim = 1;
    }

    mgb_assert(
            oshp.ndim == ishp.ndim,
            "input and output ndim mismatch for reduction: ishp=%s oshp=%s",
            ishp.to_string().c_str(), oshp.to_string().c_str());

    for (size_t i = 0; i < ishp.ndim; ++i) {
        if (ishp.shape[i] != oshp.shape[i]) {
            mgb_assert(
                    oshp.shape[i] == 1,
                    "input and output shape mismatch for reduction: "
                    "ishp=%s oshp=%s",
                    ishp.to_string().c_str(), oshp.to_string().c_str());
        }
    }

    auto remove_axis = [](TensorShape& shp, size_t ax) {
        mgb_assert(shp.ndim > 1);
        for (auto i = ax + 1; i < shp.ndim; ++i)
            shp.shape[i - 1] = shp.shape[i];
        --shp.ndim;
    };

    // collapse consecutive shape-1 axes in oshp
    for (size_t i = 0; i < oshp.ndim; ++i) {
        auto start = i;
        while (i < oshp.ndim && oshp.shape[i] == 1)
            ++i;

        if (start + 1 < i) {
            for (auto j = start + 1; j < i; ++j)
                ishp.shape[start] *= ishp.shape[j];

            for (auto j = start + 1; j < i; ++j) {
                remove_axis(ishp, start + 1);
                remove_axis(oshp, start + 1);
            }

            i = start;
        }
    }

    for (uint32_t i = 0; i < ishp.ndim; ++i) {
        if (ishp.shape[i] != oshp.shape[i]) {
            mgb_assert(oshp.shape[i] == 1);
            m_kern_param.push_back({mode, static_cast<int32_t>(i)});
        }
    }
    // sort according to reduction size, so workspace can be smaller
    small_sort(
            m_kern_param.begin(), m_kern_param.end(),
            [&](const KernParam& a, const KernParam& b) {
                return ishp.shape[a.kparam.axis] > ishp.shape[b.kparam.axis];
            });

    // init kparam input/output layout
    setup_kern_params_layout_and_mode(mode, inp_dtype, ishp, data_type);

    // init workspace size
    memset(m_workspace_spec, 0, sizeof(m_workspace_spec));

    for (auto&& i : m_kern_param) {
        opr->param() = i.kparam;
        i.workspace.size = opr->get_workspace_in_bytes(i.input.layout, i.output.layout);
        update_max(m_workspace_spec[2].size, i.workspace.size);
    }

    mgb_assert(ishp.eq_shape(oshp));

    if (m_kern_param.size() >= 2) {
        m_workspace_spec[0].size = m_kern_param[1].input.layout.span().high_byte;
    }
    if (m_kern_param.size() >= 3) {
        m_workspace_spec[1].size = m_kern_param[2].input.layout.span().high_byte;
    }

    auto align = comp_node.get_mem_addr_alignment();
    for (int i = 0; i < 2; ++i) {
        m_workspace_spec[i + 1].offset =
                get_aligned_power2(m_workspace_spec[i].end(), align);
    }

    update_kparam_for_elemwise_side_effect(comp_node, mode, data_type);

    m_shape_computed = true;
}

void Reduce::KernScheduler::update_kparam_for_elemwise_side_effect(
        CompNode comp_node, Mode mode, const Param::DataType data_type) {
    m_apply_side_effect = nullptr;
    m_elemwise_trans_opr.reset();
    m_typecvt_opr.reset();
    if (!m_kern_param.empty()) {
        // no need to set m_apply_side_effect
        return;
    } /* else */
    // case A: input.layout == output.layout
    // case B: input.total_nr_elems == 1 and output is a scalar

    if (mode == Mode::SUM_SQR) {
        m_elemwise_trans_opr =
                intl::get_megdnn_handle(comp_node)->create_operator<megdnn::Elemwise>();
        m_elemwise_trans_opr->param() = {Elemwise::Mode::MUL};
    }
    if (data_type != Param::DataType::DEFAULT) {
        m_side_affect_wkspc = DeviceTensorND{comp_node, dtype::Float32()};
        m_typecvt_opr =
                intl::get_megdnn_handle(comp_node)->create_operator<megdnn::TypeCvt>();
    }
    if (!m_typecvt_opr && !m_elemwise_trans_opr)
        return;

    m_apply_side_effect = [this](const DeviceTensorND& in, const DeviceTensorND& out) {
        if (m_typecvt_opr) {
            m_side_affect_wkspc.resize(in.shape());
        }
        if (!m_elemwise_trans_opr) {
            mgb_assert(m_typecvt_opr);
            m_typecvt_opr->exec(in.as_megdnn(), out.as_megdnn());
            return;
        }
        auto im = in.as_megdnn();
        megdnn::TensorND wm;
        if (m_typecvt_opr && in.dtype() != m_side_affect_wkspc.dtype()) {
            m_side_affect_wkspc.resize(in.shape());
            wm = m_side_affect_wkspc.as_megdnn();
            m_typecvt_opr->exec(im, wm);
        } else {
            wm = im;
        }
        if (m_typecvt_opr && wm.layout.dtype != out.dtype()) {
            m_elemwise_trans_opr->exec({wm, wm}, wm);
            m_typecvt_opr->exec(wm, out.as_megdnn());
        } else {
            auto&& wshp = wm.layout;
            if (wshp.ndim != out.layout().ndim) {
                // to ensure that wkspc.ndim equals out.ndim in the case:
                // wkspc.shape=(1, 1, ..., 1) and out.shape=(1), otherwise it
                // may lead the 'TensorShape Dimension' assertion failed in
                // the following broadcast operator
                mgb_assert(wshp.total_nr_elems() == 1 && out.layout().ndim == 1);
                wshp.ndim = 1;
            }
            m_elemwise_trans_opr->exec({wm, wm}, out.as_megdnn());
        }
    };
}

void Reduce::KernScheduler::update_ptr(
        const DeviceTensorND& input, const DeviceTensorND& dest,
        const DeviceTensorND& workspace) {
    auto dtype = dest.layout().dtype;
    mgb_assert(dtype.valid());
    mgb_assert(m_shape_computed);

    if (workspace_size()) {
        mgb_assert(
                workspace.layout().dtype == dtype::Byte() &&
                workspace.layout().ndim == 1 &&
                workspace.shape()[0] >= workspace_size());
    }

    if (m_kern_param.empty())
        return;

    mgb_assert(
            input.layout().total_nr_elems() ==
            m_kern_param[0].input.layout.total_nr_elems());
    mgb_assert(
            dest.shape().total_nr_elems() ==
            m_kern_param.back().output.layout.total_nr_elems());
    m_kern_param[0].input.raw_ptr = const_cast<dt_byte*>(input.raw_ptr());

    dt_byte *workspace_begin = workspace_size()
                                     ? const_cast<dt_byte*>(workspace.raw_ptr())
                                     : nullptr,
            *tmp_reduce_ptr[2] =
                    {workspace_begin + m_workspace_spec[0].offset,
                     workspace_begin + m_workspace_spec[1].offset},
            *kern_workspace = workspace_begin + m_workspace_spec[2].offset;
    for (size_t i = 0; i < m_kern_param.size() - 1; ++i) {
        auto optr = tmp_reduce_ptr[i % 2];
        m_kern_param[i].output.raw_ptr = optr;
        m_kern_param[i + 1].input.raw_ptr = optr;
    }
    for (auto&& i : m_kern_param)
        i.workspace.raw_ptr = kern_workspace;
    m_kern_param.back().output.raw_ptr = const_cast<dt_byte*>(dest.raw_ptr());
}

void Reduce::KernScheduler::execute(
        megdnn::Reduce* opr, const DeviceTensorND& input, const DeviceTensorND& dest) {
    if (m_apply_side_effect) {
        mgb_assert(m_kern_param.empty());
        m_apply_side_effect(input, dest);
        return;
    }

    mgb_assert(!m_kern_param.empty());

    // empty input
    if (input.shape_valid() && input.empty()) {
        auto mode = m_kern_param[0].kparam.mode;
        if (!m_fill_opr) {
            m_fill_opr = intl::get_megdnn_handle(dest.comp_node())
                                 ->create_operator<megdnn::Fill>();
        }
        std::string err_msg;
        switch (mode) {
            case Reduce::Mode::SUM:
                if (!dest.empty()) {
                    m_fill_opr->param() = 0;
                    m_fill_opr->exec(dest.as_megdnn(), {});
                }
                break;
            case Reduce::Mode::PRODUCT:
                if (!dest.empty()) {
                    m_fill_opr->param() = 1;
                    m_fill_opr->exec(dest.as_megdnn(), {});
                }
                break;
            case Reduce::Mode::MEAN:
                err_msg = "mean";
                break;
            case Reduce::Mode::MIN:
                err_msg = "min";
                break;
            case Reduce::Mode::MAX:
                err_msg = "max";
                break;
            case Reduce::Mode::SUM_SQR:
                err_msg = "sum_sqr";
                break;
            default:
                mgb_throw(MegBrainError, "bad reduce mode");
        }
        if (!err_msg.empty()) {
            mgb_throw(
                    MegBrainError, "empty input is not allowed for reduce mode: %s",
                    err_msg.c_str());
        }
        return;
    }
    mgb_assert(
            input.layout().is_contiguous() &&
            input.raw_ptr() == m_kern_param[0].input.raw_ptr &&
            dest.raw_ptr() == m_kern_param.back().output.raw_ptr);
    for (auto&& i : m_kern_param) {
        opr->param() = i.KernParam::kparam;
        opr->exec(i.input, i.output, i.workspace);
    }
}

class Reduce::OutTensorShapeExtender {
public:
    OutTensorShapeExtender(const TensorShape& ishp, const TensorShape& oshp)
            : m_oshp(oshp) {
        mgb_assert(
                oshp.ndim <= ishp.ndim,
                "output ndim should be less and equal than input ndim for "
                "reduction: "
                "ishp=%s oshp=%s",
                ishp.to_string().c_str(), oshp.to_string().c_str());
        // Ex. ishp = (a, b, c, d), oshp = (c, d)
        if (!oshp.is_scalar() && ishp.ndim != oshp.ndim) {
            size_t ndim_diff = ishp.ndim - oshp.ndim;
            auto&& canonized_oshp = m_canonized_oshp_storage.emplace(oshp);
            for (size_t i = 0; i < ishp.ndim; ++i)
                if (i < ndim_diff)
                    canonized_oshp[i] = 1;
                else
                    canonized_oshp[i] = oshp[i - ndim_diff];
            canonized_oshp.ndim = ishp.ndim;
        }
    }

    const TensorShape& get() const {
        return m_canonized_oshp_storage.valid() ? m_canonized_oshp_storage.val()
                                                : m_oshp;
    }

private:
    Maybe<TensorShape> m_canonized_oshp_storage;
    const TensorShape& m_oshp;
};

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Reduce);
Reduce::Reduce(
        VarNode* inp, VarNode* target_shape, const Param& param,
        const OperatorNodeConfig& config)
        : Super{inp->owner_graph(),
                config,
                ssprintf("reduce%d", static_cast<int>(param.mode)),
                {inp}},
          m_param{param},
          m_kern_scheduler{std::make_unique<KernScheduler>()} {
    add_input({inp});

    if (inp->dtype().enumv() == DTypeEnum::Quantized8Asymm &&
        inp->dtype().category() == DTypeCategory::QUANTIZED) {
        mgb_assert(
                param.mode != Param::Mode::PRODUCT,
                "Reduce does not support PRODUCT mode on quantized input");
        mgb_assert(
                param.mode != Param::Mode::SUM_SQR,
                "Reduce does not support SUM_SQR mode on quantized input");
        mgb_assert(
                param.mode != Param::Mode::SUM,
                "Reduce does not support SUM mode on quantized input");
    }

    DType out_dtype;
    switch (param.data_type) {
        case Param::DataType::DEFAULT:
            out_dtype = inp->dtype();
            break;
#if !MEGDNN_DISABLE_FLOAT16
        case Param::DataType::FLOAT_O16xC32:
            out_dtype = dtype::Float16();
            break;
        case Param::DataType::FLOAT_IO16xC32:
            mgb_assert(false);
#endif
        case Param::DataType::FLOAT_O32xC32:
            out_dtype = dtype::Float32();
            break;
        case Param::DataType::QUINT_I8xO32:
            out_dtype = dtype::QuantizedS32(
                    inp->dtype().param<dtype::Quantized8Asymm>().scale);
            break;
        case Param::DataType::QINT_I8xO32:
            out_dtype =
                    dtype::QuantizedS32(inp->dtype().param<dtype::QuantizedS8>().scale);
            break;
        default:
            mgb_throw(GraphError, "invalid param data_type: %d", int(param.data_type));
    }
    add_output(None)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE).dtype(out_dtype);
    cg::add_workspace_output(this);

    add_equivalence_component<PODHash<Param>>(&m_param);

    if (param.axis >= -MEGDNN_MAX_NDIM && param.axis < MEGDNN_MAX_NDIM) {
        mgb_throw_if(
                target_shape, GraphError,
                "could not specify both axis and target shape");
        m_is_symtshp = false;
    } else {
        mgb_throw_if(
                !target_shape, GraphError, "neither axis or target_shape specified");
        add_input({target_shape});
        m_is_symtshp = true;

        outshape_by_symvar_enable(0, 1);
    }
}

Reduce::~Reduce() = default;

SymbolVar Reduce::make(
        SymbolVar src, Param param, SymbolVar target_shape,
        const OperatorNodeConfig& config) {
    if (param.data_type == Param::DataType::FLOAT_IO16xC32) {
        mgb_log_warn(
                "DataType FLOAT_IO16xC32 has been deprecated "
                "use FLOAT_O16xC32 instead");
        param.data_type = Param::DataType::FLOAT_O16xC32;
    }

    if (param.mode == Mode::SUM && src.node()->owner_opr()->same_type<Elemwise>()) {
        // replace sum(x^2) by sum_sqr(x)
        auto&& opr = src.node()->owner_opr()->cast_final<Elemwise>();
        if (opr.param().mode == Elemwise::Mode::POW) {
            mgb_assert(opr.input().size() == 2);
            auto pow = SymbolVar{opr.input(1)}.as_immutable_scalar();
            if (pow.valid() && pow->get_cast<float>() == 2) {
                src = opr.input(0);
                param.mode = Mode::SUM_SQR;
            }
        }
    }
    return src.insert_single_output_opr<Reduce>(
            src.node(), target_shape.node(), param, config);
}

void Reduce::outshape_by_symvar_do_get_output_shape(
        TensorShape& dest, const ShapeInferInfo& shpinfo) {
    cg::copy_tensor_value_to_shape(dest, *shpinfo.shpval_inp_val.at(0));
}

void Reduce::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();

    // infer output shape
    if (m_is_symtshp) {
        // reduce to target shape
        Super::init_output_static_infer_desc();
    } else {
        // reduce along axis
        auto infer_shape = [this](TensorShape& dest, const InpVal& inp) {
            dest = inp.val.at(0).shape();
            mgb_assert(
                    m_param.axis < static_cast<int>(dest.ndim) &&
                            m_param.axis >= -static_cast<int>(dest.ndim),
                    "invalid axis for reduction: shape=%s axis=%d",
                    dest.to_string().c_str(), m_param.axis);
            int real_axis = m_param.axis;
            if (real_axis < 0)
                real_axis += dest.ndim;
            dest.shape[real_axis] = 1;
            return true;
        };
        mgr.register_shape_infer(
                output(0),
                {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_shape});
    }

    // infer workspace
    auto infer_workspace = [this](TensorShape& dest, const InpVal& inp) {
        init_kern_sched_shape(inp.val[0].shape(), inp.val[1].shape());
        dest.ndim = 1;
        dest.shape[0] = m_kern_scheduler->workspace_size();
        return true;
    };
    mgr.register_shape_infer(
            output(1), {SourceType::DEP,
                        {{input(0), DepType::SHAPE}, {output(0), DepType::SHAPE}},
                        infer_workspace});

    // infer value

    static StaticInferOpr<megdnn::Reduce> static_infer_opr;
    auto infer_value = [this](DeviceTensorND& dest, const InpVal& inp) {
        DeviceTensorND workspace;
        auto sopr = static_infer_opr.lock();
        perform(m_param.mode, dest, workspace, inp.val[0].value(), output(0)->dtype(),
                inp.val.at(1).shape(), sopr(), m_param.data_type);
        return true;
    };

    mgr.register_value_infer(
            output(0), {SourceType::DEP,
                        {{input(0), DepType::VALUE}, {output(0), DepType::SHAPE}},
                        infer_value});
}

void Reduce::init_kern_sched_shape(const TensorShape& ishp, const TensorShape& oshp) {
    OutTensorShapeExtender extender(ishp, oshp);
    auto&& canonized_oshp = extender.get();
    m_kern_scheduler->init_shapes(
            static_cast<megdnn::Reduce*>(megdnn_opr()), comp_node(), input(0)->dtype(),
            m_param.mode, ishp, canonized_oshp, m_param.data_type);
}

cg::OperatorNodeBase::OprEventCallback Reduce::get_opr_event_callback() {
    auto on_mem_status_changed = [this]() {
        auto&& ishp = input(0)->shape();
        auto&& oshp = output(0)->shape();
        OutTensorShapeExtender extender(ishp, oshp);
        auto&& canonized_oshp = extender.get();
        m_kern_scheduler->check_shapes(input(0)->shape(), canonized_oshp);
        m_kern_scheduler->update_ptr(
                input(0)->dev_tensor(), output(0)->dev_tensor(),
                output(1)->shape()[0] ? output(1)->dev_tensor() : DeviceTensorND{});
    };
    return {on_mem_status_changed};
}

void Reduce::mem_plan_fwd_in2out_readonly() {
    init_kern_sched_shape(input(0)->shape(), output(0)->shape());

    if (!m_kern_scheduler->has_actual_computing()) {
        // forward memory if no actual computing needed

        if (!output(0)->mem_plan().valid()) {
            // output(0) is dynamic but current is staic alloc phase (for
            // workspace)
            return;
        }
        auto&& ily = input(0)->layout();
        auto&& oly = output(0)->layout();
        const TensorLayout* fwd_spec = nullptr;
        Maybe<TensorLayout> ily_modified_storage;

        if (!ily.eq_shape(oly)) {
            auto&& ily_modified = ily_modified_storage.emplace(ily);
            mgb_assert(ily.ndim > oly.ndim);
            for (size_t i = 0; i < ily.ndim - oly.ndim; ++i)
                mgb_assert(ily.shape[i] == 1);
            ily_modified = ily_modified.reshape(oly);
            fwd_spec = &ily_modified;
        } else {
            fwd_spec = &ily;
        }
        m_mem_fwd_success = output(0)->set_fwd_in2out_readonly(
                input(0), SubTensorSpec::make_from_layout(*fwd_spec));
    }
}

void Reduce::add_input_layout_constraint() {
    if (!cg::is_static_var_shape(output(0))) {
        // output shape can not be inferred; require contiguous to be safe
        input(0)->add_layout_constraint_contiguous();
    } else {
        auto check = [this](const TensorLayout& ily) {
            auto&& mgr = owner_graph()->static_infer_manager();
            auto oshp = mgr.infer_shape(output(0));
            init_kern_sched_shape(ily, oshp);
            if (m_kern_scheduler->has_actual_computing())
                return ily.is_contiguous();
            return true;
        };
        input(0)->add_layout_constraint(check);
    }
}

void Reduce::scn_do_execute() {
    auto&& inp = input(0)->dev_tensor();
    auto&& out = output(0)->dev_tensor();
    auto&& ishp = input(0)->shape();
    auto&& oshp = output(0)->shape();
    const DeviceTensorND* out_ptr;
    Maybe<DeviceTensorND> canonized_storage;
    OutTensorShapeExtender extender(ishp, oshp);
    auto&& canonized_oshp = extender.get();
    if (canonized_oshp.ndim != out.shape().ndim) {
        auto&& canonized_out = canonized_storage.emplace(out);
        canonized_out.reset(
                canonized_out.storage(),
                canonized_out.layout().reshape(canonized_oshp));
        out_ptr = &canonized_out;
    } else {
        out_ptr = &out;
    }
    // shape initialized either in deducing workspace,
    // mem_plan_fwd_in2out_readonly, or check input layout
    m_kern_scheduler->check_shapes(inp.shape(), out_ptr->shape());

    if (m_kern_scheduler->has_actual_computing()) {
        m_kern_scheduler->execute(
                static_cast<megdnn::Reduce*>(megdnn_opr()), inp, *out_ptr);
    } else {
        // no reduction needed, just forward
        if (m_mem_fwd_success) {
            mgb_assert(
                    inp.raw_ptr() == out_ptr->raw_ptr() &&
                    out_ptr->layout().total_nr_elems() ==
                            inp.layout().total_nr_elems());
        } else {
            if (!out_ptr->shape().eq_shape(inp.shape())) {
                mgb_assert(
                        out_ptr->shape().is_scalar() &&
                        inp.shape().total_nr_elems() == 1);
                out_ptr->sub(SubTensorSpec::make_from_layout(inp.layout()))
                        .copy_from_fixlayout(inp);
            } else {
                out_ptr->copy_from_fixlayout(inp);
            }
        }
    }
}

void Reduce::perform(
        Mode mode, DeviceTensorND& dest, DeviceTensorND& workspace,
        const DeviceTensorND& input, const DType& target_dtype,
        const TensorShape& target_shape, intl::UniqPtrWithCN<megdnn::Reduce>& opr,
        const Param::DataType data_type) {
    mgb_assert(
            !dest.storage().comp_node_valid() || opr.comp_node() == dest.comp_node());
    KernScheduler ksched;
    OutTensorShapeExtender extender(input.shape(), target_shape);
    auto&& canonized_oshp = extender.get();
    ksched.init_shapes(
            opr.get(), opr.comp_node(), input.layout().dtype, mode, input.shape(),
            canonized_oshp, data_type);

    if (!ksched.has_actual_computing()) {
        mgb_assert(target_shape.total_nr_elems() == input.layout().total_nr_elems());
        dest.copy_from(input);
        dest.reset(dest.storage(), {target_shape, dest.dtype()});
        return;
    }

    workspace.comp_node(opr.comp_node()).dtype(dtype::Byte());
    size_t workspace_size = ksched.workspace_size();
    DeviceTensorND input_contig_storage;
    const DeviceTensorND* input_contig = &input;
    if (!input.layout().is_contiguous()) {
        auto offset = get_aligned_power2(
                workspace_size, opr.comp_node().get_mem_addr_alignment());
        workspace_size = offset + input.dtype().size(input.shape().total_nr_elems());

        workspace.resize({workspace_size});
        input_contig_storage
                .reset(workspace.storage().sub(offset), {input.shape(), input.dtype()})
                .copy_from(input);
        input_contig = &input_contig_storage;
    } else {
        workspace.resize({workspace_size});
    }

    opr.comp_node().activate();
    dest.comp_node(opr.comp_node()).dtype(target_dtype).resize(target_shape);
    ksched.update_ptr(*input_contig, dest, workspace);
    ksched.execute(opr.get(), *input_contig, dest);
}

Reduce::NodeProp* Reduce::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(0), NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}

void Reduce::create_megdnn_opr() {
    set_megdnn_opr(
            intl::get_megdnn_handle(comp_node())->create_operator<megdnn::Reduce>());
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Reduce) {
    for (size_t i = 1; i < opr.output().size(); ++i)
        mgb_assert(!out_grad[i]);
    if (wrt_idx || opr.input(0)->dtype().category() != DTypeCategory::FLOAT)
        return InvalidGrad::make(opr, wrt_idx);
    SymbolVar og{out_grad[0]}, iv{opr.input(0)}, ov{opr.output(0)};
    constexpr auto cmv = Elemwise::Mode::COND_LEQ_MOV;
    using Mode = Reduce::Mode;
    SymbolVar grad = [&]() {
        switch (opr.param().mode) {
            case Mode::SUM:
                return Broadcast::make(og, GetVarShape::make(iv));
            case Mode::SUM_SQR:
                return (og * og.make_scalar_dt(2) * iv);
            case Mode::PRODUCT:
                return ((og * ov) / iv);
            case Mode::MIN:
                return Elemwise::make({iv, ov, og}, cmv);
            case Mode::MAX:
                return Elemwise::make({ov, iv, og}, cmv);
            case Mode::MEAN: {
                auto og_shape = opr::GetVarShape::make(og),
                     iv_shape = opr::GetVarShape::make(iv),
                     scale =
                             div(opr::reduce_prod(og_shape, og_shape.make_scalar(1)),
                                 opr::reduce_prod(iv_shape, iv_shape.make_scalar(1)));
                return scale * Broadcast::make(og, GetVarShape::make(iv));
            }
            default:
                mgb_throw(MegBrainError, "bad reduce mode");
        }
    }();
    grad = TypeCvt::make(grad, iv.dtype());
    return grad.node();
}
#endif

void Reduce::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
    m_kern_scheduler->record_execute_deps(deps);
}

/* =========================== PowC =========================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(PowC);

PowC::PowC(VarNode* i0, const Param& param, const OperatorNodeConfig& config)
        : Super(OperatorNodeBaseCtorParam{
                  i0->owner_graph(), config, ssprintf("powc_%g", param.exp), {i0}}) {
    init_megdnn_opr(*this, param);
    add_input({i0});
    output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    intl::MegDNNOprInitPostCtor<PowC>::apply(*this);
}

SymbolVar PowC::make(
        SymbolVar x, const Param& param, const OperatorNodeConfig& config) {
    if (almost_equal(param.exp, 1.f)) {
        return x;
    }
    if (almost_equal(param.exp, 0.f)) {
        return x.make_scalar_dt(1).broadcast(x.symshape());
    }
    return x.insert_single_output_opr<PowC>(x.node(), param, config);
}

void PowC::add_input_layout_constraint() {
    input(0)->add_layout_constraint_monotone();
}

void PowC::mem_plan_fwd_in2out_writable() {
    output(0)->set_fwd_in2out_writable(input(0));
}

void PowC::init_output_static_infer_desc() {
    Super::init_output_static_infer_desc();
    static StaticInferOpr<megdnn::PowC> static_infer_opr;
    using namespace cg::static_infer;

    auto infer_value = [this](DeviceTensorND& dest, const InpVal& inp) {
        auto infer_opr_lock = static_infer_opr.lock();
        auto&& infer_opr = infer_opr_lock();
        infer_opr->param() = this->param();
        auto&& ival = inp.val[0].value().as_megdnn();
        infer_opr->exec(ival, dest.resize(ival.layout).as_megdnn());
        return true;
    };
    owner_graph()->static_infer_manager().register_value_infer(
            output(0), {SourceType::DEP, {{input(0), DepType::VALUE}}, infer_value});
}

void PowC::scn_do_execute() {
    if (input(0)->dev_tensor().empty()) {
        mgb_assert(output(0)->dev_tensor().empty());
        return;
    }
    mgb_assert(!output(0)->dev_tensor().empty());
    Super::scn_do_execute();
}

PowC::NodeProp* PowC::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(0), NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(PowC) {
    auto exp = opr.param().exp;
    return (exp * SymbolVar{out_grad[0]} *
            PowC::make(opr.input(0), exp - 1, opr.config()))
            .node();
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
