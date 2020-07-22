/**
 * \file src/jit/impl/halide/ast_hl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./ast_hl.h"

#if MGB_JIT_HALIDE

#include "megbrain/gopt/gtrans.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"

using namespace mgb;
using namespace jit;
using namespace ast_hl;

namespace {
Halide::Expr hl_type_cast(Halide::Expr src, DType dst_type) {
    switch (dst_type.enumv()) {
        case megdnn::DTypeEnum::Float32:
            return Halide::cast<float>(src);
        case megdnn::DTypeEnum::Float16:
            return Halide::cast<Halide::float16_t>(src);
        case megdnn::DTypeEnum::Int32:
            return Halide::cast<int>(src);
        default:
            mgb_throw(InternalError,
                      "dtype(%s) is not any of [Float16, Float32, Int32]",
                      dst_type.name());
    }
}

template <typename ctype>
struct ctype_to_hl_type;

template <>
struct ctype_to_hl_type<float> {
    using hl_type = float;
};

template <>
struct ctype_to_hl_type<dt_float16> {
    using hl_type = Halide::float16_t;
};

template <typename T>
using ctype_to_hl_type_t = typename ctype_to_hl_type<T>::hl_type;

Halide::Expr dispatch_elemwise_mode(
        opr::Elemwise::Mode mode, const AstNodeArray& inputs, DType out_dtype,
        const std::vector<std::vector<Halide::Expr>>& exprs_of_inps) {
    using Mode = opr::Elemwise::Mode;
    auto cv = [&](Halide::Expr a) { return hl_type_cast(a, out_dtype); };

#define inp(i) (inputs[(i)]->m_func(exprs_of_inps[(i)]))
    switch (mode) {
        // unary
        case Mode::RELU:
            return Halide::select(inp(0) <= 0, cv(0),
                                  inp(0));
        case Mode::ABS:
            return Halide::abs(inp(0));
        case Mode::ACOS:
            return Halide::acos(inp(0));
        case Mode::ASIN:
            return Halide::asin(inp(0));
        case Mode::CEIL:
            return Halide::ceil(inp(0));
        case Mode::COS:
            return Halide::cos(inp(0));
        case Mode::EXP:
            return Halide::exp(inp(0));
        case Mode::EXPM1:
            return Halide::exp(inp(0)) - cv(1);
        case Mode::FLOOR:
            return Halide::floor(inp(0));
        case Mode::LOG:
            return Halide::log(inp(0));
        case Mode::LOG1P:
            return Halide::log(inp(0) + cv(1));
        case Mode::NEGATE:
            return -inp(0);
        case Mode::SIGMOID:
            return cv(1) /
                   (cv(1) + Halide::exp(-inp(0)));
        case Mode::SIN:
            return Halide::sin(inp(0));
        case Mode::TANH:
            return Halide::tanh(inp(0));
        case Mode::ERF:
            return Halide::erf(inp(0));
        case Mode::ERFC:
            return cv(1) - Halide::erf(inp(0));
        case Mode::H_SWISH:
            return inp(0) *
                   Halide::max(Halide::min(inp(0) + cv(3), cv(6)), cv(0)) /
                   cv(6);

        // binary
        case Mode::ABS_GRAD:
            return Halide::select(inp(0) > 0, inp(1), -inp(1));
        case Mode::ADD:
            return inp(0) + inp(1);
        case Mode::FLOOR_DIV:
            return Halide::floor(inp(0) / inp(1));
        case Mode::MAX:
            return Halide::max(inp(0), inp(1));
        case Mode::MIN:
            return Halide::min(inp(0), inp(1));
        case Mode::MOD: {
            Halide::Expr e =
                    Halide::abs(inp(0)) -
                    Halide::abs(inp(1)) *
                            Halide::floor(Halide::abs(inp(0) / inp(1)));
            return Halide::select(inp(0) > 0, e, -e);
        }
        case Mode::MUL:
            return inp(0) * inp(1);
        case Mode::POW:
            return Halide::pow(inp(0), inp(1));
        case Mode::SIGMOID_GRAD:
            return inp(0) * (1 - inp(0)) * inp(1);
        case Mode::SUB:
            return inp(0) - inp(1);
        case Mode::SWITCH_GT0: {
            Halide::Expr e = inp(0) > 0;
            return e * inp(1);
        }
        case Mode::TANH_GRAD:
            return (cv(1) - inp(0) * inp(0)) * inp(1);
        case Mode::TRUE_DIV:
            return inp(0) / inp(1);
        case Mode::LOG_SUM_EXP:
            return Halide::log(Halide::exp(inp(0)) + Halide::exp(inp(1)));
        case Mode::LT:
            return cv(inp(0) < inp(1));
        case Mode::LEQ:
            return cv(inp(0) <= inp(1));
        case Mode::EQ:
            return cv(inp(0) == inp(1));
        case Mode::SHL:
            return inp(0) << inp(1);
        case Mode::SHR:
            return inp(0) >> inp(1);
        case Mode::ATAN2:
            return Halide::atan2(inp(0), inp(1));
        case Mode::H_SWISH_GRAD:
            return Halide::select(
                    inp(0) < -3, cv(0),
                    Halide::select(inp(0) > 3, inp(1),
                                   (cv(2) * inp(0) + cv(3)) * inp(1) / cv(6)));

        // ternary
        case Mode::COND_LEQ_MOV:
            return Halide::select(inp(0) <= inp(1), inp(2),
                                  cv(0));
        case Mode::FUSE_MUL_ADD3:
            return inp(0) * inp(1) + inp(2);
        case Mode::FUSE_MUL_ADD4:
            return inp(0) * inp(1) + inp(2) * inp(3);

        // misc
        case Mode::FUSE_ADD_RELU: {
            return Halide::max(inp(0) + inp(1), cv(0));
        }
        case Mode::FUSE_ADD_SIGMOID:
            return cv(1) /
                   (cv(1) +
                    Halide::exp(-(inp(0) + inp(1))));
        case Mode::FUSE_ADD_TANH:
            return Halide::tanh(inp(0) + inp(1));
        case Mode::FUSE_ADD_H_SWISH:
            return (inp(0) + inp(1)) *
                   Halide::max(Halide::min((inp(0) + inp(1)) + cv(3), cv(6)),
                               cv(0)) /
                   cv(6);
        case Mode::FAST_TANH: {
            Halide::Expr e = Halide::fast_exp(inp(0)),
                         ei = Halide::fast_inverse(e);
            return (e - ei) / (e + ei);
        }
        case Mode::FAST_TANH_GRAD:
            return (cv(1) - inp(0) * inp(0)) * inp(1);
        case Mode::ROUND:
            return Halide::round(inp(0));
        case Mode::RMULH:
            return (inp(0) * inp(1)) >> Halide::popcount(inp(0));
        case Mode::NOT:
            return cv(1) - cv(inp(0) != cv(0));
        case Mode::AND:
            return cv(inp(0) != cv(0)) * cv(inp(1) != cv(0));
        case Mode::OR:
            return cv(cv(inp(0) != cv(0)) + cv(inp(1) != cv(0)) > cv(0));
        case Mode::XOR:
            return cv(cv(inp(0) != cv(0)) + cv(inp(1) != cv(0)) == cv(1));
        default:
            mgb_throw(InternalError, "unsupported Elemwise mode(%d)",
                      static_cast<int>(mode));
    }
#undef inp
}

Halide::Expr dispatch_powc(Halide::FuncRef inp, float exp) {
    if (almost_equal(exp, .0f)) {
        return Halide::cast(inp.function().output_types()[0], 1);
    }
    auto int_neg = [exp](Halide::Expr x) {
        if (exp < 0) {
            return Halide::cast(x.type(), 1) / x;
        }
        return x;
    };
    if (almost_equal(std::abs(exp), 1.f)) {
        return int_neg(inp);
    }
    if (almost_equal(std::abs(exp), 2.f)) {
        return int_neg(inp * inp);
    }
    if (almost_equal(std::abs(exp), 3.f)) {
        return int_neg(inp * inp * inp);
    }
    if (almost_equal(std::abs(exp), 4.f)) {
        auto x = inp * inp;
        return int_neg(x * x);
    }

    if (almost_equal(exp, .5f)) {
        return Halide::sqrt(inp);
    }

    int exp_i = std::round(exp);
    if (almost_equal(static_cast<float>(exp_i), exp)) {
        auto yabs = Halide::pow(Halide::abs(inp), exp);
        if (exp_i & 1) {
            return Halide::select(inp < 0, -yabs, yabs);
        } else {
            return yabs;
        }
    }

    return Halide::pow(inp, exp);
}

}  // anonymous namespace

AstNodePtr ast_hl::make_from_opr(cg::OperatorNodeBase* opr) {
    if (SymbolVar{opr->output(0)}.as_immutable_scalar().valid()) {
        return std::make_shared<ScalarImmOp>();
    }
    auto type = opr->dyn_typeinfo();
    if (type == opr::Elemwise::typeinfo() || type == opr::PowC::typeinfo()) {
        return std::make_shared<ElemwiseOp>();
    }
    if (type == opr::TypeCvt::typeinfo()) {
        return std::make_shared<TypeCvtOp>();
    }
    if (type == opr::Reduce::typeinfo()) {
        return std::make_shared<ReduceOp>();
    }
    if (type == opr::Broadcast::typeinfo()) {
        return std::make_shared<BroadcastOp>();
    }
    mgb_throw(InternalError, "invalid JIT operator type: %s", type->name);
}

/* =================== InputHostValueShapeOp =================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(InputHostValueShapeOp);
void InputHostValueShapeOp::init(cg::OperatorNodeBase* opr) {}

/* =================== InputDevValueOp =================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(InputDevValueOp);
void InputDevValueOp::init(cg::OperatorNodeBase* opr) {
    int ndim = m_buffer.raw_buffer()->dimensions;
    std::vector<Halide::Var> vars(ndim);
    std::vector<Halide::Expr> exps(ndim);
    for (int i = 0; i < ndim; i++) {
        exps[i] = vars[i];
    }
    m_func(vars) = m_buffer(exps);
}

/* =================== ElemwiseOp =================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ElemwiseOp);
void ElemwiseOp::init(cg::OperatorNodeBase* opr) {
    megdnn::TensorLayout out_layout;
    megdnn::TensorLayoutArray inp_layouts;
    megdnn::TensorShapeArray inp_shapes;
    mgb::SmallVector<int> orig_dim;
    for (auto inp : m_inputs) {
        inp_layouts.push_back(inp->m_layout);
        inp_shapes.push_back(inp->m_layout);
        orig_dim.push_back(inp->m_layout.ndim);
    }
    megdnn::TensorShape out_shape;

    megdnn::Elemwise::deduce_shape(inp_shapes, out_shape);
    out_layout = {out_shape, opr->output()[0]->dtype()};
    out_layout.init_contiguous_stride();
    for (auto inp : inp_layouts) {
        inp = inp.broadcast(out_layout);
    }
    m_layout = out_layout;

    std::vector<Halide::Var> out_vars;
    int dim = out_layout.ndim;
    for (int i = dim - 1; i >= 0; i--) {
        out_vars.emplace_back(Halide::Var(ssprintf("d%d", i)));
    }
    std::vector<std::vector<Halide::Expr>> exprs_of_inps;
    for (size_t i = 0; i < inp_layouts.size(); i++) {
        if (inp_layouts[i].is_scalar()) {
            exprs_of_inps.push_back({Halide::Expr{0}});
        } else {
            megdnn::TensorLayout& layout = inp_layouts[i];
            int odim = orig_dim[i];
            int cur_dim = layout.ndim;
            std::vector<Halide::Expr> exprs(odim);
            mgb_assert(static_cast<int>(cur_dim) >= odim);
            for (int j = cur_dim - 1; j >= cur_dim - odim; j--) {
                if (inp_layouts[i].shape[j] != 1 &&
                    inp_layouts[i].stride[j] != 0) {
                    exprs[cur_dim - 1 - j] = out_vars[cur_dim - 1 - j];
                } else {
                    exprs[cur_dim - 1 - j] = 0;
                }
            }
            exprs_of_inps.push_back(exprs);
        }
    }

    Halide::Expr out;

    if (auto powc = gopt::try_cast_as_op<opr::PowC>(opr)) {
        out = dispatch_powc(m_inputs[0]->m_func(exprs_of_inps[0]),
                            powc->param().exp);
    } else {
        out = dispatch_elemwise_mode(
                opr->cast_final_safe<opr::Elemwise>().param().mode, m_inputs,
                out_layout.dtype, exprs_of_inps);
    }
    m_func(out_vars) = out;
}

/* =================== TypeCvtOp =================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(TypeCvtOp);
void TypeCvtOp::init(cg::OperatorNodeBase* opr) {
    auto&& type_cvt = opr->cast_final_safe<opr::TypeCvt>();
    mgb_assert(type_cvt.input().size() == 1 && m_inputs.size() == 1);
    m_layout = m_inputs[0]->m_layout;
    int ndim = m_layout.ndim;
    std::vector<Halide::Var> out_vars(ndim);
    std::vector<Halide::Expr> exprs;
    for (auto var : out_vars) {
        exprs.emplace_back(var);
    }
    auto dtype = type_cvt.output()[0]->dtype();
    m_func(out_vars) = hl_type_cast(m_inputs[0]->m_func(exprs), dtype);
    m_layout.dtype = dtype;
}

/* =================== ReduceOp =================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ReduceOp);

namespace {
namespace reduce_impl {

using Halide::FuncRef;
using Halide::Type;
using Halide::cast;

using Mode = opr::Reduce::Mode;
template <Mode, typename otype, typename ctype>
struct Trait;

#define TRAIT_IMPL_BEGIN(mode)                \
    template <typename otype, typename ctype> \
    struct Trait<mode, otype, ctype>

#define TRAIT_IMPL_COMMON                           \
    using hl_comp_type = ctype_to_hl_type_t<ctype>; \
    using hl_out_type = ctype_to_hl_type_t<otype>

TRAIT_IMPL_BEGIN(Mode::SUM) {
    TRAIT_IMPL_COMMON;

    static void init(FuncRef comp) { comp = cast<hl_comp_type>(0); }
    static void apply(FuncRef comp, Halide::FuncRef in, float) {
        comp += cast<hl_comp_type>(in);
    }
    static void on(FuncRef comp, FuncRef func) {
        func = cast<hl_out_type>(comp);
    }
};

TRAIT_IMPL_BEGIN(Mode::SUM_SQR) {
    TRAIT_IMPL_COMMON;

    static void init(FuncRef comp) { comp = cast<hl_comp_type>(0); }
    static void apply(FuncRef comp, FuncRef in, float) {
        comp += cast<hl_comp_type>(in * in);
    }
    static void on(FuncRef comp, FuncRef func) {
        func = cast<hl_out_type>(comp);
    }
};

TRAIT_IMPL_BEGIN(Mode::PRODUCT) {
    TRAIT_IMPL_COMMON;

    static void init(FuncRef comp) { comp = cast<hl_comp_type>(1); }
    static void apply(FuncRef comp, FuncRef in, float) {
        comp *= cast<hl_comp_type>(in);
    }
    static void on(FuncRef comp, FuncRef func) {
        func = cast<hl_out_type>(comp);
    }
};

TRAIT_IMPL_BEGIN(Mode::MAX) {
    TRAIT_IMPL_COMMON;

    static void init(FuncRef comp) {
        comp = cast<hl_comp_type>(Type{halide_type_of<hl_comp_type>()}.min());
    }
    static void apply(FuncRef comp, FuncRef in, float) {
        comp = cast<hl_comp_type>(max(comp, in));
    }
    static void on(FuncRef comp, FuncRef func) {
        func = cast<hl_out_type>(comp);
    }
};

TRAIT_IMPL_BEGIN(Mode::MIN) {
    TRAIT_IMPL_COMMON;

    static void init(FuncRef comp) {
        comp = cast<hl_comp_type>(Type{halide_type_of<hl_comp_type>()}.max());
    }
    static void apply(FuncRef comp, FuncRef in, float) {
        comp = cast<hl_comp_type>(min(comp, in));
    }
    static void on(FuncRef comp, FuncRef func) {
        func = cast<hl_out_type>(comp);
    }
};

TRAIT_IMPL_BEGIN(Mode::MEAN) {
    TRAIT_IMPL_COMMON;

    static void init(FuncRef comp) { comp = cast<hl_comp_type>(0); }
    static void apply(FuncRef comp, FuncRef in, float scale) {
        comp += cast<hl_comp_type>(in) / cast<hl_comp_type>(scale);
    }
    static void on(FuncRef comp, FuncRef func) {
        func = cast<hl_out_type>(comp);
    }
};

template <typename otype, typename ctype>
void dispatch_reduce_mode(Mode mode, FuncRef func, FuncRef comp, FuncRef in,
                          float scale) {
    using Mode = opr::Reduce::Mode;
#define cb(mode)                                           \
    case mode: {                                           \
        Trait<mode, otype, ctype>::init(comp);             \
        Trait<mode, otype, ctype>::apply(comp, in, scale); \
        Trait<mode, otype, ctype>::on(comp, func);         \
        break;                                             \
    }
    switch (mode) {
        cb(Mode::SUM) cb(Mode::SUM_SQR) cb(Mode::PRODUCT) cb(Mode::MAX)
                cb(Mode::MIN) cb(Mode::MEAN) default
                : mgb_throw(InternalError, "invalide reduce mode");
    }
#undef cb
}

}  // namespace reduce_impl
}  // anonymous namespace

void ReduceOp::init(cg::OperatorNodeBase* opr) {
    auto&& mgb_reduce_op = opr->cast_final_safe<opr::Reduce>();
    mgb_assert(mgb_reduce_op.output(0)->dtype().category() ==
                       megdnn::DTypeCategory::FLOAT,
               "invalid Reduce opr or dtype of output is not float32/float16");
    auto dtype = mgb_reduce_op.output(0)->dtype();
    if (m_inputs.size() == 1) {
        m_layout = m_inputs[0]->m_layout;
        int axis = mgb_reduce_op.param().axis;
        mgb_assert(axis >= 0 && axis < static_cast<int>(m_layout.ndim));
        m_layout[axis] = 1;
    } else if (m_inputs[1]->same_type<InputHostValueShapeOp>()) {
        m_layout.dtype = dtype;
        m_layout.init_contiguous_stride(m_inputs[1]->m_layout);
    } else if (auto imm = try_cast_as_op<ScalarImmOp>(m_inputs[1].get())) {
        int const_val = imm->m_val.iv;
        mgb_assert(const_val == 1,
                   "reduce target shape should be scalar, got %d", const_val);
        m_layout = {{static_cast<size_t>(const_val)},
                    mgb_reduce_op.output(0)->dtype()};
    } else {
        mgb_throw(InternalError,
                  "invalid input for Halide ReduceOp, inp size = %zu",
                  m_inputs.size());
    }

    auto&& inp_layout = m_inputs[0]->m_layout;
    // equivalent output layout, expanding scalar reduction to full ndim
    auto out_layout = m_layout;
    if (out_layout.is_scalar()) {
        out_layout.ndim = inp_layout.ndim;
        for (size_t i = 0; i < out_layout.ndim; ++i) {
            out_layout.shape[i] = 1;
        }
    }

    using DataType = opr::Reduce::Param::DataType;
    using Expr = Halide::Expr;
    using RDom = Halide::RDom;
    using Var = Halide::Var;
    using namespace reduce_impl;

    mgb_assert(inp_layout.ndim == out_layout.ndim,
               "ndim of orig shape and target shape for reduce opr mismatch, "
               "inp = %zu, out = %zu",
               inp_layout.ndim, out_layout.ndim);
    std::vector<Var> out_vars(out_layout.ndim);
    std::vector<std::pair<Expr, Expr>> ranges;
    bool need_do_reduce = false;
    for (int i = static_cast<int>(out_layout.ndim) - 1; i >= 0; i--) {
        if (out_layout[i] != inp_layout[i]) {
            mgb_assert(out_layout[i] == 1);
            need_do_reduce = true;
            ranges.push_back(std::make_pair(
                    Expr{0}, Expr{static_cast<int>(inp_layout[i])}));
        }
    }
    Halide::Func out_func;
    if (need_do_reduce) {
        RDom rvars{ranges};
        int ridx = 0;
        std::vector<Halide::Expr> exprs;
        for (int i = static_cast<int>(out_layout.ndim) - 1; i >= 0; i--) {
            if (out_layout[i] == inp_layout[i]) {
                exprs.emplace_back(out_vars[out_layout.ndim - 1 - i]);
            } else {
                exprs.emplace_back(rvars[ridx++]);
            }
        }
        float scale = inp_layout.total_nr_elems() / out_layout.total_nr_elems();
        switch (mgb_reduce_op.param().data_type) {
            case DataType::FLOAT_O32xC32:
            case DataType::DEFAULT: {
                if (dtype == dtype::Float16()) {
                    dispatch_reduce_mode<dt_float16, dt_float16>(
                            mgb_reduce_op.param().mode, out_func(out_vars),
                            m_comp(out_vars), m_inputs[0]->m_func(exprs),
                            scale);
                } else if (dtype == dtype::Float32()) {
                    dispatch_reduce_mode<float, float>(
                            mgb_reduce_op.param().mode, out_func(out_vars),
                            m_comp(out_vars), m_inputs[0]->m_func(exprs),
                            scale);
                }
                break;
            }
            case DataType::FLOAT_IO16xC32:
                mgb_log_warn(
                        "DataType::FLOAT_IO16xC32 has been deprecated, will "
                        "use FLOAT_O16xC32 instead");
                break;
            case DataType::FLOAT_O16xC32: {
                dispatch_reduce_mode<dt_float16, float>(
                        mgb_reduce_op.param().mode, out_func(out_vars),
                        m_comp(out_vars), m_inputs[0]->m_func(exprs), scale);
                break;
            }
            default:
                mgb_throw(InternalError, "invalid data type for reduce opr");
        }
    } else {
        std::vector<Halide::Expr> exprs;
        for (auto var : out_vars) {
            exprs.emplace_back(var);
        }
        Expr out_expr;
        if (dtype == dtype::Float16()) {
            out_expr = cast<Halide::float16_t>(m_inputs[0]->m_func(exprs));
        } else if (dtype == dtype::Float32()) {
            out_expr = cast<float>(m_inputs[0]->m_func(exprs));
        }
        if (mgb_reduce_op.param().mode == Mode::SUM_SQR) {
            // side effect of sum-sqr
            out_expr = out_expr * out_expr;
        }
        out_func(out_vars) = out_expr;
    }

    if (m_layout.ndim == 1 && out_layout.ndim != 1) {
        // reduce to scalar
        std::vector<Halide::Expr> exprs;
        for (size_t i = 0; i < out_layout.ndim; ++i) {
            exprs.push_back(0);
        }
        m_func(Var{}) = out_func(exprs);
    } else {
        m_func = out_func;
    }
}

/* =================== ScalarImmOp =================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ScalarImmOp);
void ScalarImmOp::init(cg::OperatorNodeBase* opr) {
    auto imm = SymbolVar{opr->output(0)}.as_immutable_scalar().val();
    auto dtype = imm.dtype();
    m_layout = {{1}, dtype};
    Halide::Var var;
    if (dtype == dtype::Int32()) {
        m_val.iv = imm.get<int>();
        m_func(var) = m_val.iv;
    } else if (dtype == dtype::Float32()) {
        m_val.fv = imm.get<float>();
        m_func(var) = m_val.fv;
    } else if (dtype == dtype::Float16()) {
        m_val.fv = imm.get<dt_float16>();
        m_func(var) = Halide::float16_t(m_val.fv);
    } else {
        mgb_throw(InternalError,
                  "dtype(%s) is not any of [float16, float32, int32]",
                  dtype.name());
    }
}

/* ================= BroadcastOp ===================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(BroadcastOp);
void BroadcastOp::init(cg::OperatorNodeBase* opr) {
    mgb_assert(m_inputs.size() == 2,
               "halide BroadcastOp should have two inputs");
    const TensorShape& tshape =
            m_inputs[1]->cast_final_safe<InputHostValueShapeOp>().m_layout;
    auto&& orig_layout = m_inputs[0]->m_layout;
    m_layout.dtype = orig_layout.dtype;
    m_layout.init_contiguous_stride(tshape);

    int ndim = m_layout.ndim;
    std::vector<Halide::Var> out_vars(ndim);
    std::vector<Halide::Expr> exprs;

    if (orig_layout.is_scalar()) {
        exprs.push_back(0);
    } else {
        mgb_assert(ndim && (orig_layout.ndim == m_layout.ndim));

        for (int i = ndim - 1; i >= 0; i++) {
            if (orig_layout[i] == m_layout[i]) {
                exprs.emplace_back(out_vars[i]);
            } else if (orig_layout[i] == 1) {
                exprs.emplace_back(0);
            } else {
                mgb_throw(InternalError,
                          "invalid boradcast shape: inpshp = %s, tshp = %s",
                          orig_layout.to_string().c_str(),
                          m_layout.to_string().c_str());
            }
        }
    }
    m_func(out_vars) = m_inputs[0]->m_func(exprs);
}

#endif  // MGB_JIT_HALIDE

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
