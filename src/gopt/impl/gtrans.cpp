/**
 * \file src/gopt/impl/gtrans.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/gopt/gtrans.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"

using namespace mgb;
using namespace gopt;
using namespace opr;

namespace {
    //! check whether *w* has shape-1 on non-channel axes
    bool check_conv_brd_shp(VarNode *w) {
        auto bshp = w->shape();
        if (!bshp.ndim)
            return false;
        for (size_t i = 0; i < bshp.ndim; ++ i) {
            if (i + 3 != bshp.ndim && bshp.shape[i] != 1) {
                // only allow non-broadcasting axis in channel
                return false;
            }
        }
        return true;
    }

    bool normalize_matmul_shape(
            const TensorShape &src, TensorShape &dst) {
        if (!src.ndim)
            return false;
        dst = src;
        if (src.ndim == 1) {
            ++ dst.ndim;
            dst[1] = dst[0];
            dst.shape[0] = 1;
        }
        mgb_assert(dst.ndim == 2);
        return true;
    }
} // anonymous namespace

/* ================ BinaryTrans20 ================ */

class BinaryTrans20::Rule {
    protected:
        ~Rule() = default;
    public:
        virtual const char* desc() = 0;
        virtual std::pair<Typeinfo*, Typeinfo*> types() = 0;
        virtual VarNode* apply(
                VarNode** internal,
                OperatorNodeBase *fop, OperatorNodeBase *gop,
                VarNode *a, VarNode *b, VarNode *c) = 0;
};

GTransResult BinaryTrans20::apply(
        OperatorNodeBase *fop, bool swap_fop_inp, bool swap_gop_inp) const {
    mgb_assert(fop->input().size() == 2);
    auto ab = fop->input(0), c = fop->input(1);
    if (swap_fop_inp) {
        mgb_assert(is_commutable_binary(fop));
        std::swap(ab, c);
    }
    auto gop = ab->owner_opr();
    mgb_assert(gop->input().size() == 2);
    auto a = gop->input(0), b = gop->input(1);
    if (swap_gop_inp) {
        mgb_assert(is_commutable_binary(gop));
        std::swap(a, b);
    }

    auto iter = m_rules.find({fop->dyn_typeinfo(), gop->dyn_typeinfo()});
    GTransResultItem ret;
    if (iter != m_rules.end()) {
        ret.result = iter->second->apply(
                ret.internal.data(), fop, gop, a, b, c);
        if (ret.result) {
            ret.msg = iter->second->desc();
            return ret;
        }
    }
    return None;
}

//! register a single rule class to given trans object
#define BINARY_TRANS_20_REG_RULE(_cls, t) \
    static _cls _cls##_ins; \
    do {  \
        auto ir = t.m_rules.insert({ \
                static_cast<Rule&>(_cls##_ins).types(), \
                &_cls##_ins}).second; \
        mgb_assert(ir); \
    } while(0)

class BinaryTrans20::AssociativeRuleReg {
    class ElemArith final: public Rule {
        using Mode = Elemwise::Mode;
        const char* desc() override {
            return mgb_cstr_log(
                    "elem(elem(x,w1),w2)->elem(x,elem(w1,w2))");
        }

        std::pair<Typeinfo*, Typeinfo*> types() override {
            return {Elemwise::typeinfo(), Elemwise::typeinfo()};
        }

        VarNode* apply(VarNode **internal,
                OperatorNodeBase *fop, OperatorNodeBase *gop,
                VarNode *a, VarNode *b, VarNode *c) override {
            auto bshp = b->shape(), cshp = c->shape();
            if (!bshp.ndim || !cshp.ndim)
                return nullptr;

            auto &&elem0 = fop->cast_final_safe<Elemwise>();
            auto &&elem1 = gop->cast_final_safe<Elemwise>();
            auto mode = elem0.param().mode;
            if (mode != elem1.param().mode)
                return nullptr;

            if (mode != Mode::ADD && mode != Mode::MUL &&
                    mode != Mode::MAX && mode != Mode::MIN) {
                return nullptr;
            }

            auto bcshp = Elemwise::get_output_var_shape(mode, {bshp, cshp});
            if (!bcshp.eq_shape(bshp) && !bcshp.eq_shape(cshp)) {
                // do not allow broadcast
                return nullptr;
            }


            return Elemwise::make(
                    {a,
                    internal[0] = Elemwise::make({b, c}, mode).node()},
                    mode).node();
        }
    };

    class ConvMul final: public Rule {
        const char* desc() override {
            return mgb_cstr_log("conv(x*k,w)->conv(x,w*k)");
        }

        std::pair<Typeinfo*, Typeinfo*> types() override {
            return {Convolution::typeinfo(), Elemwise::typeinfo()};
        }

        VarNode* apply(VarNode **internal,
                OperatorNodeBase *fop, OperatorNodeBase *gop,
                VarNode *a, VarNode *b, VarNode *c) override {
            if (!as_elem_opr(gop, Elemwise::Mode::MUL) ||
                    !check_conv_brd_shp(b))
                return nullptr;

            auto &&orig_conv = fop->cast_final_safe<Convolution>();

            SymbolVar k1 = b;
            if (orig_conv.param().sparse ==
                        opr::Convolution::Param::Sparse::GROUP &&
                !k1.shape().is_scalar()) {
                // group convolution with non-scalar multiplicand
                auto one = k1.make_scalar_dt(1);
                k1 = k1.reshape(
                        Concat::make({GetVarShape::make(c, 0), one,
                                      GetVarShape::make(c, 2), one, one},
                                     0));
            }
            return Convolution::make(
                    a, internal[0] = (k1 * SymbolVar{c}).node(),
                    orig_conv.param(), orig_conv.execution_policy()).node();
        }
    };
    class MatmulMul final: public Rule {
        const char* desc() override {
            return mgb_cstr_log("matmul(x*k,w)->matmul(x,w*k)");
        }

        std::pair<Typeinfo*, Typeinfo*> types() override {
            return {MatrixMul::typeinfo(), Elemwise::typeinfo()};
        }

        VarNode* apply(VarNode **internal,
                OperatorNodeBase *fop, OperatorNodeBase *gop,
                VarNode *a, VarNode *b, VarNode *c) override {
            if (!as_elem_opr(gop, Elemwise::Mode::MUL))
                return nullptr;
            auto &&mm = fop->cast_final_safe<opr::MatrixMul>();
            // axis that must be broadcasting
            TensorShape bshp;
            if (!normalize_matmul_shape(b->shape(), bshp) ||
                    bshp[!!mm.param().transposeA] != 1)
                return nullptr;

            SymbolVar tb{b};
            tb = tb.flatten();
            if (mm.param().transposeB) {
                tb = opr::Dimshuffle::make(tb, {-1, 0});
            } else {
                tb = opr::Dimshuffle::make(tb, {0, -1});
            }
            return MatrixMul::make(a, internal[0] = (tb * c).node(),
                    mm.param()).node();
        }
    };
    class MulConv final: public Rule {
        const char* desc() override {
            return mgb_cstr_log("conv(x,w)*k->conv(x,w*k)");
        }

        std::pair<Typeinfo*, Typeinfo*> types() override {
            return {Elemwise::typeinfo(), Convolution::typeinfo()};
        }

        VarNode* apply(VarNode **internal,
                OperatorNodeBase *fop, OperatorNodeBase *gop,
                VarNode *a, VarNode *b, VarNode *c) override {
            if (!as_elem_opr(fop, Elemwise::Mode::MUL) ||
                    !check_conv_brd_shp(c))
                return nullptr;

            SymbolVar k1{c};

            auto &&orig_conv = gop->cast_final_safe<Convolution>();
            if (orig_conv.param().sparse ==
                    opr::Convolution::Param::Sparse::GROUP) {
                auto one = k1.make_scalar_dt(1);
                auto tshp = opr::Concat::make(
                            {GetVarShape::make(b, 0), GetVarShape::make(b, 1),
                            one, one, one}, 0);
                if (k1.shape().is_scalar()) {
                    k1 = k1.broadcast(tshp);
                } else {
                    k1 = k1.reshape(tshp);
                }

            } else {
                // reshape to [-1, 1, 1, 1]
                k1 = Reshape::make(k1, TensorShape{1, 1, 1, 1}, 0);
            }

            return Convolution::make(
                    a, internal[0] = (k1 * b).node(),
                    orig_conv.param(), orig_conv.execution_policy()).node();
        }
    };
    class MulMatmul final: public Rule {
        const char* desc() override {
            return mgb_cstr_log("matmul(x,w)*k->matmul(x,w*k)");
        }

        std::pair<Typeinfo*, Typeinfo*> types() override {
            return {Elemwise::typeinfo(), MatrixMul::typeinfo()};
        }

        VarNode* apply(VarNode **internal,
                OperatorNodeBase *fop, OperatorNodeBase *gop,
                VarNode *a, VarNode *b, VarNode *c) override {
            if (!as_elem_opr(fop, Elemwise::Mode::MUL))
                return nullptr;
            TensorShape cshp;
            if (!normalize_matmul_shape(c->shape(), cshp) ||
                    cshp[0] != 1)
                return nullptr;

            auto &&mm = gop->cast_final_safe<opr::MatrixMul>();

            SymbolVar tc{c};
            tc = tc.flatten();
            if (mm.param().transposeB) {
                tc = opr::Dimshuffle::make(tc, {0, -1});
            } else {
                tc = opr::Dimshuffle::make(tc, {-1, 0});
            }
            return MatrixMul::make(a, internal[0] = (tc * b).node(),
                    mm.param()).node();
        }
    };

    public:
        AssociativeRuleReg(BinaryTrans20 &t) {
            BINARY_TRANS_20_REG_RULE(ElemArith, t);
            BINARY_TRANS_20_REG_RULE(ConvMul, t);
            BINARY_TRANS_20_REG_RULE(MatmulMul, t);
            BINARY_TRANS_20_REG_RULE(MulConv, t);
            BINARY_TRANS_20_REG_RULE(MulMatmul, t);
        }
};

class BinaryTrans20::DistributiveAddRuleReg {

    class ConvAdd final: public Rule {
        const char* desc() override {
            return mgb_cstr_log("conv(x+b,w)->conv(x,w)+b1");
        }

        std::pair<Typeinfo*, Typeinfo*> types() override {
            return {Convolution::typeinfo(), Elemwise::typeinfo()};
        }

        VarNode* apply(VarNode **internal,
                OperatorNodeBase *fop, OperatorNodeBase *gop,
                VarNode *a, VarNode *b, VarNode *c) override {
            if (!as_elem_opr(gop, Elemwise::Mode::ADD) ||
                    !check_conv_brd_shp(b))
                return nullptr;

            auto &&orig_conv = fop->cast_final_safe<Convolution>();
            auto &&param = orig_conv.param();
            if (param.pad_h || param.pad_w)
                return nullptr;

            internal[0] = Convolution::make(
                    a, c, param, orig_conv.execution_policy()).node();

            if (param.sparse == opr::Convolution::Param::Sparse::GROUP) {
                // group conv
                SymbolVar b1 = b, one = b1.make_scalar(1);
                b1 = b1.reshape(Concat::make(
                            {GetVarShape::make(c, 0), one,
                            GetVarShape::make(c, 2), one, one}, 0));
                b1 = b1 * c;
                b1 = reduce_sum(b1, Concat::make(
                            {GetVarShape::make(c, 0), GetVarShape::make(c, 1),
                            one, one, one}, 0));
                b1 = Reshape::make(b1, TensorShape{1, 1, 1, 1}, 1);
                return (b1 + internal[0]).node();
            }

            // dense conv
            SymbolVar b0{b}, w{c},
                      b1 = b0 * w,
                      ochl = GetVarShape::make(w, 0),
                      b1_tshp = Concat::make({ochl, ochl.make_scalar_dt(1)}, 0);
            b1 = Reshape::make(b1, b1_tshp, 1);
            b1 = Reduce::make(b1, {Reduce::Param::Mode::SUM}, b1_tshp);
            return (Dimshuffle::make(b1, {-1, 0, 1, -1}) + internal[0]).node();
        }
    };
    class MatmulAdd final: public Rule {
        const char* desc() override {
            return mgb_cstr_log("matmul(x+b,w)->conv(x,w)+b1");
        }

        std::pair<Typeinfo*, Typeinfo*> types() override {
            return {MatrixMul::typeinfo(), Elemwise::typeinfo()};
        }

        VarNode* apply(VarNode **internal,
                OperatorNodeBase *fop, OperatorNodeBase *gop,
                VarNode *a, VarNode *b, VarNode *c) override {
            if (!as_elem_opr(gop, Elemwise::Mode::ADD))
                return nullptr;
            TensorShape bshp;
            auto &&mm = fop->cast_final_safe<MatrixMul>();
            if (!normalize_matmul_shape(b->shape(), bshp) ||
                    bshp[!!mm.param().transposeA] != 1)
                return nullptr;
            auto &&cshp = c->shape();
            if (!cshp.ndim)
                return nullptr;

            bshp[!mm.param().transposeA] = cshp[!!mm.param().transposeB];

            auto bias = MatrixMul::make(
                    SymbolVar{b}.broadcast(bshp), c, mm.param());
            if (bias.shape().ndim)
                mgb_assert(bias.shape()[0] == 1);
            internal[0] = MatrixMul::make(a, c, mm.param()).node();
            return (bias + internal[0]).node();
        }
    };

    public:

        DistributiveAddRuleReg(BinaryTrans20 &t) {
            BINARY_TRANS_20_REG_RULE(ConvAdd, t);
            BINARY_TRANS_20_REG_RULE(MatmulAdd, t);
        }
};

BinaryTrans20& BinaryTrans20::associtive() {
    static BinaryTrans20 trans;
    static AssociativeRuleReg rule{trans};
    return trans;
}

BinaryTrans20& BinaryTrans20::distributive_add() {
    static BinaryTrans20 trans;
    static DistributiveAddRuleReg rule{trans};
    return trans;
}

/* ================ misc  ================ */

namespace mgb {
namespace gopt {

template <>
VarNode* check_is_group_inverse_opr<Elemwise::Mode::MUL>(SymbolVar x) {
    auto opr = x.node()->owner_opr();
    auto elem = as_elem_opr(opr, Elemwise::Mode::POW);
    if (!elem) {
        if (auto powc = try_cast_as_op<opr::PowC>(opr)) {
            if (almost_equal(powc->param().exp, -1.f)) {
                return powc->input(0);
            }
        }
        return nullptr;
    }
    auto exp = SymbolVar{elem->input(1)}.as_immutable_scalar_require_shape();
    if (exp.valid() && almost_equal(exp->get_cast<float>(), -1.f)) {
        return opr->input(0);
    }
    return nullptr;
}

} // namespace gopt
} // namespace mgb

VarNode* gopt::elemwise_reduce_var_list(
        const VarNodeArray &vars, opr::Elemwise::Mode mode,
        VarNodeArray *mid_results) {
    mgb_assert(!vars.empty());
    VarNode *s = vars[0];
    for (size_t i = 1; i < vars.size(); ++ i) {
        s = Elemwise::make({s, vars[i]}, mode).node();
        if (mid_results)
            mid_results->push_back(s);
    }
    return s;
}

VarNode* gopt::get_opr_single_output_var(OperatorNodeBase *opr) {
    VarNode *ret = nullptr;
    for (auto i: opr->output()) {
        if (!i->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
            if (ret)
                return nullptr;
            ret = i;
        }
    }
    return ret;
}

void gopt::visit_opr_tree(
        VarNode *endpoint,
        const thin_function<bool(VarNode*)> &check_internal,
        const thin_function<void(VarNode*)> &on_leaf,
        const thin_function<void(OperatorNodeBase*)> &on_internal_finish,
        bool allow_multi_cn) {
    struct StackFrame {
        using VarNodeArrayPtr = VarNode * const *;
        VarNodeArrayPtr var0, var1;
        OperatorNodeBase *reader_opr;
    };
    std::vector<StackFrame> stack;
    stack.push_back({&endpoint, &endpoint + 1, nullptr});
    while (!stack.empty()) {
        auto &&top = stack.back();
        if (top.var0 == top.var1) {
            if (top.reader_opr && on_internal_finish) {
                on_internal_finish(top.reader_opr);
            }
            stack.pop_back();
            continue;
        }
        VarNode* var = *(top.var0 ++);
        if (check_internal(var)) {
            auto opr = var->owner_opr();
            mgb_assert(var == opr->output(0),
                    "bad check_internal() provided to visit_opr_tree");
            auto &&inp = opr->input();
            if (!allow_multi_cn) {
                bool multi_cn = false;
                for (auto i: inp) {
                    if (i->comp_node() != var->comp_node()) {
                        multi_cn = true;
                        break;
                    }
                }
                if (multi_cn) {
                    if (on_leaf)
                        on_leaf(var);
                    continue;
                }
            }
            stack.push_back({inp.data(), inp.data() + inp.size(), opr});
        } else {
            if (on_leaf)
                on_leaf(var);
        }
    }
}

VarNodeArray gopt::extract_opr_leaves(
        VarNode *endpoint, const std::function<bool(OperatorNodeBase*)> &pred,
        bool allow_multi_cn) {
    VarNodeArray ret;
    auto check_internal = [&](VarNode *var) -> bool {
        return pred(var->owner_opr());
    };
    auto on_leaf = [&ret](VarNode *var) {
        ret.push_back(var);
    };
    visit_opr_tree(endpoint, check_internal, on_leaf);
    return ret;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

