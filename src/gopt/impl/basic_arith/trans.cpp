/**
 * \file src/gopt/impl/basic_arith/trans.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/gopt/basic_arith.h"
#include "megbrain/serialization/serializer.h"

//! TODO: here has to be know some megdnn::opr when there is produced midout.h
//! fix it if there is another graceful way.
#include "megdnn/oprs.h"

#include "megbrain/utils/hash_ct.h"
#include "midout.h"

MIDOUT_DECL(megbrain_trans)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_trans, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace gopt;

namespace {
    /*!
     * \brief helper for implementing term-rewriting for elemwise oprs
     */
    class ElemwiseRewriteImplHelper {
        void on_opr(cg::OperatorNodeBase *opr);

        protected:
            using Elemwise = opr::Elemwise;
            using Mode = Elemwise::Mode;

            /*!
             * \brief a node in elemwise chain
             *
             * An elemwise chain is a flattened tree represented as a postfix
             * expression to preserve original tree structure, consisting of
             * elemwise oprs
             */
            struct ElemwiseChainNode;
            using ElemwiseChain = std::vector<ElemwiseChainNode>;

            const Pass &m_pass;
            OptState &m_opt_state;
            SubGraph::Rewriter m_rewriter;
            UniqReaderCheck m_uniq_reader_check;

            ElemwiseRewriteImplHelper(const Pass &pass, OptState &opt_state):
                m_pass{pass}, m_opt_state{opt_state},
                m_rewriter{opt_state.graph().make_rewriter()},
                m_uniq_reader_check{opt_state.graph()}
            {
            }

            ~ElemwiseRewriteImplHelper() noexcept = default;

            /*!
             * \brief callback when an Elemwise operator is visited
             * \param elem the operator on ORIGINAL graph
             */
            virtual void on_opr_elemwise(Elemwise *elem) = 0;

            //! whether a var node can be replaced
            bool can_replace_var(VarNode *var) const {
                return m_uniq_reader_check(var);
            }

            /*!
             * \brief run the rewriter
             *
             * call on_opr_elemwise() for all elemwise oprs and apply m_rewriter
             */
            void run_elemwise_rewriter();

            /*!
             * \brief extract an elemwise chain where all internal nodes share
             *      the given mode and are replaceable
             *
             * Note: must consult can_replace_var() before replacing a leaf var
             */
            ElemwiseChain extract_elemwise_chain(VarNode *endpoint, Mode mode);

            //! reconstruct a var from given elemwise chain
            VarNode* reconstruct_elemwise_chain(const ElemwiseChain &chain);
    };
}

/* ================ ElemwiseRewriteImplHelper ================ */
struct ElemwiseRewriteImplHelper::ElemwiseChainNode {
    enum class Type {
        LEAF, INTERNAL
    };
    Type type;
    union {
        VarNode *leaf;
        Mode mode;
    } data;

    static ElemwiseChainNode make_leaf(VarNode *var) {
        ElemwiseChainNode ret;
        ret.type = Type::LEAF;
        ret.data.leaf = var;
        return ret;
    }

    static ElemwiseChainNode make_internal(Mode mode) {
        ElemwiseChainNode ret;
        ret.type = Type::INTERNAL;
        ret.data.mode = mode;
        return ret;
    }
};

void ElemwiseRewriteImplHelper::run_elemwise_rewriter() {
    using namespace std::placeholders;
    m_opt_state.graph().iter(std::bind(
                &ElemwiseRewriteImplHelper::on_opr, this, _1));
    m_rewriter.apply_inplace();
}

void ElemwiseRewriteImplHelper::on_opr(OperatorNodeBase *opr) {
    m_uniq_reader_check.update_on_opr_auto_replace(
            opr, m_rewriter.auto_replace_outputs(opr));

    if (auto elem = try_cast_as_op<Elemwise>(opr)) {
        on_opr_elemwise(elem);
    }
}

ElemwiseRewriteImplHelper::ElemwiseChain
ElemwiseRewriteImplHelper::extract_elemwise_chain(
        VarNode *endpoint, Mode mode) {
    ElemwiseChain ret;
    auto check_internal = [mode, this](VarNode *var) -> bool {
        return as_elem_opr(var, mode) && can_replace_var(var);
    };
    auto on_leaf = [&ret](VarNode *var) {
        ret.push_back(ElemwiseChainNode::make_leaf(var));
    };
    auto on_internal_finish = [&ret](OperatorNodeBase *opr) {;
        ret.push_back(ElemwiseChainNode::make_internal(
                    opr->cast_final_safe<Elemwise>().param().mode));
    };
    visit_opr_tree(endpoint, check_internal, on_leaf, on_internal_finish);
    return ret;
}

VarNode* ElemwiseRewriteImplHelper::reconstruct_elemwise_chain(
        const ElemwiseChain &chain) {
    VarNodeArray stack;
    SymbolVarArray tmp_inp;
    for (auto &&i: chain) {
        if (i.type == ElemwiseChainNode::Type::LEAF) {
            stack.push_back(i.data.leaf);
        } else {
            mgb_assert(i.type == ElemwiseChainNode::Type::INTERNAL);
            auto mode = i.data.mode;
            auto arity = megdnn::Elemwise::ModeTrait::from_mode(mode).arity;
            mgb_assert(arity <= stack.size());
            tmp_inp.resize(arity);
            for (size_t i = 0; i < arity; ++ i) {
                tmp_inp[i] = stack[stack.size() - arity + i];
            }
            stack.resize(stack.size() - arity);
            stack.push_back(Elemwise::make(tmp_inp, mode).node());
        }
    }
    mgb_assert(stack.size() == 1);
    return stack[0];
}

/* ================ ArithMulDistributePass ================ */

class ArithMulDistributePass::Impl final: public ElemwiseRewriteImplHelper {
    //! total size reduced by prev try_distribute() call
    size_t m_eliminated_computing;

    void on_opr_elemwise(Elemwise *elem) override;

    /*!
     * \brief try to distribute \p mul over terms in \p add_endpoint
     *
     * The given vars must reside on original graph.
     *
     * \return transformed var for \p add_endpoint times \p mul, or nullptr if
     *      failed
     */
    VarNode* try_distribute(VarNode *add_endpoint, VarNode *mul);

    public:
        Impl(const ArithMulDistributePass &pass, OptState &opt_state):
            ElemwiseRewriteImplHelper(pass, opt_state)
        {
            run_elemwise_rewriter();
        }
};

void ArithMulDistributePass::Impl::on_opr_elemwise(Elemwise *elem) {
    if (elem->param().mode != Mode::MUL)
        return;
    auto i0 = elem->input(0), i1 = elem->input(1), out = elem->output(0);
    auto &&shp0 = i0->shape(), &&shp1 = i1->shape(),
         &&oshp = out->shape();
    auto sz0 = shp0.total_nr_elems(), sz1 = shp1.total_nr_elems();
    if (!oshp.ndim || sz0 == sz1 || !(
                oshp.eq_shape(shp0) || oshp.eq_shape(shp1))) {
        return;
    }
    if (sz0 < sz1) {
        std::swap(i0, i1);
    }

    if (auto end = try_distribute(i0, i1)) {
        m_rewriter.replace_var(out, end,
                mgb_ssprintf_log(
                    "%zu less elemwise-computing",
                    m_eliminated_computing).c_str());
    }
}

VarNode* ArithMulDistributePass::Impl::try_distribute(
        VarNode *add_endpoint, VarNode *mul) {
    TensorShapeArray check_compatible_inp(2);
    check_compatible_inp[0] = mul->shape();
    auto shape_compatible = [&](VarNode *var) {
        check_compatible_inp[1] = var->shape();
        TensorShape tshp;
        megdnn::Elemwise::deduce_shape(check_compatible_inp, tshp);
        return tshp.eq_shape(var->shape());
    };

    mul = m_rewriter.get_var(mul);
    auto add_chain = extract_elemwise_chain(add_endpoint, Mode::ADD);

    // mul chain, combine position in mul chain
    std::vector<std::pair<ElemwiseChain, size_t>> terms;

    using Type = ElemwiseChainNode::Type;

    if (add_chain.size() < 3)
        return nullptr;

    m_eliminated_computing = add_endpoint->shape().total_nr_elems();
    for (auto &&term: add_chain) {
        if (term.type != Type::LEAF)
            continue;
        auto mul_chain = extract_elemwise_chain(term.data.leaf, Mode::MUL);
        size_t best_pos = 0, best_size = m_eliminated_computing;
        // find smallest compatible var in mul_chain
        for (size_t i = 0; i < mul_chain.size(); ++ i) {
            if (mul_chain[i].type == Type::LEAF) {
                auto var = m_rewriter.get_var(mul_chain[i].data.leaf);
                mul_chain[i].data.leaf = var;
                if (shape_compatible(var)) {
                    auto size = var->shape().total_nr_elems();
                    if (size < best_size) {
                        best_size = size;
                        best_pos = i;
                    }
                }
            }
        }
        if (best_size == m_eliminated_computing) {
            return nullptr;
        }
        m_eliminated_computing -= best_size;
        terms.push_back({{}, best_pos});
        terms.back().first = std::move(mul_chain);
    }

    auto mul_chain_iter = terms.begin();
    for (auto &&term: add_chain) {
        if (term.type != Type::LEAF)
            continue;
        auto &&var = mul_chain_iter->first[mul_chain_iter->second].data.leaf;
        var = (SymbolVar{var} * mul).node();
        term.data.leaf = reconstruct_elemwise_chain(mul_chain_iter->first);
        ++ mul_chain_iter;
    }
    mgb_assert(mul_chain_iter == terms.end());

    return reconstruct_elemwise_chain(add_chain);
}

const char* ArithMulDistributePass::name() const {
    return mgb_cstr_log("mul_distribute");
}

void ArithMulDistributePass::apply(OptState &opt) const {
    MIDOUT_B("ArithMulDistributePass::apply")
    Impl{*this, opt};
    MIDOUT_E
}

/* ================ FinalArithTransformPass ================ */

class FinalArithTransformPass::Impl final:
        public ElemwiseRewriteImplHelper, public NonCopyableObj {
    using DispatchEntry = std::pair<
        thin_function<SymbolVar(const VarNodeArray &)>,
        const char*>;

    //! for merge_negate() with ADD/SUB modes
    struct MergeNegateAddTrait;
    //! for merge_negate() with MUL/TRUE_DIV modes
    struct MergeNegateMulTrait;

    ThinHashMap<Mode, std::vector<DispatchEntry>> m_dispatch_table;

    /*!
     * \brief get neg src; also set var to current var
     * \tparam mode either Mode::ADD or Mode::MUL, to define the inv-group
     * \param[in,out] var input original var, and output currently replaced var
     * \return x if var == -x and can_replace_var(var); nullptr otherwise
     */
    template<Mode mode>
    VarNode* get_neg_repl(VarNode *&var, bool require_replaceable) const;

    /*!
     * \brief try to decompose var as a * b if var is replaceable
     *
     * Also requires shapes of a and b are known.
     *
     * \param[in,out] var input original var, and output replaced var
     * \param[out] a decomposed first term; nullptr if can not decompose
     * \param[out] b decomposed second term; nullptr if can not decompose
     */
    void as_replaceable_mul(VarNode *&var, VarNode *&a, VarNode *&b);

    /*!
     * \brief merge negate operators like (-a) + (-b) -> -(a+b)
     * \tparam Trait provides ADD, SUB and neg()
     */
    template<class Trait>
    SymbolVar merge_negate(const VarNodeArray &inp);

    void init_dispatch_table();
    void on_opr_elemwise(Elemwise *elem) override;

    public:
        Impl(const FinalArithTransformPass &pass, OptState &opt_state):
            ElemwiseRewriteImplHelper(pass, opt_state)
        {
            init_dispatch_table();
            run_elemwise_rewriter();
        }
};

void FinalArithTransformPass::Impl::on_opr_elemwise(Elemwise *elem) {
    auto mode = elem->param().mode;
    auto &&iter = m_dispatch_table.find(mode);
    if (iter != m_dispatch_table.end()) {
        for (auto &&dispatch: iter->second) {
            auto repl = dispatch.first(elem->input()).node();
            if (repl) {
                auto src = elem->output(0);
                m_rewriter.replace_var(src, repl, dispatch.second);
                return;
            }
        }
    }
}

void FinalArithTransformPass::Impl::init_dispatch_table() {
    /*
     * Note: each rule takes var on original graph as input
     */
    auto add_dispatcher = [&](Mode mode) -> DispatchEntry& {
        auto &&vec = m_dispatch_table[mode];
        vec.emplace_back();
        return vec.back();
    };

    auto add_dispatcher_with_name = [&](Mode mode, const char *name)
            -> DispatchEntry::first_type& {
        auto &&ret = add_dispatcher(mode);
        ret.second = name;
        return ret.first;
    };

#define REG(_mode, _name) add_dispatcher_with_name(\
        Mode::_mode, mgb_cstr_log(_name)) = \
    [this](const VarNodeArray &inp) -> SymbolVar

#define REG_THIS(_mode, _fn) add_dispatcher(Mode::_mode) = \
    {std::bind(&Impl::_fn, this, std::placeholders::_1), \
        mgb_cstr_log(#_fn)}

    REG_THIS(ADD, merge_negate<MergeNegateAddTrait>);
    REG_THIS(MUL, merge_negate<MergeNegateMulTrait>);

    REG(POW, "powc and exp merge") {
        auto exp_maybe = SymbolVar{inp[1]}.as_immutable_scalar_require_shape();
        if (!exp_maybe.valid()) {
            return {};
        }
        float exp = exp_maybe->get_cast<float>();
        VarNode* base = m_rewriter.get_var(inp[0]);
        Elemwise* base_pow;
        if ((base_pow = as_elem_opr(base, Mode::POW)) &&
            can_replace_var(base)) {
            // powc(pow(x, a), b) => pow(x, a * b); a is not const scalar
            VarNode* exp_new;
            VarNode* exp_old = base_pow->input(1);
            if (almost_equal(exp, -1.f)) {
                // handle reciprocal
                exp_new = get_neg_repl<Mode::ADD>(exp_old, true);
                if (!exp_new) {
                    exp_new = opr::negate(exp_old).node();
                }
            } else {
                exp_new = (SymbolVar{exp_old} * exp).node();
            }
            return opr::pow(base_pow->input(0), exp_new);
        }
        return opr::PowC::make(base, exp);
    };

#undef REG
#undef REG_THIS
}

/* ---------------- merge_negate ---------------- */
template<class Trait>
SymbolVar FinalArithTransformPass::Impl::merge_negate(const VarNodeArray &inp) {
    VarNode
        *i0 = inp[0], *i1 = inp[1],
        *neg0 = get_neg_repl<Trait::ADD>(i0, false),
        *neg1 = get_neg_repl<Trait::ADD>(i1, false);
        // always replace neg (do not check unique reader) since this does not
        // introduce new opr

    auto add = [](SymbolVar a, SymbolVar b) {
        return opr::Elemwise::make({a, b}, Trait::ADD);
    };
    auto sub = [](SymbolVar a, SymbolVar b) {
        return opr::Elemwise::make({a, b}, Trait::SUB);
    };

    if (!neg0 && !neg1)
        return {};
    if (neg0 && neg1)
        return Trait::neg(add(neg0, neg1));
    if (neg0)
        return sub(i1, neg0);
    return sub(i0, neg1);
}

struct FinalArithTransformPass::Impl::MergeNegateAddTrait {
    static constexpr Mode ADD = Mode::ADD, SUB = Mode::SUB;

    static SymbolVar neg(SymbolVar x) {
        return -x;
    }
};

struct FinalArithTransformPass::Impl::MergeNegateMulTrait {
    static constexpr Mode ADD = Mode::MUL, SUB = Mode::TRUE_DIV;

    static SymbolVar neg(SymbolVar x) {
        return opr::PowC::make(x, -1);
    }
};

/* ---------------- helpers ---------------- */
template<opr::Elemwise::Mode mode>
VarNode* FinalArithTransformPass::Impl::get_neg_repl(
        VarNode *&var, bool require_replaceable) const {
    auto new_var = m_rewriter.get_var(var);
    VarNode *ret = nullptr;
    if (!require_replaceable || can_replace_var(var)) {
        ret = check_is_group_inverse_opr<mode>(new_var);
    }
    var = new_var;
    return ret;
}

void FinalArithTransformPass::Impl::as_replaceable_mul(
        VarNode *&var, VarNode *&a, VarNode *&b) {
    a = b = nullptr;
    auto new_var = m_rewriter.get_var(var);
    Elemwise *elem = nullptr;
    if (var->shape().ndim && can_replace_var(var) &&
            (elem = as_elem_opr(new_var->owner_opr(), Mode::MUL))) {
        a = elem->input(0);
        b = elem->input(1);
    }
    var = new_var;
}

const char* FinalArithTransformPass::name() const {
    return mgb_cstr_log("final_arith_transform");
}

void FinalArithTransformPass::apply(OptState &opt) const {
    MIDOUT_B("FinalArithTransformPass::apply")
    Impl{*this, opt};
    MIDOUT_E
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

