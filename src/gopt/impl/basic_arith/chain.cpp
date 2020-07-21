/**
 * \file src/gopt/impl/basic_arith/chain.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/gopt/basic_arith.h"
#include "megbrain/gopt/gtrans.h"
#include "megbrain/opr/basic_arith_wrapper.h"

#include <deque>

//! TODO: here has to be know some megdnn::opr when there is produced midout.h
//! fix it if there is another graceful way.
#include "megdnn/oprs.h"

#include "megbrain/utils/hash_ct.h"
#include "midout.h"

MIDOUT_DECL(megbrain_chain)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_chain, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace gopt;
using namespace opr;

#define FOREACH_FUSE_ADD_MODE(cb) \
    cb(RELU) cb(SIGMOID) cb(TANH) cb(H_SWISH)

namespace {
    //! call process_opr_chain() when a chain of same mode is detected
    class ElemChainImplHelper {
        void on_opr(OperatorNodeBase *opr);

        protected:
            using Mode = Elemwise::Mode;
            OptState &m_opt_state;
            SubGraph::Rewriter m_rewriter;
            UniqReaderCheck m_uniq_reader_check;

            ElemChainImplHelper(OptState &opt_state):
                m_opt_state{opt_state},
                m_rewriter{opt_state.graph().make_rewriter()},
                m_uniq_reader_check{opt_state.graph()}
            {
            }

            ~ElemChainImplHelper() = default;

            void run_elem_chain() {
                using namespace std::placeholders;
                m_opt_state.graph().iter(
                        std::bind(&ElemChainImplHelper::on_opr, this, _1));
                m_rewriter.apply_inplace();
            }

            //! called when an opr on original graph is visited
            virtual void on_opr_visited(OperatorNodeBase *opr) {
                MGB_MARK_USED_VAR(opr);
            }

            //! called when a chain of same mode on original graph is detected
            virtual void process_chain(VarNode *endpoint, Mode mode) = 0;

            /*!
             * \brief called at the end of visiting an operator
             * \return whether this opr should be further processed by
             *      process_chain() if it is an endpoint
             */
            virtual bool on_opr_visit_finished(Elemwise *opr) {
                MGB_MARK_USED_VAR(opr);
                return true;
            }

            //! check whether a mode should be processed
            virtual bool check_mode(Mode mode) = 0;

            VarNodeArray extract_chain_terms(VarNode *endpoint, Mode mode);
    };
}

void ElemChainImplHelper::on_opr(OperatorNodeBase *opr) {
    m_uniq_reader_check.update_on_opr_auto_replace(
            opr, m_rewriter.auto_replace_outputs(opr));
    on_opr_visited(opr);

    auto elem = try_cast_as_op<Elemwise>(opr);
    Mode mode = elem ? elem->param().mode : Mode::NEGATE;

    bool inp_changed = false;
    for (auto i: opr->input()) {
        if (m_rewriter.has_manual_replace(i)) {
            inp_changed = true;
            continue;
        }

        auto ielem = try_cast_as_op<Elemwise>(i->owner_opr());
        if (ielem) {
            auto imode = ielem->param().mode;
            // To ensure that all leaves(chain terms) which found by
            // extract_chain_terms have been processed. In other word,
            // we would call process_chain in topological order.
            if ((!elem || imode != mode || !m_uniq_reader_check(i))
                    && check_mode(imode)) {
                inp_changed = true;
                m_opt_state.call_with_opr(i->owner_opr(),
                    [&]{this->process_chain(i, imode);});
            }
        }
    }
    if (inp_changed) {
        m_uniq_reader_check.update_on_opr_auto_replace(
                opr, m_rewriter.auto_replace_outputs(opr));
    }

    if (elem && on_opr_visit_finished(elem)) {
        auto ovar = opr->output(0);
        if (check_mode(mode) && m_opt_state.graph().endpoint_contain(ovar))
            process_chain(ovar, mode);
    }
}

VarNodeArray ElemChainImplHelper::extract_chain_terms(
        VarNode *endpoint, Mode mode) {
    auto pred = [mode, this, eo=endpoint->owner_opr()](OperatorNodeBase *opr) {
        return as_elem_opr(opr, mode) && (
                opr == eo || m_uniq_reader_check(opr->output(0)));
    };
    auto ret = extract_opr_leaves(endpoint, pred);
    mgb_assert(!ret.empty());
    return ret;
}

/* ================ ExpandFusedArithPass ================ */
const char* ExpandFusedArithPass::name() const {
    return mgb_cstr_log("expand_fused_arith");
}

void ExpandFusedArithPass::apply(OptState &opt) const {
    MIDOUT_B("ExpandFusedArithPass::apply")
    auto rewriter = opt.graph().make_rewriter();
    auto on_opr = [&](OperatorNodeBase *opr) {
        using Mode = Elemwise::Mode;
        auto repl_opr = rewriter.auto_replace_outputs(opr);
        auto elem = try_cast_as_op<Elemwise>(opr);
        if (elem) {
            auto src = opr->output(0);
            opr = repl_opr;
            SymbolVar out;
            const char *msg = nullptr;
            switch (elem->param().mode) {
                case Mode::FUSE_MUL_ADD3:
                    out = SymbolVar{opr->input(0)} * opr->input(1) +
                        opr->input(2);
                    msg = mgb_cstr_log("expand fma3");
                    break;
                case Mode::FUSE_MUL_ADD4:
                    out = SymbolVar{opr->input(0)} * opr->input(1) +
                        SymbolVar{opr->input(2)} * opr->input(3);
                    msg = mgb_cstr_log("expand fma4");
                    break;
#define cb(m) case Mode::FUSE_ADD_##m: \
                    out = opr::Elemwise::make( \
                            {opr::add(opr->input(0), opr->input(1))}, \
                            Mode::m); \
                    msg = mgb_cstr_log("expand FUSE_ADD_" #m); \
                    break;
                    FOREACH_FUSE_ADD_MODE(cb)
#undef cb
                default:
                    break;
            }
            if (auto dst = out.node()) {
                rewriter.replace_var(src, dst, msg);
            }
        }
    };

    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
    MIDOUT_E
}

/* ================ NormalizeArithChainPass ================ */

class NormalizeArithChainPass::Impl {
    using Mode = Elemwise::Mode;
    struct Var2CoeffRec {
        dt_max_float coeff;
        size_t order = 0;
        bool operator < (const Var2CoeffRec &rhs) const {
            return order < rhs.order;
        }
    };

    OptState &m_opt_state;
    SubGraph::Rewriter m_rewriter;
    ThinHashMap<VarNode*, size_t> m_var2nr_val_dep;
    ThinHashSet<VarNode*> m_processed_vars;

    //! passed from process_opr_chain() to sum_var2coeff()
    ThinHashMap<VarNode*, Var2CoeffRec> m_var2coeff;
    //! tmp var used by sum_var2coeff()
    std::vector<std::pair<Var2CoeffRec, VarNode*>> m_var2coeff_sort;

    void sort_var2coeff() {
        auto &&sorted = m_var2coeff_sort;
        sorted.clear();
        for (auto &&i: m_var2coeff)
            sorted.push_back({i.second, i.first});
        std::sort(sorted.begin(), sorted.end());
    }

    //! abstract operator representation
    struct AbstractOpr {
        enum class Type {
            ADD, SUB, COEFF
        };
        Type type;

        //! inputs for ADD/SUB
        VarNode *i0 = nullptr, *i1 = nullptr;

        //! input var for COEFF
        VarNode *ic;
        //! coeff mul value
        dt_max_float coeff;

        static AbstractOpr make_coeff(VarNode *ic, float coeff) {
            return {Type::COEFF, nullptr, nullptr, ic, coeff};
        }

        template<class Trait>
        static Maybe<AbstractOpr> from(VarNode* var);
    };

    struct AddTrait {
        static constexpr Mode ADD = Mode::ADD, SUB = Mode::SUB;
        static constexpr float UNIT = 0;

        static Maybe<AbstractOpr> extract_coeff(Mode mode, Elemwise *opr);
        static Maybe<AbstractOpr> extract_from_non_elemwise(OperatorNodeBase*) {
            return None;
        }

        static SymbolVar neg(SymbolVar x) {
            return -x;
        }

        static SymbolVar make_term(SymbolVar x, dt_max_float coeff) {
            return x * x.make_scalar_dt(coeff);
        }
    };
    struct MulTrait {
        static constexpr Mode ADD = Mode::MUL, SUB = Mode::TRUE_DIV;
        static constexpr float UNIT = 1;

        static Maybe<AbstractOpr> extract_coeff(Mode mode, Elemwise *opr);
        static Maybe<AbstractOpr> extract_from_non_elemwise(
                OperatorNodeBase* opr);

        static SymbolVar neg(SymbolVar x) {
            return opr::powf(x, -1);
        }

        static SymbolVar make_term(SymbolVar x, dt_max_float coeff) {
            return opr::powf(x, coeff);
        }
    };

    struct QueueNode {
        dt_max_float coeff;
        VarNode *var;
    };

    //! sum m_var2coeff
    template<class Trait>
    VarNode* sum_var2coeff();

    template<class Trait>
    void process_opr_chain(VarNode* endpoint);

    void on_opr(OperatorNodeBase *opr);

    public:
        Impl(OptState &opt_state):
            m_opt_state{opt_state},
            m_rewriter{opt_state.graph().make_rewriter()},
            m_var2nr_val_dep{opt_state.graph().get_var2nr_val_dep_oprs()}
        {
            using namespace std::placeholders;
            opt_state.graph().iter(std::bind(&Impl::on_opr, this, _1));
            m_rewriter.apply_inplace();
        }
};


Maybe<NormalizeArithChainPass::Impl::AbstractOpr>
NormalizeArithChainPass::Impl::AddTrait::extract_coeff(
        Mode mode, Elemwise *opr) {

    if (mode == Mode::NEGATE)
        return AbstractOpr::make_coeff(opr->input(0), -1);
    if (mode == Mode::MUL) {
        SymbolVar i0 = opr->input(0), i1 = opr->input(1);
        auto i0v = i0.as_immutable_scalar_require_shape();
        if (!i0v.valid()) {
            std::swap(i0, i1);
            i0v = i0.as_immutable_scalar_require_shape();
            if (!i0v.valid())
                return None;
        }
        return AbstractOpr::make_coeff(
                i1.node(), i0v->get_cast<dt_max_float>());
    }
    return None;
}

Maybe<NormalizeArithChainPass::Impl::AbstractOpr>
NormalizeArithChainPass::Impl::MulTrait::extract_coeff(
        Mode mode, Elemwise *opr) {
    if (mode != Mode::POW)
        return None;

    auto exp = SymbolVar{opr->input(1)}.as_immutable_scalar_require_shape();
    if (exp.valid()) {
        return AbstractOpr::make_coeff(
                opr->input(0), exp->get_cast<dt_max_float>());
    }
    return None;
}

Maybe<NormalizeArithChainPass::Impl::AbstractOpr>
NormalizeArithChainPass::Impl::MulTrait::extract_from_non_elemwise(
        OperatorNodeBase* opr) {
    if (auto powc = try_cast_as_op<PowC>(opr)) {
        return AbstractOpr::make_coeff(powc->input(0), powc->param().exp);
    }
    return None;
}

template <class Trait>
Maybe<NormalizeArithChainPass::Impl::AbstractOpr>
NormalizeArithChainPass::Impl::AbstractOpr::from(VarNode* var) {
    auto opr = var->owner_opr();
    auto non_elem_ret = Trait::extract_from_non_elemwise(opr);
    if (non_elem_ret.valid()) {
        return non_elem_ret;
    }
    auto elem = try_cast_as_op<Elemwise>(opr);
    if (!elem)
        return None;
    auto mode = elem->param().mode;
    if (mode == Trait::ADD || mode == Trait::SUB) {
        auto type = mode == Trait::ADD ? Type::ADD : Type::SUB;
        return AbstractOpr{type, elem->input(0), elem->input(1), nullptr, 0};
    }
    return Trait::extract_coeff(mode, elem);
}

template<class Trait>
void NormalizeArithChainPass::Impl::process_opr_chain(VarNode* endpoint) {
    if (!m_processed_vars.insert(endpoint).second)
        return;

    if (std::is_same<Trait, MulTrait>::value &&
            endpoint->dtype().category() == DTypeCategory::INT) {
        // do not normalize int mul/div, since int mul/div is not a closed group
        return;
    }

    auto &&var2coeff = m_var2coeff;
    var2coeff.clear();
    std::deque<QueueNode> queue;
    bool has_non_elem_case = false; // non-elemwise oprs should be canonized
    size_t nr_sub = 0, nr_non1_coeff = 0, nr_term = 0;
    queue.push_back({dt_max_float(1), endpoint});
    while (!queue.empty()) {
        auto qh = queue.front();
        queue.pop_front();
        VarNode* var = qh.var;
        // find leaf nodes on original graph (without applying rewriter)
        if ((var == endpoint || m_var2nr_val_dep.at(var) <= 1) &&
                var->comp_node() == endpoint->comp_node()) {
            Maybe<AbstractOpr> aopr = AbstractOpr::from<Trait>(var);
            auto append = [&](VarNode *var, dt_max_float coeff) {
                queue.push_back({qh.coeff * coeff, var});
            };
            if (aopr.valid()) {
                auto &&val = aopr.val();
                using Type = AbstractOpr::Type;
                if (!var->owner_opr()->same_type<opr::Elemwise>()) {
                    has_non_elem_case = true;
                }
                switch (val.type) {
                    case Type::ADD:
                        append(val.i0, 1);
                        append(val.i1, 1);
                        break;
                    case Type::SUB:
                        ++ nr_sub;
                        append(val.i0, 1);
                        append(val.i1, -1);
                        break;
                    case Type::COEFF:
                        if (val.coeff != 1)
                            ++ nr_non1_coeff;
                        append(val.ic, val.coeff);
                        break;
                    default:
                        mgb_assert(0);
                }
                continue;
            }
        }
        // var is a leaf node that can not be expanded

        ++ nr_term;
        var = m_rewriter.get_var(var); // apply previous trans on leaf nodes
        auto &&dest = var2coeff[var];
        dest.coeff += qh.coeff;
        if (!dest.order) {
            dest.order = nr_term;
        }
    }

    if (nr_sub || nr_non1_coeff >= 2 || nr_term > var2coeff.size() ||
        has_non_elem_case) {
        auto sum = sum_var2coeff<Trait>();
        if (endpoint != sum) {
            m_rewriter.replace_var(
                    endpoint, sum,
                    ssprintf("normalize elemwise chain with %zu terms", nr_term)
                            .c_str());
        }
    }
}

template<class Trait>
VarNode* NormalizeArithChainPass::Impl::sum_var2coeff() {
    sort_var2coeff(); // use another function to bypass GCC-5 bug
    auto &&sorted = m_var2coeff_sort;

    VarNode *sum = nullptr;

    for (auto &&var_cnt_pair: sorted) {
        SymbolVar x = var_cnt_pair.second, term;
        dt_max_float coeff = var_cnt_pair.first.coeff;
        auto eq = [coeff](dt_max_float v) {
            return almost_equal(coeff, v);
        };
        if (eq(0)) {
            term = x.fill_retain_dtype(Trait::UNIT);
        } else if (eq(1)) {
            term = x;
        } else if (eq(-1)) {
            term = Trait::neg(x);
        } else {
            // note: for power 2, 2 * x is better than x + x, because 2 * x * y
            // may be reordered to 2 * y * x, and it does not seem to cause
            // other overhead
            term = Trait::make_term(x, coeff);
        }
        if (!sum) {
            sum = term.node();
        } else {
            sum = Elemwise::make({sum, term}, Trait::ADD).node();
        }
    }

    return sum;
}

void NormalizeArithChainPass::Impl::on_opr(OperatorNodeBase *opr) {
    m_rewriter.auto_replace_outputs(opr);

    using proc_fn_t = void (Impl::*)(VarNode*);
    auto dispatch_proc_fn = [](OperatorNodeBase* opr) -> proc_fn_t {
        if (auto elem = try_cast_as_op<Elemwise>(opr)) {
            auto mode = elem->param().mode;
            if (mode == Mode::ADD || mode == Mode::SUB ||
                mode == Mode::NEGATE) {
                return &Impl::process_opr_chain<AddTrait>;
            }
            if (mode == Mode::MUL || mode == Mode::TRUE_DIV ||
                (mode == Mode::POW &&
                 SymbolVar{opr->input(1)}
                         .as_immutable_scalar_require_shape()
                         .valid())) {
                return &Impl::process_opr_chain<MulTrait>;
            }
        }
        if (opr->same_type<opr::PowC>()) {
            return &Impl::process_opr_chain<MulTrait>;
        }
        return nullptr;
    };

    VarNode* out0 = nullptr;
    auto func_self = dispatch_proc_fn(opr);
    if (func_self) {
        out0 = opr->output(0);
    }

    bool inp_changed = false;
    for (auto i: opr->input()) {
        if (m_rewriter.has_manual_replace(i)) {
            inp_changed = true;
            continue;
        }
        auto func_in = dispatch_proc_fn(i->owner_opr());
        if (func_in && (func_in != func_self || m_var2nr_val_dep.at(i) >= 2)) {
            // note: we process starting from an endpoint of a chain of the same
            // mode (either ADD or MUL) to ensure linear time complexity. An
            // endpoint is a var that must be preserved, which is either: (1)
            // received by multiple readers (2) received by an opr of different
            // mode or non-elemwise opr (3) the endpoint of the whole graph. The
            // cases (1) and (2) are handled here, and case (3) is handled
            // below by calling func_self().
            inp_changed = true;
            m_opt_state.call_with_opr(i->owner_opr(),
                [&]{(this->*func_in)(i);});
        }
    }

    if (inp_changed)
        m_rewriter.auto_replace_outputs(opr);

    if (func_self && m_opt_state.graph().endpoint_contain(out0)) {
        (this->*func_self)(out0);
    }
}

const char* NormalizeArithChainPass::name() const {
    return mgb_cstr_log("normalize_arith_expr");
}

void NormalizeArithChainPass::apply(OptState &opt) const {
    MIDOUT_B("NormalizeArithChainPass::apply")
    Impl{opt};
    MIDOUT_E
}

/* ================ ReorderArithChainPass ================ */

class ReorderArithChainPass::Impl final: public ElemChainImplHelper {
    using ShapedVars = std::vector<std::pair<TensorShape, VarNode*>>;
    ConstVarPropogate m_cvprop;

    TensorShapeArray m_tmp_inp_shp;

    //! tmp var: (shape, is_const) -> terms
    TensorShapeHashKey::Map<std::array<VarNodeArray, 2>> m_shp2terms;
    ShapedVars m_const_terms, m_nonconst_terms;

    //! reduce two terms
    static VarNode* reduce(Mode mode, VarNode *a, VarNode *b);

    //! reduce m_shp2terms into a sum var
    VarNode* reduce_shp2terms(Mode mode);

    //! merge src and dst into dst, if merging does not broadcast both
    bool merge_shape_if_compatible(const TensorShape &src, TensorShape &dst);

    //! merge compatible shapes
    void merge_shaped_terms(Mode mode, ShapedVars &vars, bool allow_compatible);

    void process_chain(VarNode *endpoint, Mode mode) override;

    void on_opr_visited(OperatorNodeBase *opr) override {
        m_cvprop.add_opr(opr);
    }

    bool check_mode(Mode mode) override {
        return mode == Mode::ADD || mode == Mode::MUL ||
            mode == Mode::MAX || mode == Mode::MIN;
    }

    public:
        Impl(const ReorderArithChainPass &pass, OptState &opt_state):
            ElemChainImplHelper(opt_state),
            m_cvprop{pass.m_const_var_type}
        {
            run_elem_chain();
        }
};

VarNode* ReorderArithChainPass::Impl::reduce(
        Mode mode, VarNode *a, VarNode *b) {
    if (!a)
        return b;
    if (!b)
        return a;
    return opr::Elemwise::make({a, b}, mode).node();
}

bool ReorderArithChainPass::Impl::merge_shape_if_compatible(
        const TensorShape &src, TensorShape &dst) {
    m_tmp_inp_shp.resize(2);
    m_tmp_inp_shp[0] = src;
    m_tmp_inp_shp[1] = dst;
    TensorShape out;
    megdnn::Elemwise::deduce_shape(m_tmp_inp_shp, out);
    if (out.eq_shape(src)) {
        dst = out;
        return true;
    }
    return out.eq_shape(dst);
}

VarNode* ReorderArithChainPass::Impl::reduce_shp2terms(Mode mode) {

    // populate m_const_terms and m_nonconst_terms
    m_const_terms.clear();
    m_nonconst_terms.clear();
    for (auto &&i: m_shp2terms) {
        if (!i.second[0].empty()) {
            m_nonconst_terms.emplace_back(
                    i.first.shape(),
                    elemwise_reduce_var_list(i.second[0], mode));
        }
        if (!i.second[1].empty()) {
            m_const_terms.emplace_back(
                    i.first.shape(),
                    elemwise_reduce_var_list(i.second[1], mode));
        }
    }

    {
        // sorted by id(), so the same set of input terms would get the
        // same reduced var
        auto cmp = [](const ShapedVars::value_type &a,
                const ShapedVars::value_type &b) {
            return a.second->id() < b.second->id();
        };
        small_sort(m_const_terms.begin(), m_const_terms.end(), cmp);
        small_sort(m_nonconst_terms.begin(), m_nonconst_terms.end(), cmp);
    }
    merge_shaped_terms(mode, m_const_terms, true);

    auto &&all_terms = m_const_terms;
    all_terms.insert(all_terms.end(),
            m_nonconst_terms.begin(), m_nonconst_terms.end());

    // merge eq shape
    merge_shaped_terms(mode, all_terms, false);
    // merge compatible shape
    merge_shaped_terms(mode, all_terms, true);

    // simple heuristic: reduce in increasing size order
    auto cmp = [](const ShapedVars::value_type &a,
            const ShapedVars::value_type &b) {
        return a.first.total_nr_elems() < b.first.total_nr_elems();
    };
    small_sort(all_terms.begin(), all_terms.end(), cmp);
    VarNode *sum = nullptr;
    for (auto &&i: all_terms) {
        sum = reduce(mode, sum, i.second);
    }
    mgb_assert(sum);
    return sum;
}

void ReorderArithChainPass::Impl::merge_shaped_terms(
        Mode mode, ShapedVars &vars, bool allow_compatible) {
    for (bool merged = true; merged;) {
        merged = false;

        for (size_t i = 0; !merged && i < vars.size(); ++ i) {
            auto &&src = vars[i];
            if (!src.first.ndim)
                continue;

            TensorShape dst_shape;
            size_t dst_idx = -1;
            auto update_dst = [&](size_t idx, const TensorShape &shp) {
                if (!dst_shape.ndim || shp.total_nr_elems() <
                        dst_shape.total_nr_elems()) {
                    dst_shape = shp;
                    dst_idx = idx;
                }
            };
            for (size_t j = 0; j < vars.size(); ++ j) {
                auto &&dst = vars[j];
                if (i == j || !dst.first.ndim)
                    continue;
                if (allow_compatible) {
                    auto tshp = dst.first;
                    if (merge_shape_if_compatible(src.first, tshp)) {
                        update_dst(j, tshp);
                    }
                } else {
                    if (src.first.eq_shape(dst.first)) {
                        update_dst(j, dst.first);
                    }
                }
            }

            if (dst_shape.ndim) {
                auto &&dst = vars[dst_idx];
                dst.first = dst_shape;
                dst.second = reduce(mode, src.second, dst.second);
                mgb_assert(
                        (!dst.second->shape().ndim &&
                         !cg::is_static_var_shape(dst.second)) ||
                        dst.second->shape().eq_shape(dst.first));
                std::swap(src, vars.back());
                vars.pop_back();
                merged = true;
                break;
            }

        }
    }
}

void ReorderArithChainPass::Impl::process_chain(VarNode *endpoint, Mode mode) {
    if (m_cvprop.is_const(endpoint))
        return;

    auto vars = extract_chain_terms(endpoint, mode);
    if (vars.size() == 1)
        return;

    // to ensure the same set of input terms get the same reduced var
    // TODO: consider maintain a cache(map) of (sorted input terms -> reduced var)
    std::sort(vars.begin(), vars.end(),
            [](VarNode *x, VarNode *y){ return x->id() < y->id(); });
    m_shp2terms.clear();
    for (auto i: vars) {
        auto inew = m_rewriter.get_var(i);
        m_shp2terms[i->shape()][m_cvprop.is_const(i)].push_back(inew);
    }

    auto sum = reduce_shp2terms(mode);

    if (m_rewriter.get_var(endpoint) != sum) {
        m_rewriter.replace_var(endpoint, sum,
                mgb_ssprintf_log("reorder %zu %s terms", vars.size(),
                    megdnn::Elemwise::ModeTrait::from_mode(mode).name).c_str());
    }
}

const char* ReorderArithChainPass::name() const {
    return mgb_cstr_log("reorder_arith_chain");
}

void ReorderArithChainPass::apply(OptState &opt) const {
    MIDOUT_B("ReorderArithChainPass::apply")
    Impl{*this, opt};
    MIDOUT_E
}

/* ================ ArithFusePass ================ */

class ArithFusePass::Impl final: public ElemChainImplHelper {
    using MulTermArray = std::vector<std::pair<VarNode*, VarNode*>>;
    class SumVars;

    size_t m_nr_fma3, m_nr_fma4;
    TensorShapeHashKey::PairMap<MulTermArray> m_mul_terms;
    TensorShapeHashKey::Map<VarNodeArray> m_bias_terms;

    bool check_mode(Mode mode) override {
        return mode == Mode::ADD;
    }

    void process_chain(VarNode *endpoint, Mode mode) override;

    VarNode* find_pop_bias_term(const TensorShape &shape) {
        auto iter = m_bias_terms.find(shape);
        if (iter != m_bias_terms.end()) {
            auto ret = elemwise_reduce_var_list(iter->second, Mode::ADD);
            m_bias_terms.erase(iter);
            return ret;
        }
        return nullptr;
    }

    VarNode* process_mul_term(MulTermArray &terms);

    bool on_opr_visit_finished(Elemwise *opr) override;

    public:
        Impl(OptState &opt_state): ElemChainImplHelper(opt_state) {
            run_elem_chain();
        }

};

class ArithFusePass::Impl::SumVars {
    VarNode *m_sum = nullptr;

    public:
    void add(SymbolVar var) {
        if (!m_sum) {
            m_sum = var.node();
        } else {
            m_sum = opr::add(m_sum, var).node();
        }
    }

    VarNode* get() const {
        return m_sum;
    }
};

void ArithFusePass::Impl::process_chain(VarNode *endpoint, Mode mode) {
    if (!endpoint->shape().ndim)
        return;
    mgb_assert(mode == Mode::ADD);
    m_mul_terms.clear();
    m_bias_terms.clear();
    m_nr_fma3 = m_nr_fma4 = 0;
    auto vars = extract_chain_terms(endpoint, mode);
    for (auto var: vars) {
        auto opr = var->owner_opr();
        Elemwise *mul;
        if (m_uniq_reader_check(var) && (mul = as_elem_opr(opr, Mode::MUL))) {
            auto a = mul->input(0), b = mul->input(1);
            if (a->shape().total_nr_elems() > b->shape().total_nr_elems()) {
                std::swap(a, b);
            }
            a = m_rewriter.get_var(a);
            b = m_rewriter.get_var(b);
            m_mul_terms[{a->shape(), b->shape()}].push_back({a, b});
        } else {
            var = m_rewriter.get_var(var);
            m_bias_terms[var->shape()].push_back(var);
        }
    }

    if (m_mul_terms.empty())
        return;

    // merge same shapes, so they can be used as bias by others
    for (auto i = m_mul_terms.begin(); i != m_mul_terms.end(); ) {
        auto &&s = i->first;
        if (s.first.shape().eq_shape(s.second.shape())) {
            auto merged = process_mul_term(i->second);
            mgb_assert(merged->shape().eq_shape(s.first.shape()));
            m_bias_terms[merged->shape()].push_back(merged);

            mgb_assert(i->second.empty());
            i = m_mul_terms.erase(i);
        } else {
            ++ i;
        }
    }

    // sort mul_terms by size
    TensorShapeArray shp_inp(2);
    using SortedTermItem = std::pair<size_t, MulTermArray*>;
    std::vector<SortedTermItem> mul_terms_sorted;
    for (auto &&i: m_mul_terms) {
        shp_inp[0] = i.first.first.shape();
        shp_inp[1] = i.first.second.shape();
        TensorShape tshp;
        megdnn::Elemwise::deduce_shape(shp_inp, tshp);
        mul_terms_sorted.push_back({tshp.total_nr_elems(), &i.second});
    }
    auto cmp = [](const SortedTermItem &a, const SortedTermItem &b) {
        return a.first < b.first || (
                a.first == b.first && a.second->size() < b.second->size());
    };
    std::sort(mul_terms_sorted.begin(), mul_terms_sorted.end(), cmp);

    // merge from smallest to largest
    for (auto &&i: mul_terms_sorted) {
        auto merged = process_mul_term(*i.second);
        mgb_assert(i.second->empty() && merged->shape().ndim);
        m_bias_terms[merged->shape()].push_back(merged);
    }

    SumVars sum_vars;
    for (auto &&i: m_bias_terms) {
        sum_vars.add(elemwise_reduce_var_list(i.second, Mode::ADD));
    }

    auto sum = sum_vars.get();
    m_rewriter.replace_var(endpoint, sum,
            mgb_ssprintf_log(
                "replace %zu fma3, %zu fma4", m_nr_fma3, m_nr_fma4).c_str());
}

VarNode* ArithFusePass::Impl::process_mul_term(MulTermArray &terms) {
    mgb_assert(!terms.empty());
    SumVars sum_vars;
    while (terms.size() >= 2) {
        auto b = terms.back();
        terms.pop_back();
        auto a = terms.back();
        terms.pop_back();
        ++ m_nr_fma4;
        sum_vars.add(Elemwise::make({a.first, a.second, b.first, b.second},
                    Mode::FUSE_MUL_ADD4));
    }
    if (!terms.empty()) {
        auto t = terms.back();
        terms.pop_back();
        auto bias = find_pop_bias_term(t.first->shape());
        if (!bias)
            bias = find_pop_bias_term(t.second->shape());
        if (bias) {
            ++ m_nr_fma3;
            sum_vars.add(Elemwise::make({t.first, t.second, bias},
                        Mode::FUSE_MUL_ADD3));
        } else {
            sum_vars.add(opr::mul(t.first, t.second));
        }
    }
    return sum_vars.get();
}

bool ArithFusePass::Impl::on_opr_visit_finished(Elemwise *opr) {
    if (opr->input().size() != 1)
        return true;

    if (!m_uniq_reader_check(opr->input(0)))
        return true;

    auto iadd = as_elem_opr(m_rewriter.get_var(opr->input(0)), Mode::ADD);
    if (!iadd)
        return true;

    if (opr->input(0)->dtype().category() == DTypeCategory::QUANTIZED)
        return true;

    Mode fmode;

    const char *msg;
    switch (opr->param().mode) {
#define cb(m) \
        case Mode::m: \
            fmode = Mode::FUSE_ADD_##m; \
            msg = mgb_cstr_log("fuse " #m "(x + y)"); \
            break;
        FOREACH_FUSE_ADD_MODE(cb)
#undef cb
        default:
            return true;
    }

    m_opt_state.call_with_opr(opr, [&]{
        auto fused = opr::Elemwise::make({iadd->input(0), iadd->input(1)},
                fmode).node();
        m_rewriter.replace_var(opr->output(0), fused, msg);
        m_uniq_reader_check.update_on_opr_auto_replace(opr, fused->owner_opr());
    });
    return false;
}

const char* ArithFusePass::name() const {
    return mgb_cstr_log("arith_fuse");
}

void ArithFusePass::apply(OptState &opt) const {
    MIDOUT_B("ArithFusePass::apply")
    Impl{opt};
    MIDOUT_E
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
