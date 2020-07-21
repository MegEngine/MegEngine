/**
 * \file src/gopt/impl/basic_arith/inplace.cpp
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
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

#include <cmath>

#include "megbrain/utils/hash_ct.h"
#include "midout.h"

MIDOUT_DECL(megbrain_inplace)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_inplace, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace opr;
using namespace gopt;

namespace {
namespace inplace_optimize {

using Mode = Elemwise::Mode;

//! elemwise optimizer
using SingleOptimizer = thin_function<SymbolVar(const SymbolVarArrayView&,
                                                const OperatorNodeConfig&)>;
//! map elemwise mode to optimizer list
using OptimizerRegistry =
        ThinHashMap<Elemwise::Mode, std::vector<SingleOptimizer>>;

OptimizerRegistry make_optimizer_registry();

//! OptimizerRegistry storage
const OptimizerRegistry& optimizer_registry();

//! broadcast src to broadcasted shape of dst_shape_var
SymbolVar broadcast_tshp(SymbolVar src,
                         const SymbolVarArrayView& dst_shape_var) {
    auto dtype = src.dtype();
    for (auto i : dst_shape_var)
        dtype = dtype_promotion(dtype, i.dtype());
    src = opr::TypeCvt::make(src, dtype);
    return opr::Broadcast::make(
            src, opr::GetVarShape::make(VarNodeArrayView{dst_shape_var}));
}

//! broadcast to ensure returned value shape is compatible with inp
static inline SymbolVar broadcast_ensure(SymbolVar value, SymbolVar inp) {
    return broadcast_tshp(value, {value, inp});
}

// a - a => 0, a / a => 1
template <int unit>
SymbolVar eq_to_unit(const SymbolVarArrayView& inp,
                     const OperatorNodeConfig& config) {
    if (inp[0].node() == inp[1].node()) {
        return inp[0].fill_retain_dtype(unit);
    }
    return {};
}

// a + 0 => a, a * 1 => a
template <int id_val>
SymbolVar identical_op(const SymbolVarArrayView& inp,
                       const OperatorNodeConfig& config) {
    auto lhs = inp[0], rhs = inp[1];
    auto k = lhs.as_immutable_scalar();
    if (!k.valid()) {
        std::swap(lhs, rhs);
        k = lhs.as_immutable_scalar();
    }
    if (k.valid() &&
        almost_equal(k->get_cast<float>(), static_cast<float>(id_val))) {
        return broadcast_tshp(rhs, inp);
    }
    return {};
}

template <int zero_val>
SymbolVar absorbing_element(const SymbolVarArrayView& inp,
                            const OperatorNodeConfig& config) {
    auto lhs = inp[0], rhs = inp[1];
    auto scalar = lhs.as_immutable_scalar();
    if (!scalar.valid()) {
        std::swap(lhs, rhs);
        scalar = lhs.as_immutable_scalar();
    }
    if (scalar.valid() &&
        almost_equal(scalar->get_cast<float>(), static_cast<float>(zero_val))) {
        return broadcast_tshp(rhs.make_scalar_dt(zero_val), inp);
    }
    return {};
}
}  // namespace inplace_optimize
}  // anonymous namespace

/* ===================== inplace optimize ===================== */

VarNode* gopt::optimize_elemwise_expr_inplace(
        const VarNodeArrayView& inputs, Elemwise::Param param,
        const OperatorNodeConfig& config) {
    using namespace inplace_optimize;

    mgb_assert(!inputs.empty());
    auto&& opt = inputs[0]->owner_graph()->options();
    auto orig_opt = opt.graph_opt_level;
    auto check_result = orig_opt < 0;

    auto&& optimizers = optimizer_registry();

    auto iter = optimizers.find(param.mode);
    if (iter != optimizers.end()) {
        for (auto&& i : iter->second) {
            auto ret = i(inputs, config).node();
            if (ret) {
                if (check_result) {
                    SymbolVar raw;
                    MGB_TRY {
                        opt.graph_opt_level = 0;
                        raw = Elemwise::make(inputs, param, config);
                    }
                    MGB_FINALLY(opt.graph_opt_level = orig_opt;);

                    opt.extra_vardeps[ret].push_back(AssertEqual::make(raw, ret)
                                                             .rename("chk_opt")
                                                             .node());
                }
                return ret;
            }
        }
    }
    return nullptr;
}

bool gopt::has_inplace_basic_arith_opt(const cg::OperatorNodeBase& opr) {
    if (!opr.owner_graph()->options().graph_opt_level)
        return false;
    auto type = opr.dyn_typeinfo();
    return type == Elemwise::typeinfo() &&
           inplace_optimize::optimizer_registry().count(
                   opr.cast_final<Elemwise>().param().mode);
}

const inplace_optimize::OptimizerRegistry&
inplace_optimize::optimizer_registry() {
    MIDOUT_B("inplace_optimize::optimizer_registry")
    static OptimizerRegistry ret = make_optimizer_registry();
    return ret;
    MIDOUT_E
}

inplace_optimize::OptimizerRegistry
inplace_optimize::make_optimizer_registry() {
    OptimizerRegistry ret;
    auto add_optimizer = [&](Mode mode) -> SingleOptimizer& {
        auto&& vec = ret[mode];
        vec.emplace_back();
        return vec.back();
    };

#define REG(_mode)                         \
    add_optimizer(Mode::_mode) = [](       \
            const SymbolVarArrayView& inp, \
            const OperatorNodeConfig& config) -> SymbolVar

    // a - a -> 0
    add_optimizer(Mode::SUB) = eq_to_unit<0>;

    // a / a -> 1
    add_optimizer(Mode::TRUE_DIV) = eq_to_unit<1>;
    add_optimizer(Mode::FLOOR_DIV) = eq_to_unit<1>;

    // a + 0 => a
    add_optimizer(Mode::ADD) = identical_op<0>;
    // a * 1 => a
    add_optimizer(Mode::MUL) = identical_op<1>;
    // a * 0 => 0
    add_optimizer(Mode::MUL) = absorbing_element<0>;

    // a ** 0 => 1, a ** 1 => a
    REG(EXP) {
        if (is_const_value(inp[0], 0)) {
            return inp[0].fill_retain_dtype(1);
        }
        return {};
    };
    REG(POW) {
        auto a = inp[0];
        auto exp = inp[1].as_immutable_scalar();
        if (exp.valid()) {
            auto fv = exp->get_cast<float>();
            // x ** 0
            if (almost_equal(fv, 0.f))
                return broadcast_tshp(a.make_scalar_dt(1), inp);

            // x ** 1
            if (almost_equal(fv, 1.f))
                return broadcast_tshp(a, inp);
        }
        return {};
    };

    // Strictly speaking, following transformations should not be inplace since
    // they remove some intermediate nodes; however these remvoed nodes are less
    // likely to be directly used (optimization can still be bypassed by
    // Identity() opr in sucn case) and they deal with numerical stability, so
    // we make them inplace here.

    // log(exp(a) */ b) -> a +- log(b)
    REG(LOG) {
        // only consider exp but now pow, since pow(a, b) can not be safely
        // converted to b * log(a) (a can be negative)

        auto opr = try_cast_as_op<Elemwise>(inp[0].node());
        if (!opr)
            return {};
        auto mode = opr->param().mode;
        if ((mode == Mode::MUL || mode == Mode::TRUE_DIV) &&
            (as_elem_opr(opr->input(0), Mode::EXP) ||
             as_elem_opr(opr->input(1), Mode::EXP))) {
            auto v0 = opr::Elemwise::make({opr->input(0)}, Mode::LOG),
                 v1 = opr::Elemwise::make({opr->input(1)}, Mode::LOG);
            return opr::Elemwise::make(
                    {v0, v1}, mode == Mode::MUL ? Mode::ADD : Mode::SUB,
                    config);
        }

        if (mode == Mode::EXP) {
            return opr->input(0);
        }

        return {};
    };

    // log(1 + x) -> log1p(x)
    REG(LOG) {
        auto opr = as_elem_opr(inp[0].node(), Mode::ADD);
        if (!opr)
            return {};
        auto i0 = opr->input(0), i1 = opr->input(1);
        if (!is_const_value(i0, 1)) {
            std::swap(i0, i1);
        }
        if (is_const_value(i0, 1)) {
            return broadcast_ensure(
                    opr::Elemwise::make({i1}, Mode::LOG1P, config), i0);
        }
        return {};
    };

    // log(exp(x) + exp(y)) -> log_sum_exp(x, y)
    REG(LOG) {
        auto add = as_elem_opr(inp[0].node(), Mode::ADD);
        if (!add)
            return {};
        Elemwise *a, *b;
        if ((a = as_elem_opr(add->input(0), Mode::EXP)) &&
            (b = as_elem_opr(add->input(1), Mode::EXP))) {
            return opr::Elemwise::make({a->input(0), b->input(0)},
                                       Mode::LOG_SUM_EXP, config);
        }
        return {};
    };

    // exp(x) - 1 -> expm1(x)
    REG(SUB) {
        auto i0 = as_elem_opr(inp[0].node(), Mode::EXP);
        if (i0 && is_const_value(inp[1], 1)) {
            return broadcast_ensure(
                    opr::Elemwise::make({i0->input(0)}, Mode::EXPM1, config),
                    inp[1]);
        }
        return {};
    };

    // float: floor_div(x, 1) -> floor(x)
    // int: floor_div(x, 1) -> x
    REG(FLOOR_DIV) {
        if (is_const_value(inp[1], 1)) {
            switch (inp[0].dtype().category()) {
                case DTypeCategory::FLOAT:
                    return broadcast_ensure(
                            opr::Elemwise::make({inp[0]}, Mode::FLOOR, config),
                            inp[1]);
                case DTypeCategory::INT:
                    return broadcast_tshp(inp[0], inp);
                default:
                    break;
            }
        }
        return {};
    };

    return ret;

#undef REG
}

/* ===================== GradSumListOptimizer ===================== */

bool GradSumListOptimizer::check_is_shapeof_wrt(VarNode* var) {
    auto opr = var->owner_opr();
    return opr->same_type<GetVarShape>() && opr->input(0) == m_wrt;
}

void GradSumListOptimizer::remove_broadcast() {
    VarNode* wrt_shp = nullptr;

    std::vector<std::pair<size_t, VarNode*>> terms;

    for (auto&& i : m_grads) {
        auto opr = i->owner_opr();
        if (opr->same_type<Broadcast>()) {
            auto bshp = opr->input(1);
            if (!wrt_shp) {
                if (!check_is_shapeof_wrt(bshp)) {
                    continue;
                }
                wrt_shp = bshp;
            } else if (wrt_shp != bshp) {
                continue;
            }
            // i == broadcast(x, shape_of(wrt))

            auto var = opr->input(0);
            auto size = var->shape().total_nr_elems();
            if (!size) {
                size = std::numeric_limits<size_t>::max();
            }
            terms.emplace_back(size, var);

            // recorded in small_terms, so do not sum it in grads
            i = nullptr;
        }
    }

    if (!wrt_shp)
        return;

    // null grads are recorded in m_small_terms
    auto nr_remove = remove_null_grads();
    mgb_assert(nr_remove == terms.size());

    m_brdcast_sum_wrt_shp = wrt_shp;

    std::sort(terms.begin(), terms.end());
    for (auto&& i : terms)
        m_grads.push_back(i.second);
}

size_t GradSumListOptimizer::remove_null_grads() {
    size_t i = 0, j = 0;
    while (j < m_grads.size()) {
        if (!m_grads[j]) {
            ++j;
        } else {
            m_grads[i++] = m_grads[j++];
        }
    }
    m_grads.resize(i);
    return j - i;
}

void GradSumListOptimizer::merge_incr_subtensor() {
    if (m_grads.size() == 1) {
        return;
    }
    for (auto&& i : m_grads) {
        auto opr = i->owner_opr();
        if (!check_is_incr_subtensor_zero(opr, true))
            continue;

        if (!check_is_shapeof_wrt(opr->input(0)->owner_opr()->input(1)))
            continue;

        // now confirmed opr is incr_sub(bcast(0, shapeof(wrt)), x)
        if (m_incr_subtensor_oprs.size() + 1 < m_grads.size()) {
            m_incr_subtensor_oprs.push_back(opr);
            i = nullptr;
        }
    }

    if (!m_incr_subtensor_oprs.empty()) {
        auto nr_remove = remove_null_grads();
        mgb_assert(nr_remove == m_incr_subtensor_oprs.size());
    }
}

GradSumListOptimizer::GradSumListOptimizer(VarNode* wrt, VarNodeArray& grads,
                                           VarNodeArray& mid_results)
        : m_wrt{wrt}, m_grads{grads} {
    remove_broadcast();
    merge_incr_subtensor();
    calc_sum(mid_results);
}

void GradSumListOptimizer::calc_sum(VarNodeArray& mid_results) {
    auto sum = elemwise_reduce_var_list(m_grads, Elemwise::Mode::ADD,
                                        &mid_results);
    auto update_sum = [&](VarNode* s) {
        sum = s;
        mid_results.push_back(s);
    };
    if (m_brdcast_sum_wrt_shp) {
        update_sum(Broadcast::make(sum, m_brdcast_sum_wrt_shp).node());
    }

    for (auto i : m_incr_subtensor_oprs) {
        update_sum(remake_incr_subtensor_zero(i, sum));
    }

    m_sum = sum;
}

/* ===================== global functions ===================== */

bool gopt::check_is_incr_subtensor_zero(cg::OperatorNodeBase* opr,
                                        bool require_brdcst) {
    auto type = opr->dyn_typeinfo();
    if (type != IncrSubtensor::typeinfo() &&
        type != IndexingIncrMultiAxisVec::typeinfo())
        return false;

    SymbolVar ivar = opr->input(0);
    if (require_brdcst) {
        auto sopr = opr->input(0)->owner_opr();
        if (!sopr->same_type<Broadcast>()) {
            return false;
        }
        ivar = sopr->input(0);
    }

    return is_const_value(ivar, 0);
}

VarNode* gopt::remake_incr_subtensor_zero(
        cg::OperatorNodeBase* orig_opr, VarNode* new_data,
        const opr::intl::FancyIndexingHelper::InputTensorReplacer&
                input_tensor_replacer) {
    auto type = orig_opr->dyn_typeinfo();
    if (!new_data)
        new_data = orig_opr->input(0);
    if (type == IncrSubtensor::typeinfo()) {
        return IncrSubtensor::make(
                       new_data, orig_opr->input(1),
                       orig_opr->cast_final<IncrSubtensor>().index_desc(),
                       orig_opr->config(), input_tensor_replacer)
                .node();
    }
    mgb_assert(type == IndexingIncrMultiAxisVec::typeinfo());
    return IndexingIncrMultiAxisVec::make(
                   new_data, orig_opr->input(1),
                   orig_opr->cast_final<IndexingIncrMultiAxisVec>()
                           .index_desc(),
                   orig_opr->config(), input_tensor_replacer)
            .node();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
