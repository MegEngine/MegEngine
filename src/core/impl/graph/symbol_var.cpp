/**
 * \file src/core/impl/graph/symbol_var.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cg_impl.h"

#include "megbrain/graph/symbol_var.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/io.h"

using namespace mgb;
using namespace cg;

SymbolVar SymbolVar::rename(const std::string &name) const {
    m_node->name(name);
    return *this;
}

SymbolVar SymbolVar::symshape() const {
    return opr::GetVarShape::make(*this);
}

SymbolVar SymbolVar::reshape(const TensorShape &tshape) const {
    return opr::Reshape::make(*this, tshape);
}

SymbolVar SymbolVar::reshape(SymbolVar tshape) const {
    return opr::Reshape::make(*this, tshape);
}

SymbolVar SymbolVar::broadcast(const TensorShape &tshape) const {
    return opr::Broadcast::make(*this, tshape);
}

SymbolVar SymbolVar::broadcast(SymbolVar tshape) const {
    return opr::Broadcast::make(*this, tshape);
}

SymbolVar SymbolVar::flatten() const {
    return opr::Reshape::make(*this, make_scalar(1), 0);
}

SymbolVar SymbolVar::add_axis(size_t idx) const {
    return opr::AxisAddRemove::make(*this,
        {opr::AxisAddRemove::AxisDesc::make_add(idx)});
}

Maybe<DTypeScalar> SymbolVar::as_immutable_scalar() const {
    using IT = static_infer::InferType;
    auto &&mgr = node()->owner_graph()->static_infer_manager();

    auto ivar = node();
    for (; ; ) {
        auto ivar_type = ivar->owner_opr()->dyn_typeinfo();
        if (ivar_type == opr::Broadcast::typeinfo() ||
                ivar_type == opr::Reshape::typeinfo()) {
            ivar = ivar->owner_opr()->input(0);
        } else {
            break;
        }
    }

    auto it = mgr.get_infer_type(ivar);
    if (it.value & IT::CONST) {
        DeviceTensorND ival = mgr.infer_value(ivar);
        // remove boradcasted axis
        auto layout = ival.layout();
        for (int i = layout.ndim - 1; i >= 0; -- i) {
            if (!layout.stride[i] && layout.ndim >= 2)
                layout.remove_axis_inplace(i);
        }
        if (layout.is_scalar() || (layout.ndim == 1 && !layout.stride[0])) {
            return DTypeScalar::make_from_raw(ival.dtype(), ival.raw_ptr());
        }
    }
    return None;
}

Maybe<DTypeScalar> SymbolVar::as_immutable_scalar_require_shape() const {
    if (!shape().is_scalar())
        return None;
    return as_immutable_scalar();
}

SymbolVar SymbolVar::operator + (const SymbolVar &rhs) const {
    return opr::add(*this, rhs);
}

SymbolVar SymbolVar::operator - (const SymbolVar &rhs) const {
    return opr::sub(*this, rhs);
}

SymbolVar SymbolVar::operator * (const SymbolVar &rhs) const {
    return opr::mul(*this, rhs);
}

SymbolVar SymbolVar::operator / (const SymbolVar &rhs) const {
    if (dtype().category() == DTypeCategory::INT &&
            rhs.dtype().category() == DTypeCategory::INT) {
        return opr::floor_div(*this, rhs);
    }
    return opr::div(*this, rhs);
}

SymbolVar SymbolVar::operator < (const SymbolVar &rhs) const {
    return opr::less_than(*this, rhs);
}

SymbolVar SymbolVar::operator <= (const SymbolVar &rhs) const {
    return opr::less_equal(*this, rhs);
}

SymbolVar SymbolVar::operator - () const {
    return opr::negate(*this);
}

SymbolVar SymbolVar::make_scalar(
        DTypeScalar value, ComputingGraph &cg, CompNode cn) {
    return opr::ImmutableTensor::make(cg, value, {cn});
}

const DeviceTensorND& SymbolVar::eager_eval_get_value() const {
#if MGB_BUILD_SLIM_SERVING
    mgb_throw(MegBrainError, "eager eval disabled at compile time");
#else
    auto og = ComputingGraphImpl::downcast(node()->owner_graph());
    mgb_assert(og->options().eager_evaluation);
    return node()->dev_tensor();
#endif
}

void VarNodeArrayView::check_idx(size_t idx) const {
    mgb_assert(m_begin + idx < m_end, "idx out of range: %zu/%td", idx,
               m_end - m_begin);
}

void SymbolVarArrayView::check_idx(size_t idx) const {
    mgb_assert(m_begin + idx < m_end, "idx out of range: %zu/%td", idx,
               m_end - m_begin);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

