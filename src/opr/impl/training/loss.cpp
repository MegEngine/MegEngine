/**
 * \file src/opr/impl/training/loss.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/opr/training/loss.h"
#include "megbrain/exception.h"
#include "megbrain/opr/indexing.h"

namespace mgb {
namespace loss {
CrossEntropyLoss::CrossEntropyLoss(
        bool with_logits, float label_smooth, ReduceMode reduce_mode, int axis)
        : m_with_logits(with_logits),
          m_label_smooth(label_smooth),
          m_reduce_mode(reduce_mode),
          m_axis(axis) {}

SymbolVar CrossEntropyLoss::operator()(
        mgb::SymbolVar symbol_pred, mgb::SymbolVar symbol_label) {
    mgb_assert(
            symbol_pred.shape().ndim >= symbol_label.shape().ndim,
            "The label must have less dimensions than the pred.");
    for (size_t i = 0; i < symbol_label.shape().ndim; i++) {
        mgb_assert(
                symbol_pred.shape()[i] == symbol_label.shape()[i] || (int)i == m_axis,
                "Unmatched shape for pred and label.");
    }
    mgb_assert(m_label_smooth >= .0f, "The label_smmoth must be positive value");

    SymbolVar symbol_loss;
    SymbolVar symbol_middle;

    SymbolVar symbol_max = opr::reduce_ax_max(symbol_pred, m_axis);
    SymbolVar symbol_primary_item =
            opr::IndexingOneHot::make(symbol_pred, symbol_label, {m_axis});
    if (m_with_logits) {
        symbol_middle = opr::reduce_ax_sum(symbol_pred, m_axis) /
                        opr::GetVarShape::make(symbol_pred, {m_axis});
        SymbolVar symbol_logits =
                symbol_max + opr::log(opr::reduce_ax_sum(
                                     opr::exp(symbol_pred - symbol_max), m_axis));

        symbol_loss = symbol_logits;
    } else {
        symbol_middle = opr::reduce_ax_sum(opr::log(symbol_pred), m_axis) /
                        opr::GetVarShape::make(symbol_pred, {m_axis});
        symbol_primary_item = opr::log(symbol_primary_item);
    }

    if (m_label_smooth > .0f) {
        symbol_loss = symbol_loss - m_label_smooth * symbol_middle -
                      (1 - m_label_smooth) * symbol_primary_item;
    } else {
        symbol_loss = symbol_loss - symbol_primary_item;
    }

    if (m_reduce_mode == ReduceMode::MEAN) {
        symbol_loss =
                opr::reduce_sum(symbol_loss.flatten(), symbol_loss.make_scalar(1)) /
                (float)(symbol_loss.shape().total_nr_elems());
    } else if (m_reduce_mode == ReduceMode::SUM) {
        symbol_loss =
                opr::reduce_sum(symbol_loss.flatten(), symbol_loss.make_scalar(1));
    }

    return symbol_loss;
}

MSELoss::MSELoss(ReduceMode reduce_mode) : m_reduce_mode(reduce_mode){};

mgb::SymbolVar MSELoss::operator()(
        mgb::SymbolVar symbol_pred, mgb::SymbolVar symol_label) {
    return opr::pow(symbol_pred - symol_label, symbol_pred.make_scalar(2));
}
}  // namespace loss

}  // namespace mgb
