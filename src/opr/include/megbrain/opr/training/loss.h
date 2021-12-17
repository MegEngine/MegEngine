/**
 * \file src/opr/include/training/loss.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/tensor.h"

namespace mgb {
namespace loss {
//! The interface of losses which should be inherited by each loss class.
class ILoss {
public:
    /*!
     * The reduce mode of loss to convert output to scalar.
     */
    enum ReduceMode { SUM = 0, MEAN = 1 };
    /*!
     * The calculation of the loss, in which the output is a scalar symbolvar
     */
    virtual mgb::SymbolVar operator()(
            mgb::SymbolVar symbol_pred, mgb::SymbolVar symol_label) = 0;
    virtual ~ILoss() = default;
};

/*!
 * The cross entropy loss. The definition could be found here:
 * https://en.wikipedia.org/wiki/Cross_entropy
 *
 * It's corresponding to the <CrossEntropy> of Python API of MegEngine.
 */
class CrossEntropyLoss : public ILoss {
public:
    CrossEntropyLoss(
            bool with_logits = true, float label_smooth = .0f,
            ReduceMode reduce_mode = ReduceMode::MEAN, int axis = 1);
    mgb::SymbolVar operator()(mgb::SymbolVar symbol_pred, mgb::SymbolVar symol_label);

protected:
    bool m_with_logits;
    float m_label_smooth;
    ReduceMode m_reduce_mode;
    int m_axis;
};

/*!
 * The MSE(Mean Square Error) loss. The definition could be found here:
 * https://en.wikipedia.org/wiki/Mean_squared_error
 *
 * It's corresponding to the <MSE> of Python API of MegEngine.
 */
class MSELoss : public ILoss {
public:
    MSELoss(ReduceMode reduce_mode = ReduceMode::MEAN);
    mgb::SymbolVar operator()(mgb::SymbolVar symbol_pred, mgb::SymbolVar symol_label);

protected:
    ReduceMode m_reduce_mode;
};
}  // namespace loss

}  // namespace mgb
