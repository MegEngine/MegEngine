/**
 * \file src/opr/include/training/optimizer.h
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
namespace optimizer {
//! The interface of optimizers which should be inherited by each optimizer.
class IOptimizer {
public:
    /*!
     * The method to add manipulations to the graph to update the weight when the
     * input is SymbolvarArrays.
     */
    virtual mgb::SymbolVarArray make_multiple(
            mgb::SymbolVarArray symbol_weights, mgb::SymbolVarArray symbol_grads,
            std::shared_ptr<mgb::cg::ComputingGraph> graph) = 0;
    /*!
     * The method to add manipulations to the graph to update the weight with a
     * certain strategy.
     * The output is expected to be the symbolvar after updating the weight.
     */
    virtual mgb::SymbolVar make(
            mgb::SymbolVar symbol_weight, mgb::SymbolVar symbol_grad,
            std::shared_ptr<mgb::cg::ComputingGraph> graph) = 0;
    virtual ~IOptimizer() = default;
};

/*!
 * An abstract class which helps to simplify the implemention of optimizers.
 * It gives a default implemention of method <make_multiple> based on the method
 * <make> defined by its derived class.
 */
class Optimizer : public IOptimizer {
public:
    mgb::SymbolVarArray make_multiple(
            mgb::SymbolVarArray symbol_weights, mgb::SymbolVarArray symbol_grads,
            std::shared_ptr<mgb::cg::ComputingGraph> graph);
    virtual mgb::SymbolVar make(
            mgb::SymbolVar symbol_weight, mgb::SymbolVar symbol_grad,
            std::shared_ptr<mgb::cg::ComputingGraph> graph) = 0;
    virtual ~Optimizer() = default;
};

/*!
 * The SGD(Stochastic gradient descent) optimizer.
 * The definition could be found here:
 * https://en.wikipedia.org/wiki/Stochastic_gradient_descent
 * It is corresponding to the <SGD> of Python API of MegEngine.
 */
class SGD : public Optimizer {
public:
    SGD() = default;
    SGD(float lr, float weight_decay = .0f, float momentum = .0f);

    SGD(const SGD& that) {
        m_lr = that.m_lr;
        m_momentum = that.m_momentum;
        m_weight_decay = that.m_weight_decay;
    }
    mgb::SymbolVar make(
            mgb::SymbolVar symbol_weight, mgb::SymbolVar symbol_grad,
            std::shared_ptr<mgb::cg::ComputingGraph> graph);

    const SGD& operator=(const SGD& that) {
        m_lr = that.m_lr;
        m_momentum = that.m_momentum;
        m_weight_decay = that.m_weight_decay;
        return *this;
    }

protected:
    float m_lr;
    float m_weight_decay;
    float m_momentum;
    std::vector<std::shared_ptr<mgb::HostTensorND>> m_pre_grads;
};

/*!
 * The Adam optimizer. The definition could be found here:
 * https://en.wikipedia.org/wiki/Stochastic_gradient_descent#:~:text=full%2Dbatches.%5B26%5D-,Adam,-%5Bedit%5D
 * It is corresponding to the <Adam> of Python API of MegEngine.
 */
class Adam : public Optimizer {
public:
    Adam() = default;
    Adam(float lr, float weight_decay = .0f,
         std::pair<float, float> betas = {0.9f, 0.999f}, float eps = 1e-8f,
         bool amsgrad = false);

    Adam(const Adam& that) {
        m_lr = that.m_lr;
        m_betas = that.m_betas;
        m_eps = that.m_eps;
        m_weight_decay = that.m_weight_decay;
        m_amsgrad = that.m_amsgrad;
    }
    mgb::SymbolVar make(
            mgb::SymbolVar symbol_weight, mgb::SymbolVar symbol_grad,
            std::shared_ptr<mgb::cg::ComputingGraph> graph);

    const Adam& operator=(const Adam& that) {
        m_lr = that.m_lr;
        m_betas = that.m_betas;
        m_eps = that.m_eps;
        m_weight_decay = that.m_weight_decay;
        m_amsgrad = that.m_amsgrad;
        return *this;
    }

protected:
    float m_lr;
    float m_weight_decay;
    std::pair<float, float> m_betas;
    float m_eps;
    bool m_amsgrad;
    std::vector<std::shared_ptr<mgb::DeviceTensorND>> m_exp_avg;
    std::vector<std::shared_ptr<mgb::DeviceTensorND>> m_exp_avg_sq;
    std::vector<std::shared_ptr<mgb::DeviceTensorND>> m_max_exp_avg_sq;
    std::shared_ptr<mgb::HostTensorND> m_correction1;
    std::shared_ptr<mgb::HostTensorND> m_correction2;
};
}  // namespace optimizer
}  // namespace mgb