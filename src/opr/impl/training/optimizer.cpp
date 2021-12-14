/**
 * \file src/opr/impl/training/optimizer.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/opr/training/optimizer.h"
#include "megbrain/exception.h"
#include "megbrain/opr/training/utils.h"

namespace mgb {
namespace optimizer {
SymbolVarArray Optimizer::make_multiple(
        SymbolVarArray symbol_weights, SymbolVarArray symbol_grads,
        std::shared_ptr<mgb::cg::ComputingGraph> graph) {
    if (symbol_weights.size() != symbol_grads.size()) {
        mgb_throw(AssertionError, "The count of weights differs with that of grads.");
    }

    SymbolVarArray r;
    for (size_t i = 0; i < symbol_weights.size(); i++) {
        r.push_back(make(symbol_weights[i], symbol_grads[i], graph));
    }
    return r;
}

SGD::SGD(float lr, float weight_decay, float momentum)
        : m_lr(lr), m_weight_decay(weight_decay), m_momentum(momentum) {
    if (m_lr <= 0) {
        mgb_throw(AssertionError, "Invalid learning rate: negative value.");
    }
    if (m_weight_decay < 0) {
        mgb_throw(AssertionError, "Invalid weight_decay value: negative value.");
    }
    if (m_momentum < 0) {
        mgb_throw(AssertionError, "Invalid momentum value: negative value.");
    }
}

SymbolVar SGD::make(
        SymbolVar symbol_weight, SymbolVar symbol_grad,
        std::shared_ptr<cg::ComputingGraph> graph) {
    SymbolVar symbol_pre_grad;
    auto pre_grad = TensorGen::zeros<dtype::Float32>(
            symbol_grad.shape(), symbol_grad.node()->comp_node());
    m_pre_grads.push_back(pre_grad);
    symbol_pre_grad = opr::SharedDeviceTensor::make(*graph, *pre_grad);

    if (m_weight_decay != .0f) {
        symbol_grad = symbol_grad + m_weight_decay * symbol_weight;
    }

    if (m_momentum != .0f) {
        symbol_pre_grad =
                opr::AddUpdate::make(symbol_pre_grad, symbol_grad, {m_momentum, 1.0f});
        return opr::AddUpdate::make(symbol_weight, -symbol_pre_grad, {1.f, m_lr});
    } else {
        return opr::AddUpdate::make(symbol_weight, -symbol_grad, {1.f, m_lr});
    }
}

Adam::Adam(
        float lr, float weight_decay, std::pair<float, float> betas, float eps,
        bool amsgrad)
        : m_lr(lr),
          m_weight_decay(weight_decay),
          m_betas(betas),
          m_eps(eps),
          m_amsgrad(amsgrad) {
    mgb_assert(m_lr > 0, "Invalid learning rate: negative value.");
    mgb_assert(m_weight_decay >= 0, "Invalid weight_decay value: negative value.");
    mgb_assert(
            m_betas.first >= 0 && m_betas.second >= 0 && m_betas.first < 1 &&
                    m_betas.second < 1,
            "Invalid betas value: negative value or larger than 1.");
}

SymbolVar Adam::make(
        SymbolVar symbol_weight, SymbolVar symbol_grad,
        std::shared_ptr<cg::ComputingGraph> graph) {
    CompNode comp_node = symbol_grad.node()->comp_node();
    DType dt = symbol_grad.dtype();
    m_correction1 = TensorGen::ones<dtype::Float32>({1}, comp_node);
    m_correction2 = TensorGen::ones<dtype::Float32>({1}, comp_node);
    std::shared_ptr<DeviceTensorND> exp_avg =
            std::make_shared<DeviceTensorND>(comp_node, symbol_grad.shape(), dt);
    mgb::fill_zero_dev_tensor(*exp_avg);
    std::shared_ptr<DeviceTensorND> exp_avg_sq =
            std::make_shared<DeviceTensorND>(comp_node, symbol_grad.shape(), dt);
    mgb::fill_zero_dev_tensor(*exp_avg_sq);
    m_exp_avg.push_back(exp_avg);
    m_exp_avg_sq.push_back(exp_avg_sq);

    SymbolVar symbol_correction1 =
            opr::SharedDeviceTensor::make(*graph, *m_correction1);
    SymbolVar symbol_correction2 =
            opr::SharedDeviceTensor::make(*graph, *m_correction2);
    SymbolVar symbol_exp_avg = opr::SharedDeviceTensor::make(*graph, exp_avg);
    SymbolVar symbol_exp_avg_sq = opr::SharedDeviceTensor::make(*graph, exp_avg_sq);

    symbol_correction1 = opr::AddUpdate::make(
            symbol_correction1, symbol_correction1, {m_betas.first, .0f});
    symbol_correction2 = opr::AddUpdate::make(
            symbol_correction2, symbol_correction2, {m_betas.second, .0f});

    if (m_weight_decay != .0f) {
        symbol_grad = symbol_grad + m_weight_decay * symbol_weight;
    }
    symbol_exp_avg = opr::AddUpdate::make(
            symbol_exp_avg, symbol_grad, {m_betas.first, 1.f - m_betas.first});
    symbol_exp_avg_sq = opr::AddUpdate::make(
            symbol_exp_avg_sq, symbol_grad * symbol_grad,
            {m_betas.second, 1.f - m_betas.second});

    SymbolVar delta;
    if (m_amsgrad) {
        std::shared_ptr<DeviceTensorND> max_exp_avg_sq =
                std::make_shared<DeviceTensorND>(comp_node, symbol_grad.shape(), dt);
        mgb::fill_zero_dev_tensor(*max_exp_avg_sq);
        SymbolVar symbol_max_exp_avg_sq =
                opr::SharedDeviceTensor::make(*graph, max_exp_avg_sq);

        symbol_max_exp_avg_sq = opr::AddUpdate::make(
                symbol_exp_avg_sq, opr::max(symbol_max_exp_avg_sq, symbol_exp_avg_sq),
                {1.0f, 1.0f});
        delta = (symbol_exp_avg / (1.f - symbol_correction1)) /
                (opr::powf(symbol_max_exp_avg_sq / (1.f - symbol_correction2), 0.5f) +
                 m_eps);
    } else {
        delta = (symbol_exp_avg / (1.f - symbol_correction1)) /
                (opr::pow(
                         symbol_exp_avg_sq / (1.f - symbol_correction2),
                         symbol_exp_avg.make_scalar(0.5f)) +
                 m_eps);
    }

    return opr::AddUpdate::make(symbol_weight, -delta, {1.0f, m_lr});
}
}  // namespace optimizer
}  // namespace mgb
