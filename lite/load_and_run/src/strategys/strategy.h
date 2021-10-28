/**
 * \file lite/load_and_run/src/strategys/strategy.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#pragma once
#include <gflags/gflags.h>
#include <string>
#include <unordered_map>
#include "helpers/common.h"
#include "models/model.h"
#include "options/option_base.h"

DECLARE_bool(fitting);

namespace lar {
/*!
 * \brief: load and run strategy base class
 */
class StrategyBase {
public:
    static std::shared_ptr<StrategyBase> create_strategy(std::string model_path);

    virtual void run() = 0;

    virtual ~StrategyBase() = default;

    RuntimeParam m_runtime_param;
    std::unordered_map<std::string, std::shared_ptr<OptionBase>> m_options;
};

/*!
 * \brief: normal strategy for running
 */
class NormalStrategy : public StrategyBase {
public:
    NormalStrategy(std::string model_path);

    //! run model with runtime parameter
    void run() override;

private:
    //! run model subline for multiple thread
    void run_subline();

    std::string m_model_path;
};

/*!
 * \brief: Fitting strategy for running
 */
class FittingStrategy : public StrategyBase {
public:
    FittingStrategy(std::string model_path);
    void run() override;
};
}  // namespace lar

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
