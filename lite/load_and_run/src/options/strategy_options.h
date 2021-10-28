/**
 * \file lite/load_and_run/src/options/strategy_options.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#include <gflags/gflags.h>
#include "models/model.h"
#include "option_base.h"
DECLARE_int32(iter);
DECLARE_int32(warmup_iter);
DECLARE_int32(thread);
DECLARE_bool(share_param_mem);

namespace lar {
/*!
 * \brief: strategy option for running model
 */
class StrategyOption final : public OptionBase {
public:
    //! creat options when option is used
    static std::shared_ptr<OptionBase> create_option();

    //! config the model, dispatch configuration for different model implement

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    //! get option name
    std::string option_name() const override { return m_option_name; };

private:
    //! Constructor
    StrategyOption();

    //! configuration for different model implement
    std::string m_option_name;

    size_t warmup_iter;  //! warm up number before running model
    size_t run_iter;     //! iteration number for running model
    size_t threads;      //! thread number for running model (NOTE:it's different
                         //! from multithread device )
};

class TestcaseOption final : public OptionBase {
public:
    //! creat options when option is used
    static std::shared_ptr<OptionBase> create_option();

    //! config the model, dispatch configuration for different model implement

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    //! get option name
    std::string option_name() const override { return m_option_name; };

private:
    //! Constructor
    TestcaseOption();

    //! configuration for different model implement
    std::string m_option_name;
};
}  // namespace lar