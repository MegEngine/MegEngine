/**
 * \file lite/load_and_run/src/options/layout_options.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#pragma once

#include <gflags/gflags.h>
#include "helpers/common.h"
#include "models/model.h"
#include "option_base.h"

DECLARE_bool(enable_nchw4);
DECLARE_bool(enable_chwn4);
DECLARE_bool(enable_nchw44);
DECLARE_bool(enable_nchw88);
DECLARE_bool(enable_nchw32);
DECLARE_bool(enable_nchw64);
DECLARE_bool(enable_nhwcd4);
DECLARE_bool(enable_nchw44_dot);

namespace lar {
/*!
 * \brief: layout option for optimization
 */
class LayoutOption final : public OptionBase {
public:
    //! check the validation  of option flag
    static bool is_valid();

    //! creat options when option is used
    static std::shared_ptr<OptionBase> create_option();

    //! config the model, dispatch configuration for different model implement
    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    //! get option name
    std::string option_name() const override { return m_option_name; };

private:
    //! Constructor
    LayoutOption();

    //! configuration for different model implement
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};

    static OptLayoutType option_flag;
    std::string m_option_name;
};
}  // namespace lar