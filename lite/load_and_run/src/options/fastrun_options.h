/**
 * \file lite/load_and_run/src/options/fastrun_options.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#pragma once

#include <gflags/gflags.h>
#include "models/model.h"
#include "option_base.h"

#if MGB_ENABLE_FASTRUN
DECLARE_bool(fast_run);
DECLARE_bool(full_run);
#endif
DECLARE_bool(reproducible);
DECLARE_bool(binary_equal_between_batch);
DECLARE_uint32(fast_run_shared_batch_size);
DECLARE_string(fast_run_algo_policy);

namespace lar {
class FastRunOption final : public OptionBase {
public:
    //! get condition for construct FastRunOption
    static bool is_valid();

    //! creat option using condition from cmdline args
    static std::shared_ptr<OptionBase> create_option();

    //! configure model for different runtime_param
    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    //! get options name for quickly search
    std::string option_name() const override { return m_option_name; }

private:
    FastRunOption();
    //! config template for different model
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>) {}

#if MGB_ENABLE_FASTRUN
    bool enable_fast_run;  //! fast run strategy flag
    bool enable_full_run;  //! full run strategy flag
#endif
    bool batch_binary_equal;       //! fast run stratgey setting
    bool enable_reproducible;      //! enable reproducible strategy
    size_t share_batch_size;       //! fast run strategy share batch size setting
    std::string m_fast_run_cache;  //! fast run cache file path
    std::string m_option_name;     //! option name
};
}  // namespace lar
