/**
 * \file lite/load_and_run/src/options/model_options.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#pragma once
#include <gflags/gflags.h>
#include "megbrain/graph/operator_node.h"
#include "models/model.h"
#include "option_base.h"

DECLARE_string(packed_model_dump);
DECLARE_string(pack_info_json);
DECLARE_string(pack_cache);
DECLARE_string(pack_info_cryption);
DECLARE_string(pack_model_cryption);

namespace lar {
class PackModelOption : public OptionBase {
public:
    static bool is_valid();
    static std::shared_ptr<OptionBase> create_option();
    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;
    std::string option_name() const override { return m_option_name; }

private:
    PackModelOption();

    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>);

    std::string m_option_name;
    std::string packed_model_dump;
    std::string pack_info_json;
    std::string pack_cache;
    std::string pack_binary_cache;
    std::string pack_info_cryption;
    std::string pack_model_cryption;
    bool is_fast_run_cache = true;
};
}  // namespace lar
