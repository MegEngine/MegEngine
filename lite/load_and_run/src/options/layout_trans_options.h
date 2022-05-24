#pragma once

#include <gflags/gflags.h>
#include "megbrain/gopt/inference.h"
#include "models/model.h"
#include "option_base.h"
DECLARE_string(layout_transform);
DECLARE_string(layout_transform_dump);

namespace lar {
class GoptLayoutOption final : public OptionBase {
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

    static void set_valid(bool val) { m_valid = val; }

    OptionValMap* get_option() override { return &m_option; }

private:
    GoptLayoutOption();
    //! config template for different model
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>) {}
    bool m_layout_transform;
    std::string m_option_name;
    std::string m_layout_transform_dump_file;
    mgb::gopt::GraphTuningOptions::Target m_layout_transform_target;
    static bool m_valid;
    OptionValMap m_option;
};
}  // namespace lar
