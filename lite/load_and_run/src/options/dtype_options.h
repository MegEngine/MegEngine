#pragma once

#include <gflags/gflags.h>
#include "helpers/common.h"
#include "models/model.h"
#include "option_base.h"

DECLARE_bool(enable_ioc16);

namespace lar {
/*!
 * \brief: dtype option for optimization
 */
class DTypeOption final : public OptionBase {
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

    static void set_valid(bool val) { m_valid = val; }

    OptionValMap* get_option() override { return &m_option; }

    void update() override;

private:
    //! Constructor
    DTypeOption() = default;

    //! configuration for different model implement
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};

    OptDTypeType m_option_flag;
    std::string m_option_name;
    static bool m_valid;
    OptionValMap m_option;
};
}  // namespace lar