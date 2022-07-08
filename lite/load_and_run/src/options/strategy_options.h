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

    OptionValMap* get_option() override { return &m_option; }

    void update() override;

private:
    //! Constructor
    StrategyOption() = default;

    //! configuration for different model implement
    std::string m_option_name;

    size_t warmup_iter;  //! warm up number before running model
    size_t run_iter;     //! iteration number for running model
    size_t threads;      //! thread number for running model (NOTE:it's different
                         //! from multithread device )
    OptionValMap m_option;
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

    void update() override;

private:
    //! Constructor
    TestcaseOption() = default;

    //! configuration for different model implement
    std::string m_option_name;
};
}  // namespace lar