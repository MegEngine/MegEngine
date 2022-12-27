#include <gflags/gflags.h>

#include "misc.h"
#include "models/model_lite.h"
#include "models/model_mdl.h"

#include "dtype_options.h"
namespace lar {
template <>
void DTypeOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
#define ENABLE_DTYPE(dtype)                            \
    LITE_LOG("enable " #dtype " optimization");        \
    model->get_config().options.enable_##dtype = true; \
    break;

        switch (m_option_flag) {
            case OptDTypeType::IOC16:
                ENABLE_DTYPE(f16_io_comp)
            default:
                LITE_THROW(
                        "Set unsupport dtype, only --enable-ioc16 is supported. "
                        "Default case is fp32.");
                break;
        }
#undef ENABLE_DTYPE
    }
}

template <>
void DTypeOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
#define ENABLE_DTYPE(dtype)                                                   \
    mgb_log("enable " #dtype " optimization");                                \
    model->get_mdl_config().comp_graph->options().graph_opt.enable_##dtype(); \
    break;

        switch (m_option_flag) {
            case OptDTypeType::IOC16:
                ENABLE_DTYPE(f16_io_comp)
            default:
                LITE_THROW(
                        "Set unsupport dtype, only --enable-ioc16 is supported. "
                        "Default case is fp32.");
                break;
        }

#undef ENABLE_DTYPE
    }
}
}  // namespace lar

using namespace lar;
bool DTypeOption::m_valid;
void DTypeOption::update() {
    m_option_name = "dtype";
    m_option_flag = static_cast<OptDTypeType>(0);
    m_option = {
            {"enable_ioc16", lar::Bool::make(false)},
    };
    std::static_pointer_cast<lar::Bool>(m_option["enable_ioc16"])
            ->set_value(FLAGS_enable_ioc16);
}

bool DTypeOption::is_valid() {
    size_t valid_flag = 0;
    if (FLAGS_enable_ioc16) {
        valid_flag |= static_cast<size_t>(OptDTypeType::IOC16);
    }
    //! only one flag is valid
    bool ret = valid_flag && !(valid_flag & (valid_flag - 1));

    return ret | m_valid;
};

std::shared_ptr<OptionBase> DTypeOption::create_option() {
    static std::shared_ptr<DTypeOption> option(new DTypeOption);
    if (DTypeOption::is_valid()) {
        option->update();
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void DTypeOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    size_t valid_flag = 0;
    if (FLAGS_enable_ioc16 ||
        std::static_pointer_cast<lar::Bool>(m_option["enable_ioc16"])->get_value()) {
        valid_flag |= static_cast<size_t>(OptDTypeType::IOC16);
    }

    mgb_throw_if(
            valid_flag && (valid_flag & (valid_flag - 1)), mgb::AssertionError,
            "invalid options of dtype transform 0x%lx", valid_flag);
    m_option_flag = static_cast<OptDTypeType>(valid_flag);
    CONFIG_MODEL_FUN;
}

DEFINE_bool(enable_ioc16, false, "enable fp16 dtype optimization!!");

REGIST_OPTION_CREATOR(dtype, lar::DTypeOption::create_option);
REGIST_OPTION_VALIDATER(dtype, lar::DTypeOption::set_valid);