#include <gflags/gflags.h>

#include "misc.h"
#include "models/model_lite.h"
#include "models/model_mdl.h"

#include "layout_options.h"
namespace lar {
template <>
void LayoutOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
#define ENABLE_LAYOUT(layout)                           \
    LITE_WARN("enable " #layout " optimization");       \
    model->get_config().options.enable_##layout = true; \
    break;

        switch (m_option_flag) {
            case OptLayoutType::NCHW4:
                ENABLE_LAYOUT(nchw4)

            case OptLayoutType::CHWN4:
                LITE_THROW("lite model unsupport chwn4 layout");
                break;
            case OptLayoutType::NCHW44:
                ENABLE_LAYOUT(nchw44)

            case OptLayoutType::NCHW88:
                ENABLE_LAYOUT(nchw88)

            case OptLayoutType::NCHW32:
                ENABLE_LAYOUT(nchw32)

            case OptLayoutType::NCHW64:
                ENABLE_LAYOUT(nchw64)

            case OptLayoutType::NHWCD4:
                ENABLE_LAYOUT(nhwcd4)

            case OptLayoutType::NCHW44_DOT:
                ENABLE_LAYOUT(nchw44_dot)
            default:
                break;
        }
#undef ENABLE_LAYOUT
    }
}

template <>
void lar::LayoutOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
#define ENABLE_LAYOUT(layout)                                                  \
    mgb_log_warn("enable " #layout " optimization");                           \
    model->get_mdl_config().comp_graph->options().graph_opt.enable_##layout(); \
    break;

        switch (m_option_flag) {
            case OptLayoutType::NCHW4:
                ENABLE_LAYOUT(nchw4)

            case OptLayoutType::CHWN4:
                ENABLE_LAYOUT(chwn4)

            case OptLayoutType::NCHW44:
                ENABLE_LAYOUT(nchw44)

            case OptLayoutType::NCHW88:
                ENABLE_LAYOUT(nchw88)

            case OptLayoutType::NCHW32:
                ENABLE_LAYOUT(nchw32)

            case OptLayoutType::NCHW64:
                ENABLE_LAYOUT(nchw64)

            case OptLayoutType::NHWCD4:
                ENABLE_LAYOUT(nhwcd4)

            case OptLayoutType::NCHW44_DOT:
                ENABLE_LAYOUT(nchw44_dot)

            default:
                break;
        }

#undef ENABLE_LAYOUT
    }
}
}  // namespace lar

using namespace lar;
bool LayoutOption::m_valid;
LayoutOption::LayoutOption() {
    m_option_name = "layout";
    m_option_flag = static_cast<OptLayoutType>(0);
    m_option = {
            {"enable_nchw4", lar::Bool::make(false)},
            {"enable_chwn4", lar::Bool::make(false)},
            {"enable_nchw44", lar::Bool::make(false)},
            {"enable_nchw88", lar::Bool::make(false)},
            {"enable_nchw32", lar::Bool::make(false)},
            {"enable_nchw64", lar::Bool::make(false)},
            {"enable_nhwcd4", lar::Bool::make(false)},
            {"enable_nchw44_dot", lar::Bool::make(false)},
    };
    std::static_pointer_cast<lar::Bool>(m_option["enable_nchw4"])
            ->set_value(FLAGS_enable_nchw4);
    std::static_pointer_cast<lar::Bool>(m_option["enable_chwn4"])
            ->set_value(FLAGS_enable_chwn4);
    std::static_pointer_cast<lar::Bool>(m_option["enable_nchw44"])
            ->set_value(FLAGS_enable_nchw44);
    std::static_pointer_cast<lar::Bool>(m_option["enable_nchw88"])
            ->set_value(FLAGS_enable_nchw88);
    std::static_pointer_cast<lar::Bool>(m_option["enable_nchw32"])
            ->set_value(FLAGS_enable_nchw32);
    std::static_pointer_cast<lar::Bool>(m_option["enable_nchw64"])
            ->set_value(FLAGS_enable_nchw64);
    std::static_pointer_cast<lar::Bool>(m_option["enable_nhwcd4"])
            ->set_value(FLAGS_enable_nhwcd4);
    std::static_pointer_cast<lar::Bool>(m_option["enable_nchw44_dot"])
            ->set_value(FLAGS_enable_nchw44_dot);
}

bool LayoutOption::is_valid() {
    size_t valid_flag = 0;
    if (FLAGS_enable_nchw4) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW4);
    }
    if (FLAGS_enable_chwn4) {
        valid_flag |= static_cast<size_t>(OptLayoutType::CHWN4);
    }
    if (FLAGS_enable_nchw44) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW44);
    }
    if (FLAGS_enable_nchw88) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW88);
    }
    if (FLAGS_enable_nchw32) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW32);
    }
    if (FLAGS_enable_nchw64) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW64);
    }
    if (FLAGS_enable_nhwcd4) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NHWCD4);
    }
    if (FLAGS_enable_nchw44_dot) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW44_DOT);
    }
    //! only one flag is valid
    bool ret = valid_flag && !(valid_flag & (valid_flag - 1));

    return ret | m_valid;
};

std::shared_ptr<OptionBase> LayoutOption::create_option() {
    static std::shared_ptr<LayoutOption> option(new LayoutOption);
    if (LayoutOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void LayoutOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    size_t valid_flag = 0;
    if (std::static_pointer_cast<lar::Bool>(m_option["enable_nchw4"])->get_value()) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW4);
    }
    if (std::static_pointer_cast<lar::Bool>(m_option["enable_chwn4"])->get_value()) {
        valid_flag |= static_cast<size_t>(OptLayoutType::CHWN4);
    }
    if (std::static_pointer_cast<lar::Bool>(m_option["enable_nchw44"])->get_value()) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW44);
    }
    if (std::static_pointer_cast<lar::Bool>(m_option["enable_nchw88"])->get_value()) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW88);
    }
    if (std::static_pointer_cast<lar::Bool>(m_option["enable_nchw32"])->get_value()) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW32);
    }
    if (std::static_pointer_cast<lar::Bool>(m_option["enable_nchw64"])->get_value()) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW64);
    }
    if (std::static_pointer_cast<lar::Bool>(m_option["enable_nhwcd4"])->get_value()) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NHWCD4);
    }
    if (std::static_pointer_cast<lar::Bool>(m_option["enable_nchw44_dot"])
                ->get_value()) {
        valid_flag |= static_cast<size_t>(OptLayoutType::NCHW44_DOT);
    }

    mgb_throw_if(
            valid_flag && (valid_flag & (valid_flag - 1)), mgb::AssertionError,
            "invalid options of layout transform 0x%lx", valid_flag);
    m_option_flag = static_cast<OptLayoutType>(valid_flag);
    CONFIG_MODEL_FUN;
}

DEFINE_bool(enable_nchw4, false, "enable nchw4 layout optimization!!");
DEFINE_bool(enable_chwn4, false, "enable chwn4 layout optimization!!");
DEFINE_bool(enable_nchw44, false, "enable nchw44 layout optimization!!");
DEFINE_bool(enable_nchw88, false, "enable nchw88 layout optimization!!");
DEFINE_bool(enable_nchw32, false, "enable nchw32 layout optimization!!");
DEFINE_bool(enable_nchw64, false, "enable nchw64 layout optimization!!");
DEFINE_bool(enable_nhwcd4, false, "enable nhwcd4 layout optimization!!");
DEFINE_bool(enable_nchw44_dot, false, "enable nchw444-dot layout optimization!!");

REGIST_OPTION_CREATOR(layout, lar::LayoutOption::create_option);
REGIST_OPTION_VALIDATER(layout, lar::LayoutOption::set_valid);