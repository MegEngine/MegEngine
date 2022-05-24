#pragma once
#include <gflags/gflags.h>
#if __linux__ || __unix__
#include <unistd.h>
#endif
#include "megbrain/plugin/cpu_dispatch_checker.h"
#include "megbrain/plugin/var_value_checker.h"

#include "helpers/common.h"
#include "helpers/text_table.h"
#include "models/model.h"

#include "option_base.h"

DECLARE_bool(check_dispatch);
DECLARE_double(range);
DECLARE_string(check_var_value);
#if MGB_ENABLE_JSON
DECLARE_string(profile);
DECLARE_string(profile_host);
#endif

DECLARE_bool(model_info);
DECLARE_bool(verbose);
DECLARE_bool(disable_assert_throw);
#if __linux__ || __unix__
DECLARE_bool(wait_gdb);
#endif
#ifndef __IN_TEE_ENV__
#if MGB_ENABLE_JSON
DECLARE_string(get_static_mem_info);
#endif
#endif

namespace lar {
class PluginOption final : public OptionBase {
public:
    static bool is_valid();

    static std::shared_ptr<OptionBase> create_option();

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    std::string option_name() const override { return m_option_name; };

private:
    PluginOption();
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};
    double range;
    bool enable_check_dispatch;
#if MGB_ENABLE_JSON
    bool enable_profile_host;
    std::string profile_path;
#endif

    std::string var_value_check_str;

    std::string m_option_name;

    std::unique_ptr<mgb::VarValueChecker> var_value_checker;
    std::unique_ptr<mgb::CPUDispatchChecker> cpu_dispatch_checker;
};

class DebugOption final : public OptionBase {
public:
    static bool is_valid();

    static std::shared_ptr<OptionBase> create_option();

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    std::string option_name() const override { return m_option_name; };

private:
    DebugOption();
    template <typename ModelImpl>
    void format_and_print(const std::string&, std::shared_ptr<ModelImpl>){};
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};
    bool enable_display_model_info;
    bool enable_verbose;
    bool disable_assert_throw;
#if __linux__ || __unix__
    bool enable_wait_gdb;
#endif
#ifndef __IN_TEE_ENV__
#if MGB_ENABLE_JSON
    std::string static_mem_log_dir_path;
#endif
#endif
    std::string m_option_name;
};
}  // namespace lar