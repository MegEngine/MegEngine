#pragma once
#include <gflags/gflags.h>
#include "helpers/utils.h"
#include "strategy.h"
DECLARE_bool(fitting);
DECLARE_string(dump_fitting_model);
#define TIME_OUT 10000
namespace lar {

class OptionsFastManager {
public:
    using ConstraintMap = std::unordered_map<std::string, bool>;

    OptionsFastManager(){};

    //! init the options value map with given options
    void init(std::shared_ptr<OptionMap>&);

    //! set next options group cyclely
    std::string set_next_options();
    std::string set_next_fixed_options();

    //! check the end of options group
    bool is_end_options();
    bool is_fixed_end();

    std::string get_curr_options_code(CoderType, bool encode_all = false);

    //! set current options with given options
    void set_options(const std::string&);

    void registe_fixed_options(const std::vector<std::string>&);

#if MGB_ENABLE_JSON
    std::vector<std::shared_ptr<mgb::json::Object>> get_json();
#endif

private:
    void reset_option();
    size_t m_option_group_cnt;
    size_t m_fixed_option_cnt;
    OptionValMap m_valid_option_vals;
    std::vector<std::vector<std::string>> m_internal_options_name;
    std::vector<std::vector<std::string>> m_fixed_options_name;

#if MGB_ENABLE_JSON
    JsonOptionsCoder m_json_coder;
#endif

    GflagsOptionsCoder m_gflags_coder;
};

//! Options proifler to get the best settings with different evaluate standard
class OptionsProfiler {
public:
    OptionsProfiler(){};

    //! run with m_options
    virtual void profile_with_given_options(
            const std::string&, std::shared_ptr<OptionMap>&, const std::string&) = 0;

    //! get the best setting and inference time
    virtual std::string get_best_setting() { return ""; }

    virtual ~OptionsProfiler() = default;
};

/**
 * profiler to get the fast setting
 */
class OptionsTimeProfiler final : public OptionsProfiler {
public:
    OptionsTimeProfiler(){};

    void profile_with_given_options(
            const std::string&, std::shared_ptr<OptionMap>&,
            const std::string&) override;

    std::string get_best_setting() override { return m_best_setting.first; }

private:
    std::unordered_map<std::string, double> m_options_profile_result;
    std::pair<std::string, double> m_best_setting = {"", TIME_OUT};
};

/**
 * parse information from user given
 */
class UserInfoParser {
public:
    UserInfoParser(){};

    void get_user_info();

    void parse_info(std::shared_ptr<OptionsFastManager>&);

    std::shared_ptr<OptionsProfiler> create_profiler() {
        switch (m_proifler_type) {
            case ProiflerType::TIME_PROFILER:
                return std::make_shared<OptionsTimeProfiler>();
            case ProiflerType::UNSPEC_PROFILER:
                return nullptr;
            default:
                return nullptr;
        }
    }

private:
    ProiflerType m_proifler_type;
    std::unordered_map<std::string, std::string> m_user_info;
};
/*!
 * \brief: Fitting strategy for running
 */
class FittingStrategy : public StrategyBase {
public:
    class AutoCleanFile {
    public:
        AutoCleanFile(
                const std::string& model_path, std::shared_ptr<OptionMap>& options);
        void dump_model();
        std::string filename() { return m_filename; }
        ~AutoCleanFile();

    private:
        std::string m_model_path;
        std::shared_ptr<OptionMap> m_options;
        std::string m_filename;
    };

    FittingStrategy(std::string model_path);

    void run() override;

    void dump_best_options_with_model();

    void dump_model();

private:
    std::string m_model_path;

    std::string m_dumped_model;
    std::shared_ptr<OptionsFastManager> m_manager;

    UserInfoParser m_info_parser;

#if MGB_ENABLE_JSON
    std::vector<std::pair<mgb::json::String, std::shared_ptr<mgb::json::Value>>>
            m_packed_info;
#endif
};
}  // namespace lar