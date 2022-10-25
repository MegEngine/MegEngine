#include "strategy_fitting.h"
#if defined(_WIN32)
#include <io.h>
#define F_OK         0
#define access(a, b) _access(a, b)
#elif __linux__ || __unix__ || __APPLE__
#include <unistd.h>
#endif
#include <fstream>
#include <iostream>
#include <list>
#include <regex>
#include <thread>
#include "lite/pack_model.h"
#include "megbrain/common.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/exception.h"
#include "megbrain/utils/timer.h"
#include "megbrain/version.h"
#include "megdnn/version.h"
#include "misc.h"
DECLARE_bool(cpu);
using namespace lar;

// /////////////////// OptionsFastManager ///////////////////
void OptionsFastManager::init(std::shared_ptr<OptionMap>& options) {
    m_option_group_cnt = 0;
    m_fixed_option_cnt = 0;
    m_internal_options_name = {
            {"enable_fuse_conv_bias_with_z"},
            {"enable_fuse_preprocess"},
            {"record_comp_seq"},
            {"const_shape"}};
    //! record the independent option value
    for (auto& option : *options) {
        auto option_vals = option.second->get_option();
        if (option_vals) {
            for (auto& item : *option_vals) {
                m_valid_option_vals.insert(item);
            }
        }
    }
};

std::string OptionsFastManager::set_next_fixed_options() {
    reset_option();
    auto& fixed_options_name = m_fixed_options_name[m_fixed_option_cnt];
    for (auto& item : fixed_options_name) {
        if (m_valid_option_vals.find(item) != m_valid_option_vals.end()) {
            auto& option_val = m_valid_option_vals[item];
            auto type = option_val->get_type();
            if (type == JsonValueType::Bool) {
                auto option_val_ptr = std::static_pointer_cast<lar::Bool>(option_val);
                option_val_ptr->set_value(true);
            } else if (type == JsonValueType::String && item == "layout_transform") {
                auto option_val_ptr = std::static_pointer_cast<lar::String>(option_val);
                //! device type
                option_val_ptr->set_value(fixed_options_name[0]);
            } else {
                mgb_log_error(
                        "invalid JsonValueType:%s to set next value for fitting mode",
                        option_val->type_string().c_str());
            }
        }
    }
    ++m_fixed_option_cnt;
    std::string code = m_gflags_coder.encode(m_valid_option_vals);
    return code;
}

std::string OptionsFastManager::set_next_options() {
    reset_option();
    auto& constraint = m_internal_options_name[m_option_group_cnt];
    for (auto& item : constraint) {
        if (m_valid_option_vals.find(item) != m_valid_option_vals.end()) {
            auto& option_val = m_valid_option_vals[item];
            auto type = option_val->get_type();
            if (type == JsonValueType::Bool) {
                auto option_val_ptr = std::static_pointer_cast<lar::Bool>(option_val);
                option_val_ptr->set_value(true);
            } else {
                mgb_log_error(
                        "invalid JsonValueType: %s to set next value for fitting mode",
                        option_val->type_string().c_str());
            }
        }
    }
    ++m_option_group_cnt;
    std::string code = m_gflags_coder.encode(m_valid_option_vals);
    return code;
}

bool OptionsFastManager::is_end_options() {
    return m_option_group_cnt == m_internal_options_name.size();
}

bool OptionsFastManager::is_fixed_end() {
    return m_fixed_option_cnt == m_fixed_options_name.size();
}

void OptionsFastManager::set_options(const std::string& code) {
    reset_option();
#if MGB_ENABLE_JSON
    const std::regex json_regex(".\\{");
#endif
    const std::regex gflags_regex("--.*=.*");
    if (std::regex_search(code, gflags_regex)) {
        m_gflags_coder.decode(code, m_valid_option_vals);
    }
#if MGB_ENABLE_JSON
    else if (std::regex_search(code, json_regex)) {
        m_json_coder.decode(code, m_valid_option_vals);
    }
#endif
    else {
        mgb_log_error("invalid options code format \"%s\" to decode", code.c_str());
    }
}

void OptionsFastManager::registe_fixed_options(
        const std::vector<std::string>& option_name) {
    m_fixed_options_name.push_back(option_name);
}

std::string OptionsFastManager::get_curr_options_code(CoderType type, bool encode_all) {
    if (type == CoderType::GFLAGS) {
        return m_gflags_coder.encode(m_valid_option_vals, encode_all);
    }
#if MGB_ENABLE_JSON
    else if (type == CoderType::JSON) {
        return m_json_coder.encode(m_valid_option_vals, encode_all);
    }
#endif
    else {
        mgb_log_error("coder should be implemented in furture");
        return "";
    }
}
#if MGB_ENABLE_JSON
std::vector<std::shared_ptr<mgb::json::Object>> OptionsFastManager::get_json() {
    std::vector<std::shared_ptr<mgb::json::Object>> ret =
            m_json_coder.encode(m_valid_option_vals);
    return ret;
}
#endif

void OptionsFastManager::reset_option() {
    for (auto& option : m_valid_option_vals) {
        option.second->reset_value();
    }
}

////////////////// OptionsTimeProfiler //////////////////

void OptionsTimeProfiler::profile_with_given_options(
        const std::string& model_path, std::shared_ptr<OptionMap>& given_options,
        const std::string& option_code) {
    RuntimeParam runtime_param;
    auto model = ModelBase::create_model(model_path);
    mgb::RealTimer timer;
    auto stage_config_model = [&]() {
        for (auto& option : *given_options) {
            mgb_assert(option.second, "invalid option %s\n", option.first.c_str());
            option.second->config_model(runtime_param, model);
        }
    };

    auto warm_up = [&]() {
        for (size_t i = 0; i < runtime_param.warmup_iter; i++) {
            auto start = timer.get_msecs();
            model->run_model();
            model->wait();
            mgb_log("warm up %ld time %f ms", i, timer.get_msecs() - start);
        }
    };
    double inference_time = 0.0;
    auto run_iter = [&]() {
        for (size_t i = 0; i < runtime_param.run_iter; i++) {
            auto start = timer.get_msecs();
            model->run_model();
            model->wait();
            auto end = timer.get_msecs();
            mgb_log("run iter %ld time %f ms", i, end - start);
            inference_time += end - start;
            mgb_throw_if(
                    inference_time > TIME_OUT, mgb::TimeoutError,
                    "time out while using fitting");
        }
    };

    //! model with testcase
    size_t case_num = runtime_param.testcase_num;

    bool exception_state = false;
    MGB_TRY {
        timer.reset();
        runtime_param.stage = RunStage::BEFORE_MODEL_LOAD;
        stage_config_model();

        runtime_param.stage = RunStage::AFTER_NETWORK_CREATED;
        model->create_network();
        stage_config_model();

        model->load_model();
        //! after load configure
        auto config_model_before_runing = [&]() {
            for (auto stage :
                 {RunStage::AFTER_MODEL_LOAD, RunStage::UPDATE_IO,
                  RunStage::GLOBAL_OPTIMIZATION, RunStage::BEFORE_OUTSPEC_SET,
                  RunStage::AFTER_OUTSPEC_SET, RunStage::MODEL_RUNNING}) {
                runtime_param.stage = stage;
                stage_config_model();
            }
        };
        timer.reset();
        for (size_t idx = 0; idx < case_num; idx++) {
            auto start = timer.get_msecs();
            config_model_before_runing();
            auto end = timer.get_msecs();
            mgb_log("config model time %f ms", end - start);
            warm_up();
            run_iter();
        }
        runtime_param.stage = RunStage::AFTER_MODEL_RUNNING;
        stage_config_model();
    }
    MGB_CATCH(std::exception & exc, {
        mgb_log_error("catch exception: %s", exc.what());
        exception_state = true;
    });

    auto average = inference_time / runtime_param.run_iter;
    if (exception_state) {
        average = TIME_OUT;
        mgb_log_error(
                "out of time (this may be caused by some exception, please checkout "
                "the log) when profile option:\n%s\n",
                option_code.c_str());
    } else {
        mgb_log("profile option:\n%s\n avg_time=%.3f ms\n", option_code.c_str(),
                average);
        //! record profile result
        m_options_profile_result.insert({option_code, average});

        //! record the best result

        if (average < m_best_setting.second) {
            m_best_setting.first = option_code;
            m_best_setting.second = average;
        }
    }
}
/////////////////////////// UserInfoParser /////////////////////////////
void UserInfoParser::get_user_info() {
    //! register user information tips
    std::vector<std::pair<std::string, std::string>> info_tips;
    m_user_info["fitting_preference"] = "Inferspeed";
}
void UserInfoParser::parse_info(std::shared_ptr<OptionsFastManager>& manager) {
    std::vector<std::string> fixed_options;
    fixed_options.push_back("enable_fuse_conv_bias_nonlinearity");
    std::vector<std::string> tmp_options;

    auto insert_common_cpu_options = [&]() {
        tmp_options = {"cpu"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cpu", "weight_preprocess", "fast_run"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cpu", "layout_transform"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cpu", "layout_transform", "weight_preprocess"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);
    };

#if (MEGDNN_AARCH64 || MEGDNN_ARMV7)
    //! arm cpu device
        insert_common_cpu_options();
        tmp_options = {"cpu", "enable_nchw44"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cpu", "enable_nchw44", "weight_preprocess", "fast_run"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cpu", "enable_nchw44_dot"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cpu", "enable_nchw44_dot", "weight_preprocess", "fast_run"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

#else
#if LITE_WITH_CUDA
    //! build with cuda and not force to use cpu device
    if (!FLAGS_cpu) {
        tmp_options = {"cuda"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cuda", "enable_nchw4"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cuda", "enable_chwn4"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cuda", "enable_nchw64"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cuda", "enable_nchw32"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cuda", "layout_transform"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cuda", "layout_transform", "weight_preprocess"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);
    }

#endif
#if LITE_WITH_CUDA
    //! build with cuda force to use cpu
    if (FLAGS_cpu) {
#endif
        //！x86 cpu options
        insert_common_cpu_options();
        tmp_options = {"cpu", "enable_nchw88"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);

        tmp_options = {"cpu", "enable_nchw88", "weight_preprocess", "fast_run"};
        tmp_options.insert(
                tmp_options.end(), fixed_options.begin(), fixed_options.end());
        manager->registe_fixed_options(tmp_options);
#if LITE_WITH_CUDA
    }
#endif
#endif

    m_proifler_type = ProiflerType::TIME_PROFILER;
}

// /////////////////// FittingStrategy //////////////////////////////////
FittingStrategy::FittingStrategy(std::string model_path) {
    m_manager = std::make_shared<OptionsFastManager>();
    m_dumped_model = FLAGS_dump_fitting_model;
    m_options = std::make_shared<OptionMap>();
    m_model_path = model_path;
    auto option_creator_map = OptionFactory::get_Instance().get_option_creator_map();
    auto option_validater_map =
            OptionFactory::get_Instance().get_option_validater_map();
    //! validate option used in fitting
    auto validate_option = [&](std::string name) -> void {
        if (option_validater_map->find(name) != option_validater_map->end()) {
            auto& validater = (*option_validater_map).at(name);
            if (validater) {
                validater(true);
            }
        }
    };

    //! construct option which is valid
    auto construct_option = [&](std::string name) -> void {
        auto& creator = (*option_creator_map)[name];
        auto option = creator();
        if (option) {
            m_options->insert({name, option});
        }
    };

    //! get all options which is valid
    for (auto& creator : *option_creator_map) {
        auto name = creator.first;
        if (m_options->count(name) == 0) {
            validate_option(name);
            construct_option(name);
        }
    }

    m_manager->init(m_options);
}

void FittingStrategy::dump_best_options_with_model() {
    std::vector<uint8_t> info_algo_policy_data;
    std::vector<uint8_t> info_binary_cache_data;
    auto model = ModelBase::create_model(m_model_path);
    RuntimeParam runtime_param;
    auto stage_config_model = [&]() {
        for (auto& option : *m_options) {
            option.second->config_model(runtime_param, model);
        }
    };
    runtime_param.stage = RunStage::BEFORE_MODEL_LOAD;
    stage_config_model();

    model->load_model();

    //! get json info vector
    std::string json_info_str;
#if MGB_ENABLE_JSON
    std::shared_ptr<mgb::json::Object> code_json = model->get_io_info();
    m_packed_info.push_back({mgb::json::String("IO"), (*code_json)["IO"]});
    auto info_json = m_manager->get_json();
    m_packed_info.push_back({mgb::json::String("options"), (*info_json[0])["options"]});
    m_packed_info.push_back({mgb::json::String("device"), (*info_json[1])["device"]});
    m_packed_info.push_back(
            {mgb::json::String("backend"), mgb::json::String::make("MGE")});
    int lite_major, lite_minor, lite_patch;
    lite::get_version(lite_major, lite_minor, lite_patch);
    std::string version = std::to_string(lite_major);
    version += ".";
    version += std::to_string(lite_minor) + ".";
    version += std::to_string(lite_patch);
    m_packed_info.push_back(
            {mgb::json::String("version"), mgb::json::String::make(version)});
    m_packed_info.push_back({mgb::json::String("valid"), mgb::json::Bool::make(true)});
    m_packed_info.push_back(
            {mgb::json::String("name"), mgb::json::String::make("packed_model")});
    auto obj = mgb::json::Object::make(m_packed_info);
    json_info_str = obj->to_string();
#endif
    std::vector<uint8_t> json_info(json_info_str.begin(), json_info_str.end());

    //! get model binary data after optimized
    for (auto stage :
         {RunStage::AFTER_MODEL_LOAD, RunStage::UPDATE_IO,
          RunStage::GLOBAL_OPTIMIZATION, RunStage::BEFORE_OUTSPEC_SET,
          RunStage::AFTER_OUTSPEC_SET, RunStage::MODEL_RUNNING}) {
        runtime_param.stage = stage;
        stage_config_model();
    }
    model->run_model();
    model->wait();
    std::vector<uint8_t> model_data = model->get_model_data();
    mgb_log("model_data size=%zu", model_data.size());
    mgb_log("json_info size=%zu", json_info.size());
    mgb_log("info_algo_policy_data size=%zu", info_algo_policy_data.size());
    mgb_log("info_binary_cache_data size=%zu", info_binary_cache_data.size());
    lite::ModelPacker packer(
            model_data, m_dumped_model, json_info, info_algo_policy_data,
            info_binary_cache_data);
    packer.set_header("NONE", "NONE", info_binary_cache_data.empty());
    packer.pack_model();
}
///////////////////////// AutoCleanFile///////////////////////////
FittingStrategy::AutoCleanFile::AutoCleanFile(
        const std::string& model_path, std::shared_ptr<OptionMap>& options)
        : m_model_path(model_path), m_options(options) {
    m_filename = "fitting_tmp_model";
    if (!access(m_filename.c_str(), F_OK)) {
        remove(m_filename.c_str());
    }
}

FittingStrategy::AutoCleanFile::~AutoCleanFile() {
    if (!access(m_filename.c_str(), F_OK)) {
        remove(m_filename.c_str());
    }
}

void FittingStrategy::AutoCleanFile::dump_model() {
    auto model = ModelBase::create_model(m_model_path);
    RuntimeParam runtime_param;
    auto stage_config_model = [&]() {
        for (auto& option : *m_options) {
            option.second->config_model(runtime_param, model);
        }
    };
    runtime_param.stage = RunStage::BEFORE_MODEL_LOAD;
    stage_config_model();

    model->load_model();
    //! get model binary data after optimized
    for (auto stage :
         {RunStage::AFTER_MODEL_LOAD, RunStage::UPDATE_IO,
          RunStage::GLOBAL_OPTIMIZATION, RunStage::BEFORE_OUTSPEC_SET,
          RunStage::AFTER_OUTSPEC_SET, RunStage::MODEL_RUNNING}) {
        runtime_param.stage = stage;
        stage_config_model();
    }
    model->run_model();
    model->wait();

    std::vector<uint8_t> model_data = model->get_model_data();
    mgb_log("dumped model_data size=%zu\n", model_data.size());
    auto fp = fopen(m_filename.c_str(), "wb");
    fwrite(model_data.data(), 1, model_data.size(), fp);
    fclose(fp);
}

void FittingStrategy::run() {
    auto mgb_version = mgb::get_version();
    auto dnn_version = megdnn::get_version();
    mgb_log("megbrain/lite/load_and_run:\nusing MegBrain "
            "%d.%d.%d(%d) and MegDNN %d.%d.%d\n",
            mgb_version.major, mgb_version.minor, mgb_version.patch, mgb_version.is_dev,
            dnn_version.major, dnn_version.minor, dnn_version.patch);
    // ! create profiler with given user info
    m_info_parser.get_user_info();
    m_info_parser.parse_info(m_manager);
    auto profiler = m_info_parser.create_profiler();
    mgb_throw_if(
            profiler == nullptr, mgb::AssertionError,
            "get empty profiler for fittting\n");
    //! profile model with fixed options
    while (!m_manager->is_fixed_end()) {
        std::string option_str = m_manager->set_next_fixed_options();
        profiler->profile_with_given_options(m_model_path, m_options, option_str);
#if (MEGDNN_AARCH64 || MEGDNN_ARMV7)
        //! sleep to keep machine with stable cpu frequence
        usleep(500000);
#endif
    }
    std::string m_tmp_model = m_model_path;
    const std::regex layout_regex("layout_transform");
    auto best_fixed_options = profiler->get_best_setting();
    m_manager->set_options(best_fixed_options);
    //! dump model for global layout transform
    auto m_tmp_file = AutoCleanFile(m_model_path, m_options);
    if (std::regex_search(best_fixed_options, layout_regex)) {
        m_tmp_file.dump_model();
        m_model_path = m_tmp_file.filename();
    }
    //! profile model with given profiler
    while (!m_manager->is_end_options()) {
        std::string curr_option_str = m_manager->set_next_options();
        //! set option with current option and fixed options
        if (m_model_path == m_tmp_model) {
            auto total_option_str = curr_option_str + best_fixed_options;
            m_manager->set_options(total_option_str);
        }

        curr_option_str += best_fixed_options;
        profiler->profile_with_given_options(m_model_path, m_options, curr_option_str);
#if (MEGDNN_AARCH64 || MEGDNN_ARMV7)
        usleep(500000);
#endif
    }
    //! set with best options and inference
    m_model_path = m_tmp_model;
    auto best_options = profiler->get_best_setting();

    m_manager->set_options(best_options);
    profiler->profile_with_given_options(m_model_path, m_options, best_options);

    //! save best options into given dir
    std::cout << "the best options:\n" << best_options << std::endl;

    if (!m_dumped_model.empty()) {
        dump_best_options_with_model();
    }
}
DEFINE_bool(
        fitting, false, "use the fitting mode profile and get the best option set.");
DEFINE_string(dump_fitting_model, "", "dump the best option and algo cache into model");
