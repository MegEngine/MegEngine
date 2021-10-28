/**
 * \file lite/load_and_run/src/options/fastrun_options.cpp
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#include <gflags/gflags.h>

#if defined(_WIN32)
#include <io.h>
#define F_OK         0
#define access(a, b) _access(a, b)
#elif __linux__ || __unix__ || __APPLE__
#include <unistd.h>
#endif
#include "fastrun_options.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/utils/infile_persistent_cache.h"
#include "misc.h"
#include "models/model_lite.h"
#include "models/model_mdl.h"

namespace lar {

template <>
void FastRunOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        //! set the algo policy before model load
        using Strategy = ModelLite::Strategy;
        uint32_t strategy = 0;
#if MGB_ENABLE_FASTRUN
        if (enable_full_run) {
            LITE_WARN("enable full-run strategy for algo profile");
            strategy = static_cast<uint32_t>(Strategy::LITE_ALGO_PROFILE) | strategy;
        } else if (enable_fast_run) {
            LITE_WARN("enable fast-run strategy for algo profile");
            strategy = static_cast<uint32_t>(Strategy::LITE_ALGO_PROFILE) |
                       static_cast<uint32_t>(Strategy::LITE_ALGO_OPTIMIZED) | strategy;
        } else {
            strategy = static_cast<uint32_t>(Strategy::LITE_ALGO_HEURISTIC) | strategy;
        }
#else
        strategy = static_cast<uint32_t>(Strategy::LITE_ALGO_HEURISTIC) | strategy;
#endif
        if (batch_binary_equal || enable_reproducible) {
            LITE_WARN("enable reproducible strategy for algo profile");
            if (batch_binary_equal)
                strategy = static_cast<uint32_t>(Strategy::LITE_ALGO_REPRODUCIBLE) |
                           strategy;
        }
        auto lite_strategy = static_cast<Strategy>(strategy);
        model->set_lite_strategy(lite_strategy);
    } else if (runtime_param.stage == RunStage::AFTER_MODEL_LOAD) {
        auto lite_network = model->get_lite_network();
        auto lite_strategy = model->get_lite_strategy();
        //! set algo policy for model
        lite::Runtime::set_network_algo_policy(
                lite_network, lite_strategy, share_batch_size, batch_binary_equal);
        if (!m_fast_run_cache.empty()) {
            if (!access(m_fast_run_cache.c_str(), F_OK)) {
                lite::set_persistent_cache(m_fast_run_cache);
            } else {
                lite::set_persistent_cache(m_fast_run_cache, true);
            }
            //! TODO:this is from mdl model settings but not matched settings in
            //! lite model
            // if (!enable_full_run && !enable_fast_run)
            //     mgb::gopt::enable_opr_use_profiling_cache_inplace(vars);
        }
    } else if (runtime_param.stage == RunStage::AFTER_MODEL_RUNNING) {
#if MGB_ENABLE_FASTRUN
        //! dump algo cache
        if (!m_fast_run_cache.empty()) {
            lite::dump_persistent_cache(m_fast_run_cache);
        }
#endif
    }
}

template <>
void FastRunOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        //! set the algo policy before model load
        using Strategy = ModelMdl::Strategy;
        auto strategy = static_cast<Strategy>(0);
#if MGB_ENABLE_FASTRUN
        if (enable_full_run) {
            mgb_log_warn("enable full-run strategy for algo profile");
            strategy = Strategy::PROFILE | strategy;
        } else if (enable_fast_run) {
            mgb_log_warn("enable fast-run strategy for algo profile");
            strategy = Strategy::PROFILE | Strategy::OPTIMIZED | strategy;
        } else {
            strategy = Strategy::HEURISTIC | strategy;
        }
#else
        strategy = Strategy::HEURISTIC | strategy;
#endif
        if (batch_binary_equal || enable_reproducible) {
            mgb_log_warn("enable reproducible strategy for algo profile");
            strategy = Strategy::REPRODUCIBLE | strategy;
        }
        model->set_mdl_strategy(strategy);

        //! set binary_equal_between_batch and shared_batch_size
        if (batch_binary_equal) {
            mgb_log_warn("enable batch binary equal");
            model->get_mdl_config()
                    .comp_graph->options()
                    .fast_run_config.binary_equal_between_batch = true;
        }
        if (share_batch_size > 0) {
            mgb_log_warn("set shared shared batch");
            model->get_mdl_config()
                    .comp_graph->options()
                    .fast_run_config.shared_batch_size = share_batch_size;
        }
    } else if (runtime_param.stage == RunStage::AFTER_MODEL_LOAD) {
        auto vars = model->get_mdl_load_result().output_var_list;
        auto strategy = model->get_mdl_strategy();
        mgb::gopt::modify_opr_algo_strategy_inplace(vars, strategy);
        // set algo cache path
        if (!m_fast_run_cache.empty()) {
            if (!access(m_fast_run_cache.c_str(), F_OK)) {
                mgb::PersistentCache::set_impl(
                        std::make_shared<mgb::InFilePersistentCache>(
                                m_fast_run_cache.c_str()));
            } else {
                mgb::PersistentCache::set_impl(
                        std::make_shared<mgb::InFilePersistentCache>());
            }
#if MGB_ENABLE_FASTRUN
            if (!enable_full_run && !enable_fast_run)
#endif
                mgb::gopt::enable_opr_use_profiling_cache_inplace(vars);
        }
    } else if (runtime_param.stage == RunStage::AFTER_MODEL_RUNNING) {
#if MGB_ENABLE_FASTRUN
        //! dump algo cache
        if (!m_fast_run_cache.empty()) {
            static_cast<mgb::InFilePersistentCache&>(mgb::PersistentCache::inst())
                    .dump_cache(m_fast_run_cache.c_str());
        }
#endif
    }
}

}  // namespace lar

using namespace lar;

FastRunOption::FastRunOption() {
    m_option_name = "fastrun";
#if MGB_ENABLE_FASTRUN
    enable_fast_run = FLAGS_fast_run;
    enable_full_run = FLAGS_full_run;
#endif
    batch_binary_equal = FLAGS_binary_equal_between_batch;
    enable_reproducible = FLAGS_reproducible;
    m_fast_run_cache = FLAGS_fast_run_algo_policy;
    share_batch_size = FLAGS_fast_run_shared_batch_size;
#if MGB_ENABLE_FASTRUN
    //! while fastrun cache file path is not empty and can't be accessed
    if (!m_fast_run_cache.empty() && access(m_fast_run_cache.c_str(), F_OK)) {
        mgb_assert(
                enable_full_run || enable_fast_run,
                "--fast-run or --full-run should be enabled");
    }
    if (share_batch_size) {
        mgb_assert(
                enable_full_run || enable_fast_run || !m_fast_run_cache.empty(),
                "--fast-run-shared-batch-size should be used with "
                "--fast-run|--full-run|--fast-run-algo-policy");
    }
#endif
}

bool FastRunOption::is_valid() {
    bool ret = false;
#if MGB_ENABLE_FASTRUN
    ret = ret || FLAGS_fast_run;
    ret = ret || FLAGS_full_run;
#endif
    ret = ret || FLAGS_binary_equal_between_batch;
    ret = ret || FLAGS_fast_run_shared_batch_size > 0;
    ret = ret || FLAGS_reproducible;
    ret = ret || FLAGS_fast_run_algo_policy.size() > 0;

    return ret;
}

std::shared_ptr<OptionBase> FastRunOption::create_option() {
    static std::shared_ptr<FastRunOption> option(new FastRunOption);
    if (FastRunOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void FastRunOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}

#if MGB_ENABLE_FASTRUN
DEFINE_bool(fast_run, false, "whether to use fast-run in model run");
DEFINE_bool(full_run, false, "whether to use full-run in model run");
#endif

DEFINE_bool(
        binary_equal_between_batch, false,
        "Each batch of output is promised binary equal if each batch of "
        "input is binary equal\n Note that if this option is turned on, "
        "`--reproducible` will also be turned on.");
DEFINE_bool(
        reproducible, false,
        "Enable choose algo which is reproducible. It mainly used for "
        "cudnn algos.See "
        "https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/"
        "index.html#reproducibility"
        "for more details.");
DEFINE_uint32(fast_run_shared_batch_size, 0, "Set the batch size used during fastrun");
DEFINE_string(fast_run_algo_policy, "", "fast-run cache path.");

REGIST_OPTION_CREATOR(fastrun, lar::FastRunOption::create_option);