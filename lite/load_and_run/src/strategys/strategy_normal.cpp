/**
 * \file lite/load_and_run/src/strategys/strategy_normal.cpp
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */
#include <iostream>
#include <thread>
#include "megbrain/common.h"
#include "megbrain/utils/timer.h"
#include "megbrain/version.h"
#include "megdnn/version.h"
#include "misc.h"
#include "strategy.h"

using namespace lar;

NormalStrategy::NormalStrategy(std::string model_path) {
    mgb::set_log_level(mgb::LogLevel::WARN);
    lite::set_log_level(LiteLogLevel::WARN);
    m_model_path = model_path;
    auto option_creator_map = OptionFactory::get_Instance().get_option_creator_map();
    mgb_log_debug("option map size: %lu", option_creator_map->size());
    auto construct_option = [&](std::string name) -> void {
        auto& creator = (*option_creator_map)[name];
        auto option = creator();
        if (option) {
            m_options.insert({name, option});
        }
    };

    for (auto& creator : *option_creator_map) {
        auto name = creator.first;
        if (m_options.count(name) == 0) {
            construct_option(name);
        }
    }
}

void NormalStrategy::run_subline() {
    auto model = ModelBase::create_model(m_model_path);
    mgb_assert(model != nullptr, "create model failed!!");

    auto stage_config_model = [&]() {
        for (auto& option : m_options) {
            option.second->config_model(m_runtime_param, model);
        }
    };
    //! execute before load config
    m_runtime_param.stage = RunStage::BEFORE_MODEL_LOAD;
    stage_config_model();

    mgb::RealTimer timer;
    model->load_model();
    printf("load model: %.3fms\n", timer.get_msecs_reset());

    //! after load configure
    m_runtime_param.stage = RunStage::AFTER_MODEL_LOAD;
    stage_config_model();

    m_runtime_param.stage = RunStage::GLOBAL_OPTIMIZATION;
    stage_config_model();

    m_runtime_param.stage = RunStage::BEFORE_OUTSPEC_SET;
    stage_config_model();

    // for get static memmory information options
    m_runtime_param.stage = RunStage::AFTER_OUTSPEC_SET;
    stage_config_model();

    auto warm_up = [&]() {
        auto warmup_num = m_runtime_param.warmup_iter;
        for (size_t i = 0; i < warmup_num; i++) {
            printf("=== prepare: %.3fms; going to warmup\n\n", timer.get_msecs_reset());
            model->run_model();
            model->wait();
            printf("warm up %lu  %.3fms\n", i, timer.get_msecs_reset());
            m_runtime_param.stage = RunStage::AFTER_RUNNING_WAIT;
            stage_config_model();
        }
    };

    auto run_iter = [&](int idx) {
        double time_sqrsum = 0, time_sum = 0,
               min_time = std::numeric_limits<double>::max(), max_time = 0;
        auto run_num = m_runtime_param.run_iter;
        for (size_t i = 0; i < run_num; i++) {
            timer.reset();
            model->run_model();
            auto exec_time = timer.get_msecs();
            model->wait();
            m_runtime_param.stage = RunStage::AFTER_RUNNING_WAIT;
            stage_config_model();
            auto cur = timer.get_msecs();
            printf("iter %lu/%lu: %.3f ms (exec=%.3f ms)\n", i, run_num, cur,
                   exec_time);
            time_sum += cur;
            time_sqrsum += cur * cur;
            fflush(stdout);
            min_time = std::min(min_time, cur);
            max_time = std::max(max_time, cur);
        }
        printf("\n=== finished test #%u: time=%.3f ms avg_time=%.3f ms "
               "sexec=%.3f ms min=%.3f ms max=%.3f ms\n\n",
               idx, time_sum, time_sum / run_num,
               std::sqrt(
                       (time_sqrsum * run_num - time_sum * time_sum) /
                       (run_num * (run_num - 1))),
               min_time, max_time);
        return time_sum;
    };

    //! model with testcase
    size_t iter_num = m_runtime_param.testcase_num;

    double tot_time = 0;
    for (size_t idx = 0; idx < iter_num; idx++) {
        //! config when running model
        mgb_log_warn("run testcase: %zu ", idx);
        m_runtime_param.stage = RunStage::MODEL_RUNNING;
        stage_config_model();

        if (idx == 0) {
            warm_up();
        }
        tot_time += run_iter(idx);

        m_runtime_param.stage = RunStage::AFTER_RUNNING_ITER;
        stage_config_model();
    }

    printf("=== total time: %.3fms\n", tot_time);
    //! execute after run
    m_runtime_param.stage = RunStage::AFTER_MODEL_RUNNING;
    stage_config_model();
};

void NormalStrategy::run() {
    auto v0 = mgb::get_version();
    auto v1 = megdnn::get_version();
    printf("megbrain/lite/load_and_run:\nusing MegBrain "
           "%d.%d.%d(%d) and MegDNN %d.%d.%d\n",
           v0.major, v0.minor, v0.patch, v0.is_dev, v1.major, v1.minor, v1.patch);

    size_t thread_num = m_runtime_param.threads;
    auto run_sub = [&]() { run_subline(); };
    if (thread_num == 1) {
        run_sub();
    } else if (thread_num > 1) {
#if MGB_HAVE_THREAD
        std::vector<std::thread> threads;

        for (size_t i = 0; i < thread_num; ++i) {
            threads.emplace_back(run_sub);
        }
        for (auto&& i : threads) {
            i.join();
        }
#else
        mgb_log_error(
                "%d threads requested, but load_and_run was compiled "
                "without <thread> support.",
                thread_num);
#endif
    } else {
        mgb_assert(false, "--thread must input a positive number!!");
    }
    //! execute before run
}
