/**
 * \file lite/load_and_run/src/options/device_options.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */
#pragma once
#include <gflags/gflags.h>
#include "models/model.h"
#include "option_base.h"

DECLARE_bool(cpu);
#if MGB_CUDA || LITE_WITH_CUDA
DECLARE_bool(cuda);
#endif
DECLARE_bool(cpu_default);
DECLARE_int32(multithread);
DECLARE_int32(multithread_default);
DECLARE_string(multi_thread_core_ids);
namespace lar {

class XPUDeviceOption final : public OptionBase {
public:
    static bool is_valid();
    static std::shared_ptr<OptionBase> create_option();
    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;
    std::string option_name() const override { return m_option_name; };

private:
    XPUDeviceOption();
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};
    bool enable_cpu;
#if MGB_CUDA || LITE_WITH_CUDA
    bool enable_cuda;
#endif
    bool enable_cpu_default;
    bool enable_multithread;
    bool enable_multithread_default;
    bool enable_set_core_ids;
    size_t thread_num;
    std::vector<int> core_ids;
    std::string m_option_name;
};
}  // namespace lar
