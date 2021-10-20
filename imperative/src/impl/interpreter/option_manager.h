/**
 * \file imperative/src/impl/interpreter/option_manager.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <string>
#include <unordered_map>

#include "megbrain/common.h"

namespace mgb::imperative::interpreter::intl {

struct OptionManager {
private:
    std::unordered_map<std::string, size_t*> m_option_map = {};

public:
#define DEF_OPTION(name, env_key, default_value, desc) \
    size_t name =                                      \
            (m_option_map[#name] = &name,              \
             get_option_from_env(env_key, default_value));

    // Environ value would be read only when interpreter initializing.
    DEF_OPTION(
            async_level, "MEGENGINE_INTERP_ASYNC_LEVEL", 2,
            "config whether raise error exactly when invoking op.\n"
            "level 2: both device and user side errors are async;\n"
            "level 1: user side errors are sync;\n"
            "level 0: both sync.");
    DEF_OPTION(enable_swap, "MEGENGINE_ENABLE_SWAP", 0, "");
    DEF_OPTION(enable_drop, "MEGENGINE_ENABLE_DROP", 0, "");
    DEF_OPTION(max_recompute_time, "MEGENGINE_MAX_RECOMP_TIME", 1, "");
    DEF_OPTION(
            catch_worker_execption, "MEGENGINE_CATCH_WORKER_EXEC", 1,
            "catch worker exception if enabled, close it when debugging");
    DEF_OPTION(
            buffer_length, "MEGENGINE_COMMAND_BUFFER_LENGTH", 3,
            "set command buffer length.");
    DEF_OPTION(
            enable_host_compute, "MEGENGINE_HOST_COMPUTE", 1,
            "enable host compute, thus computation may be done in host event if it's "
            "device is gpu.");
    DEF_OPTION(enable_dtr_auto_drop, "MEGENGINE_DTR_AUTO_DROP", 0, "");
    DEF_OPTION(enable_dtr_sqrt_sampling, "MEGENGINE_DTR_SQRT_SAMPLING", 0, "");
    DEF_OPTION(
            dtr_eviction_threshold, "MEGENGINE_DTR_EVICTION_THRESHOLD", 0,
            "auto drop will start whenever gpu memory usage exceeds this value.");
    DEF_OPTION(
            dtr_evictee_minimum_size, "MEGENGINE_DTR_EVICTEE_MINIMUM_SIZE", 1048576,
            "the minimum memory value of a tensor added to the candidate set");
    DEF_OPTION(record_computing_path, "MEGENGINE_RECORD_COMPUTING_PATH", 0, "");

#undef DEF_OPTION

    void set_option(const std::string& name, size_t value) {
        *m_option_map[name] = value;
    }

    size_t get_option(const std::string& name) const { return *m_option_map.at(name); }

    static size_t get_option_from_env(const std::string& name, size_t default_value) {
        if (const char* env_val = MGB_GETENV(name.c_str())) {
            sscanf(env_val, "%zu", &default_value);
        }
        return default_value;
    }
};

}  // namespace mgb::imperative::interpreter::intl
