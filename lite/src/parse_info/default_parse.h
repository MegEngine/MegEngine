/**
 * \file src/parse_info/default_parse.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "../misc.h"

#include "lite/global.h"
#include "lite/network.h"
#include "nlohmann/json.hpp"

namespace lite {
//! The LITE_default parse info function
bool default_parse_info(
        const void* info_ptr, size_t length, const std::string& model_name,
        Config& config, NetworkIO& network_io,
        std::unordered_map<std::string, LiteAny>& separate_config_map,
        std::string& extra_info) {
    using json = nlohmann::json;
    std::string json_string(static_cast<const char*>(info_ptr), length);
    auto info = json::parse(json_string);

    if (!info["valid"]) {
        return false;
    }
    auto info_model_name = info["name"];
    if (info_model_name != model_name) {
        LITE_THROW(ssprintf(
                "infomation of model name is not match, packed model "
                "is %s, but json info get %s.",
                model_name.c_str(), static_cast<std::string>(info_model_name).c_str()));
    }
    //! check version
    std::string model_version = info["version"];
    int major = std::stoi(model_version.substr(0, model_version.find(".")));
    int start = model_version.find(".") + 1;
    int minor = std::stoi(model_version.substr(start, model_version.find(".", start)));
    start = model_version.find(".", start) + 1;
    int patch = std::stoi(model_version.substr(start));
    int lite_major, lite_minor, lite_patch;
    lite::get_version(lite_major, lite_minor, lite_patch);
    size_t model_version_sum = (major * 10000 + minor) * 100 + patch;
    size_t lite_version_sum = (lite_major * 10000 + lite_minor) * 100 + lite_patch;
    if (model_version_sum > lite_version_sum) {
        LITE_WARN("Lite load the future version model !!!!!!!!!!!!!");
    }

    if (info.contains("has_compression")) {
        config.has_compression = info["has_compression"];
    }
    if (info.contains("backend")) {
        if (info["backend"] == "MGE") {
            config.backend = LiteBackend::LITE_DEFAULT;
        }
    }

    auto get_device_type = [](std::string type) -> LiteDeviceType {
        if (type == "CPU")
            return LiteDeviceType::LITE_CPU;
        if (type == "CUDA")
            return LiteDeviceType::LITE_CUDA;
        if (type == "ATLAS")
            return LiteDeviceType::LITE_ATLAS;
        if (type == "NPU")
            return LiteDeviceType::LITE_NPU;
        else {
            LITE_THROW(ssprintf("LITE not support device type of %s.", type.c_str()));
        }
    };
    if (info.contains("device")) {
        auto device_json = info["device"];
        config.device_type = get_device_type(device_json["type"]);
        if (device_json.contains("device_id")) {
            separate_config_map["device_id"] =
                    static_cast<int>(device_json["device_id"]);
        }
        if (device_json.contains("number_threads")) {
            separate_config_map["number_threads"] =
                    static_cast<uint32_t>(device_json["number_threads"]);
        }
        if (device_json.contains("enable_inplace_model")) {
            separate_config_map["enable_inplace_model"] =
                    static_cast<bool>(device_json["enable_inplace_model"]);
        }
        if (device_json.contains("use_tensorrt")) {
            separate_config_map["use_tensorrt"] =
                    static_cast<bool>(device_json["use_tensorrt"]);
        }
    }
    //! options
    if (info.contains("options")) {
        auto options = info["options"];
        if (options.contains("weight_preprocess"))
            config.options.weight_preprocess = options["weight_preprocess"];
        if (options.contains("fuse_preprocess"))
            config.options.fuse_preprocess = options["fuse_preprocess"];
        if (options.contains("fake_next_exec"))
            config.options.fake_next_exec = options["fake_next_exec"];
        if (options.contains("var_sanity_check_first_run"))
            config.options.var_sanity_check_first_run =
                    options["var_sanity_check_first_run"];
        if (options.contains("const_shape"))
            config.options.const_shape = options["const_shape"];
        if (options.contains("force_dynamic_alloc"))
            config.options.force_dynamic_alloc = options["force_dynamic_alloc"];
        if (options.contains("force_output_dynamic_alloc"))
            config.options.force_output_dynamic_alloc =
                    options["force_output_dynamic_alloc"];
        if (options.contains("no_profiling_on_shape_change"))
            config.options.no_profiling_on_shape_change =
                    options["no_profiling_on_shape_change"];
        if (options.contains("jit_level"))
            config.options.jit_level = options["jit_level"];
        if (options.contains("comp_node_seq_record_level"))
            config.options.comp_node_seq_record_level =
                    options["comp_node_seq_record_level"];
        if (options.contains("graph_opt_level"))
            config.options.graph_opt_level = options["graph_opt_level"];
        if (options.contains("async_exec_level"))
            config.options.async_exec_level = options["async_exec_level"];
    }
    //! IO
    auto get_io_type = [](std::string type) -> LiteIOType {
        if (type == "value")
            return LiteIOType::LITE_IO_VALUE;
        if (type == "shape")
            return LiteIOType::LITE_IO_SHAPE;
        else {
            LITE_THROW(ssprintf("LITE not support IO type of %s.", type.c_str()));
        }
    };
    auto get_data_type = [](std::string type) -> LiteDataType {
        if (type == "float32")
            return LiteDataType::LITE_FLOAT;
        if (type == "float16")
            return LiteDataType::LITE_HALF;
        if (type == "int32")
            return LiteDataType::LITE_INT;
        if (type == "int16")
            return LiteDataType::LITE_INT16;
        if (type == "int8")
            return LiteDataType::LITE_INT8;
        if (type == "uint8")
            return LiteDataType::LITE_UINT8;
        else {
            LITE_THROW(ssprintf("LITE not support data type of %s.", type.c_str()));
        }
    };
#define SET_SHAPE(shape_json_, config_)                                       \
    do {                                                                      \
        int ndim = 0;                                                         \
        for (int i = 0; i < 4; i++) {                                         \
            if (shape_json_.contains(shape_name[i])) {                        \
                ndim++;                                                       \
                config_.config_layout.shapes[i] = shape_json_[shape_name[i]]; \
            } else {                                                          \
                break;                                                        \
            }                                                                 \
        }                                                                     \
        config_.config_layout.ndim = ndim;                                    \
    } while (0)

#define Config_IO(io_json_, io_config_)                                        \
    if (io_json_.contains("is_host"))                                          \
        io_config_.is_host = io_json_["is_host"];                              \
    if (io_json_.contains("io_type"))                                          \
        io_config_.io_type = get_io_type(io_json_["io_type"]);                 \
    if (io_json_.contains("dtype"))                                            \
        io_config_.config_layout.data_type = get_data_type(io_json_["dtype"]); \
    if (io_json_.contains("shape")) {                                          \
        auto shape_json = io_json_["shape"];                                   \
        SET_SHAPE(shape_json, io_config_);                                     \
    }

    const std::string shape_name[] = {"dim0", "dim1", "dim2", "dim3"};
    if (info.contains("IO")) {
        auto IOs = info["IO"];
        if (IOs.contains("inputs")) {
            auto inputs = IOs["inputs"];
            for (size_t i = 0; i < inputs.size(); i++) {
                auto input_json = inputs[i];
                bool found = false;
                for (auto&& io_config : network_io.inputs) {
                    if (io_config.name == input_json["name"]) {
                        found = true;
                        Config_IO(input_json, io_config);
                    }
                }
                if (!found) {
                    IO input;
                    input.name = input_json["name"];
                    Config_IO(input_json, input);
                    network_io.inputs.push_back(input);
                }
            }
        }
        if (IOs.contains("outputs")) {
            auto outputs = IOs["outputs"];
            for (size_t i = 0; i < outputs.size(); i++) {
                auto output_json = outputs[i];
                bool found = false;
                for (auto&& io_config : network_io.outputs) {
                    if (io_config.name == output_json["name"]) {
                        found = true;
                        Config_IO(output_json, io_config);
                    }
                }
                if (!found) {
                    IO output;
                    output.name = output_json["name"];
                    Config_IO(output_json, output);
                    network_io.outputs.push_back(output);
                }
            }
        }
    }
    //! extra_info
    if (info.contains("extra_info")) {
        extra_info = info["extra_info"].dump();
    }
    return true;
#undef GET_BOOL
#undef Config_IO
}

}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
