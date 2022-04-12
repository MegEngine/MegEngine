/**
 * \file src/model_parser.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "../network_impl_base.h"
#include "lite/global.h"

#include <flatbuffers/flatbuffers.h>
#include "pack_model_generated.h"

#include <unordered_map>

namespace lite {

/*!
 * \brief parse the model and decyt
 */
class ModelParser {
public:
    ModelParser(std::shared_ptr<void> model_ptr, size_t model_length)
            : m_model(model_ptr), m_total_length(model_length) {
        //! parse the header
        parse_header();
    }

    //! parse the Info part of the model, update the network_config and
    //! network_io
    bool parse_model_info(
            Config& network_config, NetworkIO& network_io,
            std::unordered_map<std::string, LiteAny>& isolated_config_map,
            std::string& extra_info) const;

    //! parse the model and decrypt the model
    std::shared_ptr<void> parse_model(size_t& model_length, const Config& config) const;

private:
    //! parse the header of the model and store the model related information
    //! to the menber data
    void parse_header();

    //! decrypt a memory with length of length and decryption method name
    //! decrypt_name
    std::shared_ptr<void> decrypt_memory(
            const uint8_t* data, size_t length, const std::string decryption_name,
            size_t& result_length) const;

private:
    std::string m_model_name;
    //! the info and model decryption method name,  the
    //! decryption func can be found through this name
    std::string m_info_decryption_name;
    std::string m_model_decryption_name;
    //! the function name to parse the model info
    std::string m_info_parse_func_name;
    std::string m_info_cache_parse_func_name;
    bool m_is_fast_run_cache;
    //! if a model is not added json info to the model is not crypted, the
    //! model is a bare model
    bool m_is_bare_model = true;

    const model_parse::ModelInfo* m_info = nullptr;
    const model_parse::ModelData* m_model_data = nullptr;

    std::shared_ptr<void> m_model;
    size_t m_total_length;

    static std::string sm_model_tag;
};
}  // namespace lite
   // vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
