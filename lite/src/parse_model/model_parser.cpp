/**
 * \file src/model_parser.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "model_parser.h"
#include "decryption/decrypt_base.h"
#include "parse_info/cache_parse.h"
#include "parse_info/parse_info_base.h"

using namespace lite;
using namespace model_parse;

std::string ModelParser::sm_model_tag = "packed_model";

void ModelParser::parse_header() {
    size_t tag_length = sm_model_tag.size();

    //! parse model tag
    const char* ptr = static_cast<char*>(m_model.get());
    std::string tag(static_cast<const char*>(ptr), tag_length);
    if (sm_model_tag == tag) {
        m_is_bare_model = false;
    } else {
        //! if no tag, the model is bare model, return
        m_is_bare_model = true;
        return;
    }

    uint8_t* buffer = static_cast<uint8_t*>(m_model.get()) + tag_length;
    auto packed_model = GetPackModel(buffer);
    auto models = packed_model->models();
    LITE_ASSERT(models->size() == 1, "Now only support one model");
    auto model = models->Get(0);
    m_model_name = model->header()->name()->c_str();
    m_model_decryption_name = model->header()->model_decryption_method()->c_str();
    m_info_decryption_name = model->header()->info_decryption_method()->c_str();
    m_info_parse_func_name = model->header()->info_parse_method()->c_str();
    if (model->header()->info_cache_parse_method())
        m_info_cache_parse_func_name =
                model->header()->info_cache_parse_method()->c_str();
    m_is_fast_run_cache = model->header()->is_fast_run_cache();

    m_info = model->info();
    m_model_data = model->data();
}

bool ModelParser::parse_model_info(
        Config& network_config, NetworkIO& network_io,
        std::unordered_map<std::string, LiteAny>& isolated_config_map,
        std::string& extra_info) const {
    //! no model info, no parse, direct return
    if (m_is_bare_model || !m_info) {
        return false;
    }
    //! parse ModelInfo::data
    if (m_info->data()) {
        size_t info_length = m_info->data()->size();
        const uint8_t* info_data = m_info->data()->Data();
        //! decryption the info
        auto info_ptr = decrypt_memory(
                info_data, info_length, m_info_decryption_name, info_length);
        //! parse the info
        LITE_LOCK_GUARD(parse_info_static_data().map_mutex);
        auto it_parse = parse_info_static_data().parse_info_methods.find(
                m_info_parse_func_name);
        if (it_parse == parse_info_static_data().parse_info_methods.end()) {
            LITE_THROW(ssprintf(
                    "can't find model info parse function %s.",
                    m_info_parse_func_name.c_str()));
        }
        auto model_info_parse_func =
                parse_info_static_data().parse_info_methods[m_info_parse_func_name];
        //! convert for NetworkIOInner to NetworkIO
        if (model_info_parse_func) {
            model_info_parse_func(
                    info_ptr.get(), info_length, m_model_name, network_config,
                    network_io, isolated_config_map, extra_info);
        } else {
            LITE_THROW(ssprintf(
                    "model info parse function of  %s is empty",
                    m_info_parse_func_name.c_str()));
        }
    }
    //! parse ModelInfo::algo_policy
    if (m_info->algo_policy()) {
        size_t cache_length = m_info->algo_policy()->size();
        const uint8_t* cache = m_info->algo_policy()->Data();
        if (m_info_cache_parse_func_name == "LITE_parse_cache") {
            if (m_is_fast_run_cache) {
                parse_info_cache(cache, cache_length);
            } else if (m_info->binary_cache()) {
                size_t binary_cache_length = m_info->binary_cache()->size();
                const uint8_t* binary_cache = m_info->binary_cache()->Data();
                parse_info_cache(
                        cache, cache_length, m_is_fast_run_cache, binary_cache,
                        binary_cache_length);
            } else {
                LITE_THROW("opencl binary cache is not given");
            }
        }
    }
    return true;
}

std::shared_ptr<void> ModelParser::parse_model(
        size_t& model_length, const Config& config) const {
    if (m_is_bare_model) {
        if (config.bare_model_cryption_name.size() == 0) {
            model_length = m_total_length;
            return m_model;
        } else {
            return decrypt_memory(
                    static_cast<uint8_t*>(m_model.get()), m_total_length,
                    config.bare_model_cryption_name, model_length);
        }
    }
    LITE_ASSERT(m_model_data, "packed model parse error!");
    model_length = m_model_data->data()->size();
    const uint8_t* model_data = m_model_data->data()->Data();
    LITE_ASSERT(model_length > 0, "The loaded model is of zero length.");
    return decrypt_memory(
            model_data, model_length, m_model_decryption_name, model_length);
}

std::shared_ptr<void> ModelParser::decrypt_memory(
        const uint8_t* data, size_t length, const std::string decryption_name,
        size_t& result_length) const {
    const uint8_t* memory_ptr = data;
    if (decryption_name == "NONE") {
        result_length = length;
        return std::shared_ptr<void>(const_cast<uint8_t*>(memory_ptr), [](void*) {});
    }
    LITE_LOCK_GUARD(decryption_static_data().map_mutex);
    auto it = decryption_static_data().decryption_methods.find(decryption_name);
    if (it == decryption_static_data().decryption_methods.end()) {
        LITE_THROW(ssprintf(
                "The decryption method %s is not registed yet.",
                decryption_name.c_str()));
    }
    auto&& func = it->second.first;
    auto&& key = it->second.second;
    if (func) {
        auto model_vector = func(memory_ptr, length, *key);
        result_length = model_vector.size();
        auto tmp_model_vector = new std::vector<uint8_t>(std::move(model_vector));
        return std::shared_ptr<void>(
                tmp_model_vector->data(),
                [tmp_model_vector](void*) { delete tmp_model_vector; });
    } else {
        LITE_THROW(ssprintf(
                "No decryption function in %s method.", decryption_name.c_str()));
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
