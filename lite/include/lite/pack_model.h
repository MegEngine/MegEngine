/**
 * \file inlude/lite/pack_model.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include <string>
#include <vector>
namespace lite {

struct FeatureBits32 {
    uint32_t is_fast_run_cache : 1;
    //! reserved for new fields
    uint32_t : 31;
};

struct Header {
    std::string name;  //! model name
    std::string
            model_decryption_method;  //! model encryption method name, this is used to
                                      //! find the right decryption method. [
                                      //! AES_default | RC4_default |
                                      //! SIMPLE_FAST_RC4_default ], default is NONE.
    std::string info_decryption_method;  //! info data encryption method name, this is
                                         //! used to find the right decryption method. [
                                         //! AES_default | RC4_default |
                                         //! SIMPLE_FAST_RC4_default ], default is NONE.
    std::string info_parse_method = "LITE_default";  //! info parse method name.
    std::string info_cache_parse_method =
            "LITE_parse_cache";  //! fastrun cache parse method name.
    FeatureBits32 fb32;
};

class FbsHelper;

class ModelPacker {
public:
    ModelPacker(
            std::string model_path, std::string packed_model_path,
            std::string info_data_path = "", std::string info_algo_policy_path = "",
            std::string info_binary_cache_path = "");
    ModelPacker(
            std::vector<uint8_t> model_data, std::string packed_model_path,
            std::vector<uint8_t> info_data = {},
            std::vector<uint8_t> info_algo_policy_data = {},
            std::vector<uint8_t> info_binary_cache_data = {});

    void set_header(
            std::string model_decryption_method = "NONE",
            std::string info_decryption_method = "NONE", bool is_fast_run_cache = true);

    void pack_model();

private:
    std::vector<uint8_t> m_info_data;
    //! fastrun cache / algo policy
    std::vector<uint8_t> m_algo_policy_data;
    //! binary cache
    std::vector<uint8_t> m_binary_cache_data;
    std::string m_packed_model_path;
    Header m_header;

    friend class FbsHelper;
    FbsHelper* m_fbs_helper;
};

}  // namespace lite