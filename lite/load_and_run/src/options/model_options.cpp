/**
 * \file lite/load_and_run/src/options/model_options.cpp
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#include "model_options.h"
#include "device_options.h"
#include "lite/pack_model.h"
#include "misc.h"
#include "models/model_lite.h"
#include "models/model_mdl.h"

namespace lar {
template <typename ModelImpl>
void PackModelOption::config_model_internel(
        RuntimeParam& runtime_param, std::shared_ptr<ModelImpl> model) {
    if (runtime_param.stage == RunStage::AFTER_MODEL_RUNNING) {
        lite::ModelPacker packer(
                model->get_model_path(), packed_model_dump, pack_info_json, pack_cache,
                pack_binary_cache);
        packer.set_header(pack_info_cryption, pack_model_cryption, is_fast_run_cache);
        packer.pack_model();
    }
}
}  // namespace lar
using namespace lar;
////////////////////// PackModel options ////////////////////////

PackModelOption::PackModelOption() {
    m_option_name = "pack_model";
    if (!FLAGS_packed_model_dump.empty())
        packed_model_dump = FLAGS_packed_model_dump;
    if (!FLAGS_pack_info_json.empty())
        pack_info_json = FLAGS_pack_info_json;
    if (!FLAGS_pack_cache.empty())
        pack_cache = FLAGS_pack_cache;
    if (!FLAGS_pack_info_cryption.empty())
        pack_info_cryption = FLAGS_pack_info_cryption;
    if (!FLAGS_pack_model_cryption.empty())
        pack_model_cryption = FLAGS_pack_model_cryption;
}

bool PackModelOption::is_valid() {
    return !FLAGS_packed_model_dump.empty();
}

std::shared_ptr<OptionBase> PackModelOption::create_option() {
    static std::shared_ptr<PackModelOption> option(new PackModelOption);
    if (PackModelOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void PackModelOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}

////////////////////// PackModel gflags ////////////////////////

DEFINE_string(packed_model_dump, "", "The output file path of packed model.");
DEFINE_string(
        pack_info_json, "",
        "An encrypted or not encrypted json format file to pack into the model.");
DEFINE_string(pack_cache, "", "Pack the fastrun cache or algo policy into the model.");
DEFINE_string(
        pack_info_cryption, "NONE",
        "The info data encryption method name, this is used to find the right "
        "decryption method. --pack-info-cryption [ AES_default | RC4_default | "
        "SIMPLE_FAST_RC4_default ], default is NONE. See "
        "https://megengine.megvii-inc.com/user-guide/deployment/lite/advance/"
        "pack-lite-model.html for more details.");
DEFINE_string(
        pack_model_cryption, "NONE",
        "The model encryption method name, this is used to find the right decryption "
        "method. --pack-model-cryption [ AES_default | RC4_default | "
        "SIMPLE_FAST_RC4_default ], default is NONE. See "
        "https://megengine.megvii-inc.com/user-guide/deployment/lite/advance/"
        "pack-lite-model.html for more details.");

REGIST_OPTION_CREATOR(pack_model, lar::PackModelOption::create_option);
