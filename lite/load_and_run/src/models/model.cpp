#include "model.h"
#include <iostream>
#include <memory>
#include "model_lite.h"
#include "model_mdl.h"

using namespace lar;

ModelType ModelBase::get_model_type(std::string model_path) {
    //! read magic number of dump file
    FILE* fin = fopen(model_path.c_str(), "rb");
    mgb_assert(fin, "failed to open %s: %s", model_path.c_str(), strerror(errno));
    char buf[16];
    mgb_assert(fread(buf, 1, 16, fin) == 16, "read model failed");
    fclose(fin);

    // get model type
    std::string tag(buf);
    ModelType type;
    if (tag.substr(0, 7) == std::string("mgb0001") ||
        tag.substr(0, 8) == std::string("mgb0000a") ||
        tag.substr(0, 4) == std::string("MGBS") ||
        tag.substr(0, 4) == std::string("MGBC") ||
        tag.substr(0, 8) == std::string("mgbtest0")) {
        type = ModelType::MEGDL_MODEL;

    } else {
        type = ModelType::LITE_MODEL;
    }

    return type;
}

std::shared_ptr<ModelBase> ModelBase::create_model(std::string model_path) {
    mgb_log_debug("model path %s\n", model_path.c_str());

    auto model_type = get_model_type(model_path);

    if (ModelType::LITE_MODEL == model_type) {
        return std::make_shared<ModelLite>(model_path);
    } else if (ModelType::MEGDL_MODEL == model_type) {
        if (FLAGS_lite)
            return std::make_shared<ModelLite>(model_path);
        else
            return std::make_shared<ModelMdl>(model_path);
    } else {
        return nullptr;
    }
}
DEFINE_bool(lite, false, "use megengine lite interface to run model");
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
