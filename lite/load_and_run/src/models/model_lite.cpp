/**
 * \file lite/load_and_run/src/models/model_lite.cpp
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */
#include "model_lite.h"
#include <gflags/gflags.h>
#include <cstring>
#include "misc.h"

DECLARE_bool(share_param_mem);

using namespace lar;
ModelLite::ModelLite(const std::string& path) : model_path(path) {
    LITE_WARN("creat lite model use CPU as default comp node");
};
void ModelLite::load_model() {
    m_network = std::make_shared<lite::Network>(config, IO);
    if (share_model_mem) {
        //! WARNNING:maybe not right to share param memmory for this
        LITE_WARN("enable share model memory");

        FILE* fin = fopen(model_path.c_str(), "rb");
        LITE_ASSERT(fin, "failed to open %s: %s", model_path.c_str(), strerror(errno));
        fseek(fin, 0, SEEK_END);
        size_t size = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        void* ptr = malloc(size);
        std::shared_ptr<void> buf{ptr, free};
        auto nr = fread(buf.get(), 1, size, fin);
        LITE_ASSERT(nr == size, "read model file failed");
        fclose(fin);

        m_network->load_model(buf.get(), size);
    } else {
        m_network->load_model(model_path);
    }
}

void ModelLite::run_model() {
    m_network->forward();
}

void ModelLite::wait() {
    m_network->wait();
}
