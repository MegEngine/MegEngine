/**
 * \file test/src/megdnn_helper.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/test/megdnn_helper.h"
#include "megbrain/common.h"

#define MEGCORE_CHECK(expr)                  \
    do {                                     \
        auto _code = expr;                   \
        mgb_assert(_code == megcoreSuccess); \
    } while (0)

namespace {
struct MegDNNHandleBundle {
    std::unique_ptr<megdnn::Handle> handle;
    megcoreDeviceHandle_t dev_hdl;
    megcoreComputingHandle_t comp_hdl;

    MegDNNHandleBundle() {
        MEGCORE_CHECK(megcoreCreateDeviceHandle(&dev_hdl, megcorePlatformCPU));
        MEGCORE_CHECK(megcoreCreateComputingHandle(&comp_hdl, dev_hdl));
        handle = megdnn::Handle::make(comp_hdl, 2);
    }

    ~MegDNNHandleBundle() {
        MEGCORE_CHECK(megcoreDestroyComputingHandle(comp_hdl));
        MEGCORE_CHECK(megcoreDestroyDeviceHandle(dev_hdl));
    }
};
}  // anonymous namespace

megdnn::Handle* mgb::megdnn_naive_handle() {
    static MegDNNHandleBundle handle;
    return handle.handle.get();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
