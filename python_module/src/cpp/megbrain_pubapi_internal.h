/**
 * \file python_module/src/cpp/megbrain_pubapi_internal.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief internal helpers related to pubapi. Implemented in pubapi.cpp
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain_pubapi.h"
#include "megbrain/graph.h"

namespace mgb {
    /*!
     * \brief fill fields in \p dest with information from other tensors
     *
     * Note that exactly one of \p tensor and \p var must be non-null
     */
    void init_pubapi_dev_tensor(
            pubapi::DeviceTensor &dest,
            DeviceTensorND *tensor, VarNode *var, bool readonly);

    //! convert megbrain dtype to pubapi dtype
    pubapi::DeviceTensor::DataType dtype_mgb2pubapi(DType dtype);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
