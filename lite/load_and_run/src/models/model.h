/**
 * \file lite/load_and_run/src/models/model.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#pragma once
#include <gflags/gflags.h>
#include <string>
#include "helpers/common.h"

DECLARE_bool(lite);

namespace lar {
/*!
 * \brief: base class of model
 */
class ModelBase {
public:
    //! get model type by the magic number in dump file
    static ModelType get_model_type(std::string model_path);

    //! create model by different model type
    static std::shared_ptr<ModelBase> create_model(std::string model_path);

    //! type of the model
    virtual ModelType type() = 0;

    //! set model load state

    virtual void set_shared_mem(bool state) = 0;

    //! load model interface for load and run strategy
    virtual void load_model() = 0;

    //! run model interface for load and run strategy
    virtual void run_model() = 0;

    //! wait asynchronous function interface for load and run strategy
    virtual void wait() = 0;

    virtual ~ModelBase() = default;
};
}  // namespace lar

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
