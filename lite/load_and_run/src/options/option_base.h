/**
 * \file lite/load_and_run/src/options/option_base.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "megbrain/common.h"

#include "helpers/common.h"
#include "models/model.h"

namespace lar {
/*!
 * \brief: base class of options
 */
class OptionBase {
public:
    //! configure  model in different runtime state
    virtual void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) = 0;
    //! get depend options
    virtual std::vector<std::string> depend_option() const { return {}; };

    //! get option name
    virtual std::string option_name() const = 0;

    virtual ~OptionBase() = default;
};

/*!
 * \brief: Singleton option factory for register options before main function
 */
class OptionFactory {
public:
    using OptionCreator = std::function<std::shared_ptr<OptionBase>()>;
    using OptionMap = std::unordered_map<std::string, OptionCreator>;

    //! get Singleton option factory
    static OptionFactory& get_Instance() {
        static OptionFactory instance;
        return instance;
    }

    //! registe option creator into option map
    void registe_options(std::string name, OptionCreator creator) {
        if (option_creator_map.count(name) == 0) {
            option_creator_map[name] = creator;
        }
    }

    //! get creator map
    OptionMap* get_option_creator_map() { return &option_creator_map; }

private:
    OptionFactory(){};
    OptionMap option_creator_map;
};

}  // namespace lar

#define REGIST_OPTION_CREATOR(name_, creator_)                                    \
    struct OptionRegister_##name_ {                                               \
        OptionRegister_##name_() {                                                \
            lar::OptionFactory::get_Instance().registe_options(#name_, creator_); \
        }                                                                         \
    };                                                                            \
    OptionRegister_##name_ name_;

#define CONFIG_MODEL_FUN                                                    \
    if (model->type() == ModelType::LITE_MODEL) {                           \
        config_model_internel<ModelLite>(                                   \
                runtime_param, std::static_pointer_cast<ModelLite>(model)); \
    } else if (model->type() == ModelType::MEGDL_MODEL) {                   \
        config_model_internel<ModelMdl>(                                    \
                runtime_param, std::static_pointer_cast<ModelMdl>(model));  \
    }
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}