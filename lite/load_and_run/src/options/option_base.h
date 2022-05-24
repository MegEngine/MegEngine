#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "megbrain/common.h"

#include "helpers/common.h"
#include "helpers/utils.h"
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

    //! get option map
    virtual OptionValMap* get_option() { return nullptr; }

    virtual ~OptionBase() = default;
};

/*!
 * \brief: Singleton option factory for register options before main function
 */
class OptionFactory {
public:
    using OptionCreator = std::function<std::shared_ptr<OptionBase>()>;
    using OptionValidater = std::function<void(bool)>;

    using OptionCreatorMap = std::unordered_map<std::string, OptionCreator>;
    using OptionValidaterMap = std::unordered_map<std::string, OptionValidater>;

    //! get Singleton option factory
    static OptionFactory& get_Instance() {
        static OptionFactory instance;
        return instance;
    }

    //! registe option creator into option map
    void registe_options_creator(std::string name, OptionCreator creator) {
        if (m_option_creator_map.count(name) == 0) {
            m_option_creator_map[name] = creator;
        }
    }
    //! registe option validater into option map
    void registe_options_validater(std::string name, OptionValidater validater) {
        if (m_option_validater_map.count(name) == 0) {
            m_option_validater_map[name] = validater;
        }
    }

    //! get creator map
    OptionCreatorMap* get_option_creator_map() { return &m_option_creator_map; }

    //! get validater map
    OptionValidaterMap* get_option_validater_map() { return &m_option_validater_map; }

private:
    OptionFactory(){};
    OptionCreatorMap m_option_creator_map;
    OptionValidaterMap m_option_validater_map;
};

}  // namespace lar

#define REGIST_OPTION_CREATOR(_name, _creator)                          \
    struct CreatorRegister_##_name {                                    \
        CreatorRegister_##_name() {                                     \
            lar::OptionFactory::get_Instance().registe_options_creator( \
                    #_name, _creator);                                  \
        }                                                               \
    };                                                                  \
    CreatorRegister_##_name creator_##_name;

#define REGIST_OPTION_VALIDATER(_name, _validater)                        \
    struct ValitaterRegister_##_name {                                    \
        ValitaterRegister_##_name() {                                     \
            lar::OptionFactory::get_Instance().registe_options_validater( \
                    #_name, _validater);                                  \
        }                                                                 \
    };                                                                    \
    ValitaterRegister_##_name validater_##_name;

#define CONFIG_MODEL_FUN                                                    \
    if (model->type() == ModelType::LITE_MODEL) {                           \
        config_model_internel<ModelLite>(                                   \
                runtime_param, std::static_pointer_cast<ModelLite>(model)); \
    } else if (model->type() == ModelType::MEGDL_MODEL) {                   \
        config_model_internel<ModelMdl>(                                    \
                runtime_param, std::static_pointer_cast<ModelMdl>(model));  \
    }
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}