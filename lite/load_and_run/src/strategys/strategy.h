#pragma once
#include <string>
#include <unordered_map>
#include "helpers/common.h"
#include "models/model.h"
#include "options/option_base.h"
namespace lar {
using OptionMap = std::unordered_map<std::string, std::shared_ptr<OptionBase>>;
/*!
 * \brief: load and run strategy base class
 */
class StrategyBase {
public:
    static std::shared_ptr<StrategyBase> create_strategy(std::string model_path);

    virtual void run() = 0;

    virtual ~StrategyBase() = default;

    RuntimeParam m_runtime_param;

    std::shared_ptr<OptionMap> m_options;
};

}  // namespace lar

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
