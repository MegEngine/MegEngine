#pragma once

#include "strategy.h"

namespace lar {
/*!
 * \brief: normal strategy for running
 */
class NormalStrategy : public StrategyBase {
public:
    NormalStrategy(std::string model_path);

    //! run model with runtime parameter
    void run() override;

private:
    //! run model subline for multiple thread
    void run_subline();

    std::string m_model_path;
};
}  // namespace lar