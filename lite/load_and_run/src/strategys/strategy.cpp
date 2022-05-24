#include "strategy.h"
#include <iostream>
#include "strategy_fitting.h"
#include "strategy_normal.h"

using namespace lar;
DECLARE_bool(fitting);
std::shared_ptr<StrategyBase> StrategyBase::create_strategy(std::string model_path) {
    if (FLAGS_fitting) {
        return std::make_shared<FittingStrategy>(model_path);
    } else {
        return std::make_shared<NormalStrategy>(model_path);
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}