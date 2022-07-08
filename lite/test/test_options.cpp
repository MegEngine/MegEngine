#include "test_options.h"
using namespace lar;

void lar::run_NormalStrategy(std::string model_path) {
    auto origin_level = mgb::get_log_level();
    mgb::set_log_level(mgb::LogLevel::WARN);
    NormalStrategy strategy(model_path);
    strategy.run();
    mgb::set_log_level(origin_level);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
