#include <gflags/gflags.h>
#include <string>
#include "strategys/strategy.h"

std::string simple_usage = R"(
load_and_run: load_and_run <model_path> [options Flags...]

  Flags from lite/load_and_run/src/models/model.cpp:
    -lite           type: bool      default: false      use megengine lite interface to run model

  Flags from lite/load_and_run/src/options/strategy_options.cpp:
    -iter           type: int32     default: 10         iteration number for run model
    -thread         type: int32     default: 1          thread number for run model when <thread> is supported
    -warmup_iter    type: int32     default: 1          iteration number for warm up model before run
    
  Flags from com_github_gflags_gflags/src/gflags.cc:
    -flagfile       type: string      default: ""       load flags from file
    -fromenv        type: string      default: ""       set flags from the environment [use 'export FLAGS_flag1=value']
    ...

  Flags from com_github_gflags_gflags/src/gflags_reporting.cc:
    -help           type: bool        default: false    show help on all flags
    -helpmatch      type: string      default: ""       show help on modules whose name contains the specified substr
    -version        type: bool        default: false    show version and build info and exit
    ...

More details using "--help" to get!!

)";

int main(int argc, char** argv) {
    std::string usage = "load_and_run <model_path> [options Flags...]";
    if (argc < 2) {
        printf("usage: %s\n", simple_usage.c_str());
        return -1;
    }
    gflags::SetUsageMessage(usage);
    gflags::SetVersionString("1.0");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::string model_path = argv[1];
    auto strategy = lar::StrategyBase::create_strategy(model_path);
    strategy->run();
    gflags::ShutDownCommandLineFlags();

    return 0;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
