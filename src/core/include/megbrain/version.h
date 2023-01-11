#pragma once

#include "megbrain_build_config.h"

#define MGE_MAJOR 1
#define MGE_MINOR 12
#define MGE_PATCH 1

// for rc version, could be like "rc1", "rc2", etc
#define MGE_EXTRA_NAME ""

//! whether it is development version
#ifndef MGB_IS_DEV
#define MGB_IS_DEV 0
#endif  // MGB_IS_DEV

namespace mgb {
struct Version {
    int major, minor, patch, is_dev;
};

MGE_WIN_DECLSPEC_FUC Version get_version();
MGE_WIN_DECLSPEC_FUC int get_cuda_version();
MGE_WIN_DECLSPEC_FUC int get_cudnn_version();
MGE_WIN_DECLSPEC_FUC int get_tensorrt_version();
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
