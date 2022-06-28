#pragma once

#include "megbrain_build_config.h"

#define MGE_MAJOR 1
#define MGE_MINOR 10
#define MGE_PATCH 0

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
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
