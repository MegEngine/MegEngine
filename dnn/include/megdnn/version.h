#pragma once

#define MEGDNN_MAJOR 9
#define MEGDNN_MINOR 3
#define MEGDNN_PATCH 0

#include "megbrain_build_config.h"
#include "megdnn/internal/visibility_prologue.h"

namespace megdnn {
struct Version {
    int major, minor, patch;
};

//! get megdnn version of the binary
MGE_WIN_DECLSPEC_FUC Version get_version();
}  // namespace megdnn

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
