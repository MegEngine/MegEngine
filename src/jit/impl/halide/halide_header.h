#pragma once

#include "megbrain_build_config.h"

#if MGB_JIT_HALIDE
#if !MGB_JIT
#error "MGB_JIT must be set if MGB_JIT_HALIDE is enabled"
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include <Halide.h>
#pragma GCC diagnostic pop

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
