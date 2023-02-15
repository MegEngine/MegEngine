#ifdef MEGDNN_DIAGNOSTIC_PROLOGUE_INCLUDED
#error "diagnostic_prologue.h included twice without including diagnostic_epilogue.h"
#else
#define MEGDNN_DIAGNOSTIC_PROLOGUE_INCLUDED
#endif
#include <cuda.h>

//! see
//! https://stackoverflow.com/questions/49836419/how-to-hide-nvccs-function-was-declared-but-never-referenced-warnings
//! for more details.
#ifdef __GNUC__
#if CUDA_VERSION < 9020
#pragma GCC diagnostic push
#pragma diag_suppress 177  // suppress "function was declared but never referenced
                           // warning"
#endif
#endif

// vim: syntax=cpp.doxygen
