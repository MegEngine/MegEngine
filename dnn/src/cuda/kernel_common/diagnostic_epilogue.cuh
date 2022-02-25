#ifdef __GNUC__
#if CUDA_VERSION < 9020
#pragma GCC diagnostic pop
#endif
#endif

#ifdef MEGDNN_DIAGNOSTIC_PROLOGUE_INCLUDED
#undef MEGDNN_DIAGNOSTIC_PROLOGUE_INCLUDED
#else
#error "diagnostic_epilogue.h must be included after diagnostic_prologue.h"
#endif

// vim: syntax=cpp.doxygen
