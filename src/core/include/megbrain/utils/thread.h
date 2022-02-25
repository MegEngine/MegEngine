#pragma once

#include "megbrain_build_config.h"
#if MGB_HAVE_THREAD
#include "./thread_impl_1.h"
#include "./thread_local.h"
#else
#include "./thread_impl_0.h"
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
