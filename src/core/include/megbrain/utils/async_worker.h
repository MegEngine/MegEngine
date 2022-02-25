#pragma once

#include "megbrain_build_config.h"
#if MGB_HAVE_THREAD
#include "./async_worker_impl_1.h"
#else
#include "./async_worker_impl_0.h"
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
