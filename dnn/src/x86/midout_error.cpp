/**
 * \file dnn/src/x86/midout_error.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#if defined(MIDOUT_GENERATED) || defined(MIDOUT_PROFILING)
#error "midout should not be enabled on x86, because current x86 implemention requires all possible inputs to be passed in midout, which is essentially impossible in production as the input spatial size is unfixed."
#endif

// vim: syntax=cpp.doxygen
