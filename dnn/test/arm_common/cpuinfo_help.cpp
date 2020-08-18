/**
 * \file dnn/test/arm_common/cpuinfo_help.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/common/utils.h"
#include "test/arm_common/cpuinfo_help.h"
#if MGB_ENABLE_CPUINFO
std::mutex CpuInfoTmpReplace::m_cpuinfo_lock;
#endif
// vim: syntax=cpp.doxygen