/**
 * \file dnn/test/arm_common/cpuinfo_help.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include <mutex>
#include <vector>
#include "src/common/utils.h"

#if MGB_ENABLE_CPUINFO
#include "cpuinfo.h"
extern const struct cpuinfo_core** cpuinfo_linux_cpu_to_core_map;
class CpuInfoTmpReplace {
public:
    CpuInfoTmpReplace(enum cpuinfo_uarch arch) {
        m_cpuinfo_lock.lock();
        for (uint32_t i = 0; i < cpuinfo_get_cores_count(); ++i) {
            m_arch_bak_vec.push_back(cpuinfo_linux_cpu_to_core_map[i]->uarch);
            ((struct cpuinfo_core**)cpuinfo_linux_cpu_to_core_map)[i]->uarch =
                    arch;
        }
    }
    ~CpuInfoTmpReplace() {
        if (m_arch_bak_vec.size() > 0) {
            for (uint32_t i = 0; i < cpuinfo_get_cores_count(); ++i) {
                ((struct cpuinfo_core**)cpuinfo_linux_cpu_to_core_map)[i]
                        ->uarch = m_arch_bak_vec[i];
            }
        }

        m_cpuinfo_lock.unlock();
    }

private:
    static std::mutex m_cpuinfo_lock;
    std::vector<cpuinfo_uarch> m_arch_bak_vec;
};
#endif

// vim: syntax=cpp.doxygen