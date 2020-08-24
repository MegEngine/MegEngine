/**
 * \file dnn/src/cambricon/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cambricon/utils.h"
#include "src/cambricon/utils.mlu.h"

#include "src/cambricon/handle.h"
#include "src/common/utils.h"

#include <mutex>
#include <unordered_map>

using namespace megdnn;
using namespace cambricon;

namespace {
struct DeviceInfoRecord {
    bool init = false;
    cnrtDeviceInfo_t device_info;
    std::mutex mtx;
};
std::unordered_map<cnrtDev_t, int> dev2device_id;
std::mutex dev2device_id_mtx;
constexpr int MAX_NR_DEVICE = 64;
DeviceInfoRecord device_info_rec[MAX_NR_DEVICE];
}  // namespace

void cambricon::__throw_cnrt_error__(cnrtRet_t err, const char* msg) {
    auto s = ssprintf("cnrt return %s(%d) occurred; expr: %s",
                      cnrtGetErrorStr(err), int(err), msg);
    megdnn_throw(s.c_str());
}

cnrtDeviceInfo_t cambricon::current_device_info() {
    static bool dev2device_id_init = false;
    {
        std::lock_guard<std::mutex> lock(dev2device_id_mtx);
        if (!dev2device_id_init) {
            unsigned int dev_num = 0;
            cnrt_check(cnrtGetDeviceCount(&dev_num));
            for (unsigned int dev_id = 0; dev_id < dev_num; ++dev_id) {
                cnrtDev_t dev;
                cnrt_check(cnrtGetDeviceHandle(&dev, dev_id));
                dev2device_id[dev] = dev_id;
            }
            dev2device_id_init = true;
        }
    }

    cnrtDev_t dev;
    cnrt_check(cnrtGetCurrentDevice(&dev));
    {
        std::lock_guard<std::mutex> lock(dev2device_id_mtx);
        int dev_id = dev2device_id.at(dev);
        auto& rec = device_info_rec[dev_id];
        {
            std::lock_guard<std::mutex> lock(rec.mtx);
            if (!rec.init) {
                cnrt_check(cnrtGetDeviceInfo(&rec.device_info, dev_id));
                rec.init = true;
            }
        }
        return rec.device_info;
    }
}

// vim: syntax=cpp.doxygen

