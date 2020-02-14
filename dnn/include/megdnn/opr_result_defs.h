/**
 * \file dnn/include/megdnn/opr_result_defs.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <stdint.h>

namespace megdnn {
namespace opr_result {

    struct Checksum {
        uint32_t checksum;
        union {
            int32_t iv;
            float fv;
        } last_val;

        bool operator == (const Checksum &rhs) const {
            return checksum == rhs.checksum &&
                last_val.iv == rhs.last_val.iv;
        }

        bool operator != (const Checksum &rhs) const {
            return !operator==(rhs);
        }
    };

} // namespace opr_result
} // namespace megdnn


// vim: syntax=cpp.doxygen
