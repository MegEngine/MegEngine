/**
 * \file dnn/src/common/indexing_multi_axis_vec_kdef.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/arch.h"

#if MEGDNN_CC_HOST && !defined(__device__)
#define __device__
#define def_device  1
#endif

namespace megdnn {
namespace indexing_multi_axis_vec_kdef {

struct OprFwd {
    template<typename ctype>
    __device__ static void apply(ctype data, ctype &value) {
        value = data;
    }
};

struct OprSet {
    template<typename ctype>
    __device__ static void apply(ctype &data, ctype value) {
        data = value;
    }
};

struct OprIncr {
    template<typename ctype>
    __device__ static void apply(ctype &data, ctype value) {
        data += value;
    }
};

}
}

#if def_device
#undef __device__
#undef def_device
#endif

// vim: syntax=cpp.doxygen
