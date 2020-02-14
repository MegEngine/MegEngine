/**
 * \file src/opr/include/megbrain/opr/internal/param_tag_defs.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/dtype.h"
#include <cstdint>

namespace mgb {
namespace opr {

    namespace param_tag {
        enum ParamTag: uint32_t {
            ADD_UPDATE = 1,
            DIMSHUFFLE,
            AXIS_ADD_REMOVE,
            HOST2DEVICE_COPY,
            SUBTENSOR_INDEX_DESC,
            LOOP,
            LOOP_INPUT_MAKER,
            SLEEP,
            NNLIB_EMPTY_CONST
        };
    }

} // namespace opr
} // namespace mgb



// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

