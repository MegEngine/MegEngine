/**
 * \file dnn/src/common/postprocess.h
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
namespace megdnn {
enum class PostprocessMode : uint8_t {
    FLOAT = 0,   ///< support all biasmode and no_nonlinemode
    NO_PROCESS,  ///< support  non bias and identity
    QUANTIZED,   ///< support  NOBIAS ,BROADCAST_CHANNEL_BIAS and relu hswish
                 ///< identify nonline mode
    ADD_BIAS,    ///< only add bias
};
}