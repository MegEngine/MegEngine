/**
 * \file dnn/src/common/cv/enums.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

enum BorderMode {
    BORDER_REPLICATE = 0,
    BORDER_REFLECT = 1,
    BORDER_REFLECT_101 = 2,
    BORDER_WRAP = 3,
    BORDER_CONSTANT = 4,
    BORDER_TRANSPARENT = 5,
    BORDER_ISOLATED = 6
};
enum InterpolationMode {
    INTER_NEAREST = 0,
    INTER_LINEAR = 1,
    INTER_AREA = 2,
    INTER_CUBIC = 3,
    INTER_LANCZOS4 = 4
};

// vim: syntax=cpp.doxygen
