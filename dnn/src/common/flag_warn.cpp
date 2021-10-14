/**
 * \file dnn/src/common/flag_warn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/config/config.h"

#if !MEGDNN_ENABLE_MANGLING
#pragma message "Mangling is disabled."
#endif  // MEGDNN_ENABLE_MANGLING

// vim: syntax=cpp.doxygen
