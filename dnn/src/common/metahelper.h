/**
 * \file dnn/src/common/metahelper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

namespace megdnn {
/*!
 * \brief base class for non-copyable objects
 */
class NonCopyableObj {
    NonCopyableObj(const NonCopyableObj&) = delete;
    NonCopyableObj& operator=(const NonCopyableObj&) = delete;

public:
    NonCopyableObj() = default;
};

}  // namespace megdnn

// vim: syntax=cpp.doxygen
