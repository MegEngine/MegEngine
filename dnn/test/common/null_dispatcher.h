/**
 * \file dnn/test/common/null_dispatcher.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megcore.h"

namespace megdnn {
namespace test {

class NullDispatcher final : public MegcoreCPUDispatcher {
public:
    ~NullDispatcher() {}
    void dispatch(Task&&) override {}
    void dispatch(MultiThreadingTask&&, size_t) override {}
    void sync() override {}
    size_t nr_threads() override { return 1; }
};

}  // namespace test
}  // namespace megdnn
