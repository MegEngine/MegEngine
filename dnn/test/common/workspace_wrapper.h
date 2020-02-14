/**
 * \file dnn/test/common/workspace_wrapper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/basic_types.h"
#include "megdnn/handle.h"

namespace megdnn {
namespace test {

class WorkspaceWrapper {
public:
    WorkspaceWrapper();
    WorkspaceWrapper(Handle* handle, size_t size_in_bytes = 0);
    ~WorkspaceWrapper();

    void update(size_t size_in_bytes);

    bool valid() const { return m_handle != nullptr; }
    Workspace workspace() const { return m_workspace; }

private:
    Handle* m_handle;
    Workspace m_workspace;
};

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
