/**
 * \file dnn/test/common/workspace_wrapper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/workspace_wrapper.h"

#include "test/common/utils.h"

namespace megdnn {
namespace test {

WorkspaceWrapper::WorkspaceWrapper():
    WorkspaceWrapper(nullptr, 0)
{

}

WorkspaceWrapper::WorkspaceWrapper(Handle *handle, size_t size_in_bytes):
    m_handle(handle)
{
    m_workspace.size = size_in_bytes;
    if (m_workspace.size > 0) {
        m_workspace.raw_ptr = static_cast<dt_byte *>(
                megdnn_malloc(handle, size_in_bytes));
    } else {
        m_workspace.raw_ptr = nullptr;
    }
}

void WorkspaceWrapper::update(size_t size_in_bytes)
{
    megdnn_assert(this->valid());
    if (size_in_bytes > m_workspace.size) {
        // free workspace
        if (m_workspace.size > 0) {
            megdnn_free(m_handle, m_workspace.raw_ptr);
            m_workspace.raw_ptr = nullptr;
        }
        // alloc new workspace
        m_workspace.size = size_in_bytes;
        if (m_workspace.size > 0) {
            m_workspace.raw_ptr = static_cast<dt_byte *>(
                    megdnn_malloc(m_handle, size_in_bytes));
        } else {
            m_workspace.raw_ptr = nullptr;
        }
    }
}

WorkspaceWrapper::~WorkspaceWrapper()
{
    if (m_workspace.size > 0) {
        megdnn_free(m_handle, m_workspace.raw_ptr);
        m_workspace.raw_ptr = nullptr;
    }
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
