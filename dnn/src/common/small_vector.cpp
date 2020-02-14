/**
 * \file dnn/src/common/small_vector.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/thin/small_vector.h"
#include "src/common/utils.h"

using namespace megdnn;

void SmallVectorBase::on_invalid_at(size_t idx, size_t size) {
    megdnn_throw(ssprintf("invalid vector at(): idx=%zu size=%zu", idx, size));
    MEGDNN_MARK_USED_VAR(idx);
    MEGDNN_MARK_USED_VAR(size);
}

void SmallVectorBase::grow_pod(void* first_elm_ptr, size_t min_sz_in_bytes,
                               size_t type_size) {
    size_t cur_sz_in_bytes = size_in_bytes();
    size_t new_capacity_in_bytes = 2 * capacity_in_bytes() + type_size;
    if (new_capacity_in_bytes < min_sz_in_bytes) {
        new_capacity_in_bytes = min_sz_in_bytes;
    }
    void* new_begin;
    if (first_elm_ptr == m_begin_ptr) {
        new_begin = malloc(new_capacity_in_bytes);
        memcpy(new_begin, m_begin_ptr, cur_sz_in_bytes);
    } else {
        new_begin = realloc(this->m_begin_ptr, new_capacity_in_bytes);
    }
    this->m_begin_ptr = new_begin;
    this->m_end_ptr = static_cast<char*>(this->m_begin_ptr) + cur_sz_in_bytes;
    this->m_capacity_ptr =
            static_cast<char*>(this->m_begin_ptr) + new_capacity_in_bytes;
}

// vim: syntax=cpp.doxygen
