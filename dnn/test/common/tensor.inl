/**
 * \file dnn/test/common/tensor.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./tensor.h"

#include "megdnn/basic_types.h"
#include "test/common/index.h"
#include "test/common/get_dtype_from_static_type.h"
#include "test/common/utils.h"
#include <memory>

namespace megdnn {
namespace test {

template <typename T, typename C>
Tensor<T, C>::Tensor(Handle *handle, TensorLayout layout):
    m_handle(handle),
    m_comparator(C())
{
    if (!layout.dtype.valid())
        layout.dtype = get_dtype_from_static_type<T>();
    m_tensornd.raw_ptr = megdnn_malloc(m_handle, layout.span().dist_byte());
    m_tensornd.layout = layout;
}

template <typename T, typename C>
Tensor<T, C>::~Tensor()
{
    megdnn_free(m_handle, m_tensornd.raw_ptr);
}

template <typename T, typename C>
T *Tensor<T, C>::ptr()
{
    return m_tensornd.ptr<T>();
}

template <typename T, typename C>
const T *Tensor<T, C>::ptr() const
{
    return m_tensornd.ptr<T>();
}

template <typename T, typename C>
TensorLayout Tensor<T, C>::layout() const
{
    return m_tensornd.layout;
}

template <typename T, typename C> template <typename C_>
void Tensor<T, C>::check_with(const Tensor<T, C_> &rhs) const
{
    // compare layout
    ASSERT_TRUE(this->m_tensornd.layout.eq_layout(rhs.m_tensornd.layout))
        << "this->layout is " << this->m_tensornd.layout.to_string()
        << "rhs.layout is " << rhs.m_tensornd.layout.to_string();
    // compare value
    auto n = m_tensornd.layout.total_nr_elems();
    auto p0 = this->ptr(), p1 = rhs.ptr();
    for (size_t linear_idx = 0; linear_idx < n; ++linear_idx) {
        auto index = Index(m_tensornd.layout, linear_idx);
        auto offset = index.positive_offset();
        ASSERT_TRUE(m_comparator.is_same(p0[offset], p1[offset]))
                << "Index is " << index.to_string() << "; layout is "
                << m_tensornd.layout.to_string() << "; this->ptr()[offset] is "
                << this->ptr()[offset] << "; rhs.ptr()[offset] is "
                << rhs.ptr()[offset];
    }
}

template <typename T, typename C>
SyncedTensor<T, C>::SyncedTensor(Handle *dev_handle, TensorLayout layout):
    m_handle_host(create_cpu_handle(2, false)),
    m_handle_dev(dev_handle),
    m_tensor_host(m_handle_host.get(), layout),
    m_tensor_dev(m_handle_dev, layout),
    m_sync_state(SyncState::UNINITED)
{
}

template <typename T, typename C>
const T *SyncedTensor<T, C>::ptr_host()
{
    ensure_host();
    return m_tensor_host.tensornd().template ptr<T>();
}

template <typename T, typename C>
const T *SyncedTensor<T, C>::ptr_dev()
{
    ensure_dev();
    return m_tensor_dev.tensornd().template ptr<T>();
}

template <typename T, typename C>
T *SyncedTensor<T, C>::ptr_mutable_host()
{
    ensure_host();
    m_sync_state = SyncState::HOST;
    return m_tensor_host.tensornd().template ptr<T>();
}

template <typename T, typename C>
T *SyncedTensor<T, C>::ptr_mutable_dev()
{
    ensure_dev();
    m_sync_state = SyncState::DEV;
    return m_tensor_dev.tensornd().template ptr<T>();
}

template <typename T, typename C>
TensorND SyncedTensor<T, C>::tensornd_host()
{
    ensure_host();
    m_sync_state = SyncState::HOST;
    return m_tensor_host.tensornd();
}

template <typename T, typename C>
TensorND SyncedTensor<T, C>::tensornd_dev()
{
    ensure_dev();
    m_sync_state = SyncState::DEV;
    return m_tensor_dev.tensornd();
}

template <typename T, typename C>
TensorLayout SyncedTensor<T, C>::layout() const
{
    return m_tensor_host.tensornd().layout;
}

template <typename T, typename C> template <typename C_>
void SyncedTensor<T, C>::check_with(SyncedTensor<T, C_> &rhs)
{
    this->ensure_host();
    rhs.ensure_host();
    this->m_tensor_host.check_with(rhs.m_tensor_host);
}

template <typename T, typename C>
void SyncedTensor<T, C>::ensure_host()
{
    if (m_sync_state == SyncState::HOST || m_sync_state == SyncState::SYNCED) {
        return;
    }
    if (m_sync_state == SyncState::DEV) {
        megdnn_memcpy_D2H(m_handle_dev,
                m_tensor_host.ptr(), m_tensor_dev.ptr(),
                m_tensor_host.layout().span().dist_byte());
    }
    m_sync_state = SyncState::SYNCED;
}

template <typename T, typename C>
void SyncedTensor<T, C>::ensure_dev()
{
    if (m_sync_state == SyncState::DEV || m_sync_state == SyncState::SYNCED) {
        return;
    }
    if (m_sync_state == SyncState::HOST) {
        megdnn_memcpy_H2D(m_handle_dev,
                m_tensor_dev.ptr(), m_tensor_host.ptr(),
                m_tensor_host.layout().span().dist_byte());
    }
    m_sync_state = SyncState::SYNCED;
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen
