/**
 * \file dnn/test/common/tensor.h
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
#include <gtest/gtest.h>

#include <memory>
#include "test/common/comparator.h"

namespace megdnn {
namespace test {

/**
 * \brief A thin wrapper over megdnn::TensorND.
 *
 * dtype is determined by T; layout.dtype is ignored.
 */
template <typename T = dt_float32, typename Comparator = DefaultComparator<T>>
class Tensor {
public:
    Tensor(Handle* handle, TensorLayout layout);
    ~Tensor();

    T* ptr();
    const T* ptr() const;

    TensorND tensornd() const { return m_tensornd; }

    TensorLayout layout() const;

    template <typename C>
    void check_with(const Tensor<T, C>& rhs) const;

private:
    Handle* m_handle;
    TensorND m_tensornd;
    Comparator m_comparator;
};

/**
 * \brief A wrapper over host and device tensor.
 *
 * dtype is determined by T; layout.dtype is ignored.
 */
template <typename T = dt_float32, typename Comparator = DefaultComparator<T>>
class SyncedTensor {
public:
    SyncedTensor(Handle* dev_handle, TensorLayout layout);
    SyncedTensor(Handle* dev_handle, const TensorShape& shape)
            : SyncedTensor(
                      dev_handle,
                      TensorLayout{shape, typename DTypeTrait<T>::dtype()}) {}
    ~SyncedTensor() = default;

    const T* ptr_host();
    const T* ptr_dev();

    T* ptr_mutable_host();
    T* ptr_mutable_dev();

    TensorND tensornd_host();
    TensorND tensornd_dev();

    TensorLayout layout() const;

    template <typename C>
    void check_with(SyncedTensor<T, C>& rhs);

private:
    std::unique_ptr<Handle> m_handle_host;
    Handle* m_handle_dev;
    Tensor<T, Comparator> m_tensor_host, m_tensor_dev;

    enum class SyncState {
        HOST,
        DEV,
        SYNCED,
        UNINITED,
    } m_sync_state;

    void ensure_host();
    void ensure_dev();
};

//! make a device tensor on handle by value on host tensor
std::shared_ptr<TensorND> make_tensor_h2d(Handle* handle,
                                          const TensorND& htensor);

//! make a host tensor from device tensor on handle
std::shared_ptr<TensorND> make_tensor_d2h(Handle* handle,
                                          const TensorND& dtensor);

//! load tensors onto host from file (can be dumpped by megbrain tests)
std::vector<std::shared_ptr<TensorND>> load_tensors(const char* fpath);

void init_gaussian(SyncedTensor<dt_float32>& tensor, dt_float32 mean = 0.0f,
                   dt_float32 stddev = 1.0f);

}  // namespace test
}  // namespace megdnn

#include "test/common/tensor.inl"

// vim: syntax=cpp.doxygen
