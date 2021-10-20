/**
 * \file inlude/lite/tensor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "common_enum_c.h"
#include "macro.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace lite {

/*!
 * \brief the simple layout description
 */
struct LITE_API Layout {
    static constexpr uint32_t MAXDIM = 7;
    size_t shapes[MAXDIM];
    size_t ndim = 0;
    LiteDataType data_type = LiteDataType::LITE_FLOAT;

    //! get the total byte of a layout
    size_t get_elem_size() const;

    //! compare whether the two layout is equal
    bool operator==(const Layout& other) const;
};

/*!
 * \brief warpper of the MegEngine Tensor
 *
 * The memory is not alloc directly, when call get_memory_ptr() the memory
 * will be allocated in tensor implement, which will be deleted automatically
 *
 * Note: if the tensor memory is set through reset() interface, the memory is
 * managed by the user, it will not be freed by the tensor
 *
 * If the device or layout is not set, when copy form other source tensor, its
 * device and layout will be copy form the source tensor
 *
 * if is_pinned_host is set, the storage memory of the tensor is pinned memory,
 * this is used to Optimize the H2D or D2H memory copy, if the device or layout
 * is not set, when copy form other device(CUDA) tensor, this tensor
 * will be automatically set to pinned tensor
 */
class LITE_API Tensor {
    class TensorImpl;

public:
    class TensorImplBase;

    Tensor();
    Tensor(LiteDeviceType device_type, bool is_pinned_host = false);
    Tensor(LiteDeviceType device_type, const Layout& layout,
           bool is_pinned_host = false);
    Tensor(int device_id, LiteDeviceType device_type, const Layout& layout = {},
           bool is_pinned_host = false);
    Tensor(int device_id, int stream_id, LiteDeviceType device_type,
           bool is_pinned_host = false);
    Tensor(LiteBackend backend, LiteDeviceType device_type = LiteDeviceType::LITE_CPU,
           int device_id = 0, const Layout& layout = {}, bool is_pinned_host = false);
    ~Tensor();

    LiteDeviceType get_device_type() const { return m_device_type; };

    int get_device_id() const { return m_device_id; };

    Layout get_layout() const { return m_layout; };

    bool is_pinned_host() const { return m_is_pinned_host; };

    //! set layout will change the layout and reallocate memory of the tensor
    void set_layout(const Layout& layout);

    //! which will trigger memory alloc in tensor implement
    void* get_memory_ptr() const;

    //! get the memory with the offset describe in idx
    void* get_memory_ptr(const std::vector<size_t>& idx) const;

    //! get the tensor capacity in byte
    size_t get_tensor_total_size_in_byte() const;

    //! use the user allocated data to reset the memory of the tensor, the
    //! memory will not be managed by the lite, later, the user should delete
    //! it.
    void reset(void* prepared_data, size_t data_length_in_byte);

    //! use the user allocated data and corresponding layout to reset the data
    //! and layout of the tensor, the memory will not be managed by lite, later,
    //! the user should delete it.
    void reset(void* prepared_data, const Layout& layout);

    //! reshape the tensor with new shape, keep the data_type the same
    void reshape(const std::vector<int>& shape);

    //! get a new tensor slice from the origin tensor
    std::shared_ptr<Tensor> slice(
            const std::vector<size_t>& start, const std::vector<size_t>& end,
            const std::vector<size_t>& step = {});

    //! set the tensor memory with zero
    void fill_zero();

    //! copy tensor form other tensor
    //! Note: the best way for tensor copy is just set the dst device, left
    //! layout empty, when copying the dst layout will be set the same with
    //! src
    void copy_from(const Tensor& src);

    //! share memory with other tensor
    void share_memory_with(const Tensor& src_tensor);

    //! whether the memory of tensor is continue
    bool is_continue_memory() const;

    //! update the menbers from the implement
    void update_from_implement();

public:
    friend class TensorHelper;

private:
    std::shared_ptr<TensorImplBase> m_tensor_impl;

    //! flag whether the storage of the tensor is pinned, this is only used
    //! when the compnode is not in CPU
    bool m_is_pinned_host = false;
    int m_device_id = 0;
    Layout m_layout;
    //! the device of the tensor should not be changed after the tensor has
    //! constructed
    LiteDeviceType m_device_type = LiteDeviceType::LITE_CPU;
};

/**
 * \brief a class can hold any type data, but not check whether the visit type
 * is valid
 */
class LITE_API LiteAny {
public:
    LiteAny() = default;
    template <class T>
    LiteAny(T value) : m_holder(new AnyHolder<T>(value)) {
        m_is_string = std::is_same<std::string, T>();
    }

    LiteAny(const LiteAny& any) {
        m_holder = any.m_holder->clone();
        m_is_string = any.is_string();
    }
    LiteAny& operator=(const LiteAny& any) {
        m_holder = any.m_holder->clone();
        m_is_string = any.is_string();
        return *this;
    }
    bool is_string() const { return m_is_string; }

    class HolderBase {
    public:
        virtual ~HolderBase() = default;
        virtual std::shared_ptr<HolderBase> clone() = 0;
        virtual size_t type_length() const = 0;
    };

    template <class T>
    class AnyHolder : public HolderBase {
    public:
        AnyHolder(const T value) : m_value(value) {}
        virtual std::shared_ptr<HolderBase> clone() override {
            return std::make_shared<AnyHolder>(m_value);
        }
        virtual size_t type_length() const override { return sizeof(T); }

    public:
        T m_value;
    };
    //! if type is miss matching, it will throw
    void type_missmatch(size_t expect, size_t get) const;

    //! only check the storage type and the visit type length, so it's not safe
    template <class T>
    T unsafe_cast() const {
        if (sizeof(T) != m_holder->type_length()) {
            type_missmatch(m_holder->type_length(), sizeof(T));
        }
        return static_cast<LiteAny::AnyHolder<T>*>(m_holder.get())->m_value;
    }
    //! only check the storage type and the visit type length, so it's not safe
    void* cast_void_ptr() const {
        return &static_cast<LiteAny::AnyHolder<char>*>(m_holder.get())->m_value;
    }

private:
    std::shared_ptr<HolderBase> m_holder;
    bool m_is_string = false;
};

/*********************** special tensor function ***************/
class LITE_API TensorUtils {
public:
    //! concat all the input tensor to one on the specified dim, the result
    //! tensor reside in dst_device_id of dst_device, if dst_device is
    //! LITE_DEVICE_DEFAULT, the device will get from the first tensor
    static std::shared_ptr<Tensor> concat(
            const std::vector<Tensor>& tensors, int dim,
            LiteDeviceType dst_device = LiteDeviceType::LITE_DEVICE_DEFAULT,
            int dst_device_id = -1);
};
}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
