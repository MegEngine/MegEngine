/**
 * \file src/core/include/megbrain/tensor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/common.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/comp_node.h"
#include "megbrain/dtype.h"

#include "megdnn/basic_types.h"


#include <memory>
#include <limits>

namespace mgb {

using ::megdnn::TensorShape;
using ::megdnn::TensorLayout;
using ::megdnn::TensorFormat;

using ::megdnn::TensorShapeArray;
using ::megdnn::TensorLayoutArray;
using ::megdnn::TensorFormatArray;

/*!
 * \brief specify how a subtensor resides in a larger one
 */
class SubTensorSpec {
    TensorLayout m_layout;

    ptrdiff_t m_offset_elem = 0;

    SubTensorSpec(const TensorLayout &l, ptrdiff_t o):
        m_layout{l}, m_offset_elem{o}
    {}

    public:
        SubTensorSpec() = default;

        //! make a SubTensorSpec from given layout and zero offset
        static SubTensorSpec make_from_layout(const TensorLayout &layout) {
            return make_from_offset_elem(layout, 0);
        }

        //! make a SubTensorSpec from given layout and offset
        static SubTensorSpec make_from_offset_elem(
                const TensorLayout &layout, ptrdiff_t offset_elem);

        //! get underlying layout
        const TensorLayout& layout() const {
            return m_layout;
        }

        //! get offset in number of logical elements in the layout
        ptrdiff_t offset_elem() const {
            return m_offset_elem;
        }

        //! get offset measured in bytes
        ptrdiff_t offset_byte() const {
            return m_offset_elem * m_layout.dtype.size();
        }

        /*!
         * \brief merge with another SubTensorSpec: accum offset, and replace
         *      layout by rhs
         */
        void merge_with(const SubTensorSpec &rhs);
};

/*!
 * \brief slice along some axis; index as in Python, with negative indices
 *      supported
 */
class Slice {
    Maybe<ptrdiff_t> m_begin, m_end, m_step;

    public:
        Slice(Maybe<ptrdiff_t> begin = None,
                Maybe<ptrdiff_t> end = None,
                Maybe<ptrdiff_t> step = None):
            m_begin{begin}, m_end{end}, m_step{step}
        { }

        /*!
         * \brief apply this slice on given tensor layout, and get corresponding
         *      subtensor
         * \param axis the axis to apply this slice; -1 can be used for
         *      flattened layout
         */
        SubTensorSpec apply(TensorLayout layout, int axis) const;
};

template <class Trait> class TensorStorage;

class DeviceTensorStorageTrait;
class HostTensorStorageTrait;

using HostTensorStorage = TensorStorage<HostTensorStorageTrait>;
using DeviceTensorStorage = TensorStorage<DeviceTensorStorageTrait>;

/*!
 * \brief manager for raw tensor memory
 *
 * It contains no dtype information and all sizes are measured in bytes.
 *
 * Note that ensure_size() is lazy, and memory allocation only happens when
 * ptr() or sub() is called
 */
template <class Trait>
class TensorStorage {
    public:
        using RawStorage = std::shared_ptr<dt_byte>;

        TensorStorage() = default;

        TensorStorage(CompNode comp_node):
            m_comp_node(comp_node)
        {}

        TensorStorage(TensorStorage&&) noexcept = default;
        TensorStorage& operator = (TensorStorage&&) noexcept = default;

        TensorStorage(const TensorStorage& rhs) {
            *this = rhs;
        }

        TensorStorage& operator = (const TensorStorage& rhs);

        /*!
         * \brief whether given tensor span is valid in this storage
         */
        bool valid_span(const TensorLayout::Span &span) const {
            return m_comp_node.valid() &&
                static_cast<ptrdiff_t>(m_offset) + span.low_byte >= 0 &&
                span.high_byte <= size();
        }

        /*!
         * \brief ensure that its space could hold at least sz bytes
         *
         * Note
         * 1. This method is lazy; size would only be changed when memory
         *    must be accessed.
         * 2. This method would only grow storage, but it would not release
         *    memory
         */
        TensorStorage& ensure_size(size_t sz);

        /*!
         * \brief return a subtensor that shares the memory; the returned
         *      subtensor is not allowed to realloc
         * \param offset offset given in bytes
         */
        TensorStorage sub(ptrdiff_t offset) const;

        //! apply lazy resize and get ptr
        dt_byte* ptr() const {
            return const_cast<TensorStorage*>(this)->apply_lazy_and_get_ptr();
        }

        /*!
         * \brief usable size in bytes until end of allocated block
         */
        size_t size() const {
            return m_size;
        }

        //! get underlying comp node; error would be raised if it is invalid
        CompNode comp_node() const {
            check_comp_node_valid();
            return m_comp_node;
        }

        //! get underlying comp node and allow it to be invalid
        CompNode comp_node_allow_invalid() const { return m_comp_node; }

        /*!
         * \brief whether underlying comp_node is valid
         */
        bool comp_node_valid() const {
            return m_comp_node.valid();
        }

        /*!
         * \brief whether this tensor has no valid element (either due to
         *      reaching end of mem chunk or no mem allocated)
         */
        bool empty() const {
            return !m_size;
        }

        /*!
         * \brief chain-style computing node setter
         *
         * note that if allow_mem_node_change is true and memory node is
         * changed, the underlying data would be released and this tensor would
         * become empty
         */
        TensorStorage& comp_node(
                CompNode node, bool allow_mem_node_change = false);

        /*!
         * \brief copy from another TensorStorage, possibly of other storage
         *      type
         *
         * This storage must have been initialized
         *
         * \param size number of bytes to be copied; must not exceed size of
         *      this or src
         */
        template<class RTrait>
        void copy_from(const TensorStorage<RTrait> &src, size_t size) const;

        /*!
         * \brief reset the tensor storage to given memory area
         */
        void reset(CompNode node, size_t size, RawStorage data);

        /*!
         * \brief make a TensorStorage that shares memory with another
         *      TensorStorage some different storage type
         *
         * This method can be used to convert between HostTensorStorage and
         * DeviceTensorStorage; \p src must be on CPU memory node.
         */
        template<class RTrait, typename = typename
            std::enable_if<!std::is_same<Trait, RTrait>::value>::type>
        static TensorStorage make_proxy(const TensorStorage<RTrait> &src);

        /*!
         * \brief make a DeviceTensorStorage on default_cpu
         *      that shares memory with this
         *
         * this must be a HostTensorStorage. Alignment not checked.
         */
        template<bool x = true, typename = std::enable_if_t<x && std::is_same<Trait, HostTensorStorageTrait>::value>>
        DeviceTensorStorage proxy_to_default_cpu() const {
            ptr();
            return {true, CompNode::default_cpu(), m_size, m_capacity, m_offset, m_data};
        }

        //! shortcut for raw_storage().use_count(), but won't trigger lazy alloc
        size_t use_count() const {
            if (m_size > m_capacity) {
                return 1;
            }
            return raw_storage().use_count();
        }

        //! whether current capacity is 0 (so we are waiting for lazy init)
        bool has_no_real_storage() const { return !m_capacity; }

        //! get underlying raw reference-counted storage
        const RawStorage& raw_storage() const {
            ptr();  // apply lazy resize
            return m_data;
        }

    private:
        template<class T> friend class TensorStorage;

        bool m_allow_realloc = true;
        CompNode m_comp_node;

        //! current logical size; may exceed m_capacity and in such case memory
        //! would be allocate when ptr() is called
        size_t m_size = 0;

        //! usable size until end of allocated data block, excluding offset
        size_t m_capacity = 0;

        //! offset on m_data
        size_t m_offset = 0;

        RawStorage m_data;

        //! used internally for returning a predefined TensorStorage
        TensorStorage(bool allow_realloc,
                CompNode comp_node,
                size_t size, size_t capacity, size_t offset,
                const RawStorage &data):
            m_allow_realloc(allow_realloc),
            m_comp_node(comp_node),
            m_size(size), m_capacity(capacity), m_offset(offset), m_data(data)
        {}

        void check_comp_node_valid() const {
            if (mgb_unlikely(!m_comp_node.valid()))
                on_invalid_comp_node();
        }

        dt_byte* apply_lazy_and_get_ptr();

        [[noreturn]] static void on_invalid_comp_node();
};


template<class TensorStorage> class TensorND;

using HostTensorND = TensorND<HostTensorStorage>;
using DeviceTensorND = TensorND<DeviceTensorStorage>;

/*!
 * \brief n-dimensional tensor
 *
 * Note that TensorND is built on TensorStorage, which has some lazy behavior.
 */
template<class TensorStorage>
class TensorND {
    TensorStorage m_storage;
    TensorLayout m_layout;

    public:
        using ChainReturnType = TensorND<TensorStorage>;

        TensorND();

        explicit TensorND(CompNode node);

        explicit TensorND(DType dtype);

        TensorND(CompNode node, DType dtype);

        //! allocate contiguous tensor
        TensorND(CompNode node, const TensorShape& shape,
                 DType dtype = dtype::Float32{}, TensorFormat format = {});

        //! allocate contiguous tensor from given comp node and layout; layout
        //! is required to be contiguous, and its dtype and format would be used
        TensorND(CompNode node, const TensorLayout &layout);

        /* ================= shape and basic functionality =================  */

        //! get subtensor according to given slices
        ChainReturnType operator[](std::initializer_list<Slice> slice) const;

        //! get subtensor according to spec
        ChainReturnType sub(const SubTensorSpec &spec) const;


        //! whether underlying storage is empty
        bool empty() const {
            return m_storage.empty();
        }

        //! whether tensor shape is valid (i.e. ndim != 0)
        bool shape_valid() const {
            return m_layout.ndim;
        }

        const TensorShape& shape() const {
            return m_layout;
        }

        const TensorLayout& layout() const {
            return m_layout;
        }

        //! shape at given dimension, with boundary check
        size_t shape(size_t dim) const {
            mgb_assert(dim < m_layout.ndim);
            return m_layout.shape[dim];
        }

        //! get ptr at given index
        template<typename T, typename Iter>
        T* ptr(Iter idx_begin, Iter idx_end) {
            auto ptr = this->template ptr<T>();
            size_t nidx = 0;
            while (idx_begin != idx_end) {
                mgb_assert(nidx < m_layout.ndim);
                size_t idx = *idx_begin;
                mgb_assert(idx < m_layout.shape[nidx]);
                ptr += m_layout.stride[nidx] * idx;

                ++ idx_begin;
                ++ nidx;
            }
            return ptr;
        }

        template<typename T>
        T* ptr(std::initializer_list<size_t> idx) {
            return ptr<T>(idx.begin(), idx.end());
        }

        template<typename T>
        const T* ptr(std::initializer_list<size_t> dim) const {
            return const_cast<TensorND&>(*this).ptr<T>(dim);
        }

        //! get ptr of buffer start; *T* must match dtype
        template<typename T>
        T* ptr() const {
            m_layout.dtype.assert_is_ctype<T>();
            return m_storage.ptr()->template as<T>();
        }

        dt_byte* raw_ptr() const {
            return m_storage.ptr();
        }

        /*!
         * \brief change the shape without retaining old data, and initialize as
         *      contiguous stride
         *
         * dtype and format would not be changed
         */
        ChainReturnType& resize(const TensorShape& shape);

        /*!
         * \brief totally reset the tensor to given storage and layout
         */
        ChainReturnType& reset(
                TensorStorage storage, const TensorLayout &layout);

        /* ================= getter and setters =================  */

        /*!
         * \brief change comp node; see TensorStorage::comp_node()
         */
        ChainReturnType& comp_node(
                CompNode comp_node, bool allow_mem_node_change = false);

        CompNode comp_node() const {
            return m_storage.comp_node();
        }

        const TensorStorage& storage() const {
            return m_storage;
        }

        /*!
         * \brief change the storage and invalidate all data, resulting in an
         *      empty tensor
         */
        ChainReturnType& storage(const TensorStorage &storage);

        //! get data type
        DType dtype() const {
            return m_layout.dtype;
        }

        //! get tensor format
        TensorFormat format() const {
            return m_layout.format;
        }

        /*!
         * \brief change underlying dtype
         *
         * layout would be cleared (reset to ndim=0) if dtype actually changes
         */
        ChainReturnType& dtype(DType dtype);

        /*!
         * \brief change underlying tensor format
         *
         * layout would be cleared (reset to ndim=0) if format actually changes
         */
        ChainReturnType& format(TensorFormat format);

        /*!
         * \brief copy from another tensor and initialize contiguous layout
         *
         * Note:
         * 1. If the computing node is empty, it would be copied from src
         * 2. To copy from device to host, if the two tensors reside on
         *    different computing nodes, the caller is responsible to perform
         *    sync before copying; a better way is to set empty computing node
         *    to host tensor.
         * 3. For cross-device copy: copy would be synced on comp node of this,
         *    and the caller is responsible to sync this comp node with src comp
         *    node.
         * 4. If dtype is valid, it would be checked to match the dtype of src.
         * 5. Format would be reset to default and layout would be initialized
         *    to be contiguous.
         */
        template<class RStorage>
        ChainReturnType& copy_from(const TensorND<RStorage> &src);

        /*!
         * \brief copy from another tensor of the same shape, retaining current
         *      layout
         *
         * If storage type of src and this are different and src is not
         * contiguous, a temporary storage would be allocated to first make src
         * contiguous.
         */
        template <class RStorage>
        const ChainReturnType& copy_from_fixlayout(
                const TensorND<RStorage>& src) const;

        //! non-const version of copy_from_fixlayout
        template <class RStorage>
        ChainReturnType& copy_from_fixlayout(const TensorND<RStorage>& src) {
            return const_cast<ChainReturnType&>(
                    static_cast<const ChainReturnType*>(this)
                            ->copy_from_fixlayout(src));
        }

        //! convert to megdnn::TensorND
        megdnn::TensorND as_megdnn() const {
            return {const_cast<void*>(static_cast<const void*>(raw_ptr())),
                m_layout};
        }

        /* ================= misc =================  */

        /*!
         * \brief block host thread to synchronize with the CompNode
         */
        const ChainReturnType& sync() const {
            comp_node().sync();
            return static_cast<const ChainReturnType&>(*this);
        }

        ChainReturnType& sync() {
            return const_cast<ChainReturnType&>(
                    static_cast<const ChainReturnType*>(this)->sync());
        }

        //! similar to TensorStorage<>::make_proxy
        template<class RStorage,
            typename = typename std::enable_if<
                !std::is_same<TensorStorage, RStorage>::value>::type>
        static ChainReturnType make_proxy(const TensorND<RStorage> &src) {
            ChainReturnType ret;
            ret.reset(TensorStorage::make_proxy(src.storage()), src.layout());
            return ret;
        }

        //! similar to HostTensorStorage::proxy_to_default_cpu
        template<bool x = true, typename = std::enable_if_t<x && std::is_same<TensorStorage, HostTensorStorage>::value>>
        DeviceTensorND proxy_to_default_cpu() const {
            DeviceTensorND ret;
            ret.reset(storage().proxy_to_default_cpu(), layout());
            return ret;
        }
};

/*!
 * \brief call memset in the data of a device tensor
 */
void dev_tensor_memset(const DeviceTensorND& tensor, int val);

/*!
 * \brief fill zeros in the content of a dev tensor
 */
static inline void fill_zero_dev_tensor(const DeviceTensorND& tensor) {
    dev_tensor_memset(tensor, 0);
}

} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
