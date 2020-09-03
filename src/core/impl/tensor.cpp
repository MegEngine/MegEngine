/**
 * \file src/core/impl/tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */


#include "megbrain/tensor.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/opr/param_defs.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "megdnn/oprs.h"

#include <thread>

#include <cstring>
#include <cmath>

using namespace mgb;

namespace {

    //! implement non-contiguous d2d copy
    void noncont_tensor_copy(
            const DeviceTensorND &dest, const DeviceTensorND &src,
            bool contig_dest, bool contig_src) {
        auto src_cn = src.comp_node();
        auto dst_cn = dest.comp_node();
        if (src_cn.device_type() == dst_cn.device_type()) {
            // perform relayout op for better performance when src and dst are
            // placed on comp nodes with the same device type
            auto &&src_env = CompNodeEnv::from_comp_node(src.comp_node());
            auto relayout = opr::intl::get_megdnn_global_opr<megdnn::Relayout>(
                    dst_cn);
            dst_cn.activate();
            relayout->exec(
                    const_cast<DeviceTensorND&>(src).as_megdnn(),
                    dest.as_megdnn(), MegDNNHandle::get(src_env).handle());
        } else {
            if (contig_src) {
                mgb_assert(!contig_dest);
                DeviceTensorND tmp{dst_cn};
                tmp.copy_from(src);
                dest.copy_from_fixlayout(tmp);
                return;
            }
            DeviceTensorND tmp;
            tmp.copy_from(src);
            dest.copy_from_fixlayout(tmp);
        }
    }

    //! implement non-contiguous h2h copy
    void noncont_tensor_copy(
            const HostTensorND &dest, const HostTensorND &src, bool, bool) {
        auto opr = opr::intl::get_megdnn_global_opr<megdnn::Relayout>(
                CompNode::default_cpu());

        opr->exec(
                const_cast<HostTensorND&>(src).as_megdnn(),
                dest.as_megdnn());
    }

    //! implement non-contiguous d2h copy
    void noncont_tensor_copy(
            const HostTensorND &dest, const DeviceTensorND &src,
            bool contig_dest, bool contig_src) {
        if (contig_src) {
            mgb_assert(!contig_dest);
            HostTensorND tmp;
            tmp.copy_from(src).sync();
            dest.copy_from_fixlayout(tmp);  // sync not needed for h2h copy
            return;
        }
        DeviceTensorND tmp;
        tmp.copy_from(src);
        dest.copy_from_fixlayout(tmp);
    }

    //! implement non-contiguous h2d copy
    void noncont_tensor_copy(
            const DeviceTensorND &dest, const HostTensorND &src,
            bool contig_dest, bool contig_src) {
        if (contig_src) {
            mgb_assert(!contig_dest);
            DeviceTensorND tmp;
            // no need to sync because device free is async-safe with respect to
            // host thread
            tmp.copy_from(src);
            dest.copy_from_fixlayout(tmp);
            return;
        }
        HostTensorND tmp;
        tmp.copy_from(src);
        dest.copy_from_fixlayout(tmp).sync();
    }
} // anonymous namespace

/* ============= Slice and SubTensorSpec ============= */

SubTensorSpec SubTensorSpec::make_from_offset_elem(
        const TensorLayout &layout, ptrdiff_t offset_elem) {
    mgb_assert(layout.ndim && layout.dtype.valid());
    return {layout, offset_elem};
}

SubTensorSpec Slice::apply(TensorLayout layout, int axis) const {
    mgb_assert(layout.ndim > 0 && layout.dtype.valid());
    if (axis == megdnn::param::OptionalAxisV1::INVALID_AXIS) {
        axis = 0;
        layout = layout.collapse_contiguous();
        mgb_assert(layout.ndim == 1,
                   "apply Slice with axis==INVALID_AXIS on non-contig layout");
    }
    // axis in [-ndim, ndim) is available
    if (axis < 0)
        axis += layout.ndim;
    mgb_assert(axis >= 0 && static_cast<size_t>(axis) < layout.ndim,
            "invalid axis: %d; ndim=%zu", axis, layout.ndim);

    ptrdiff_t size_ax = layout.shape[axis];
    ptrdiff_t begin, end, step = m_step.val_with_default(1);
    mgb_assert(step, "Slice step can not be zero");

    auto tostr = [](const Maybe<ptrdiff_t> &v) -> std::string {
        if (!v.valid())
            return "None";
        return std::to_string(v.val());
    };
    auto mod_size = [size_ax](ptrdiff_t v) {
        return v < 0 ? v + size_ax : v;
    };
    MGB_MARK_USED_VAR(tostr);

#define CHECK(cond) \
    mgb_assert(cond, \
            "index out of bound: layout=%s; request begin=%s end=%s step=%s " \
            "axis=%d", \
            layout.to_string().c_str(), tostr(m_begin).c_str(), \
            tostr(m_end).c_str(), tostr(m_step).c_str(), axis)

    if (step > 0) {
        begin = mod_size(m_begin.val_with_default(0));
        end = mod_size(m_end.val_with_default(size_ax));
        CHECK(begin >= 0 && end >= begin && end <= size_ax);
    } else {
        begin = mod_size(m_begin.val_with_default(size_ax - 1));
        end = m_end.valid() ? mod_size(m_end.val()) : -1;
        CHECK(step < 0 && begin >= 0 && end <= begin && begin < size_ax &&
              end >= -1);
    }
    auto step_abs = std::abs(step);
    layout.shape[axis] = (std::abs(end - begin) + step_abs - 1) / step_abs;
    auto orig_stride = layout.stride[axis];
    layout.stride[axis] *= step;

    // make stride as contiguous as possible
    if (layout.shape[axis] != 1 && axis)
        -- axis;
    if (layout.shape[axis] == 1) {
        auto stride = layout.stride[axis] =
            axis + 1 < static_cast<int>(layout.ndim) ?
            layout.stride[axis + 1] * layout.shape[axis + 1] : 1;

        for (int i = axis - 1; i >= 0; -- i) {
            if (layout.shape[i] == 1) {
                layout.stride[i] = stride;
            } else {
                break;
            }
        }
    }

    auto offset_elem = layout.is_empty() ? 0 : orig_stride * begin;
    return SubTensorSpec::make_from_offset_elem(layout, offset_elem);

#undef CHECK
}

void SubTensorSpec::merge_with(const SubTensorSpec &rhs) {
    mgb_assert(m_layout.dtype.valid() && m_layout.dtype == rhs.m_layout.dtype &&
            rhs.m_layout.ndim);
    m_offset_elem += rhs.m_offset_elem;
    m_layout = rhs.m_layout;
}

/* ===================== TensorStorage ===================== */

class mgb::HostTensorStorageTrait {
    public:
        static void* alloc(CompNode node, size_t size) {
            return node.alloc_host(size);
        }

        static void free(CompNode node, void *data) {
            node.free_host(data);
        }
};

class mgb::DeviceTensorStorageTrait {
    public:
        static void* alloc(CompNode node, size_t size) {
            return node.alloc_device(size);
        }

        static void free(CompNode node, void *data) {
            node.free_device(data);
        }
};

template<class Trait>
TensorStorage<Trait>& TensorStorage<Trait>::operator = (
        const TensorStorage& rhs) {
    if (rhs.m_size > rhs.m_capacity) {
        rhs.ptr();
    }
    m_allow_realloc = rhs.m_allow_realloc;
    m_comp_node = rhs.m_comp_node;
    m_size = rhs.m_size;
    m_capacity = rhs.m_capacity;
    m_offset = rhs.m_offset;
    m_data = rhs.m_data;
    return *this;
}

template<class Trait>
TensorStorage<Trait>& TensorStorage<Trait>::ensure_size(size_t sz) {
    if (sz > m_size) {
        mgb_throw_if(!m_allow_realloc || m_offset, MegBrainError,
                "can not grow a tensor that does not allow realloc");
        check_comp_node_valid();
    }
    m_size = sz;
    return *this;
}

template<class Trait>
TensorStorage<Trait> TensorStorage<Trait>::sub(
        ptrdiff_t offset) const {
    ptr(); // apply lazy resize
    ptrdiff_t toff = offset + m_offset;
    if (offset == static_cast<ptrdiff_t>(m_size)) {
        return {false, m_comp_node, 0, 0, 0, RawStorage{}};
    }
    mgb_assert(toff >= 0 && offset < static_cast<ptrdiff_t>(m_size),
            "bad subtensor: offset=%td m_offset=%zu m_size=%zu",
            offset, m_offset, m_size);
    return {false, m_comp_node, m_size - offset, m_capacity - offset,
        static_cast<size_t>(toff), m_data};
}

template<class Trait>
dt_byte* TensorStorage<Trait>::apply_lazy_and_get_ptr() {
    check_comp_node_valid();
    if (m_size > m_capacity) {
        mgb_assert(m_allow_realloc && !m_offset);
        m_data.reset(); // free old ptr
        m_capacity = 0; // to be exception safe
        auto ptr = static_cast<dt_byte*>(Trait::alloc(m_comp_node, m_size));
        mgb_throw_if(!ptr, SystemError, "failed to allocate memory");
        CompNode cn = m_comp_node;
        m_data.reset(ptr, [cn](void *p){Trait::free(cn, p);});
        m_capacity = m_size;
    }
    return m_data.get() + m_offset;
}

template<class Trait>
TensorStorage<Trait>& TensorStorage<Trait>::comp_node(
        CompNode node, bool allow_mem_node_change) {
    mgb_assert(node.valid());
    if (m_comp_node.valid() && node.mem_node() != m_comp_node.mem_node()) {
        mgb_assert(allow_mem_node_change);
        m_allow_realloc = true;
        m_size = m_capacity = m_offset = 0;
        m_data.reset();
    }
    m_comp_node = node;
    return *this;
}

template<class Trait>
void TensorStorage<Trait>::reset(CompNode node, size_t size,
        RawStorage data) {
    mgb_assert(m_allow_realloc);
    m_comp_node = node;
    m_size = size;
    m_capacity = size;
    m_offset = 0;
    m_data = std::move(data);
}

template<class Trait>
template<class RTrait, typename>
TensorStorage<Trait> TensorStorage<Trait>::make_proxy(
        const TensorStorage<RTrait> &src) {
    mgb_assert(src.comp_node().mem_node() == CompNode::default_cpu().mem_node(),
            "proxy source should be on CPU; got %s",
            src.comp_node().to_string().c_str());
    src.ptr();
    return {true, src.m_comp_node, src.m_size, src.m_capacity, src.m_offset,
        src.m_data};
}

template<class Trait>
void TensorStorage<Trait>::on_invalid_comp_node() {
    mgb_throw(MegBrainError, "trying to acccess TensorStorage with invalid "
            "comp node");
}

namespace mgb {

// host to host
template<> template<>
void TensorStorage<HostTensorStorageTrait>::copy_from(
        const TensorStorage<HostTensorStorageTrait> &src, size_t size) const {
    mgb_assert(size <= this->size() && size <= src.size());
    memcpy(ptr(), src.ptr(), size);
}

// device to host
template<> template<>
void TensorStorage<HostTensorStorageTrait>::copy_from(
        const TensorStorage<DeviceTensorStorageTrait> &src, size_t size) const {
    bool need_sync = false;
    mgb_assert(size <= this->size() && size <= src.size());
    if (m_comp_node != src.comp_node()) {
        auto default_cpu = CompNode::default_cpu();
        if (src.comp_node() != default_cpu) {
            mgb_assert(m_comp_node == default_cpu,
                    "inconsistent D2H copy:"
                    " copy from device to host using different comp nodes:"
                    " device_node=%s host_node=%s",
                    src.comp_node().to_string().c_str(),
                    m_comp_node.to_string().c_str());
            // copy_from() should use m_comp_node, and default_cpu is
            // synchronous with current thread, so this copy has no
            // synchronizing ambiguity and we only need to sync on host
            need_sync = true;
        }
    }
    src.comp_node().copy_to_host(ptr(), src.ptr(), size);
    if (need_sync)
        src.comp_node().sync();
}

// host to device
template<> template<>
void TensorStorage<DeviceTensorStorageTrait>::copy_from(
        const TensorStorage<HostTensorStorageTrait> &src, size_t size) const {
    mgb_assert(size <= this->size() && size <= src.size());
    m_comp_node.copy_to_device(ptr(), src.ptr(), size);
}

// device to device
template<> template<>
void TensorStorage<DeviceTensorStorageTrait>::copy_from(
        const TensorStorage<DeviceTensorStorageTrait> &src, size_t size) const {
    mgb_assert(size <= this->size() && size <= src.size());
    if (src.comp_node().device_type() == CompNode::DeviceType::CPU &&
        comp_node().device_type() == CompNode::DeviceType::CUDA) {
        // current thread(i.e. cuda dispatcher thread) should wait for all
        // operations on src's comp_node to finish, otherwise a race condition
        // might occur between the worker thread of src's comp_node and the
        // thread responsible for copying pageable memory in \p src to a pinned
        // buffer, refer to
        // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
        //
        // Note: it is highly recommended that copy tensor from cpu to cuda
        // with asynchronized disaptching(see graph option async_exec_level),
        // or main thread might be blocked by worker thread corresponding to
        // the src's comp_node, resulting in bad performance
        //
        // TODO: consider using cudaMallocHost or cudaHostRegister
        // to pin the memory of src tensor, so it does not require synchronization
        // and is more efficient
        src.comp_node().sync();
        comp_node().copy_to_device(ptr(), src.ptr(), size);
    } else {
        src.comp_node().peer_copy_to(m_comp_node, ptr(), src.ptr(), size);
    }
}


// proxy host to device
template TensorStorage<DeviceTensorStorageTrait>
TensorStorage<DeviceTensorStorageTrait>::
make_proxy<HostTensorStorageTrait, void>(
        const TensorStorage<HostTensorStorageTrait>&);

// proxy device to host
template TensorStorage<HostTensorStorageTrait>
TensorStorage<HostTensorStorageTrait>::
make_proxy<DeviceTensorStorageTrait, void>(
        const TensorStorage<DeviceTensorStorageTrait>&);

}

/* ===================== TensorND ===================== */

// ctor def {

#define DEF                        \
    template <class TensorStorage> \
    TensorND<TensorStorage>::TensorND
DEF() = default;

DEF(CompNode node) : m_storage{node} {}

DEF(DType dtype) : m_layout{dtype} {}

DEF(CompNode node, DType dtype) : m_storage{node}, m_layout{dtype} {}

//! allocate contiguous from given comp node, shape and dtype
DEF(CompNode node, const TensorShape& shape, DType dtype, TensorFormat format)
        : m_storage{node}, m_layout{dtype, format} {
    resize(shape);
}

//! allocate contiguous from given comp node and layout (strides not
//! used)
DEF(CompNode node, const TensorLayout& layout)
        : TensorND(node, layout, layout.dtype, layout.format) {
    mgb_assert(layout.is_contiguous(),
               "non-contiguous layout used for initializing a tensor: %s",
               layout.to_string().c_str());
}

#undef DEF
// ctor def }

// def {
#define DEF(name, ret) \
template<class TensorStorage> \
typename TensorND<TensorStorage>::ChainReturnType ret \
TensorND<TensorStorage>::name

DEF(resize, &)(const TensorShape& shape) {
    mgb_assert(m_layout.dtype.valid());
    auto nr_elems = m_layout.init_contiguous_stride(shape);
    m_storage.ensure_size(m_layout.dtype.size(nr_elems));
    return static_cast<ChainReturnType&>(*this);
}

DEF(reset, &)(TensorStorage storage, const TensorLayout &layout) {
    //! The storage to be reset is either satisfy the layout or empty.
    //! Empty storage is used after weight preprocess for saving memory and
    //! checking layout when running
    mgb_assert(!layout.ndim || storage.valid_span(layout.span()) ||
               storage.empty());
    m_storage = std::move(storage);
    m_layout = layout;
    return static_cast<ChainReturnType&>(*this);
}

DEF(comp_node, &)(CompNode comp_node, bool allow_mem_node_change) {
    auto orig_cn = m_storage.comp_node_allow_invalid();
    m_storage.comp_node(comp_node, allow_mem_node_change);
    if (orig_cn.valid() && orig_cn.mem_node() != comp_node.mem_node()) {
        m_layout.ndim = 0;
    }
    return static_cast<ChainReturnType&>(*this);
}

DEF(storage, &)(const TensorStorage &storage) {
    if (m_storage.empty() || storage.empty() ||
            m_storage.ptr() != storage.ptr()) {
        m_storage = storage;
        m_layout.ndim = 0;
    }
    return static_cast<ChainReturnType&>(*this);
}

DEF(dtype, &)(DType dtype) {
    if (m_layout.dtype != dtype) {
        m_layout.dtype = dtype;
        m_layout.ndim = 0;
    }
    return static_cast<ChainReturnType&>(*this);
}

DEF(format, &)(TensorFormat format) {
    if (m_layout.format != format) {
        m_layout.format = format;
        m_layout.ndim = 0;
    }
    return static_cast<ChainReturnType&>(*this);
}

DEF(operator[], ) (std::initializer_list<Slice> slice) const {
    auto subspec = SubTensorSpec::make_from_offset_elem(m_layout, 0);
    size_t axis = 0;
    for (auto &&i: slice) {
        subspec.merge_with(i.apply(subspec.layout(), axis));
        axis ++;
    }
    return sub(subspec);
}

DEF(sub, )(const SubTensorSpec &spec) const {
    mgb_assert(
            spec.layout().dtype == dtype() && spec.layout().format == format(),
            "invalid subtensor spec: sub_layout=%s self=%s",
            spec.layout().to_string().c_str(), m_layout.to_string().c_str());
    ChainReturnType rst;
    rst.reset(m_storage.sub(spec.offset_byte()), spec.layout());
    return rst;
}

#undef DEF

// def }

/* ===================== TensorND::copy_from ===================== */
namespace {
/**
 * \brief determine whether to check overlap of two tensors.
 * \return true : when HostStorage || (DeviceStorage && SUPPORT_UNIFIED_ADDRESS)
 * \note when both support unified address, we can treat them both on CPU. So,
 * overlap check should be done
 */
template <typename TensorStorage, typename RStorage>
inline bool should_check_overlap(const TensorND<TensorStorage>& dst,
                                 const TensorND<RStorage>& src) {
    return true;
}

template <>
inline bool should_check_overlap<HostTensorStorage, DeviceTensorStorage>(
        const HostTensorND& dst, const DeviceTensorND& src) {
    return src.comp_node().contain_flag(
            CompNode::Flag::SUPPORT_UNIFIED_ADDRESS);
}

template <>
inline bool should_check_overlap<DeviceTensorStorage, HostTensorStorage>(
        const DeviceTensorND& dst, const HostTensorND& src) {
    return dst.comp_node().contain_flag(
            CompNode::Flag::SUPPORT_UNIFIED_ADDRESS);
}

/**
 * \brief D2D tensor copy should check overlap when
 * 1. They are on the same mem node. But note that the address must be logical
 * comparable. i.e. the original address alloc on enflame is uncomparable.
 * 2. They both support unified address, so can be treated as CPU address.
 */
template <>
inline bool should_check_overlap<DeviceTensorStorage, DeviceTensorStorage>(
        const DeviceTensorND& dst, const DeviceTensorND& src) {
    bool is_same_memnode =
            dst.comp_node().mem_node() == src.comp_node().mem_node();
    bool unified_address = src.comp_node().contain_flag(
                                   CompNode::Flag::SUPPORT_UNIFIED_ADDRESS) &&
                           dst.comp_node().contain_flag(
                                   CompNode::Flag::SUPPORT_UNIFIED_ADDRESS);
    return is_same_memnode || unified_address;
}

/**
 * \brief check overlap of two tensors. throw exception when overlapped
 */
inline void check_overlapped(const dt_byte* dst_min, const dt_byte* dst_max,
                             const dt_byte* src_min, const dt_byte* src_max) {
    mgb_throw_if(src_min < dst_max && dst_min < src_max, TensorCopyOverlapError,
                 "cound not perform copy between overlapped tensors");
}
}  // namespace

template<class TensorStorage>
template<class RStorage>
typename TensorND<TensorStorage>::ChainReturnType&
TensorND<TensorStorage>::copy_from(const TensorND<RStorage> &src) {
    if (!m_storage.comp_node_valid())
        m_storage.comp_node(src.comp_node());

    if (m_layout.dtype.valid())
        m_layout.dtype.assert_is(src.dtype());
    else
        m_layout.dtype = src.dtype();
    m_layout.format = {};

    size_t size_bytes = dtype().size(
            m_layout.init_contiguous_stride(src.shape()));
    m_storage.ensure_size(size_bytes);
    if (!size_bytes) {
        return static_cast<ChainReturnType&>(*this);
    }
    if (src.layout().is_physical_contiguous()) {
        if (should_check_overlap(*this, src)) {
            check_overlapped(m_storage.ptr(),
                             m_storage.ptr() + size_bytes,
                             src.storage().ptr(),
                             src.storage().ptr() + size_bytes);
        }
        m_storage.copy_from(src.storage(), size_bytes);
        return static_cast<ChainReturnType&>(*this);
    }
    return const_cast<ChainReturnType&>(copy_from_fixlayout(src));
}

template <class TensorStorage>
template <class RStorage>
const typename TensorND<TensorStorage>::ChainReturnType&
TensorND<TensorStorage>::copy_from_fixlayout(
        const TensorND<RStorage>& src) const {

    dtype().assert_is(src.dtype());
    mgb_assert(m_layout.eq_shape(src.layout()),
            "shape differs in copy_from_fixlayout: %s vs %s",
            static_cast<const TensorShape&>(m_layout).to_string().c_str(),
            static_cast<const TensorShape&>(src.layout()).to_string().c_str());

    if (src.empty()) {
        return static_cast<const ChainReturnType&>(*this);
    }

    mgb_assert(m_layout.is_non_overlapping_strong(),
            "copy dest must have non-overlapping layout");

    TensorLayout::Span
        src_span = src.layout().span(),
        dst_span = layout().span();

    if (should_check_overlap(*this, src)) {
        check_overlapped(this->raw_ptr() + dst_span.low_byte,
                         this->raw_ptr() + dst_span.high_byte,
                         src.raw_ptr() + src_span.low_byte,
                         src.raw_ptr() + src_span.high_byte);
    }

    bool self_contig = m_layout.is_physical_contiguous(),
         src_contig = src.layout().is_physical_contiguous();
    if (self_contig && src_contig) {
        if (m_layout.format.is_default() && src.layout().format.is_default()) {
            mgb_assert(src_span.low_byte == 0 && dst_span.low_byte == 0 &&
                       src_span.high_byte == dst_span.high_byte);
            m_storage.copy_from(src.storage(), src_span.high_byte);
        } else {
            mgb_assert(src_span.low_byte == 0 && dst_span.low_byte == 0);
            m_storage.copy_from(src.storage(), std::min(src_span.high_byte,
                                                        dst_span.high_byte));
        }
        return static_cast<const ChainReturnType&>(*this);
    }
    noncont_tensor_copy(*this, src, self_contig, src_contig);
    return static_cast<const ChainReturnType&>(*this);
}

/* =================== misc =================== */

void mgb::dev_tensor_memset(const DeviceTensorND& tensor, int val) {
    auto&& env = CompNodeEnv::from_comp_node(tensor.comp_node());
    env.activate();
    void* ptr = tensor.raw_ptr();
    size_t size = tensor.layout().span().dist_byte();
    switch (env.property().type) {
#if MGB_CUDA
        case CompNode::DeviceType::CUDA:
            MGB_CUDA_CHECK(
                    cudaMemsetAsync(ptr, val, size, env.cuda_env().stream));
            break;
#endif
#if MGB_ATLAS
       case CompNode::DeviceType::ATLAS:
#if MGB_USE_ATLAS_ASYNC_API
           MGB_ATLAS_CHECK(aclrtMemsetAsync(ptr, -1, val, size,
                                            env.atlas_env().stream));
#else
           MGB_ATLAS_CHECK(aclrtMemset(ptr, -1, val, size));
#endif
           break;
#endif
#if MGB_CAMBRICON
       case CompNode::DeviceType::CAMBRICON:
           MGB_CNRT_CHECK(cnrtSyncQueue(env.cnrt_env().queue));
           MGB_CNRT_CHECK(cnrtMemset(ptr, val, size));
           break;
#endif
       case CompNode::DeviceType::CPU: {
            auto fill = [ptr, size, val]() { std::memset(ptr, val, size); };
            env.cpu_env().dispatch(fill);
        } break;
        default:
            mgb_throw(MegBrainError,
                      "unhandled comp node in dev_tensor_memset: %s",
                      tensor.comp_node().to_string().c_str());
    }
}

namespace mgb {
    template class TensorStorage<HostTensorStorageTrait>;
    template class TensorStorage<DeviceTensorStorageTrait>;
    template class TensorND<TensorStorage<HostTensorStorageTrait>>;
    template class TensorND<TensorStorage<DeviceTensorStorageTrait>>;

    /* ===== copy_from related ===== */

#define HT_RAW TensorND<HostTensorStorage>
#define DT_RAW TensorND<DeviceTensorStorage>
#define HT(f) f<HostTensorStorage>(const HT_RAW&)
#define DT(f) f<DeviceTensorStorage> (const DT_RAW&)


#define INST(f, c) \
    template c HostTensorND& HT_RAW::HT(f) c; \
    template c HostTensorND& HT_RAW::DT(f) c; \
    template c DeviceTensorND& DT_RAW::HT(f) c; \
    template c DeviceTensorND& DT_RAW::DT(f) c

    INST(copy_from, );
    INST(copy_from_fixlayout, const);

#undef INST
#undef DT
#undef HT
#undef DT_RAW
#undef HT_RAW

}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
