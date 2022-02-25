#include "lite_build_config.h"

#if LITE_BUILD_WITH_MGE
#include "common.h"
#include "tensor_impl.h"

#include "lite/tensor.h"

#include "megbrain/comp_node.h"
#include "megbrain/tensor.h"

#include <memory>

using namespace lite;

/**********************TensorImpl****************************/

LITE_DYN_TYPE_OBJ_FINAL_IMPL(TensorImplDft);

TensorImplDft::TensorImplDft() {
    m_host_tensor = std::make_shared<mgb::HostTensorND>(mgb::CompNode::default_cpu());
}

TensorImplDft::TensorImplDft(LiteDeviceType device, bool is_pinned_host) {
    auto cn = mgb::CompNode::load(to_compnode_locator(device));
    if (device == LiteDeviceType::LITE_DEVICE_DEFAULT) {
        device = LiteDeviceType::LITE_CPU;
    }
    if (device == LiteDeviceType::LITE_CPU) {
        m_host_tensor =
                std::make_shared<mgb::HostTensorND>(mgb::CompNode::default_cpu());
    } else if (is_pinned_host) {
        m_host_tensor = std::make_shared<mgb::HostTensorND>(cn);
    } else {
        m_dev_tensor = std::make_shared<mgb::DeviceTensorND>(cn);
    }
}

TensorImplDft::TensorImplDft(
        LiteDeviceType device, const Layout& layout, bool is_pinned_host) {
    auto cn = mgb::CompNode::load(to_compnode_locator(device));
    auto mge_layout = to_impl_layout(layout);
    if (device == LiteDeviceType::LITE_DEVICE_DEFAULT) {
        device = LiteDeviceType::LITE_CPU;
    }
    if (device == LiteDeviceType::LITE_CPU) {
        m_host_tensor = std::make_shared<mgb::HostTensorND>(
                mgb::CompNode::default_cpu(), mge_layout);
    } else if (is_pinned_host) {
        m_host_tensor = std::make_shared<mgb::HostTensorND>(cn, mge_layout);
    } else {
        m_dev_tensor = std::make_shared<mgb::DeviceTensorND>(cn, mge_layout);
    }
}

TensorImplDft::TensorImplDft(
        int device_id, LiteDeviceType device_type, const Layout& layout,
        bool is_pinned_host) {
    auto locator = to_compnode_locator(device_type);
    locator.device = device_id;
    auto cn = mgb::CompNode::load(locator);
    if (device_type == LiteDeviceType::LITE_DEVICE_DEFAULT) {
        device_type = LiteDeviceType::LITE_CPU;
    }
    if (layout.ndim) {
        auto mge_layout = to_impl_layout(layout);
        if (device_type == LiteDeviceType::LITE_CPU) {
            m_host_tensor = std::make_shared<mgb::HostTensorND>(
                    mgb::CompNode::default_cpu(), mge_layout);
        } else if (is_pinned_host) {
            m_host_tensor = std::make_shared<mgb::HostTensorND>(cn, mge_layout);
        } else {
            m_dev_tensor = std::make_shared<mgb::DeviceTensorND>(cn, mge_layout);
        }
    } else {
        if (device_type == LiteDeviceType::LITE_CPU) {
            m_host_tensor =
                    std::make_shared<mgb::HostTensorND>(mgb::CompNode::default_cpu());
        } else if (is_pinned_host) {
            m_host_tensor = std::make_shared<mgb::HostTensorND>(cn);
        } else {
            m_dev_tensor = std::make_shared<mgb::DeviceTensorND>(cn);
        }
    }
}

TensorImplDft::TensorImplDft(
        int device_id, int stream_id, LiteDeviceType device_type, bool is_pinned_host) {
    auto locator = to_compnode_locator(device_type);
    locator.device = device_id;
    locator.stream = stream_id;
    auto cn = mgb::CompNode::load(locator);
    if (get_device_from_locator(locator) == LiteDeviceType::LITE_CPU) {
        m_host_tensor =
                std::make_shared<mgb::HostTensorND>(mgb::CompNode::default_cpu());
    } else if (is_pinned_host) {
        m_host_tensor = std::make_shared<mgb::HostTensorND>(cn);
    } else {
        m_dev_tensor = std::make_shared<mgb::DeviceTensorND>(cn);
    }
}

LiteDeviceType TensorImplDft::get_device_type() const {
    if (is_host()) {
        return LiteDeviceType::LITE_CPU;
    } else {
        return get_device_from_locator(m_dev_tensor->comp_node().locator());
    }
}

int TensorImplDft::get_device_id() const {
    if (is_host()) {
        return m_host_tensor->comp_node().locator().device;
    } else {
        return m_dev_tensor->comp_node().locator().device;
    }
}

bool TensorImplDft::is_pinned_host() const {
    return is_host() && get_device_from_locator(m_host_tensor->comp_node().locator()) !=
                                LiteDeviceType::LITE_CPU;
}

void TensorImplDft::set_mge_tensor_compnode(const mgb::CompNode& comp_node) {
    if (is_host()) {
        m_host_tensor->comp_node(comp_node, true);
    } else {
        m_dev_tensor->comp_node(comp_node, true);
    }
}

Layout TensorImplDft::get_layout() const {
    if (is_host()) {
        return to_lite_layout(m_host_tensor->layout());
    } else {
        return to_lite_layout(m_dev_tensor->layout());
    }
}

void* TensorImplDft::get_memory_ptr() const {
    if (m_get_memory_callback) {
        m_get_memory_callback(const_cast<TensorImplDft*>(this));
    }
    if (is_host()) {
        return static_cast<void*>(m_host_tensor->raw_ptr());
    } else {
        return static_cast<void*>(m_dev_tensor->raw_ptr());
    }
}

void* TensorImplDft::get_memory_ptr(const std::vector<size_t>& idx) const {
    if (m_get_memory_callback) {
        m_get_memory_callback(const_cast<TensorImplDft*>(this));
    }
    if (is_host()) {
        auto elemsize_log = m_host_tensor->layout().dtype.size_log();
        switch (elemsize_log) {
            case 0:
                return static_cast<void*>(
                        m_host_tensor->ptr<uint8_t>(idx.begin(), idx.end()));
                break;
            case 1:
                return static_cast<void*>(
                        m_host_tensor->ptr<short>(idx.begin(), idx.end()));
                break;
            case 2:
                return static_cast<void*>(
                        m_host_tensor->ptr<float>(idx.begin(), idx.end()));
                break;
            default:
                LITE_THROW("not supported data_type.");
        }
    } else {
        auto elemsize_log = m_dev_tensor->layout().dtype.size_log();
        switch (elemsize_log) {
            case 0:
                return static_cast<void*>(
                        m_dev_tensor->ptr<uint8_t>(idx.begin(), idx.end()));
                break;
            case 1:
                return static_cast<void*>(
                        m_dev_tensor->ptr<short>(idx.begin(), idx.end()));
                break;
            case 2:
                return static_cast<void*>(
                        m_dev_tensor->ptr<float>(idx.begin(), idx.end()));
                break;
            default:
                LITE_THROW("not supported data_type.");
        }
    }
}

std::shared_ptr<Tensor> TensorImplDft::slice(
        const std::vector<size_t>& start, const std::vector<size_t>& end,
        const std::vector<size_t>& step) {
    Layout layout;
    mgb::TensorLayout layout_mge;
    if (is_host()) {
        layout_mge = m_host_tensor->layout();
        layout = to_lite_layout(m_host_tensor->layout());
    } else {
        layout_mge = m_dev_tensor->layout();
        layout = to_lite_layout(m_dev_tensor->layout());
    }

    size_t length = start.size();
    LITE_ASSERT(
            length == end.size() && length <= layout.ndim,
            "The start and end must be the same size and less than layout "
            "ndim.");
    std::vector<mgb::Slice> slices;
    if (step.size()) {
        LITE_ASSERT(length == step.size(), "The start and step must be the same size.");
        for (size_t i = 0; i < length; i++) {
            slices.push_back(mgb::Slice{start[i], end[i], step[i]});
        }
    } else {
        for (size_t i = 0; i < length; i++) {
            slices.push_back(mgb::Slice{start[i], end[i]});
        }
    }
    auto subspec = mgb::SubTensorSpec::make_from_offset_elem(layout_mge, 0);
    size_t axis = 0;
    for (auto&& i : slices) {
        subspec.merge_with(i.apply(subspec.layout(), axis));
        axis++;
    }
    auto ret = std::make_shared<Tensor>();
    auto& impl = TensorHelper::implement(ret)->cast_final_safe<TensorImplDft>();
    if (is_host()) {
        *impl.m_host_tensor = m_host_tensor->sub(subspec);
    } else {
        impl.m_dev_tensor =
                std::make_shared<mgb::DeviceTensorND>(m_dev_tensor->sub(subspec));
        impl.m_host_tensor = nullptr;
    }
    LITE_ASSERT(is_host() == impl.is_host());
    return ret;
}

void TensorImplDft::fill_zero() {
    if (is_host()) {
        auto mge_layout = m_host_tensor->layout();
        if (m_host_tensor->layout().is_physical_contiguous()) {
            auto ptr = get_memory_ptr();
            std::memset(ptr, 0, mge_layout.dtype.size(mge_layout.total_nr_elems()));
        } else {
            TensorImplDft tmp(
                    LiteDeviceType::LITE_CPU, to_lite_layout(mge_layout), true);
            tmp.fill_zero();
            this->copy_from(&tmp);
        }
    } else {
        mgb::dev_tensor_memset(*m_dev_tensor, 0);
        m_dev_tensor->sync();
    }
}

void TensorImplDft::share_memory_with(const TensorImplBase* src_tensor_impl) {
    auto src_dft_tensor = static_cast<const TensorImplDft*>(src_tensor_impl);
    LITE_ASSERT(
            is_host() == src_dft_tensor->is_host(),
            "share memory must happen in same device");
    //! make shape the src memory is ready
    src_tensor_impl->get_memory_ptr();
    if (is_host()) {
        *m_host_tensor = *src_dft_tensor->m_host_tensor;
    } else {
        *m_dev_tensor = *src_dft_tensor->m_dev_tensor;
    }
}

void TensorImplDft::set_layout(const Layout& layout) {
    bool host = is_host();
    auto mgb_layout = to_impl_layout(layout);
    if (host) {
        m_host_tensor->dtype(mgb_layout.dtype);
        m_host_tensor->resize(mgb_layout);
    } else {
        m_dev_tensor->dtype(mgb_layout.dtype);
        m_dev_tensor->resize(mgb_layout);
    }
}

void TensorImplDft::reshape(const Layout& layout) {
    auto mgb_layout = to_impl_layout(layout);
    bool host = is_host();
    if (host) {
        m_host_tensor->resize(mgb_layout);
    } else {
        m_dev_tensor->resize(mgb_layout);
    }
}

void TensorImplDft::reset(void* prepared_data) {
    auto raw_ptr = static_cast<mgb::dt_byte*>(prepared_data);
    auto raw_storage = std::shared_ptr<mgb::dt_byte>(raw_ptr, [](void*) {});
    bool host = is_host();
    if (host) {
        auto cn = m_host_tensor->comp_node();
        auto mge_layout = m_host_tensor->layout();
        size_t size = mge_layout.span().dist_byte();
        mgb::HostTensorStorage storage;
        storage.reset(cn, size, raw_storage);
        if (m_record_reset) {
            m_host_tensor->only_reset_raw_storage(storage);
        } else {
            m_host_tensor->reset(storage, mge_layout);
        }
    } else {
        auto cn = m_dev_tensor->comp_node();
        auto mge_layout = m_dev_tensor->layout();
        size_t size = mge_layout.span().dist_byte();
        mgb::DeviceTensorStorage storage;
        storage.reset(cn, size, raw_storage);
        if (m_record_reset) {
            m_dev_tensor->only_reset_raw_storage(storage);
        } else {
            m_dev_tensor->reset(storage, mge_layout);
        }
    }
    if (m_reset_callback) {
        m_reset_callback(this);
    }
}

void TensorImplDft::reset(void* prepared_data, const Layout& layout) {
    set_layout(layout);
    reset(prepared_data);
}

bool TensorImplDft::is_continue_memory() const {
    if (is_host()) {
        return m_host_tensor->layout().is_physical_contiguous();
    } else {
        return m_dev_tensor->layout().is_physical_contiguous();
    }
}

void TensorImplDft::copy_from(const TensorImplBase* src_impl) {
    if (is_continue_memory()) {
        copy_from_continue(src_impl);
    } else {
        copy_from_fixlayout(src_impl);
    }
}

void TensorImplDft::copy_from_continue(const TensorImplBase* src_impl) {
    auto src = static_cast<const TensorImplDft*>(src_impl);
    if (is_host()) {
        //! host to host
        if (src->is_host()) {
            m_host_tensor->copy_from(*src->m_host_tensor);
            //! device to host
        } else {
            auto src_cn = src->m_dev_tensor->comp_node();
            auto dst_cn = m_host_tensor->comp_node();
            if (src_cn != dst_cn && m_host_tensor->layout().ndim > 0) {
                LITE_WARN(
                        "The dst tensor memroy is alloced before coping, "
                        "then pinned memroy would not use to optmize the "
                        "copy performance.");
                //! When D2H in megbrain and the compnode of src and dst is not
                //! equal, there must be one compnode that is cpu-default, so
                //! here, we use temp tensor for transition
                auto tmp_impl = std::make_shared<TensorImplDft>();
                tmp_impl->set_mge_tensor_compnode(src_cn);
                tmp_impl->m_host_tensor->copy_from(*src->m_dev_tensor).sync();
                m_host_tensor->copy_from(*tmp_impl->m_host_tensor);
            } else {
                //! if dst compnode is not valid(memory is not alloced), the
                //! tensor is pinned host tensor
                m_host_tensor->comp_node(src_cn, true);
                m_host_tensor->copy_from(*src->m_dev_tensor).sync();
            }
        }
    } else {
        //! host to device
        if (src->is_host()) {
            m_dev_tensor->copy_from(*src->m_host_tensor).sync();
            //! device to device
        } else {
            m_dev_tensor->copy_from(*src->m_dev_tensor).sync();
        }
    }
}

void TensorImplDft::copy_from_fixlayout(const TensorImplBase* src_impl) {
    auto src = static_cast<const TensorImplDft*>(src_impl);
    if (is_host()) {
        //! host to host
        if (src->is_host()) {
            m_host_tensor->copy_from_fixlayout(*src->m_host_tensor);
            //! device to host
        } else {
            auto src_cn = src->m_dev_tensor->comp_node();
            auto dst_cn = m_host_tensor->comp_node();
            if (src_cn != dst_cn && m_host_tensor->layout().ndim > 0) {
                LITE_WARN(
                        "The dst tensor memroy is alloced before coping, "
                        "then pinned memroy would not use to optmize the "
                        "copy performance.");
                //! When D2H in megbrain and the compnode of src and dst is not
                //! equal, there must be one compnode that is cpu-default, so
                //! here, we use temp tensor for transition
                auto tmp_impl = std::make_shared<TensorImplDft>();
                tmp_impl->set_mge_tensor_compnode(src_cn);
                tmp_impl->m_host_tensor->copy_from(*src->m_dev_tensor).sync();
                m_host_tensor->copy_from_fixlayout(*tmp_impl->m_host_tensor);
            } else {
                //! if dst compnode is not valid(memory is not alloced), the
                //! tensor is pinned host tensor
                m_host_tensor->comp_node(src_cn, true);
                m_host_tensor->copy_from_fixlayout(*src->m_dev_tensor).sync();
            }
        }
    } else {
        //! host to device
        if (src->is_host()) {
            m_dev_tensor->copy_from_fixlayout(*src->m_host_tensor).sync();
            //! device to device
        } else {
            m_dev_tensor->copy_from_fixlayout(*src->m_dev_tensor).sync();
        }
    }
}

void TensorImplDft::copy_from_mge_tensor(const mgb::DeviceTensorND& dv) {
    if (is_host()) {
        auto src_cn = dv.comp_node();
        m_host_tensor->comp_node(src_cn, true);
        m_host_tensor->copy_from(dv);
    } else {
        m_dev_tensor->copy_from(dv);
    }
}

void TensorImplDft::set_reset_callback(const std::function<void(TensorImplDft*)>& cb) {
    m_reset_callback = cb;
}

void TensorImplDft::set_get_memory_callback(
        const std::function<void(TensorImplDft*)>& cb) {
    m_get_memory_callback = cb;
}

void TensorImplDft::device_share_host_memory() {
    if (is_host()) {
        if (!m_dev_tensor) {
            m_dev_tensor = std::make_shared<mgb::DeviceTensorND>(
                    m_host_tensor->comp_node(), m_host_tensor->layout());
        }
        if (m_host_tensor->raw_ptr() != m_dev_tensor->raw_ptr()) {
            auto&& storage =
                    mgb::DeviceTensorStorage::make_proxy(m_host_tensor->storage());
            m_dev_tensor->only_reset_raw_storage(storage);
        }
    }
}

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
