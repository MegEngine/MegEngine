#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <variant>

#include "megbrain/graph.h"
#include "megbrain/imperative/resource_manager.h"
#include "megbrain/tensor.h"
#include "megbrain/utils/metahelper.h"

namespace mgb {
namespace imperative {

/************************** Tensor *****************************/
class Blob;
using BlobPtr = std::shared_ptr<Blob>;

class BlobManagerImpl;
class OwnedBlob;

class Blob : public NonCopyableObj, public std::enable_shared_from_this<Blob> {
protected:
    CompNode m_comp_node;
    size_t m_size;

    Blob(CompNode cn, size_t size) : m_comp_node(cn), m_size(size) {}

public:
    virtual ~Blob() = default;

    template <typename... Args>
    static std::shared_ptr<OwnedBlob> make(Args&&... args) {
        return std::make_shared<OwnedBlob>(std::forward<Args>(args)...);
    }

    const CompNode& comp_node() const { return m_comp_node; }
    size_t size() const { return m_size; }
    using RawStorage = DeviceTensorStorage::RawStorage;
    virtual const RawStorage& storage() = 0;
    virtual BlobPtr borrow_to(CompNode) = 0;
    virtual bool storage_is_unique() = 0;
    virtual void* raw_ptr_not_for_readwrite() = 0;
};

class OwnedBlob final : public Blob {
    friend class Blob;

public:
    OwnedBlob(const DeviceTensorStorage& s);
    OwnedBlob(CompNode cn, size_t sz);
    ~OwnedBlob() override;

    const RawStorage& storage() override;
    BlobPtr borrow_to(CompNode) override;
    bool storage_is_unique() override;
    void* raw_ptr_not_for_readwrite() override;

private:
    friend class BlobManagerImpl;
    RawStorage m_storage;
    size_t m_id;
};

class BorrowedBlob final : public Blob {
    std::mutex m_mtx;
    std::shared_ptr<OwnedBlob> m_owner;
    uint64_t m_event;
    bool m_initialized = false;

public:
    BorrowedBlob(CompNode, std::shared_ptr<OwnedBlob>);
    ~BorrowedBlob() override;

    const RawStorage& storage() override;
    BlobPtr borrow_to(CompNode) override;
    bool storage_is_unique() override;
    void* raw_ptr_not_for_readwrite() override;
};

struct EventDeleter {
    void operator()(CompNode::Event*);
};
using EventPtr = std::unique_ptr<CompNode::Event, EventDeleter>;

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
class Tensor : public NonCopyableObj {
public:
    Tensor() = default;
    Tensor(BlobPtr blob, const TensorLayout& layout, size_t offset = 0,
           const HostTensorND& hv = {});
    Tensor(BlobPtr blob, const TensorLayout& layout, const HostTensorND& hv = {})
            : Tensor(std::move(blob), layout, 0, hv){};
    Tensor(const HostTensorND& hv);
    Tensor(const DeviceTensorND& dv, const HostTensorND& hv = {});
    Tensor(const TensorLayout& layout, const CompNode& cn);
    Tensor(const BlobPtr blob, const size_t offset, const TensorLayout& layout);

    static TensorPtr make(const HostTensorND& hv);

    template <
            typename T,
            typename = std::enable_if_t<std::is_same_v<std::decay_t<T>, HostTensorND>>>
    static TensorPtr make(T&& hv) {
        TensorPtr (*f)(const HostTensorND&) = &make;
        return f(std::forward<T>(hv));
    };

    template <typename... Args>
    static TensorPtr make(Args&&... args) {
        return std::make_shared<Tensor>(std::forward<Args>(args)...);
    }

    CompNode comp_node() const {
        mgb_assert(m_blob, "uninitialized tensor.");
        return m_cn;
    }

    DType dtype() const { return m_dtype; }

    TensorLayout layout() const { return m_layout; }

    const TensorShape& shape() const { return m_shape; }

    size_t offset() const { return m_offset; }

    void to_contiguous_inplace(VarNode::LayoutConstraintCallback&);

    void to_contiguous_inplace();

    DeviceTensorND dev_tensor(bool contiguous = true);

    void assign_from_dev_tensor(DeviceTensorND);

    megdnn::TensorND dnn_tensor();

    static TensorPtr make_scalar(DTypeScalar value, CompNode cn);

    TensorPtr make_scalar(DTypeScalar value) const {
        mgb_assert(m_blob, "uninitialized tensor.");
        return make_scalar(value, m_blob->comp_node());
    }

    BlobPtr& blob() { return m_blob; }

    void* raw_ptr_not_for_readwrite() { return m_blob->raw_ptr_not_for_readwrite(); }

    void fetch_value();
    bool value_fetched();
    TensorPtr sub(size_t offset, TensorShape shape);

    // m_value is set once readonly afterwards
    // so the return value is thread safe
    const HostTensorND& get_value();
    // return a pointer instead of a reference to ensure thread safety
    const HostTensorND* try_get_value();

    void set_ready_event(uint64_t event) { m_produced_at = event; }
    uint64_t get_ready_event();

    bool storage_is_unique();

    // Make sure all static objects required to destruct a tensor has completed
    // construction. All static storage duration object that holds tensors must
    // call this method before their constructors completes.
    static void static_initialize();

private:
    size_t m_offset;
    const CompNode m_cn;
    const TensorShape m_shape;
    const DType m_dtype;

    std::mutex m_blob_mtx;
    BlobPtr m_blob;
    TensorLayout m_layout;

    std::mutex m_value_mtx;
    HostTensorND m_value;
    EventPtr m_value_ready = nullptr;
    uint64_t m_produced_at = 0;
};

/*!
 * \brief record a virtual event
 * \param doitnow also record a real event
 */
uint64_t record_event(CompNode cn, bool doitnow = false);

//! make a device wait on a virtual event
void device_wait_event(CompNode waiter, CompNode waitee, uint64_t event);

//! hold a blob until a virtual event on a device is completed
void async_release(CompNode cn, uint64_t event, BlobPtr blob);

//! hold a host tensor until a virtual event on a device is completed
void async_release(CompNode cn, uint64_t event, HostTensorStorage::RawStorage storage);

inline void async_release(CompNode cn, uint64_t event, Tensor& tensor) {
    async_release(cn, event, tensor.blob());
}

inline void async_release(CompNode cn, uint64_t event, const HostTensorND& hnd) {
    async_release(cn, event, hnd.storage().raw_storage());
}

inline void async_release(const HostTensorND& hnd) {
    auto cn = hnd.comp_node();
    async_release(cn, record_event(cn, true), hnd);
}

// Cache for small blobs
// 1. A blob has to be seen twice (within a window) to be eligible for cache
// 2. Cache eviction occurs when cache size reaches a threshold, in least frequently
// used order
class ConstTensorCache {
public:
    struct Entry {
        size_t hitcnt = 0;
        std::unique_ptr<dt_byte[]> data;
        size_t size;
        BlobPtr blob;

        Entry() = default;
        Entry(const dt_byte* ptr, size_t size_, BlobPtr blob_)
                : data(new dt_byte[size_]), size(size_), blob(blob_) {
            memcpy(data.get(), ptr, size);
        }

        // does not check input
        bool match(const HostTensorND& hv) {
            return 0 == memcmp(data.get(), hv.raw_ptr(), hv.layout().span().high_byte);
        }
    };

    using KV = std::pair<uint64_t, Entry>;

    bool check(const HostTensorND& hv) {
        auto&& layout = hv.layout();
        auto&& span = layout.span();
        return hv.format().is_default() && !hv.empty() && layout.is_contiguous() &&
               span.low_byte == 0 && span.high_byte <= max_bytes;
    }

    // hash storage; does not check input
    static uint64_t hash(const HostTensorND& hv) {
        auto&& span = hv.layout().span();
        return XXHash{}.update(hv.raw_ptr(), span.high_byte).digest();
    }

    BlobPtr lookup(const HostTensorND& hv) {
        if (!check(hv)) {
            return {};
        }
        auto h = hash(hv);
        MGB_LOCK_GUARD(mtx);
        // lookup in g1
        auto it = g1.find(h);
        if (it != g1.end()) {
            if (!it->second.match(hv)) {
                mgb_log_warn("hash collision in const tensor cache");
                return {};
            }
            it->second.hitcnt += 1;
            return it->second.blob;
        }
        // lookup in g0
        if (!g0.extract(h) && !g0b.extract(h)) {
            maybe_collect_g0();
            g0.emplace(h);
            return {};
        }
        // add new entry to g1
        maybe_collect_g1();
        Entry entry(hv.raw_ptr(), hv.layout().span().high_byte, Tensor(hv).blob());
        it = g1.emplace_hint(it, h, std::move(entry));
        it->second.hitcnt += 1;
        return it->second.blob;
    }

    void clear() {
        MGB_LOCK_GUARD(mtx);
        g0.clear();
        g0b.clear();
        g1.clear();
    }

    std::mutex mtx;
    const size_t hwm = 1024, lwm = 512, max_bytes = TensorShape::MAX_NDIM * 8,
                 window = 65536;

private:
    void maybe_collect_g0() {
        if (g0.size() > window) {
            std::swap(g0, g0b);
            g0.clear();
        }
    }
    void maybe_collect_g1() {
        if (g1.size() < hwm)
            return;

        tmp.clear();
        for (auto&& kv : g1) {
            tmp.emplace_back(kv.first, std::move(kv.second));
        }
        std::nth_element(
                tmp.begin(), tmp.begin() + lwm, tmp.end(),
                [](const KV& lhs, const KV& rhs) {
                    return lhs.second.hitcnt > rhs.second.hitcnt;
                });
        tmp.resize(lwm);
        g1.clear();
        for (auto&& kv : tmp) {
            kv.second.hitcnt = 0;
            g1.emplace(std::move(kv));
        }
    }

    // g0: records blobs which have been seen at least once (within a window)
    // g0b: backup of g0
    // g1: records the most frequently used blobs which have been seen at least
    // twice. When `g1.size() == hwm`, it will be refreshed and only the top
    // `lhw` frequently used blobs will be kept.
    std::unordered_set<uint64_t> g0, g0b;
    std::unordered_map<uint64_t, Entry> g1;
    std::vector<KV> tmp;

public:
    ConstTensorCache() {
        g0.reserve(window), g0b.reserve(window);
        g1.reserve(hwm), tmp.reserve(hwm);
    }
};

struct MultiCNConstTensorCache : CompNodeDepedentObject {
    std::mutex mtx;
    CompNode::UnorderedMap<ConstTensorCache> cn2cache;

    std::shared_ptr<void> on_comp_node_finalize() {
        MGB_LOCK_GUARD(mtx);
        cn2cache.clear();
        return {};
    }

    BlobPtr lookup(const HostTensorND& hv) {
        MGB_LOCK_GUARD(mtx);
        return cn2cache[hv.comp_node()].lookup(hv);
    }

    static MultiCNConstTensorCache& inst() {
        static auto* sl_inst =
                ResourceManager::create_global<MultiCNConstTensorCache>();
        return *sl_inst;
    }
};

struct LogicalTensorDesc {
    TensorLayout layout;
    CompNode comp_node;
    DeviceTensorND value;  // cpu:default
};
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
