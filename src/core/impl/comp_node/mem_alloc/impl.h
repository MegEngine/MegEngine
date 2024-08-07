#pragma once

#include "megbrain/comp_node/alloc.h"

#include <atomic>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace mgb {
namespace mem_alloc {

class DevMemAllocImpl;

class MemAllocImplHelper : virtual public MemAllocBase {
    friend class DevMemAllocImpl;

protected:
    struct MemAddr {
        //! whether it is head of a chunk from raw allocator; if true, it
        //! could not be merged with chunks with lower address
        bool is_head = false;
        size_t addr = -1;

        void* addr_ptr() const { return reinterpret_cast<void*>(addr); }

        bool operator<(const MemAddr& rhs) const { return addr < rhs.addr; }

        MemAddr operator+(size_t delta) const { return {false, addr + delta}; }
    };

    struct FreeBlock {
        MemAddr addr;
        size_t size = -1;

        size_t end() const { return addr.addr + size; }
    };

    struct FreeCmpBySize {
        bool operator()(const FreeBlock& a, const FreeBlock& b) const {
            // prefer more recent (hotter) block
            return a.size < b.size || (a.size == b.size && a.addr < b.addr);
        }
    };

    struct BlkByAddrIter;
    struct FreeBlockAddrInfo;

    //! free blocks sorted by size, and map to corresponding iterator in
    //! m_free_blk_addr
    std::map<FreeBlock, BlkByAddrIter, FreeCmpBySize> m_free_blk_size;

    //! map from address to size and size iter
    std::map<size_t, FreeBlockAddrInfo> m_free_blk_addr;

    MGB_MUTEX m_mutex;

    struct BlkByAddrIter {
        decltype(m_free_blk_addr.begin()) aiter;
    };

    struct FreeBlockAddrInfo {
        bool is_head;  //! always equals to siter->first.addr.is_head
        size_t size;
        decltype(m_free_blk_size.begin()) siter;
    };

    /*!
     * \brief merge a block into free list, without locking
     */
    void merge_free_unsafe(FreeBlock block);

    /*!
     * \brief directly insert a free block into m_free_blk_size and
     *      m_free_blk_addr, without merging
     */
    virtual void insert_free_unsafe(const FreeBlock& block);

    /*!
     * \brief allocate from parent allocator; this method must either return
     *      a valid address or throw an exception
     *
     * m_free_blk_addr and m_free_blk_size must be maintained if necessary
     */
    virtual MemAddr alloc_from_parent(size_t size) = 0;

    /*!
     * \brief get name of this allocator
     */
    virtual std::string get_name() const = 0;

    MemAddr do_alloc(
            size_t size, bool allow_from_parent, bool log_stat_on_error = false);

    //! get free mem for this allocator, without locking
    FreeMemStat get_free_memory_self_unsafe();

#if !MGB_BUILD_SLIM_SERVING
    size_t get_max_block_size_available_unsafe();

    std::pair<size_t, size_t> get_free_left_and_right(
            size_t begin_ptr, size_t end_ptr) override;
#endif

public:
    void print_memory_state() override;

    FreeMemStat get_free_memory() override final;

#if !MGB_BUILD_SLIM_SERVING
    size_t get_max_block_size_available() override final;
#endif

    std::string str_free_info(size_t threshold);
};

class StreamMemAllocImpl final : public StreamMemAlloc, public MemAllocImplHelper {
    struct AllocatedBlock {
        bool is_head;
        size_t size;
        size_t size_ahead;  // allocated size ahead of ptr
    };

    DevMemAllocImpl* m_dev_alloc;
    int m_stream_id;

    //! map from address to block info
    std::unordered_map<void*, AllocatedBlock> m_allocated_blocks;

    void* alloc(size_t size) override;

    void free(void* addr) override;

    void get_mem_info(size_t& free, size_t& tot) override;

    std::string get_name() const override;

    MemAddr alloc_from_parent(size_t size) override;
    size_t get_used_memory() override;
    FreeMemStat get_free_memory_dev() override;

public:
    void print_memory_state() override;
    StreamMemAllocImpl(DevMemAllocImpl* dev_alloc, int stream_id)
            : m_dev_alloc(dev_alloc), m_stream_id(stream_id) {}
};

/*!
 * \Note: DevMemAlloc has two-level structure, but when only one stream was
 * registered into the DevMemAlloc, the DevMemAlloc would behave like a
 * single-level allocator(i.e. only the FreeBlock pool in its child stream
 * allocator will be used) for better performance
 */
class DevMemAllocImpl final : public DevMemAlloc, public MemAllocImplHelper {
    friend class StreamMemAllocImpl;
    int m_device;
    std::shared_ptr<RawAllocator> m_raw_allocator;
    std::shared_ptr<DeviceRuntimePolicy> m_runtime_policy;
    ThinHashMap<StreamKey, std::unique_ptr<StreamMemAllocImpl>> m_stream_alloc;

    //!< blocks allocated from raw alloc, addr to size
    std::unordered_map<void*, size_t> m_alloc_from_raw;

    size_t m_tot_allocated_from_raw = 0;
    std::atomic_size_t m_used_size{0};
    std::atomic_size_t m_max_used_size{0};

    /*!
     * \brief gather all free blocks from child streams, and release full chunks
     *      back to parent allocator
     * \return number of bytes released
     */
    size_t gather_stream_free_blk_and_release_full() override;

    StreamMemAlloc* add_stream(StreamKey stream) override;

    MemAddr alloc_from_parent(size_t size) override;

    std::string get_name() const override {
        return ssprintf("dev allocator %d", m_device);
    }

    const std::shared_ptr<RawAllocator>& raw_allocator() const override {
        return m_raw_allocator;
    }

    const std::shared_ptr<DeviceRuntimePolicy>& device_runtime_policy() const override {
        return m_runtime_policy;
    }

    size_t get_used_memory() override { return m_used_size.load(); }

    size_t get_max_used_memory() override { return m_max_used_size.load(); }

    void reset_max_used_memory() override { m_max_used_size = 0; }

    void insert_free_unsafe(const FreeBlock& block) override;

    /*!
     * \brief return stream allocator if DevMemAlloc has single child,
     * otherwise return nullptr
     */
    StreamMemAllocImpl* get_single_child_stream_unsafe();

public:
    DevMemAllocImpl(
            int device, size_t reserve_size,
            const std::shared_ptr<mem_alloc::RawAllocator>& raw_allocator,
            const std::shared_ptr<mem_alloc::DeviceRuntimePolicy>& runtime_policy);

    ~DevMemAllocImpl();

    int device() const { return m_device; }

    MemAddr alloc(size_t size);

    void print_memory_state() override;

    FreeMemStat get_free_memory_dev() override;
};

class SimpleCachingAllocImpl : public SimpleCachingAlloc, public MemAllocImplHelper {
    struct AllocatedBlock {
        bool is_head;
        size_t size;
        size_t size_ahead;  // allocated size ahead of ptr
    };

    std::unique_ptr<RawAllocator> m_raw_alloc;
    std::unordered_map<void*, size_t> m_alloc_from_raw;
    std::unordered_map<void*, AllocatedBlock> m_allocated_blocks;
    size_t m_used_size = 0;

public:
    SimpleCachingAllocImpl(std::unique_ptr<RawAllocator> m_raw_alloc);
    ~SimpleCachingAllocImpl();

    void* alloc(size_t size) override;
    void free(void* ptr) override;
    size_t get_used_memory() override;
    FreeMemStat get_free_memory_dev() override;

protected:
    MemAddr alloc_from_parent(size_t size) override;
    std::string get_name() const override;
};

}  // namespace mem_alloc
}  // namespace mgb
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
