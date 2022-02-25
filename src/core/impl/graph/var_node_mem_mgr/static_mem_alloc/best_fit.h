#pragma once

#include <map>
#include <set>
#include "./impl.h"
#include "megbrain/utils/metahelper.h"

namespace mgb {
namespace cg {

class StaticMemAllocBestFit final : public StaticMemAllocImplHelper {
    struct Chunk {
        size_t addr, size;
        size_t addr_end() const { return addr + size; }
    };
    struct FreeBlockBySizeAddrAligned;
    struct FreeBlockByAddr;

    std::set<FreeBlockBySizeAddrAligned> m_free_by_size_addr_align;
    std::map<size_t, FreeBlockByAddr> m_free_by_addr;

    using FreeByAddrIter = decltype(m_free_by_addr.begin());

    struct FreeBlockBySizeAddrAligned {
        FreeByAddrIter& aiter() { return m_aiter_storage.get(); }

        const FreeByAddrIter& aiter() const { return m_aiter_storage.get(); }

        size_t addr_aligned, size;

        FreeBlockBySizeAddrAligned(size_t addr, size_t size)
                : addr_aligned(addr), size(size) {}

        bool operator<(const FreeBlockBySizeAddrAligned& rhs) const {
            return size < rhs.size ||
                   (size == rhs.size && addr_aligned < rhs.addr_aligned);
        }

    private:
        IncompleteObjStorageMock<FreeByAddrIter, std::set<int>::iterator>
                m_aiter_storage;
    };

    struct FreeBlockByAddr {
        decltype(m_free_by_size_addr_align.begin()) siter;
        size_t size;

        explicit FreeBlockByAddr(const Chunk& chk) : size(chk.size) {}
    };

    size_t m_top = 0;

    // from addr to size
    ThinHashMap<size_t, size_t> m_allocated_chunk;

    void remove_free_by_aiter(FreeByAddrIter aiter);

    void merge_free_and_insert(Chunk chk);
    void insert_free(const Chunk& chk);

    void free(size_t addr);

    /*!
     * \brief alloc new chunk with aligned address
     */
    size_t alloc_aligned_addr(size_t size);

    /*!
     * \brief alloc on given address
     */
    void alloc_placement(size_t addr, size_t size);

public:
    void do_solve() override;

    size_t tot_alloc() const override { return m_top; }
};

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
