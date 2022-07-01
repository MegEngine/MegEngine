#pragma once

#include "megbrain/imperative/physical_tensor.h"
#include "megbrain/imperative/utils/helper.h"

namespace mgb {
namespace imperative {

class BlobManager : public NonCopyableObj {
public:
    using allocator_t =
            std::function<DeviceTensorStorage::RawStorage(CompNode, size_t)>;
    virtual ~BlobManager() = default;

    static BlobManager* inst();

    virtual void alloc_direct(OwnedBlob* blob, size_t size) = 0;

    virtual bool try_alloc_direct(OwnedBlob* blob, size_t size) {
        try {
            alloc_direct(blob, size);
            return true;
        } catch (MemAllocError&) {
            return false;
        }
    }

    virtual void alloc_with_defrag(OwnedBlob* blob, size_t size) = 0;

    virtual void set_allocator(allocator_t allocator) = 0;

    virtual void register_blob(OwnedBlob* blob) = 0;

    virtual void unregister_blob(OwnedBlob* blob) = 0;

    virtual void defrag(const CompNode& cn) = 0;
};

}  // namespace imperative
}  // namespace mgb
