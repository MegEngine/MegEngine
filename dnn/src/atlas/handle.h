#pragma once
#include "megcore_atlas.h"
#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "megdnn/oprs/general.h"

#include "src/atlas/megcore/device_context.hpp"
#include "src/common/handle_impl.h"
#include "src/common/megcore/common/device_context.hpp"
#include "src/common/utils.h"

#include <atomic>
#include <mutex>

#include "acl/acl_rt.h"

namespace megdnn {
namespace atlas {

class HandleImpl : public HandleImplHelper {
public:
    HandleImpl(megcoreComputingHandle_t computing_handle);
    ~HandleImpl() noexcept;

    size_t alignment_requirement() const override;

    template <typename Opr>
    std::unique_ptr<Opr> create_operator();

    const megcore::AtlasContext& megcore_context() const { return m_megcore_context; }

    int device_id() const { return m_device_id; }

    aclrtStream stream() const { return megcore_context().stream; }

    void* alloc(size_t size, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
    void free(void* ptr);

    //! global matmul opr
    Checksum* checksum_opr() override final {
        return get_helper_opr<Checksum, 0>(this);
    }

private:
    int m_device_id;
    //! MegDNN handle does not manage the lifetime of cnrt queue.
    megcore::AtlasContext m_megcore_context;
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
