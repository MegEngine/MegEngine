#pragma once
#include "megcore_cambricon.h"
#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "megdnn/oprs/general.h"

#include "src/common/handle_impl.h"
#include "src/common/utils.h"

#include <atomic>
#include <mutex>

#include <cnnl.h>
#include <cnrt.h>

namespace megdnn {
namespace cambricon {

class HandleImpl : public HandleImplHelper {
public:
    HandleImpl(megcoreComputingHandle_t computing_handle);
    ~HandleImpl() noexcept;

    size_t alignment_requirement() const override;

    const cnrtDeviceProp_t& device_info() const { return m_device_info; }

    template <typename Opr>
    std::unique_ptr<Opr> create_operator();

    const megcore::CambriconContext& megcore_context() const {
        return m_megcore_context;
    }

    int device_id() const { return m_device_id; }

    cnrtQueue_t queue() const { return megcore_context().queue; }
    cnnlHandle_t cnnl_handle() const { return megcore_context().cnnl_handle; };

    void* alloc(size_t size);
    void free(void* ptr);

    //! global matmul opr
    Checksum* checksum_opr() override final {
        return get_helper_opr<Checksum, 0>(this);
    }

private:
    int m_device_id;
    //! MegDNN handle does not manage the lifetime of cnrt queue.
    megcore::CambriconContext m_megcore_context;

    cnrtDeviceProp_t m_device_info;
};

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
