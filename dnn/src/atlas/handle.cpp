#include "src/atlas/handle.h"
#include "megcore_atlas.h"
#include "src/atlas/checksum/opr_impl.h"
#include "src/common/handle_impl.h"

#include <acl/acl.h>

namespace megdnn {
namespace atlas {

HandleImpl::HandleImpl(megcoreComputingHandle_t comp_handle)
        : HandleImplHelper(comp_handle, HandleType::ATLAS) {
    // Get megcore device handle
    megcoreDeviceHandle_t dev_handle;
    megcoreGetDeviceHandle(comp_handle, &dev_handle);

    int dev_id;
    megcoreGetDeviceID(dev_handle, &dev_id);
    m_device_id = dev_id;
    megcore::getAtlasContext(comp_handle, &m_megcore_context);
}

HandleImpl::~HandleImpl() noexcept = default;

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    megdnn_throw(
            "unsupported atlas opr, try export RUNTIME_OVERRIDE_LOG_LEVEL=0 to get "
            "more info");
    return nullptr;
}

size_t HandleImpl::alignment_requirement() const {
    //! because memcpyasync api requires that the memory is 128bytes alignment
    return 64;
}

MEGDNN_SPECIALIZE_CREATE_OPERATOR(ChecksumForward);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
