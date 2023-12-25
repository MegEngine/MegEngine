#include "src/atlas/handle.h"
#include "megcore_atlas.h"
#include "src/atlas/checksum/opr_impl.h"
#include "src/atlas/conv_bias/opr_impl.h"
#include "src/atlas/elemwise/opr_impl.h"
#include "src/atlas/fill/opr_impl.h"
#include "src/atlas/type_cvt/opr_impl.h"
#include "src/atlas/utils.h"
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

void* HandleImpl::alloc(size_t size, aclrtMemMallocPolicy policy) {
    auto mem_mgr = megcore_context().mem_mgr;
    if (size <= 0) {
        return nullptr;
    }
    if (mem_mgr) {
        return mem_mgr->alloc(size);
    } else {
        int32_t dev_id = -1;
        auto err = aclrtGetDevice(&dev_id);
        if (err == ACL_ERROR_INVALID_DEVICE || err == ACL_ERROR_RT_CONTEXT_NULL ||
            device_id() != dev_id) {
            acl_check(aclrtSetDevice(device_id()));
        }

        void* ptr = nullptr;
        acl_check(aclrtMalloc(&ptr, size, policy));
        return ptr;
    }
}

void HandleImpl::free(void* ptr) {
    auto mem_mgr = megcore_context().mem_mgr;
    if (!ptr) {
        return;
    }
    if (mem_mgr) {
        return mem_mgr->free(ptr);
    } else {
        int32_t dev_id = -1;
        auto err = aclrtGetDevice(&dev_id);
        if (err == ACL_ERROR_INVALID_DEVICE || err == ACL_ERROR_RT_CONTEXT_NULL ||
            device_id() != dev_id) {
            acl_check(aclrtSetDevice(device_id()));
        }
        acl_check(aclrtFree(ptr));
    }
}

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
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ElemwiseForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Fill);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TypeCvt);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvBiasForward);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
