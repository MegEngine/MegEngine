#include "megcore.h"

#include "src/cambricon/utils.h"
#include "src/common/utils.h"

#include "src/cambricon/megcore/cambricon_computing_context.hpp"

using namespace megcore;
using namespace megcore::cambricon;

CambriconComputingContext::CambriconComputingContext(
        megcoreDeviceHandle_t dev_handle, unsigned int flags,
        const CambriconContext& ctx)
        : ComputingContext(dev_handle, flags),
          own_queue{ctx.queue == nullptr},
          own_cnnl_handle(ctx.cnnl_handle == nullptr),
          context_{ctx} {
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_assert(platform == megcorePlatformCambricon);
    if (own_queue) {
        cnrt_check(cnrtQueueCreate(&context_.queue));
    }
    if (own_cnnl_handle) {
        cnnl_check(cnnlCreate(&context_.cnnl_handle));
        cnnl_check(cnnlSetQueue(context_.cnnl_handle, context_.queue));
    }
}

CambriconComputingContext::~CambriconComputingContext() {
    if (own_cnnl_handle) {
        cnnl_check(cnnlDestroy(context_.cnnl_handle));
    }
    if (own_queue) {
        cnrt_check(cnrtQueueDestroy(context_.queue));
    }
}

void CambriconComputingContext::memcpy(
        void* dst, const void* src, size_t size_in_bytes, megcoreMemcpyKind_t kind) {
    cnrtMemTransDir_t dir;
    switch (kind) {
        case megcoreMemcpyDeviceToHost:
            dir = CNRT_MEM_TRANS_DIR_DEV2HOST;
            break;
        case megcoreMemcpyHostToDevice:
            dir = CNRT_MEM_TRANS_DIR_HOST2DEV;
            break;
        case megcoreMemcpyDeviceToDevice:
            dir = CNRT_MEM_TRANS_DIR_DEV2DEV;
            break;
        default:
            megdnn_throw("bad cnrt mem trans dir");
    }
    cnrt_check(cnrtMemcpyAsync(
            dst, const_cast<void*>(src), size_in_bytes, context_.queue, dir));
}

void CambriconComputingContext::memcpy_peer_async_d2d(
        void* dst, int dst_dev, const void* src, int src_dev, size_t size_in_bytes) {
    unsigned int can_access = -1;
    cnrt_check(cnrtGetPeerAccessibility(&can_access, src_dev, dst_dev));
    if (can_access == -1) {
        megdnn_throw("there is no enough MLU devices for memory copy");
    }

    cnrt_check(cnrtMemcpyPeerAsync(
            dst, dst_dev, const_cast<void*>(src), src_dev, size_in_bytes,
            context_.queue));
}

void CambriconComputingContext::memset(void* dst, int value, size_t size_in_bytes) {
    cnrt_check(cnrtMemsetAsync(dst, value, size_in_bytes, context_.queue));
}

void CambriconComputingContext::synchronize() {
    cnrt_check(cnrtQueueSync(context_.queue));
}

// vim: syntax=cpp.doxygen
