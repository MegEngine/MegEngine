#include "megbrain/imperative/cpp_cupti.h"

#include <cinttypes>
#include <cstddef>
#include <cstdlib>

#include "megbrain/exception.h"
#include "megbrain/imperative/profiler.h"
#include "megbrain/imperative/utils/platform.h"

#include "./profiler/events.h"

#if MGB_CUPTI
#include "cupti.h"

#define CUPTI_CALL(call)                                                     \
    do {                                                                     \
        CUptiResult _status = call;                                          \
        if (_status != CUPTI_SUCCESS) {                                      \
            const char* errstr;                                              \
            cuptiGetResultString(_status, &errstr);                          \
            mgb_assert(_status == CUPTI_SUCCESS, "cupti error: %s", errstr); \
        }                                                                    \
    } while (0)
#endif

namespace mgb::imperative::cupti {

#if MGB_CUPTI
namespace {
CUpti_SubscriberHandle cuptiSubscriber;

void cuptiSubscriberCallback(
        void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cb_id,
        const void* cb_info) {
    using namespace profiler;
    switch (domain) {
        case CUPTI_CB_DOMAIN_DRIVER_API: {
            auto cb_data = (const CUpti_CallbackData*)cb_info;
            switch (cb_id) {
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: {
                    if (cb_data->callbackSite == CUPTI_API_ENTER) {
                        MGB_RECORD_EVENT(
                                CUPTIKernelLaunchEvent, cb_data->correlationId,
                                cb_data->symbolName);
                    } else if (cb_data->callbackSite == CUPTI_API_EXIT) {
                        MGB_RECORD_EVENT(
                                CUPTIKernelLaunchFinishEvent, cb_data->correlationId,
                                cb_data->symbolName);
                    }
                    break;
                }
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA: {
                }
                case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync: {
                    if (cb_data->callbackSite == CUPTI_API_ENTER) {
                        MGB_RECORD_EVENT(
                                CUPTIMemcpyLaunchEvent, cb_data->correlationId);
                    } else if (cb_data->callbackSite == CUPTI_API_EXIT) {
                        MGB_RECORD_EVENT(
                                CUPTIMemcpyLaunchFinishEvent, cb_data->correlationId);
                    }
                    break;
                }
                default: {
                    if (cb_data->callbackSite == CUPTI_API_ENTER) {
                        MGB_RECORD_EVENT(
                                CUPTIDriverEvent, cb_data->correlationId,
                                cb_data->functionName);
                    } else if (cb_data->callbackSite == CUPTI_API_EXIT) {
                        MGB_RECORD_EVENT(
                                CUPTIDriverFinishEvent, cb_data->correlationId,
                                cb_data->functionName);
                    }
                }
            }
            break;
        }
        case CUPTI_CB_DOMAIN_RUNTIME_API: {
            auto cb_data = (const CUpti_CallbackData*)cb_info;
            if (cb_data->callbackSite == CUPTI_API_ENTER) {
                MGB_RECORD_EVENT(
                        CUPTIRuntimeEvent, cb_data->correlationId,
                        cb_data->functionName);
            } else if (cb_data->callbackSite == CUPTI_API_EXIT) {
                MGB_RECORD_EVENT(
                        CUPTIRuntimeFinishEvent, cb_data->correlationId,
                        cb_data->functionName);
            }
            break;
        }
    }
}

void handleActivity(CUpti_Activity* record) {
    using namespace std::chrono_literals;
    auto delta = 16ns;
    switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            auto kernel = cupti::activity<CUpti_ActivityKernel4>(record);
            MGB_RECORD_EVENT(
                    profiler::CUPTIKernelExecuteEvent, kernel->correlationId,
                    kernel->name, kernel.stream(), kernel.start(),
                    kernel.end() - delta);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto memcpy = cupti::activity<CUpti_ActivityMemcpy>(record);
            MGB_RECORD_EVENT(
                    profiler::CUPTIMemcpyEvent, memcpy->correlationId, memcpy->srcKind,
                    memcpy->dstKind, memcpy->bytes, memcpy.stream(), memcpy.start(),
                    memcpy.end());
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            auto memset = cupti::activity<CUpti_ActivityMemset>(record);
            MGB_RECORD_EVENT(
                    profiler::CUPTIMemsetEvent, memset->correlationId, memset->value,
                    memset->bytes, memset.stream(), memset.start(),
                    memset.end() - delta);
            break;
        }
        default:
            break;
    }
}

using activity_buffer_t =
        std::aligned_storage_t<8 * 1024 * 1024, ACTIVITY_RECORD_ALIGNMENT>;

void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *buffer = reinterpret_cast<uint8_t*>(new activity_buffer_t());
    *size = sizeof(activity_buffer_t);
    *maxNumRecords = 0;
}

void bufferCompleted(
        CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size,
        size_t validSize) {
    CUptiResult status;
    CUpti_Activity* record = NULL;

    if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                handleActivity(record);
            } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else {
                CUPTI_CALL(status);
            }
        } while (1);

        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        mgb_assert(dropped == 0, "%zu records dropped", dropped);
    }

    delete reinterpret_cast<activity_buffer_t*>(buffer);
}

static bool initialized = false;
}  // namespace

bool available() {
    uint32_t compiletime_version = (CUPTI_API_VERSION);
    uint32_t runtime_version;
    CUPTI_CALL(cuptiGetVersion(&runtime_version));
    if (compiletime_version != runtime_version) {
        static std::once_flag once;
        std::call_once(once, [&] {
            mgb_log_warn(
                    "CuPTI version %d mismatch against compiletime version %d. "
                    "This may caused by user config LD_LIBRARY_PATH"
                    "at unix-like env or config PATH at Windows env",
                    (int)compiletime_version, (int)runtime_version);
        });
        return false;
    }
    return true;
}

void enable() {
    // not thread safe
    mgb_assert(!initialized, "cupti already initialized");
    // callback
    CUPTI_CALL(cuptiSubscribe(
            &cuptiSubscriber, (CUpti_CallbackFunc)cuptiSubscriberCallback,
            (void*)nullptr));
    CUPTI_CALL(cuptiEnableDomain(1, cuptiSubscriber, CUPTI_CB_DOMAIN_DRIVER_API));
    CUPTI_CALL(cuptiEnableDomain(1, cuptiSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

    // activity
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    initialized = true;
}

void disable() {
    mgb_assert(initialized, "cupti not initialized yet");
    flush();
    CUPTI_CALL(cuptiFinalize());
    initialized = false;
}

void flush() {
    if (initialized) {
        CUPTI_CALL(cuptiActivityFlushAll(1));
    }
}

bool enabled() {
    return initialized;
}

time_point clock::now() {
    uint64_t timestamp;
    CUPTI_CALL(cuptiGetTimestamp(&timestamp));
    using namespace std::chrono;
    // overflow?
    return time_point(duration((int64_t)timestamp));
}

#else

class CuPTIUnavailableError : public MegBrainError {
public:
    CuPTIUnavailableError()
            : MegBrainError(
#if MGB_CUDA
                      "CuPTI disabled at compile time"
#else
                      "CuPTI unsupported on non cuda platform"
#endif
              ) {
    }
};

bool available() {
    return false;
}

void enable() {
    throw CuPTIUnavailableError();
}

void disable() {
    throw CuPTIUnavailableError();
}

void flush() {}

bool enabled() {
    return false;
}

time_point clock::now() {
    throw CuPTIUnavailableError();
}

#endif
}  // namespace mgb::imperative::cupti
