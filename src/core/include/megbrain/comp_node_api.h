#pragma once

#include <cstddef>
#if defined(_WIN32)
#define MGB_API __declspec(dllexport)
#else
#define MGB_API __attribute__((visibility("default")))
#endif
namespace mgb {
namespace pubapi {

typedef struct _MgbComputeNode* mgbComputeNode_t;
struct DeviceLocator {
    int device = -1;
    int stream = -1;
};

MGB_API mgbComputeNode_t load_cuda_cn(int device_id, int stream_id);
MGB_API void unload_cuda_cn(mgbComputeNode_t);
MGB_API void* alloc(mgbComputeNode_t cn, size_t);
MGB_API void dealloc(mgbComputeNode_t cn, void* addr);
MGB_API void* get_cuda_stream(mgbComputeNode_t cn);
MGB_API DeviceLocator get_physical_location(mgbComputeNode_t);
MGB_API void sync(mgbComputeNode_t cn);
MGB_API bool is_finalize();
MGB_API void log_xla_mem_states();
MGB_API void reset_xla_mem_states();
MGB_API bool is_xla_used();
}  // namespace pubapi
}  // namespace mgb
