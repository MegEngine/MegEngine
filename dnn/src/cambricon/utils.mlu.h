#pragma once

#include "src/common/utils.cuh"

#include <stdint.h>

#include <cnnl.h>
#include <cnrt.h>
#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define cndrv_check(_x)                                            \
    do {                                                           \
        CNresult _ret = (_x);                                      \
        if (_ret != CN_SUCCESS) {                                  \
            ::megdnn::cambricon::__throw_cndrv_error__(_ret, #_x); \
        }                                                          \
    } while (0)

#define cnnl_check(_x)                                            \
    do {                                                          \
        cnnlStatus_t _ret = (_x);                                 \
        if (_ret != CNNL_STATUS_SUCCESS) {                        \
            ::megdnn::cambricon::__throw_cnnl_error__(_ret, #_x); \
        }                                                         \
    } while (0)

#define cnrt_check(_x)                                            \
    do {                                                          \
        cnrtRet_t _ret = (_x);                                    \
        if (_ret != CNRT_RET_SUCCESS) {                           \
            ::megdnn::cambricon::__throw_cnrt_error__(_ret, #_x); \
        }                                                         \
    } while (0)

#define after_kernel_launch()           \
    do {                                \
        cnrt_check(cnrtGetLastError()); \
    } while (0)

namespace megdnn {
namespace cambricon {

struct BangHandle {
    cnrtQueue_t queue;
    uint32_t num_clusters;
    uint32_t num_cores_per_cluster;

    BangHandle(cnrtQueue_t q, int num_clusters, int num_cores_per_cluster)
            : queue(q),
              num_clusters(num_clusters),
              num_cores_per_cluster(num_cores_per_cluster) {}
};

//! Error handling funcions
MEGDNN_NORETURN void __throw_cndrv_error__(CNresult err, const char* msg);
MEGDNN_NORETURN void __throw_cnrt_error__(cnrtRet_t err, const char* msg);
MEGDNN_NORETURN void __throw_cnnl_error__(cnnlStatus_t err, const char* msg);

static inline void callback_free(CNqueue queue, CNresult status, void* userData) {
    cndrv_check(status);
    free(userData);
}

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
