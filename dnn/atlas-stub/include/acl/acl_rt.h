/**
* @file acl_rt.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_RT_H_
#define INC_EXTERNAL_ACL_ACL_RT_H_

#include <stdint.h>
#include <stddef.h>
#include "acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_EVENT_TIME_LINE 0x00000008u

typedef enum aclrtRunMode {
    ACL_DEVICE,
    ACL_HOST,
} aclrtRunMode;

typedef enum aclrtTsId {
    ACL_TS_ID_AICORE   = 0,
    ACL_TS_ID_AIVECTOR = 1,
    ACL_TS_ID_RESERVED = 2,
} aclrtTsId;

typedef enum aclrtEventStatus {
    ACL_EVENT_STATUS_COMPLETE  = 0,
    ACL_EVENT_STATUS_NOT_READY = 1,
    ACL_EVENT_STATUS_RESERVED  = 2,
} aclrtEventStatus;

typedef enum aclrtEventRecordedStatus {
    ACL_EVENT_RECORDED_STATUS_NOT_READY = 0,
    ACL_EVENT_RECORDED_STATUS_COMPLETE = 1,
} aclrtEventRecordedStatus;

typedef enum aclrtEventWaitStatus {
    ACL_EVENT_WAIT_STATUS_COMPLETE  = 0,
    ACL_EVENT_WAIT_STATUS_NOT_READY = 1,
    ACL_EVENT_WAIT_STATUS_RESERVED  = 0xffff,
} aclrtEventWaitStatus;

typedef enum aclrtCallbackBlockType {
    ACL_CALLBACK_NO_BLOCK,
    ACL_CALLBACK_BLOCK,
} aclrtCallbackBlockType;

typedef enum aclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_DEVICE_TO_DEVICE,
} aclrtMemcpyKind;

typedef enum aclrtMemMallocPolicy {
    ACL_MEM_MALLOC_HUGE_FIRST,
    ACL_MEM_MALLOC_HUGE_ONLY,
    ACL_MEM_MALLOC_NORMAL_ONLY,
    ACL_MEM_MALLOC_HUGE_FIRST_P2P,
    ACL_MEM_MALLOC_HUGE_ONLY_P2P,
    ACL_MEM_MALLOC_NORMAL_ONLY_P2P,
} aclrtMemMallocPolicy;

typedef enum aclrtMemAttr {
    ACL_DDR_MEM,
    ACL_HBM_MEM,
    ACL_DDR_MEM_HUGE,
    ACL_DDR_MEM_NORMAL,
    ACL_HBM_MEM_HUGE,
    ACL_HBM_MEM_NORMAL,
    ACL_DDR_MEM_P2P_HUGE,
    ACL_DDR_MEM_P2P_NORMAL,
    ACL_HBM_MEM_P2P_HUGE,
    ACL_HBM_MEM_P2P_NORMAL,
} aclrtMemAttr;

typedef enum aclrtGroupAttr {
    ACL_GROUP_AICORE_INT,
    ACL_GROUP_AIV_INT,
    ACL_GROUP_AIC_INT,
    ACL_GROUP_SDMANUM_INT,
    ACL_GROUP_ASQNUM_INT,
    ACL_GROUP_GROUPID_INT
} aclrtGroupAttr;

typedef struct tagRtGroupInfo aclrtGroupInfo;

typedef struct rtExceptionInfo aclrtExceptionInfo;

typedef void (*aclrtCallback)(void *userData);

typedef void (*aclrtExceptionInfoCallback)(aclrtExceptionInfo *exceptionInfo);

/**
 * @ingroup AscendCL
 * @brief Set a callback function to handle exception information
 *
 * @param callback [IN] callback function to handle exception information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback);

/**
 * @ingroup AscendCL
 * @brief Get task id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The task id from exception information
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetTaskIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get stream id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The stream id from exception information
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetStreamIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get thread id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The thread id of fail task
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetThreadIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get device id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The thread id of fail task
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetDeviceIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief The thread that handles the callback function on the Stream
 *
 * @param threadId [IN] thread ID
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Add a callback function to be executed on the host
 *        to the task queue of the Stream
 *
 * @param fn [IN]   Specify the callback function to be added
 *                  The function prototype of the callback function is:
 *                  typedef void (*aclrtCallback)(void *userData);
 * @param userData [IN]   User data to be passed to the callback function
 * @param blockType [IN]  callback block type
 * @param stream [IN]     stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType,
                                                 aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief After waiting for a specified time, trigger callback processing
 *
 * @par Function
 *  The thread processing callback specified by
 *  the aclrtSubscribeReport interface
 *
 * @param timeout [IN]   timeout value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSubscribeReport
 */
ACL_FUNC_VISIBILITY aclError aclrtProcessReport(int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Cancel thread registration,
 *        the callback function on the specified Stream
 *        is no longer processed by the specified thread
 *
 * @param threadId [IN]   thread ID
 * @param stream [IN]     stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create context and associates it with the calling thread
 *
 * @par Function
 * The following use cases are supported:
 * @li If you don't call the aclrtCreateContext interface
 * to explicitly create the context,
 * the system will use the default context, which is implicitly created
 * when the aclrtSetDevice interface is called.
 * @li If multiple contexts are created in a process
 * (there is no limit on the number of contexts),
 * the current thread can only use one of them at the same time.
 * It is recommended to explicitly specify the context of the current thread
 * through the aclrtSetCurrentContext interface to increase.
 * the maintainability of the program.
 *
 * @param  context [OUT]    point to the created context
 * @param  deviceId [IN]    device to create context on
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSetDevice | aclrtSetCurrentContext
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief destroy context instance
 *
 * @par Function
 * Can only destroy context created through aclrtCreateContext interface
 *
 * @param  context [IN]   the context to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateContext
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyContext(aclrtContext context);

/**
 * @ingroup AscendCL
 * @brief set the context of the thread
 *
 * @par Function
 * The following scenarios are supported:
 * @li If the aclrtCreateContext interface is called in a thread to explicitly
 * create a Context (for example: ctx1), the thread's Context can be specified
 * without calling the aclrtSetCurrentContext interface.
 * The system uses ctx1 as the context of thread1 by default.
 * @li If the aclrtCreateContext interface is not explicitly created,
 * the system uses the default context as the context of the thread.
 * At this time, the aclrtDestroyContext interface cannot be used to release
 * the default context.
 * @li If the aclrtSetCurrentContext interface is called multiple times to
 * set the thread's Context, the last one prevails.
 *
 * @par Restriction
 * @li If the cevice corresponding to the context set for the thread
 * has been reset, you cannot set the context as the context of the thread,
 * otherwise a business exception will result.
 * @li It is recommended to use the context created in a thread.
 * If the aclrtCreateContext interface is called in thread A to create a context,
 * and the context is used in thread B,
 * the user must guarantee the execution order of tasks in the same stream
 * under the same context in two threads.
 *
 * @param  context [IN]   the current context of the thread
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateContext | aclrtDestroyContext
 */
ACL_FUNC_VISIBILITY aclError aclrtSetCurrentContext(aclrtContext context);

/**
 * @ingroup AscendCL
 * @brief get the context of the thread
 *
 * @par Function
 * If the user calls the aclrtSetCurrentContext interface
 * multiple times to set the context of the current thread,
 * then the last set context is obtained
 *
 * @param  context [OUT]   the current context of the thread
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSetCurrentContext
 */
ACL_FUNC_VISIBILITY aclError aclrtGetCurrentContext(aclrtContext *context);

/**
 * @ingroup AscendCL
 * @brief Specify the device to use for the operation
 * implicitly create the default context and the default stream
 *
 * @par Function
 * The following use cases are supported:
 * @li Device can be specified in the process or thread.
 * If you call the aclrtSetDevice interface multiple
 * times to specify the same device,
 * you only need to call the aclrtResetDevice interface to reset the device.
 * @li The same device can be specified for operation
 *  in different processes or threads.
 * @li Device is specified in a process,
 * and multiple threads in the process can share this device to explicitly
 * create a Context (aclrtCreateContext interface).
 * @li In multi-device scenarios, you can switch to other devices
 * through the aclrtSetDevice interface in the process.
 *
 * @param  deviceId [IN]  the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtResetDevice |aclrtCreateContext
 */
ACL_FUNC_VISIBILITY aclError aclrtSetDevice(int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief Reset the current operating Device and free resources on the device,
 * including the default context, the default stream,
 * and all streams created under the default context,
 * and synchronizes the interface.
 * If the task under the default context or stream has not been completed,
 * the system will wait for the task to complete before releasing it.
 *
 * @par Restriction
 * @li The Context, Stream, and Event that are explicitly created
 * on the device to be reset. Before resetting,
 * it is recommended to follow the following interface calling sequence,
 * otherwise business abnormalities may be caused.
 * @li Interface calling sequence:
 * call aclrtDestroyEvent interface to release Event or
 * call aclrtDestroyStream interface to release explicitly created Stream->
 * call aclrtDestroyContext to release explicitly created Context->
 * call aclrtResetDevice interface
 *
 * @param  deviceId [IN]   the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetDevice(int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief get target device of current thread
 *
 * @param deviceId [OUT]  the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDevice(int32_t *deviceId);

/**
 * @ingroup AscendCL
 * @brief get target side
 *
 * @param runMode [OUT]    the run mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetRunMode(aclrtRunMode *runMode);

/**
 * @ingroup AscendCL
 * @brief Wait for compute device to finish
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeDevice(void);

/**
 * @ingroup AscendCL
 * @brief Set Scheduling TS
 *
 * @param tsId [IN]   the ts id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetTsDevice(aclrtTsId tsId);

/**
 * @ingroup AscendCL
 * @brief get total device number.
 *
 * @param count [OUT]    the device number
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceCount(uint32_t *count);

/**
 * @ingroup AscendCL
 * @brief create event instance
 *
 * @param event [OUT]   created event
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateEvent(aclrtEvent *event);

/**
 * @ingroup AscendCL
 * @brief create event instance with flag
 *
 * @param event [OUT]   created event
 * @param flag [IN]     event flag
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief destroy event instance
 *
 * @par Function
 *  Only events created through the aclrtCreateEvent interface can be
 *  destroyed, synchronous interfaces. When destroying an event,
 *  the user must ensure that the tasks involved in the aclrtSynchronizeEvent
 *  interface or the aclrtStreamWaitEvent interface are completed before
 *  they are destroyed.
 *
 * @param  event [IN]   event to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateEvent | aclrtSynchronizeEvent | aclrtStreamWaitEvent
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyEvent(aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief Record an Event in the Stream
 *
 * @param event [IN]    event to record
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Reset an event
 *
 * @par Function
 *  Users need to make sure to wait for the tasks in the Stream
 *  to complete before resetting the Event
 *
 * @param event [IN]    event to reset
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream);

 /**
 * @ingroup AscendCL
 * @brief Queries an event's status
 *
 * @param  event [IN]    event to query
 * @param  status [OUT]  event status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
// ACL_DEPRECATED_MESSAGE("aclrtQueryEvent is deprecated, use aclrtQueryEventStatus instead")
ACL_FUNC_VISIBILITY aclError aclrtQueryEvent(aclrtEvent event, aclrtEventStatus *status);

/**
 * @ingroup AscendCL
 * @brief Queries an event's status
 *
 * @param  event [IN]    event to query
 * @param  status [OUT]  event recorded status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status);

/**
* @ingroup AscendCL
* @brief Queries an event's wait-status
*
* @param  event [IN]    event to query
* @param  status [OUT]  event wait-status
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus *status);

/**
 * @ingroup AscendCL
 * @brief Block Host Running, wait event to be complete
 *
 * @param  event [IN]   event to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeEvent(aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief computes the elapsed time between events.
 *
 * @param ms [OUT]     time between start and end in ms
 * @param start [IN]   starting event
 * @param end [IN]     ending event
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateEvent | aclrtRecordEvent | aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent);

/**
 * @ingroup AscendCL
 * @brief alloc memory on device
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aclrtMalloc interface needs to be released
 * through the aclrtFree interface.
 * @li Before calling the media data processing interface,
 * if you need to apply memory on the device to store input or output data,
 * you need to call acldvppMalloc to apply for memory.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFree | acldvppMalloc | aclrtMallocCached
 */
ACL_FUNC_VISIBILITY aclError aclrtMalloc(void **devPtr,
                                         size_t size,
                                         aclrtMemMallocPolicy policy);

/**
 * @ingroup AscendCL
 * @brief allocate memory on device with cache
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aclrtMallocCached interface needs to be released
 * through the aclrtFree interface.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFree | aclrtMalloc
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocCached(void **devPtr,
                                               size_t size,
                                               aclrtMemMallocPolicy policy);

/**
 * @ingroup AscendCL
 * @brief flush cache data to ddr
 *
 * @param devPtr [IN]  the pointer that flush data to ddr
 * @param size [IN]    flush size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemFlush(void *devPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief invalidate cache data
 *
 * @param devPtr [IN]  pointer to invalidate cache data
 * @param size [IN]    invalidate size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemInvalidate(void *devPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief free device memory
 *
 * @par Function
 *  can only free memory allocated through the aclrtMalloc interface
 *
 * @param  devPtr [IN]  Pointer to memory to be freed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMalloc
 */
ACL_FUNC_VISIBILITY aclError aclrtFree(void *devPtr);

/**
 * @ingroup AscendCL
 * @brief alloc memory on host
 *
 * @par Restriction
 * @li The requested memory cannot be used in the Device
 * and needs to be explicitly copied to the Device.
 * @li The memory requested by the aclrtMallocHost interface
 * needs to be released through the aclrtFreeHost interface.
 *
 * @param  hostPtr [OUT] pointer to pointer to allocated memory on the host
 * @param  size [IN]     alloc memory size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFreeHost
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocHost(void **hostPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief free host memory
 *
 * @par Function
 *  can only free memory allocated through the aclrtMallocHost interface
 *
 * @param  hostPtr [IN]   free memory pointer
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMallocHost
 */
ACL_FUNC_VISIBILITY aclError aclrtFreeHost(void *hostPtr);

/**
 * @ingroup AscendCL
 * @brief synchronous memory replication between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param destMax [IN]   Max length of the destination address memory
 * @param src [IN]       source address pointer
 * @param count [IN]     the length of byte to copy
 * @param kind [IN]      memcpy type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy(void *dst,
                                         size_t destMax,
                                         const void *src,
                                         size_t count,
                                         aclrtMemcpyKind kind);

/**
 * @ingroup AscendCL
 * @brief Initialize memory and set contents of memory to specified value
 *
 * @par Function
 *  The memory to be initialized is on the Host or device side,
 *  and the system determines whether
 *  it is host or device according to the address
 *
 * @param devPtr [IN]    Starting address of memory
 * @param maxCount [IN]  Max length of destination address memory
 * @param value [IN]     Set value
 * @param count [IN]     The length of memory
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count);

/**
 * @ingroup AscendCL
 * @brief  Asynchronous memory replication between Host and Device
 *
 * @par Function
 *  After calling this interface,
 *  be sure to call the aclrtSynchronizeStream interface to ensure that
 *  the task of memory replication has been completed
 *
 * @par Restriction
 * @li For on-chip Device-to-Device memory copy,
 *     both the source and destination addresses must be 64-byte aligned
 *
 * @param dst [IN]     destination address pointer
 * @param destMax [IN] Max length of destination address memory
 * @param src [IN]     source address pointer
 * @param count [IN]   the number of byte to copy
 * @param kind [IN]    memcpy type
 * @param stream [IN]  asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsync(void *dst,
                                              size_t destMax,
                                              const void *src,
                                              size_t count,
                                              aclrtMemcpyKind kind,
                                              aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief synchronous memory replication of two-dimensional matrix between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param dpitch [IN]    pitch of destination memory
 * @param src [IN]       source address pointer
 * @param spitch [IN]    pitch of source memory
 * @param width [IN]     width of matrix transfer
 * @param height [IN]    height of matrix transfer
 * @param kind [IN]      memcpy type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy2d(void *dst,
                                           size_t dpitch,
                                           const void *src,
                                           size_t spitch,
                                           size_t width,
                                           size_t height,
                                           aclrtMemcpyKind kind);

/**
 * @ingroup AscendCL
 * @brief asynchronous memory replication of two-dimensional matrix between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param dpitch [IN]    pitch of destination memory
 * @param src [IN]       source address pointer
 * @param spitch [IN]    pitch of source memory
 * @param width [IN]     width of matrix transfer
 * @param height [IN]    height of matrix transfer
 * @param kind [IN]      memcpy type
 * @param stream [IN]    asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy2dAsync(void *dst,
                                                size_t dpitch,
                                                const void *src,
                                                size_t spitch,
                                                size_t width,
                                                size_t height,
                                                aclrtMemcpyKind kind,
                                                aclrtStream stream);

/**
* @ingroup AscendCL
* @brief Asynchronous initialize memory
* and set contents of memory to specified value async
*
* @par Function
 *  The memory to be initialized is on the Host or device side,
 *  and the system determines whether
 *  it is host or device according to the address
 *
* @param devPtr [IN]      destination address pointer
* @param maxCount [IN]    Max length of destination address memory
* @param value [IN]       set value
* @param count [IN]       the number of byte to set
* @param stream [IN]      asynchronized task stream
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*
* @see aclrtSynchronizeStream
*/
ACL_FUNC_VISIBILITY aclError aclrtMemsetAsync(void *devPtr,
                                              size_t maxCount,
                                              int32_t value,
                                              size_t count,
                                              aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief  create stream instance
 *
 * @param  stream [OUT]   the created stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateStream(aclrtStream *stream);

/**
 * @ingroup AscendCL
 * @brief destroy stream instance
 *
 * @par Function
 * Can only destroy streams created through the aclrtCreateStream interface
 *
 * @par Restriction
 * Before calling the aclrtDestroyStream interface to destroy
 * the specified Stream, you need to call the aclrtSynchronizeStream interface
 * to ensure that the tasks in the Stream have been completed.
 *
 * @param stream [IN]  the stream to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateStream | aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyStream(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief block the host until all tasks
 * in the specified stream have completed
 *
 * @param  stream [IN]   the stream to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStream(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Blocks the operation of the specified Stream until
 * the specified Event is completed.
 * Support for multiple streams waiting for the same event.
 *
 * @param  stream [IN]   the wait stream If using thedefault Stream, set NULL
 * @param  event [IN]    the event to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief set group
 *
 * @par Function
 *  set the task to the corresponding group
 *
 * @param groupId [IN]   group id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount | aclrtGetAllGroupInfo | aclrtGetGroupInfoDetail
 */
ACL_FUNC_VISIBILITY aclError aclrtSetGroup(int32_t groupId);

/**
 * @ingroup AscendCL
 * @brief get the number of group
 *
 * @par Function
 *  get the number of group. if the number of group is zero,
 *  it means that group is not supported or group is not created.
 *
 * @param count [OUT]   the number of group
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 */
ACL_FUNC_VISIBILITY aclError aclrtGetGroupCount(uint32_t *count);

/**
 * @ingroup AscendCL
 * @brief create group information
 *
 * @retval null for failed.
 * @retval OtherValues success.
 *
 * @see aclrtDestroyGroupInfo
 */
ACL_FUNC_VISIBILITY aclrtGroupInfo *aclrtCreateGroupInfo();

/**
 * @ingroup AscendCL
 * @brief destroy group information
 *
 * @param groupInfo [IN]   pointer to group information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateGroupInfo
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyGroupInfo(aclrtGroupInfo *groupInfo);

/**
 * @ingroup AscendCL
 * @brief get all group information
 *
 * @param groupInfo [OUT]   pointer to group information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount
 */
ACL_FUNC_VISIBILITY aclError aclrtGetAllGroupInfo(aclrtGroupInfo *groupInfo);

/**
 * @ingroup AscendCL
 * @brief get detail information of group
 *
 * @param groupInfo [IN]    pointer to group information
 * @param groupIndex [IN]   group index value
 * @param attr [IN]         group attribute
 * @param attrValue [OUT]   pointer to attribute value
 * @param valueLen [IN]     length of attribute value
 * @param paramRetSize [OUT]   pointer to real length of attribute value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount | aclrtGetAllGroupInfo
 */
ACL_FUNC_VISIBILITY aclError aclrtGetGroupInfoDetail(const aclrtGroupInfo *groupInfo,
                                                     int32_t groupIndex,
                                                     aclrtGroupAttr attr,
                                                     void *attrValue,
                                                     size_t valueLen,
                                                     size_t *paramRetSize);

/**
 * @ingroup AscendCL
 * @brief checking whether current device and peer device support the p2p feature
 *
 * @param canAccessPeer [OUT]   pointer to save the checking result
 * @param deviceId [IN]         current device id
 * @param peerDeviceId [IN]     peer device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceEnablePeerAccess | aclrtDeviceDisablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId);

/**
 * @ingroup AscendCL
 * @brief enable the peer device to support the p2p feature
 *
 * @param peerDeviceId [IN]   the peer device id
 * @param flags [IN]   reserved field, now it must be zero
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceCanAccessPeer | aclrtDeviceDisablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags);

/**
 * @ingroup AscendCL
 * @brief disable the peer device to support the p2p function
 *
 * @param peerDeviceId [IN]   the peer device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceCanAccessPeer | aclrtDeviceEnablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceDisablePeerAccess(int32_t peerDeviceId);

/**
 * @ingroup AscendCL
 * @brief Obtain the free memory and total memory of specified attribute.
 * the specified memory include normal memory and huge memory.
 *
 * @param attr [IN]    the memory attribute of specified device
 * @param free [OUT]   the free memory of specified device
 * @param total [OUT]  the total memory of specified device.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total);

/**
 * @ingroup AscendCL
 * @brief Set the timeout interval for waitting of op
 *
 * @param timeout [IN]   op wait timeout
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpWaitTimeout(uint32_t timeout);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_RT_H_

