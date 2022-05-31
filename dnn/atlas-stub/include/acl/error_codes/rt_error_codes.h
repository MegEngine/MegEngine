/**
* @file rt_error_codes.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef __INC_EXTERNEL_RT_ERROR_CODES_H__
#define __INC_EXTERNEL_RT_ERROR_CODES_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

static const int32_t ACL_RT_SUCCESS                          = 0; // success

static const int32_t ACL_ERROR_RT_PARAM_INVALID              = 107000; // param invalid
static const int32_t ACL_ERROR_RT_INVALID_DEVICEID           = 107001; // invalid device id
static const int32_t ACL_ERROR_RT_CONTEXT_NULL               = 107002; // current context null
static const int32_t ACL_ERROR_RT_STREAM_CONTEXT             = 107003; // stream not in current context
static const int32_t ACL_ERROR_RT_MODEL_CONTEXT              = 107004; // model not in current context
static const int32_t ACL_ERROR_RT_STREAM_MODEL               = 107005; // stream not in model
static const int32_t ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID    = 107006; // event timestamp invalid
static const int32_t ACL_ERROR_RT_EVENT_TIMESTAMP_REVERSAL   = 107007; // event timestamp reversal
static const int32_t ACL_ERROR_RT_ADDR_UNALIGNED             = 107008; // memory address unaligned
static const int32_t ACL_ERROR_RT_FILE_OPEN                  = 107009; // open file failed
static const int32_t ACL_ERROR_RT_FILE_WRITE                 = 107010; // write file failed
static const int32_t ACL_ERROR_RT_STREAM_SUBSCRIBE           = 107011; // error subscribe stream
static const int32_t ACL_ERROR_RT_THREAD_SUBSCRIBE           = 107012; // error subscribe thread
static const int32_t ACL_ERROR_RT_GROUP_NOT_SET              = 107013; // group not set
static const int32_t ACL_ERROR_RT_GROUP_NOT_CREATE           = 107014; // group not create
static const int32_t ACL_ERROR_RT_STREAM_NO_CB_REG           = 107015; // callback not register to stream
static const int32_t ACL_ERROR_RT_INVALID_MEMORY_TYPE        = 107016; // invalid memory type
static const int32_t ACL_ERROR_RT_INVALID_HANDLE             = 107017; // invalid handle
static const int32_t ACL_ERROR_RT_INVALID_MALLOC_TYPE        = 107018; // invalid malloc type
static const int32_t ACL_ERROR_RT_WAIT_TIMEOUT               = 107019; // wait timeout
static const int32_t ACL_ERROR_RT_TASK_TIMEOUT               = 107020; // task timeout

static const int32_t ACL_ERROR_RT_FEATURE_NOT_SUPPORT        = 207000; // feature not support
static const int32_t ACL_ERROR_RT_MEMORY_ALLOCATION          = 207001; // memory allocation error
static const int32_t ACL_ERROR_RT_MEMORY_FREE                = 207002; // memory free error
static const int32_t ACL_ERROR_RT_AICORE_OVER_FLOW           = 207003; // aicore over flow
static const int32_t ACL_ERROR_RT_NO_DEVICE                  = 207004; // no device
static const int32_t ACL_ERROR_RT_RESOURCE_ALLOC_FAIL        = 207005; // resource alloc fail
static const int32_t ACL_ERROR_RT_NO_PERMISSION              = 207006; // no permission
static const int32_t ACL_ERROR_RT_NO_EVENT_RESOURCE          = 207007; // no event resource
static const int32_t ACL_ERROR_RT_NO_STREAM_RESOURCE         = 207008; // no stream resource
static const int32_t ACL_ERROR_RT_NO_NOTIFY_RESOURCE         = 207009; // no notify resource
static const int32_t ACL_ERROR_RT_NO_MODEL_RESOURCE          = 207010; // no model resource
static const int32_t ACL_ERROR_RT_NO_CDQ_RESOURCE            = 207011; // no cdq resource
static const int32_t ACL_ERROR_RT_OVER_LIMIT                 = 207012; // over limit
static const int32_t ACL_ERROR_RT_QUEUE_EMPTY                = 207013; // queue is empty
static const int32_t ACL_ERROR_RT_QUEUE_FULL                 = 207014; // queue is full
static const int32_t ACL_ERROR_RT_REPEATED_INIT              = 207015; // repeated init
static const int32_t ACL_ERROR_RT_AIVEC_OVER_FLOW            = 207016; // aivec over flow

static const int32_t ACL_ERROR_RT_INTERNAL_ERROR             = 507000; // runtime internal error
static const int32_t ACL_ERROR_RT_TS_ERROR                   = 507001; // ts internel error
static const int32_t ACL_ERROR_RT_STREAM_TASK_FULL           = 507002; // task full in stream
static const int32_t ACL_ERROR_RT_STREAM_TASK_EMPTY          = 507003; // task empty in stream
static const int32_t ACL_ERROR_RT_STREAM_NOT_COMPLETE        = 507004; // stream not complete
static const int32_t ACL_ERROR_RT_END_OF_SEQUENCE            = 507005; // end of sequence
static const int32_t ACL_ERROR_RT_EVENT_NOT_COMPLETE         = 507006; // event not complete
static const int32_t ACL_ERROR_RT_CONTEXT_RELEASE_ERROR      = 507007; // context release error
static const int32_t ACL_ERROR_RT_SOC_VERSION                = 507008; // soc version error
static const int32_t ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT      = 507009; // task type not support
static const int32_t ACL_ERROR_RT_LOST_HEARTBEAT             = 507010; // ts lost heartbeat
static const int32_t ACL_ERROR_RT_MODEL_EXECUTE              = 507011; // model execute failed
static const int32_t ACL_ERROR_RT_REPORT_TIMEOUT             = 507012; // report timeout
static const int32_t ACL_ERROR_RT_SYS_DMA                    = 507013; // sys dma error
static const int32_t ACL_ERROR_RT_AICORE_TIMEOUT             = 507014; // aicore timeout
static const int32_t ACL_ERROR_RT_AICORE_EXCEPTION           = 507015; // aicore exception
static const int32_t ACL_ERROR_RT_AICORE_TRAP_EXCEPTION      = 507016; // aicore trap exception
static const int32_t ACL_ERROR_RT_AICPU_TIMEOUT              = 507017; // aicpu timeout
static const int32_t ACL_ERROR_RT_AICPU_EXCEPTION            = 507018; // aicpu exception
static const int32_t ACL_ERROR_RT_AICPU_DATADUMP_RSP_ERR     = 507019; // aicpu datadump response error
static const int32_t ACL_ERROR_RT_AICPU_MODEL_RSP_ERR        = 507020; // aicpu model operate response error
static const int32_t ACL_ERROR_RT_PROFILING_ERROR            = 507021; // profiling error
static const int32_t ACL_ERROR_RT_IPC_ERROR                  = 507022; // ipc error
static const int32_t ACL_ERROR_RT_MODEL_ABORT_NORMAL         = 507023; // model abort normal
static const int32_t ACL_ERROR_RT_KERNEL_UNREGISTERING       = 507024; // kernel unregistering
static const int32_t ACL_ERROR_RT_RINGBUFFER_NOT_INIT        = 507025; // ringbuffer not init
static const int32_t ACL_ERROR_RT_RINGBUFFER_NO_DATA         = 507026; // ringbuffer no data
static const int32_t ACL_ERROR_RT_KERNEL_LOOKUP              = 507027; // kernel lookup error
static const int32_t ACL_ERROR_RT_KERNEL_DUPLICATE           = 507028; // kernel register duplicate
static const int32_t ACL_ERROR_RT_DEBUG_REGISTER_FAIL        = 507029; // debug register failed
static const int32_t ACL_ERROR_RT_DEBUG_UNREGISTER_FAIL      = 507030; // debug unregister failed
static const int32_t ACL_ERROR_RT_LABEL_CONTEXT              = 507031; // label not in current context
static const int32_t ACL_ERROR_RT_PROGRAM_USE_OUT            = 507032; // program register num use out
static const int32_t ACL_ERROR_RT_DEV_SETUP_ERROR            = 507033; // device setup error
static const int32_t ACL_ERROR_RT_VECTOR_CORE_TIMEOUT        = 507034; // vector core timeout
static const int32_t ACL_ERROR_RT_VECTOR_CORE_EXCEPTION      = 507035; // vector core exception
static const int32_t ACL_ERROR_RT_VECTOR_CORE_TRAP_EXCEPTION = 507036; // vector core trap exception
static const int32_t ACL_ERROR_RT_CDQ_BATCH_ABNORMAL         = 507037; // cdq alloc batch abnormal
static const int32_t ACL_ERROR_RT_DIE_MODE_CHANGE_ERROR      = 507038; // can not change die mode
static const int32_t ACL_ERROR_RT_DIE_SET_ERROR              = 507039; // single die mode can not set die
static const int32_t ACL_ERROR_RT_INVALID_DIEID              = 507040; // invalid die id
static const int32_t ACL_ERROR_RT_DIE_MODE_NOT_SET           = 507041; // die mode not set

static const int32_t ACL_ERROR_RT_DRV_INTERNAL_ERROR         = 507899; // drv internal error
static const int32_t ACL_ERROR_RT_AICPU_INTERNAL_ERROR       = 507900; // aicpu internal error
static const int32_t ACL_ERROR_RT_SOCKET_CLOSE               = 507901; // hdc disconnect

#ifdef __cplusplus
}
#endif
#endif // __INC_EXTERNEL_RT_ERROR_CODES_H__
