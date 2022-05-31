/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INC_EXTERNAL_GE_GE_ERROR_CODES_H_
#define INC_EXTERNAL_GE_GE_ERROR_CODES_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
static const uint32_t ACL_ERROR_GE_PARAM_INVALID = 145000U;
static const uint32_t ACL_ERROR_GE_EXEC_NOT_INIT = 145001U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID = 145002U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_ID_INVALID = 145003U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID = 145006U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID = 145007U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID = 145008U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED = 145009U;
static const uint32_t ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID = 145011U;
static const uint32_t ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID = 145012U;
static const uint32_t ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID = 145013U;
static const uint32_t ACL_ERROR_GE_AIPP_BATCH_EMPTY = 145014U;
static const uint32_t ACL_ERROR_GE_AIPP_NOT_EXIST = 145015U;
static const uint32_t ACL_ERROR_GE_AIPP_MODE_INVALID = 145016U;
static const uint32_t ACL_ERROR_GE_OP_TASK_TYPE_INVALID = 145017U;
static const uint32_t ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID = 145018U;
static const uint32_t ACL_ERROR_GE_PLGMGR_PATH_INVALID = 145019U;
static const uint32_t ACL_ERROR_GE_FORMAT_INVALID = 145020U;
static const uint32_t ACL_ERROR_GE_SHAPE_INVALID = 145021U;
static const uint32_t ACL_ERROR_GE_DATATYPE_INVALID = 145022U;
static const uint32_t ACL_ERROR_GE_MEMORY_ALLOCATION = 245000U;
static const uint32_t ACL_ERROR_GE_MEMORY_OPERATE_FAILED = 245001U;
static const uint32_t ACL_ERROR_GE_INTERNAL_ERROR = 545000U;
static const uint32_t ACL_ERROR_GE_LOAD_MODEL = 545001U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED = 545002U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED = 545003U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED = 545004U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED = 545005U;
static const uint32_t ACL_ERROR_GE_EXEC_RELEASE_MODEL_DATA = 545006U;
static const uint32_t ACL_ERROR_GE_COMMAND_HANDLE = 545007U;
static const uint32_t ACL_ERROR_GE_GET_TENSOR_INFO = 545008U;
static const uint32_t ACL_ERROR_GE_UNLOAD_MODEL = 545009U;

#ifdef __cplusplus
}  // namespace ge
#endif
#endif  // INC_EXTERNAL_GE_GE_ERROR_CODES_H_
