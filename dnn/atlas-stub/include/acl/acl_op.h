/**
* @file acl_op.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef INC_EXTERNAL_ACL_ACL_OP_H_
#define INC_EXTERNAL_ACL_ACL_OP_H_

#include "acl_base.h"
#include "acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aclopHandle aclopHandle;
typedef struct aclopAttr aclopAttr;
typedef struct aclopKernelDesc aclopKernelDesc;

typedef void (*aclDataDeallocator)(void *data, size_t length);

const int ACL_COMPILE_FLAG_BIN_SELECTOR = 1;

typedef enum aclEngineType {
    ACL_ENGINE_SYS,
    ACL_ENGINE_AICORE,
    ACL_ENGINE_VECTOR,
} aclopEngineType;

/**
 * @ingroup AscendCL
 * @brief Set base directory that contains single op models
 *
 * @par Restriction
 * The aclopSetModelDir interface can be called only once in a process.
 * @param  modelDir [IN]   path of the directory
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetModelDir(const char *modelDir);

/**
 * @ingroup AscendCL
 * @brief load single op models from memory
 *
 * @par Restriction
 * The aclopLoad interface can be called more than one times in a process.
 * @param model [IN]        address of single op models
 * @param modelSize [IN]    size of single op models
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopLoad(const void *model,  size_t modelSize);

/**
 * @ingroup AscendCL
 * @brief create data of type aclopAttr
 *
 * @retval pointer to created instance.
 * @retval nullptr if run out of memory
 */
ACL_FUNC_VISIBILITY aclopAttr *aclopCreateAttr();

/**
 * @ingroup AscendCL
 * @brief destroy data of typ aclopAttr
 *
 * @param attr [IN]   pointer to the instance of aclopAttr
 */
ACL_FUNC_VISIBILITY void aclopDestroyAttr(const aclopAttr *attr);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is bool
 *
 * @param attr [IN]        pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 *                         false if attrValue is 0, true otherwise.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is int64_t
 *
 * @param attr [IN]        pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrInt(aclopAttr *attr, const char *attrName, int64_t attrValue);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is float
 *
 * @param attr [IN]        pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrFloat(aclopAttr *attr, const char *attrName, float attrValue);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is string
 *
 * @param attr [IN]        pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrString(aclopAttr *attr, const char *attrName, const char *attrValue);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is list of bools
 *
 * @param attr [IN]        pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values. false if attrValue is 0, true otherwise.
 * @param values [IN]      pointer to values
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrListBool(aclopAttr *attr, const char *attrName, int numValues,
    const uint8_t *values);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is list of ints
 *
 * @param attr [IN]        pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values
 * @param values [IN]      pointer to values
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrListInt(aclopAttr *attr, const char *attrName, int numValues,
    const int64_t *values);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is list of floats
 *
 * @param attr [IN]        pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values
 * @param values [IN]      pointer to values
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrListFloat(aclopAttr *attr, const char *attrName, int numValues,
    const float *values);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is list of strings
 *
 * @param attr [IN]        pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values
 * @param values [IN]      pointer to values
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrListString(aclopAttr *attr, const char *attrName, int numValues,
    const char **values);

/**
 * @ingroup AscendCL
 * @brief set an attribute. the type of the attribute is list of list of ints
 *
 * @param attr [IN]        pointer to the instance of aclopAttr
 * @param attrName [IN]    attribute name
 * @param numLists [IN]    number of lists
 * @param numValues [IN]   pointer to number of values of each list
 * @param values [IN]      pointer to values
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetAttrListListInt(aclopAttr *attr,
                                                     const char *attrName,
                                                     int numLists,
                                                     const int *numValues,
                                                     const int64_t *const values[]);

/**
 * @ingroup AscendCL
 * @brief Load and execute the specified operator asynchronously
 *
 * @par Restriction
 * @li The input and output organization of each operator is different,
 * and the application needs to organize the operator strictly
 * according to the operator input and output parameters when calling.
 * @li When the user calls aclopExecute,
 * the ACL finds the corresponding task according to the optype,
 * the description of the input tesnsor,
 * the description of the output tesnsor, and attr, and issues the execution.
 * @param opType [IN]      type of op
 * @param numInputs [IN]   number of inputs
 * @param inputDesc [IN]   pointer to array of input tensor descriptions
 * @param inputs [IN]      pointer to array of input buffers
 * @param numOutputs [IN]  number of outputs
 * @param outputDesc [IN]  pointer to array of output tensor descriptions
 * @param outputs [OUT]    pointer to array of output buffers
 * @param attr [IN]        pointer to instance of aclopAttr.
 *                         may pass nullptr if the op has no attribute
 * @param stream [IN]      stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopExecute(const char *opType,
                                          int numInputs,
                                          const aclTensorDesc *const inputDesc[],
                                          const aclDataBuffer *const inputs[],
                                          int numOutputs,
                                          const aclTensorDesc *const outputDesc[],
                                          aclDataBuffer *const outputs[],
                                          const aclopAttr *attr,
                                          aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create a instance of aclopHandle.
 *
 * @param opType [IN]      type of op
 * @param numInputs [IN]   number of inputs
 * @param inputDesc [IN]   pointer to array of input tensor descriptions
 * @param numOutputs [IN]  number of outputs
 * @param outputDesc [IN]  pointer to array of output tensor descriptions
 * @param opAttr [IN]      pointer to instance of aclopAttr.
 *                         may pass nullptr if the op has no attribute
 * @param handle [OUT]     pointer to the pointer to the handle
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopCreateHandle(const char *opType,
                                               int numInputs,
                                               const aclTensorDesc *const inputDesc[],
                                               int numOutputs,
                                               const aclTensorDesc *const outputDesc[],
                                               const aclopAttr *opAttr,
                                               aclopHandle **handle);

/**
 * @ingroup AscendCL
 * @brief destroy aclopHandle instance
 *
 * @param handle [IN]   pointer to the instance of aclopHandle
 */
ACL_FUNC_VISIBILITY void aclopDestroyHandle(aclopHandle *handle);

/**
 * @ingroup AscendCL
 * @brief execute an op with the handle.
 *        can save op model matching cost compared with aclopExecute
 *
 * @param handle [IN]      pointer to the instance of aclopHandle.
 *                         The aclopCreateHandle interface has been called
 *                         in advance to create aclopHandle type data.
 * @param numInputs [IN]   number of inputs
 * @param inputs [IN]      pointer to array of input buffers.
 *                         The aclCreateDataBuffer interface has been called
 *                         in advance to create aclDataBuffer type data.
 * @param numOutputs [IN]  number of outputs
 * @param outputs [IN]     pointer to array of output buffers
 * @param stream [IN]      stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclopCreateHandle | aclCreateDataBuffer
 */
ACL_FUNC_VISIBILITY aclError aclopExecWithHandle(aclopHandle *handle,
                                                 int numInputs,
                                                 const aclDataBuffer *const inputs[],
                                                 int numOutputs,
                                                 aclDataBuffer *const outputs[],
                                                 aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief cast data type
 *
 * @param srcDesc [IN]     source tensor desc
 * @param srcBuffer [IN]   source tensor buffer
 * @param dstDesc [IN]     destination tensor desc
 * @param dstBuffer [OUT]  destination tensor buffer
 * @param truncate [IN]    do not truncate if value is 0, truncate otherwise
 * @param stream [IN]      stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopCast(const aclTensorDesc *srcDesc,
                                       const aclDataBuffer *srcBuffer,
                                       const aclTensorDesc *dstDesc,
                                       aclDataBuffer *dstBuffer,
                                       uint8_t truncate,
                                       aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create a handle for casting datatype
 *
 * @param srcDesc [IN]     source tensor desc
 * @param dstDesc [IN]     destination tensor desc
 * @param truncate [IN]    do not truncate if value is 0, truncate otherwise
 * @param handle [IN]     pointer to the pointer to the handle
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopCreateHandleForCast(aclTensorDesc *srcDesc,
                                                      aclTensorDesc *dstDesc,
                                                      uint8_t truncate,
                                                      aclopHandle **handle);


/**
 * @ingroup AscendCL
 * @brief create kernel
 *
 * @param opType [IN]           op type
 * @param kernelId [IN]         kernel id
 * @param kernelName [IN]       kernel name
 * @param binData [IN]          kernel bin data
 * @param binSize [IN]          kernel bin size
 * @param enginetype [IN]       enigne type
 * @param deallocator [IN]      callback function for deallocating bin data,
 *                              null if bin data to be deallocated by caller
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclopCompile
 */
ACL_FUNC_VISIBILITY aclError aclopCreateKernel(const char *opType,
                                               const char *kernelId,
                                               const char *kernelName,
                                               void *binData,
                                               int binSize,
                                               aclopEngineType enginetype,
                                               aclDataDeallocator deallocator);


/**
 * @ingroup AscendCL
 * @brief create kernel
 *
 * @param numInputs [IN]            number of inputs
 * @param inputDesc [IN]            pointer to array of input tensor descriptions
 * @param numOutputs [IN]           number of outputs
 * @param outputDesc [IN]           pointer to array of output tensor descriptions
 * @param opAttr [IN]               pointer to instance of aclopAttr
 * @param aclopKernelDesc [IN]      pointer to instance of aclopKernelDesc
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
typedef aclError (*aclopCompileFunc)(int numInputs,
                                     const aclTensorDesc *const inputDesc[],
                                     int numOutputs,
                                     const aclTensorDesc *const outputDesc[],
                                     const aclopAttr *opAttr,
                                     aclopKernelDesc *aclopKernelDesc);

/**
 * @ingroup AscendCL
 * @brief register compile function
 *
 * @param opType [IN]         op type
 * @param func [IN]           compile function
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclopUnregisterCompileFunc
 */
ACL_FUNC_VISIBILITY aclError aclopRegisterCompileFunc(const char *opType, aclopCompileFunc func);

/**
 * @ingroup AscendCL
 * @brief unregister compile function
 *
 * @param opType [IN]         op type
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopUnregisterCompileFunc(const char *opType);

/**
 * @ingroup AscendCL
 * @brief set kernel args
 *
 * @param kernelDesc [IN]               pointer to instance of aclopKernelDesc
 * @param kernelId [IN]                 kernel id
 * @param blockDim [IN]                 block dim
 * @param args [IN]                     args
 * @param argSize [IN]                  size in bytes of args
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetKernelArgs(aclopKernelDesc *kernelDesc,
                                                const char *kernelId,
                                                uint32_t blockDim,
                                                const void *args,
                                                uint32_t argSize);

/**
 * @ingroup AscendCL
 * @brief set workspace sizes
 *
 * @param kernelDesc [IN]               pointer to instance of aclopKernelDesc
 * @param numWorkspaces [IN]            number of workspaces
 * @param workspaceSizes [IN]           pointer to array of sizes of workspaces
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetKernelWorkspaceSizes(aclopKernelDesc *kernelDesc, int numWorkspaces,
    size_t *workspaceSizes);

/**
 * @ingroup AscendCL
 * @brief compile op with dynamic shape
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN]       pointer to array of output tensor descriptions
 * @param attr [IN]             pointer to instance of aclopAttr.
 *                              may pass nullptr if the op has no attribute
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopUpdateParams(const char *opType,
                                               int numInputs,
                                               const aclTensorDesc *const inputDesc[],
                                               int numOutputs,
                                               const aclTensorDesc *const outputDesc[],
                                               const aclopAttr *attr);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_OP_H_
