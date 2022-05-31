/**
* @file acl_mdl.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_MODEL_H_
#define INC_EXTERNAL_ACL_ACL_MODEL_H_

#include <stddef.h>
#include <stdint.h>

#include "acl_base.h"
#include "acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_MAX_DIM_CNT          128
#define ACL_MAX_TENSOR_NAME_LEN  128
#define ACL_MAX_BATCH_NUM        128
#define ACL_MAX_HW_NUM           128
#define ACL_MAX_SHAPE_COUNT      128
#define ACL_INVALID_NODE_INDEX   0xFFFFFFFF

#define ACL_MDL_LOAD_FROM_FILE            1
#define ACL_MDL_LOAD_FROM_FILE_WITH_MEM   2
#define ACL_MDL_LOAD_FROM_MEM             3
#define ACL_MDL_LOAD_FROM_MEM_WITH_MEM    4
#define ACL_MDL_LOAD_FROM_FILE_WITH_Q     5
#define ACL_MDL_LOAD_FROM_MEM_WITH_Q      6

#define ACL_DYNAMIC_TENSOR_NAME "ascend_mbatch_shape_data"
#define ACL_DYNAMIC_AIPP_NAME "ascend_dynamic_aipp_data"
#define ACL_ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES "_datadump_original_op_names"

typedef struct aclmdlDataset aclmdlDataset;
typedef struct aclmdlDesc aclmdlDesc;
typedef struct aclmdlAIPP aclmdlAIPP;
typedef struct aclAippExtendInfo aclAippExtendInfo;
typedef struct aclmdlConfigHandle aclmdlConfigHandle;

typedef enum {
    ACL_YUV420SP_U8 = 1,
    ACL_XRGB8888_U8,
    ACL_RGB888_U8,
    ACL_YUV400_U8,
    ACL_NC1HWC0DI_FP16,
    ACL_NC1HWC0DI_S8,
    ACL_ARGB8888_U8,
    ACL_YUYV_U8,
    ACL_YUV422SP_U8,
    ACL_AYUV444_U8,
    ACL_RAW10,
    ACL_RAW12,
    ACL_RAW16,
    ACL_RAW24,
    ACL_AIPP_RESERVED = 0xffff,
} aclAippInputFormat;

typedef enum {
    ACL_MDL_PRIORITY_INT32 = 0,
    ACL_MDL_LOAD_TYPE_SIZET,
    ACL_MDL_PATH_PTR, /**< pointer to model load path with deep copy */
    ACL_MDL_MEM_ADDR_PTR, /**< pointer to model memory with shallow copy */
    ACL_MDL_MEM_SIZET,
    ACL_MDL_WEIGHT_ADDR_PTR, /**< pointer to weight memory of model with shallow copy */
    ACL_MDL_WEIGHT_SIZET,
    ACL_MDL_WORKSPACE_ADDR_PTR, /**< pointer to worksapce memory of model with shallow copy */
    ACL_MDL_WORKSPACE_SIZET,
    ACL_MDL_INPUTQ_NUM_SIZET,
    ACL_MDL_INPUTQ_ADDR_PTR, /**< pointer to inputQ with shallow copy */
    ACL_MDL_OUTPUTQ_NUM_SIZET,
    ACL_MDL_OUTPUTQ_ADDR_PTR /**< pointer to outputQ with shallow copy */
} aclmdlConfigAttr;

typedef enum {
    ACL_DATA_WITHOUT_AIPP = 0,
    ACL_DATA_WITH_STATIC_AIPP,
    ACL_DATA_WITH_DYNAMIC_AIPP,
    ACL_DYNAMIC_AIPP_NODE
} aclmdlInputAippType;

typedef struct aclmdlIODims {
    char name[ACL_MAX_TENSOR_NAME_LEN]; /**< tensor name */
    size_t dimCount;  /**< dim array count */
    int64_t dims[ACL_MAX_DIM_CNT]; /**< dim data array */
} aclmdlIODims;

typedef struct aclAippDims {
    aclmdlIODims srcDims; /**< input dims before model transform */
    size_t srcSize; /**< input size before model transform */
    aclmdlIODims aippOutdims; /**< aipp output dims */
    size_t aippOutSize; /**< aipp output size */
} aclAippDims;

typedef struct aclmdlBatch {
    size_t batchCount; /**< batch array count */
    uint64_t batch[ACL_MAX_BATCH_NUM]; /**< batch data array */
} aclmdlBatch;

typedef struct aclmdlHW {
    size_t hwCount; /**< height&width array count */
    uint64_t hw[ACL_MAX_HW_NUM][2]; /**< height&width data array */
} aclmdlHW;

typedef struct aclAippInfo {
    aclAippInputFormat inputFormat;
    int32_t srcImageSizeW;
    int32_t srcImageSizeH;
    int8_t cropSwitch;
    int32_t loadStartPosW;
    int32_t loadStartPosH;
    int32_t cropSizeW;
    int32_t cropSizeH;
    int8_t resizeSwitch;
    int32_t resizeOutputW;
    int32_t resizeOutputH;
    int8_t paddingSwitch;
    int32_t leftPaddingSize;
    int32_t rightPaddingSize;
    int32_t topPaddingSize;
    int32_t bottomPaddingSize;
    int8_t cscSwitch;
    int8_t rbuvSwapSwitch;
    int8_t axSwapSwitch;
    int8_t singleLineMode;
    int32_t matrixR0C0;
    int32_t matrixR0C1;
    int32_t matrixR0C2;
    int32_t matrixR1C0;
    int32_t matrixR1C1;
    int32_t matrixR1C2;
    int32_t matrixR2C0;
    int32_t matrixR2C1;
    int32_t matrixR2C2;
    int32_t outputBias0;
    int32_t outputBias1;
    int32_t outputBias2;
    int32_t inputBias0;
    int32_t inputBias1;
    int32_t inputBias2;
    int32_t meanChn0;
    int32_t meanChn1;
    int32_t meanChn2;
    int32_t meanChn3;
    float minChn0;
    float minChn1;
    float minChn2;
    float minChn3;
    float varReciChn0;
    float varReciChn1;
    float varReciChn2;
    float varReciChn3;
    aclFormat srcFormat;
    aclDataType srcDatatype;
    size_t srcDimNum;
    size_t shapeCount;
    aclAippDims outDims[ACL_MAX_SHAPE_COUNT];
    aclAippExtendInfo *aippExtend; /**< reserved parameters, current version needs to be null */
} aclAippInfo;

/**
 * @ingroup AscendCL
 * @brief Create data of type aclmdlDesc
 *
 * @retval the aclmdlDesc pointer
 */
ACL_FUNC_VISIBILITY aclmdlDesc *aclmdlCreateDesc();

/**
 * @ingroup AscendCL
 * @brief destroy data of type aclmdlDesc
 *
 * @param modelDesc [IN]   Pointer to almdldlDesc to be destroyed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc);

/**
 * @ingroup AscendCL
 * @brief Get aclmdlDesc data of the model according to the model ID
 *
 * @param  modelDesc [OUT]   aclmdlDesc pointer
 * @param  modelId [IN]      model id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId);

/**
 * @ingroup AscendCL
 * @brief Get the number of the inputs of
 *        the model according to data of aclmdlDesc
 *
 * @param  modelDesc [IN]   aclmdlDesc pointer
 *
 * @retval input size with aclmdlDesc
 */
ACL_FUNC_VISIBILITY size_t aclmdlGetNumInputs(aclmdlDesc *modelDesc);

/**
 * @ingroup AscendCL
 * @brief Get the number of the output of
 *        the model according to data of aclmdlDesc
 *
 * @param  modelDesc [IN]   aclmdlDesc pointer
 *
 * @retval output size with aclmdlDesc
 */
ACL_FUNC_VISIBILITY size_t aclmdlGetNumOutputs(aclmdlDesc *modelDesc);

/**
 * @ingroup AscendCL
 * @brief Get the size of the specified input according to
 *        the data of type aclmdlDesc
 *
 * @param  modelDesc [IN]  aclmdlDesc pointer
 * @param  index [IN] the size of the number of inputs to be obtained,
 *         the index value starts from 0
 *
 * @retval Specify the size of the input
 */
ACL_FUNC_VISIBILITY size_t aclmdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief Get the size of the specified output according to
 *        the data of type aclmdlDesc
 *
 * @param modelDesc [IN]   aclmdlDesc pointer
 * @param index [IN]  the size of the number of outputs to be obtained,
 *        the index value starts from 0
 *
 * @retval Specify the size of the output
 */
ACL_FUNC_VISIBILITY size_t aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief Create data of type aclmdlDataset
 *
 * @retval the aclmdlDataset pointer
 */
ACL_FUNC_VISIBILITY aclmdlDataset *aclmdlCreateDataset();

/**
 * @ingroup AscendCL
 * @brief destroy data of type aclmdlDataset
 *
 * @param  dataset [IN]  Pointer to aclmdlDataset to be destroyed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlDestroyDataset(const aclmdlDataset *dataset);

/**
 * @ingroup AscendCL
 * @brief Add aclDataBuffer to aclmdlDataset
 *
 * @param dataset [OUT]    aclmdlDataset address of aclDataBuffer to be added
 * @param dataBuffer [IN]  aclDataBuffer address to be added
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief Set aclTensorDesc to aclmdlDataset
 *
 * @param dataset [OUT]    aclmdlDataset address of aclDataBuffer to be added
 * @param tensorDesc [IN]  aclTensorDesc address to be added
 * @param index [IN]       index of tensorDesc which to be added
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetDatasetTensorDesc(aclmdlDataset *dataset,
                                                        aclTensorDesc *tensorDesc,
                                                        size_t index);

/**
 * @ingroup AscendCL
 * @brief Get aclTensorDesc from aclmdlDataset
 *
 * @param dataset [IN]    aclmdlDataset pointer;
 * @param index [IN]      index of tensorDesc
 *
 * @retval Get address of aclTensorDesc when executed successfully.
 * @retval Failure return NULL
 */
ACL_FUNC_VISIBILITY aclTensorDesc *aclmdlGetDatasetTensorDesc(const aclmdlDataset *dataset, size_t index);

/**
 * @ingroup AscendCL
 * @brief Get the number of aclDataBuffer in aclmdlDataset
 *
 * @param dataset [IN]   aclmdlDataset pointer
 *
 * @retval the number of aclDataBuffer
 */
ACL_FUNC_VISIBILITY size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset *dataset);

/**
 * @ingroup AscendCL
 * @brief Get the aclDataBuffer in aclmdlDataset by index
 *
 * @param dataset [IN]   aclmdlDataset pointer
 * @param index [IN]     the index of aclDataBuffer
 *
 * @retval Get successfully, return the address of aclDataBuffer
 * @retval Failure return NULL
 */
ACL_FUNC_VISIBILITY aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index);

/**
 * @ingroup AscendCL
 * @brief Load offline model data from files
 * and manage memory internally by the system
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations
 *
 * @param modelPath [IN]   Storage path for offline model files
 * @param modelId [OUT]    Model ID generated after
 *        the system finishes loading the model
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromFile(const char *modelPath, uint32_t *modelId);

/**
 * @ingroup AscendCL
 * @brief Load offline model data from memory and manage the memory of
 * model running internally by the system
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations
 *
 * @param model [IN]      Model data stored in memory
 * @param modelSize [IN]  model data size
 * @param modelId [OUT]   Model ID generated after
 *        the system finishes loading the model
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromMem(const void *model,  size_t modelSize,
                                               uint32_t *modelId);

/**
 * @ingroup AscendCL
 * @brief Load offline model data from a file,
 * and the user manages the memory of the model run by itself
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations.
 * @param modelPath [IN]   Storage path for offline model files
 * @param modelId [OUT]    Model ID generated after finishes loading the model
 * @param workPtr [IN]     A pointer to the working memory
 *                         required by the model on the Device,can be null
 * @param workSize [IN]    The amount of working memory required by the model
 * @param weightPtr [IN]   Pointer to model weight memory on Device
 * @param weightSize [IN]  The amount of weight memory required by the model
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromFileWithMem(const char *modelPath,
                                                       uint32_t *modelId, void *workPtr, size_t workSize,
                                                       void *weightPtr, size_t weightSize);

/**
 * @ingroup AscendCL
 * @brief Load offline model data from memory,
 * and the user can manage the memory of model running
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations
 * @param model [IN]      Model data stored in memory
 * @param modelSize [IN]  model data size
 * @param modelId [OUT]   Model ID generated after finishes loading the model
 * @param workPtr [IN]    A pointer to the working memory
 *                        required by the model on the Device,can be null
 * @param workSize [IN]   work memory size
 * @param weightPtr [IN]  Pointer to model weight memory on Device,can be null
 * @param weightSize [IN] The amount of weight memory required by the model
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromMemWithMem(const void *model, size_t modelSize,
                                                      uint32_t *modelId, void *workPtr, size_t workSize,
                                                      void *weightPtr, size_t weightSize);

/**
 * @ingroup AscendCL
 * @brief load model from file with async queue
 *
 * @param modelPath  [IN] model path
 * @param modelId [OUT]   return model id if load success
 * @param inputQ [IN]     input queue pointer
 * @param inputQNum [IN]  input queue num
 * @param outputQ [IN]    output queue pointer
 * @param outputQNum [IN] output queue num
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromFileWithQ(const char *modelPath, uint32_t *modelId, const uint32_t *inputQ,
                                                     size_t inputQNum, const uint32_t *outputQ, size_t outputQNum);

/**
 * @ingroup AscendCL
 * @brief load model from memory with async queue
 *
 * @param model [IN]      model memory which user manages
 * @param modelSize [IN]  model size
 * @param modelId [OUT]   return model id if load success
 * @param inputQ [IN]     input queue pointer
 * @param inputQNum [IN]  input queue num
 * @param outputQ [IN]    output queue pointer
 * @param outputQNum [IN] output queue num
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlLoadFromMemWithQ(const void *model, size_t modelSize, uint32_t *modelId,
                                                    const uint32_t *inputQ, size_t inputQNum,
                                                    const uint32_t *outputQ, size_t outputQNum);

/**
 * @ingroup AscendCL
 * @brief Execute model synchronous inference until the inference result is returned
 *
 * @param  modelId [IN]   ID of the model to perform inference
 * @param  input [IN]     Input data for model inference
 * @param  output [OUT]   Output data for model inference
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output);

/**
 * @ingroup AscendCL
 * @brief Execute model asynchronous inference until the inference result is returned
 *
 * @param  modelId [IN]   ID of the model to perform inference
 * @param  input [IN]     Input data for model inference
 * @param  output [OUT]   Output data for model inference
 * @param  stream [IN]    stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem
 */
ACL_FUNC_VISIBILITY aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset *input,
                                                aclmdlDataset *output, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief unload model with model id
 *
 * @param  modelId [IN]   model id to be unloaded
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlUnload(uint32_t modelId);

/**
 * @ingroup AscendCL
 * @brief Get the weight memory size and working memory size
 * required for model execution according to the model file
 *
 * @param  fileName [IN]     Model path to get memory information
 * @param  workSize [OUT]    The amount of working memory for model executed
 * @param  weightSize [OUT]  The amount of weight memory for model executed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlQuerySize(const char *fileName, size_t *workSize, size_t *weightSize);

/**
 * @ingroup AscendCL
 * @brief Obtain the weights required for
 * model execution according to the model data in memory
 *
 * @par Restriction
 * The execution and weight memory is Device memory,
 * and requires user application and release.
 * @param  model [IN]        model memory which user manages
 * @param  modelSize [IN]    model data size
 * @param  workSize [OUT]    The amount of working memory for model executed
 * @param  weightSize [OUT]  The amount of weight memory for model executed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlQuerySizeFromMem(const void *model, size_t modelSize, size_t *workSize,
                                                    size_t *weightSize);

/**
 * @ingroup AscendCL
 * @brief In dynamic batch scenarios,
 * it is used to set the number of images processed
 * at one time during model inference
 *
 * @param  modelId [IN]     model id
 * @param  dataset [IN|OUT] data for model inference
 * @param  index [IN]       index of dynamic tensor
 * @param  batchSize [IN]   Number of images processed at a time during model
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetDynamicBatchSize(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                       uint64_t batchSize);

/**
 * @ingroup AscendCL
 * @brief Sets the H and W of the specified input of the model
 *
 * @param  modelId [IN]     model id
 * @param  dataset [IN|OUT] data for model inference
 * @param  index [IN]       index of dynamic tensor
 * @param  height [IN]      model height
 * @param  width [IN]       model width
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetDynamicHWSize(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                    uint64_t height, uint64_t width);

/**
 * @ingroup AscendCL
 * @brief Sets the dynamic dims of the specified input of the model
 *
 * @param  modelId [IN]     model id
 * @param  dataset [IN|OUT] data for model inference
 * @param  index [IN]       index of dynamic dims
 * @param  dims [IN]        value of dynamic dims
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetInputDynamicDims(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                       const aclmdlIODims *dims);

/**
 * @ingroup AscendCL
 * @brief get input dims info
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  input tensor index
 * @param dims [OUT]  dims info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlGetInputDimsV2
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

/**
 * @ingroup AscendCL
 * @brief get input dims info(version 2), especially for static aipp
 * it is the same with aclmdlGetInputDims while model without static aipp
 *
 * @param modelDesc [IN] model description
 * @param index [IN]     input tensor index
 * @param dims [OUT]     dims info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlGetInputDims
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetInputDimsV2(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

/**
 * @ingroup AscendCL
 * @brief get output dims info
 *
 * @param modelDesc [IN] model description
 * @param index [IN]     output tensor index
 * @param dims [OUT]     dims info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

/**
 * @ingroup AscendCL
 * @brief get current output dims info
 *
 * @par Function
 * The following use cases are supported:
 * @li Get current output shape when model is dynamic and
 * dynamic shape info is set
 * @li Get max output shape when model is dynamic and
 * dynamic shape info is not set
 * @li Get actual output shape when model is static
 *
 * @param modelDesc [IN] model description
 * @param index [IN]     output tensor index
 * @param dims [OUT]     dims info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetCurOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

/**
 * @ingroup AscendCL
 * @brief get attr value by op name
 *
 * @param modelDesc [IN]   model description
 * @param opName [IN]      op name
 * @param attr [IN]        attr name
 * 
 * @retval the attr value
 */
ACL_FUNC_VISIBILITY const char *aclmdlGetOpAttr(aclmdlDesc *modelDesc, const char *opName, const char *attr);

/**
 * @ingroup AscendCL
 * @brief get input name by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      intput tensor index
 *
 * @retval input tensor name,the same life cycle with modelDesc
 */
ACL_FUNC_VISIBILITY const char *aclmdlGetInputNameByIndex(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get output name by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      output tensor index
 *
 * @retval output tensor name,the same life cycle with modelDesc
 */
ACL_FUNC_VISIBILITY const char *aclmdlGetOutputNameByIndex(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get input format by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      intput tensor index
 *
 * @retval input tensor format
 */
ACL_FUNC_VISIBILITY aclFormat aclmdlGetInputFormat(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get output format by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      output tensor index
 *
 * @retval output tensor format
 */
ACL_FUNC_VISIBILITY aclFormat aclmdlGetOutputFormat(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get input data type by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  intput tensor index
 *
 * @retval input tensor data type
 */
ACL_FUNC_VISIBILITY aclDataType aclmdlGetInputDataType(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get output data type by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  output tensor index
 *
 * @retval output tensor data type
 */
ACL_FUNC_VISIBILITY aclDataType aclmdlGetOutputDataType(const aclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get input tensor index by name
 *
 * @param modelDesc [IN]  model description
 * @param name [IN]    intput tensor name
 * @param index [OUT]  intput tensor index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetInputIndexByName(const aclmdlDesc *modelDesc, const char *name, size_t *index);

/**
 * @ingroup AscendCL
 * @brief get output tensor index by name
 *
 * @param modelDesc [IN]  model description
 * @param name [IN]  output tensor name
 * @param index [OUT]  output tensor index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetOutputIndexByName(const aclmdlDesc *modelDesc, const char *name, size_t *index);

/**
 * @ingroup AscendCL
 * @brief get dynamic batch info
 *
 * @param modelDesc [IN]  model description
 * @param batch [OUT]  dynamic batch info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetDynamicBatch(const aclmdlDesc *modelDesc, aclmdlBatch *batch);

/**
 * @ingroup AscendCL
 * @brief get dynamic height&width info
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  input tensor index
 * @param hw [OUT]  dynamic height&width info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetDynamicHW(const aclmdlDesc *modelDesc, size_t index, aclmdlHW *hw);

/**
 * @ingroup AscendCL
 * @brief get dynamic gear count
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  unused, must be -1
 * @param gearCount [OUT]  dynamic gear count
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetInputDynamicGearCount(const aclmdlDesc *modelDesc, size_t index,
                                                            size_t *gearCount);

/**
 * @ingroup AscendCL
 * @brief get dynamic dims info
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  unused, must be -1
 * @param dims [OUT]  value of dynamic dims
 * @param gearCount [IN]  dynamic gear count
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlGetInputDynamicDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims,
                                                       size_t gearCount);

/**
 * @ingroup AscendCL
 * @brief Create data of type aclmdlAIPP
 *
 * @param batchSize [IN]    batchsizes of model
 *
 * @retval the aclmdlAIPP pointer
 */
ACL_FUNC_VISIBILITY aclmdlAIPP *aclmdlCreateAIPP(uint64_t batchSize);

/**
 * @ingroup AscendCL
 * @brief destroy data of type aclmdlAIPP
 *
 * @param aippParmsSet [IN]    Pointer for aclmdlAIPP to be destroyed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlDestroyAIPP(const aclmdlAIPP *aippParmsSet);

/**
 * @ingroup AscendCL
 * @brief set InputFormat of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param inputFormat [IN]    The inputFormat of aipp
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPInputFormat(aclmdlAIPP *aippParmsSet, aclAippInputFormat inputFormat);

/**
 * @ingroup AscendCL
 * @brief set cscParms of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]    Pointer for aclmdlAIPP
 * @param csc_switch [IN]       Csc switch
 * @param cscMatrixR0C0 [IN]    Csc_matrix_r0_c0
 * @param cscMatrixR0C1 [IN]    Csc_matrix_r0_c1
 * @param cscMatrixR0C2 [IN]    Csc_matrix_r0_c2
 * @param cscMatrixR1C0 [IN]    Csc_matrix_r1_c0
 * @param cscMatrixR1C1 [IN]    Csc_matrix_r1_c1
 * @param cscMatrixR1C2 [IN]    Csc_matrix_r1_c2
 * @param cscMatrixR2C0 [IN]    Csc_matrix_r2_c0
 * @param cscMatrixR2C1 [IN]    Csc_matrix_r2_c1
 * @param cscMatrixR2C2 [IN]    Csc_matrix_r2_c2
 * @param cscOutputBiasR0 [IN]  Output Bias for RGB to YUV, element of row 0, unsigned number
 * @param cscOutputBiasR1 [IN]  Output Bias for RGB to YUV, element of row 1, unsigned number
 * @param cscOutputBiasR2 [IN]  Output Bias for RGB to YUV, element of row 2, unsigned number
 * @param cscInputBiasR0 [IN]   Input Bias for YUV to RGB, element of row 0, unsigned number
 * @param cscInputBiasR1 [IN]   Input Bias for YUV to RGB, element of row 1, unsigned number
 * @param cscInputBiasR2 [IN]   Input Bias for YUV to RGB, element of row 2, unsigned number
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPCscParams(aclmdlAIPP *aippParmsSet, int8_t cscSwitch,
                                                    int16_t cscMatrixR0C0, int16_t cscMatrixR0C1, int16_t cscMatrixR0C2,
                                                    int16_t cscMatrixR1C0, int16_t cscMatrixR1C1, int16_t cscMatrixR1C2,
                                                    int16_t cscMatrixR2C0, int16_t cscMatrixR2C1, int16_t cscMatrixR2C2,
                                                    uint8_t cscOutputBiasR0, uint8_t cscOutputBiasR1,
                                                    uint8_t cscOutputBiasR2, uint8_t cscInputBiasR0,
                                                    uint8_t cscInputBiasR1, uint8_t cscInputBiasR2);

/**
 * @ingroup AscendCL
 * @brief set rb/ub swap switch of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param rbuvSwapSwitch [IN] rb/ub swap switch
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPRbuvSwapSwitch(aclmdlAIPP *aippParmsSet, int8_t rbuvSwapSwitch);

/**
 * @ingroup AscendCL
 * @brief set RGBA->ARGB, YUVA->AYUV swap switch of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param axSwapSwitch [IN]   RGBA->ARGB, YUVA->AYUV swap switch
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPAxSwapSwitch(aclmdlAIPP *aippParmsSet, int8_t axSwapSwitch);

/**
 * @ingroup AscendCL
 * @brief set source image of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param srcImageSizeW [IN]  Source image width
 * @param srcImageSizeH [IN]  Source image height
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPSrcImageSize(aclmdlAIPP *aippParmsSet, int32_t srcImageSizeW,
                                                       int32_t srcImageSizeH);

/**
 * @ingroup AscendCL
 * @brief set resize switch of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param scfSwitch [IN]      Resize switch
 * @param scfInputSizeW [IN]  Input width of scf
 * @param scfInputSizeH [IN]  Input height of scf
 * @param scfOutputSizeW [IN] Output width of scf
 * @param scfOutputSizeH [IN] Output height of scf
 * @param batchIndex [IN]     Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPScfParams(aclmdlAIPP *aippParmsSet,
                                                    int8_t scfSwitch,
                                                    int32_t scfInputSizeW,
                                                    int32_t scfInputSizeH,
                                                    int32_t scfOutputSizeW,
                                                    int32_t scfOutputSizeH,
                                                    uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set cropParams of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]  Pointer for aclmdlAIPP
 * @param cropSwitch [IN]     Crop switch
 * @param cropStartPosW [IN]  The start horizontal position of cropping
 * @param cropStartPosH [IN]  The start vertical position of cropping
 * @param cropSizeW [IN]      Crop width
 * @param cropSizeH [IN]      Crop height
 * @param batchIndex [IN]     Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPCropParams(aclmdlAIPP *aippParmsSet,
                                                     int8_t cropSwitch,
                                                     int32_t cropStartPosW,
                                                     int32_t cropStartPosH,
                                                     int32_t cropSizeW,
                                                     int32_t cropSizeH,
                                                     uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set paddingParams of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]      Pointer for aclmdlAIPP
 * @param paddingSwitch [IN]      Padding switch
 * @param paddingSizeTop [IN]     Top padding size
 * @param paddingSizeBottom [IN]  Bottom padding size
 * @param paddingSizeLeft [IN]    Left padding size
 * @param paddingSizeRight [IN]   Right padding size
 * @param batchIndex [IN]         Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPPaddingParams(aclmdlAIPP *aippParmsSet, int8_t paddingSwitch,
                                                        int32_t paddingSizeTop, int32_t paddingSizeBottom,
                                                        int32_t paddingSizeLeft, int32_t paddingSizeRight,
                                                        uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set DtcPixelMean of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]      Pointer for aclmdlAIPP
 * @param dtcPixelMeanChn0 [IN]   Mean value of channel 0
 * @param dtcPixelMeanChn1 [IN]   Mean value of channel 1
 * @param dtcPixelMeanChn2 [IN]   Mean value of channel 2
 * @param dtcPixelMeanChn3 [IN]   Mean value of channel 3
 * @param batchIndex [IN]         Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPDtcPixelMean(aclmdlAIPP *aippParmsSet,
                                                       int16_t dtcPixelMeanChn0,
                                                       int16_t dtcPixelMeanChn1,
                                                       int16_t dtcPixelMeanChn2,
                                                       int16_t dtcPixelMeanChn3,
                                                       uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set DtcPixelMin of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]    Pointer for aclmdlAIPP
 * @param dtcPixelMinChn0 [IN]  Min value of channel 0
 * @param dtcPixelMinChn1 [IN]  Min value of channel 1
 * @param dtcPixelMinChn2 [IN]  Min value of channel 2
 * @param dtcPixelMinChn3 [IN]  Min value of channel 3
 * @param batchIndex [IN]       Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPDtcPixelMin(aclmdlAIPP *aippParmsSet,
                                                      float dtcPixelMinChn0,
                                                      float dtcPixelMinChn1,
                                                      float dtcPixelMinChn2,
                                                      float dtcPixelMinChn3,
                                                      uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set PixelVarReci of type aclmdlAIPP
 *
 * @param aippParmsSet [OUT]       Pointer for aclmdlAIPP
 * @param dtcPixelVarReciChn0 [IN] sfr_dtc_pixel_variance_reci_ch0
 * @param dtcPixelVarReciChn1 [IN] sfr_dtc_pixel_variance_reci_ch1
 * @param dtcPixelVarReciChn2 [IN] sfr_dtc_pixel_variance_reci_ch2
 * @param dtcPixelVarReciChn3 [IN] sfr_dtc_pixel_variance_reci_ch3
 * @param batchIndex [IN]          Batch parameter index
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPPixelVarReci(aclmdlAIPP *aippParmsSet,
                                                       float dtcPixelVarReciChn0,
                                                       float dtcPixelVarReciChn1,
                                                       float dtcPixelVarReciChn2,
                                                       float dtcPixelVarReciChn3,
                                                       uint64_t batchIndex);

/**
 * @ingroup AscendCL
 * @brief set aipp parameters to model
 *
 * @param modelId [IN]        model id
 * @param dataset [IN]        Pointer of dataset
 * @param index [IN]          index of input for aipp data(ACL_DYNAMIC_AIPP_NODE)
 * @param aippParmsSet [IN]   Pointer for aclmdlAIPP
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName | aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetInputAIPP(uint32_t modelId,
                                                aclmdlDataset *dataset,
                                                size_t index,
                                                const aclmdlAIPP *aippParmsSet);

/**
 * @ingroup AscendCL
 * @brief set aipp parameters to model
 *
 * @param modelId [IN]        model id
 * @param dataset [IN]        Pointer of dataset
 * @param index [IN]          index of input for data which linked dynamic aipp(ACL_DATA_WITH_DYNAMIC_AIPP)
 * @param aippParmsSet [IN]   Pointer for aclmdlAIPP
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName | aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPByInputIndex(uint32_t modelId,
                                                       aclmdlDataset *dataset,
                                                       size_t index,
                                                       const aclmdlAIPP *aippParmsSet);

/**
 * @ingroup AscendCL
 * @brief get input aipp type
 *
 * @param modelId [IN]        model id
 * @param index [IN]          index of input
 * @param type [OUT]          aipp type for input.refrer to aclmdlInputAippType(enum)
 * @param dynamicAttachedDataIndex [OUT]     index for dynamic attached data(ACL_DYNAMIC_AIPP_NODE)
 *        valid when type is ACL_DATA_WITH_DYNAMIC_AIPP, invalid value is ACL_INVALID_NODE_INDEX
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName | aclmdlCreateAIPP
*/
ACL_FUNC_VISIBILITY aclError aclmdlGetAippType(uint32_t modelId,
                                               size_t index,
                                               aclmdlInputAippType *type,
                                               size_t *dynamicAttachedDataIndex);

/**
 * @ingroup AscendCL
 * @brief get static aipp parameters from model
 *
 * @param modelId [IN]        model id
 * @param index [IN]          index of tensor
 * @param aippInfo [OUT]      Pointer for static aipp info
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_MODEL_AIPP_NOT_EXIST The tensor of index is not configured with aipp
 * @retval OtherValues Failure
 *
 * @see aclmdlLoadFromFile | aclmdlLoadFromMem | aclmdlLoadFromFileWithMem |
 * aclmdlLoadFromMemWithMem | aclmdlGetInputIndexByName
*/
ACL_FUNC_VISIBILITY aclError aclmdlGetFirstAippInfo(uint32_t modelId, size_t index, aclAippInfo *aippInfo);

/**
 * @ingroup AscendCL
 * @brief get op description info
 *
 * @param deviceId [IN]       device id
 * @param streamId [IN]       stream id
 * @param taskId [IN]         task id
 * @param opName [OUT]        pointer to op name
 * @param opNameLen [IN]      the length of op name
 * @param inputDesc [OUT]     pointer to input description
 * @param numInputs [OUT]     the number of input tensor
 * @param outputDesc [OUT]    pointer to output description
 * @param numOutputs [OUT]    the number of output tensor
 *
 * @retval ACL_SUCCESS The function is successfully executed
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlCreateAndGetOpDesc(uint32_t deviceId, uint32_t streamId,
    uint32_t taskId, char *opName, size_t opNameLen, aclTensorDesc **inputDesc, size_t *numInputs,
    aclTensorDesc **outputDesc, size_t *numOutputs);

/**
 * @ingroup AscendCL
 * @brief init dump
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlInitDump();

/**
 * @ingroup AscendCL
 * @brief set param of dump
 *
 * @param dumpCfgPath [IN]   the path of dump config
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetDump(const char *dumpCfgPath);

/**
 * @ingroup AscendCL
 * @brief finalize dump.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlFinalizeDump();

/**
 * @ingroup AscendCL
 * @brief load model with config
 *
 * @param handle [IN]    pointer to model config handle
 * @param modelId [OUT]  pointer to model id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlLoadWithConfig(const aclmdlConfigHandle *handle, uint32_t *modelId);

/**
 * @ingroup AscendCL
 * @brief create model config handle of type aclmdlConfigHandle
 *
 * @retval the aclmdlConfigHandle pointer
 *
 * @see aclmdlDestroyConfigHandle
*/
ACL_FUNC_VISIBILITY aclmdlConfigHandle *aclmdlCreateConfigHandle();

/**
 * @ingroup AscendCL
 * @brief destroy data of type aclmdlConfigHandle
 *
 * @param handle [IN]   pointer to model config handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclmdlCreateConfigHandle
 */
ACL_FUNC_VISIBILITY aclError aclmdlDestroyConfigHandle(aclmdlConfigHandle *handle);

/**
 * @ingroup AscendCL
 * @brief set config for model load
 *
 * @param handle [OUT]    pointer to model config handle
 * @param attr [IN]       config attr in model config handle to be set
 * @param attrValue [IN]  pointer to model config value
 * @param valueSize [IN]  memory size of attrValue
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlSetConfigOpt(aclmdlConfigHandle *handle, aclmdlConfigAttr attr,
    const void *attrValue, size_t valueSize);

/**
 * @ingroup AscendCL
 * @brief get real tensor name from modelDesc
 *
 * @param modelDesc [IN]  pointer to modelDesc
 * @param name [IN]       tensor name
 *
 * @retval the pointer of real tensor name
 * @retval Failure return NULL
 */
ACL_FUNC_VISIBILITY const char *aclmdlGetTensorRealName(const aclmdlDesc *modelDesc, const char *name);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_MODEL_H_
