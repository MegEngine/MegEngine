/* *
 * @file acl_fv.h
 *
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef INC_EXTERNAL_ACL_OPS_ACL_RETR_H_
#define INC_EXTERNAL_ACL_OPS_ACL_RETR_H_

#include "../acl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aclfvFeatureInfo aclfvFeatureInfo;
typedef struct aclfvRepoRange aclfvRepoRange;
typedef struct aclfvQueryTable aclfvQueryTable;
typedef struct aclfvSearchInput aclfvSearchInput;
typedef struct aclfvSearchResult aclfvSearchResult;

// search operation type
enum aclfvSearchType {
    SEARCH_1_N, // 1:N operation type
    SEARCH_N_M  // N:M operation type
};

/* *
 * @ingroup AscendCL
 * @brief Create fv feature info.
 * @param id0 [IN]: The first level library id0
 * @param id1 [IN]: Secondary library id1
 * @param offset [IN]: The offset of the first feature in the library
 * @param featureLen [IN]: Single feature length
 * @param featureCount [IN]: Single feature count
 * @param featureData [IN/OUT]: Feature value list
 * @param featureDataLen [IN]: Feature value list length
 * @retval null for failed.
 * @retval OtherValues success.
 */
ACL_FUNC_VISIBILITY aclfvFeatureInfo *aclfvCreateFeatureInfo(uint32_t id0, uint32_t id1, uint32_t offset,
    uint32_t featureLen, uint32_t featureCount, uint8_t *featureData, uint32_t featureDataLen);

/* *
 * @ingroup AscendCL
 * @brief Destroy fv feature info.
 *
 * @par Function
 * Can only destroy fv feature info information created
 * through aclfvCreateFeatureInfo interface.
 * @param featureInfo [IN/OUT]     fv feature info.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclfvCreateFeatureInfo
 */
ACL_FUNC_VISIBILITY aclError aclfvDestroyFeatureInfo(aclfvFeatureInfo *featureInfo);

/* *
 * @ingroup AscendCL
 * @brief Create fv repo range.
 * @param id0Min [IN]: id0 start value
 * @param id0Min [IN]: id0 max
 * @param id1Min [IN]: id0 start value
 * @param id1Max [IN]: id1 max
 * @retval null for failed. OtherValues success
 */
ACL_FUNC_VISIBILITY aclfvRepoRange *aclfvCreateRepoRange(uint32_t id0Min, uint32_t id0Max, uint32_t id1Min,
    uint32_t id1Max);

/* *
 * @ingroup AscendCL
 * @brief Destroy fv repo range.
 *
 * @par Function
 * Can only destroy fv repo range information created
 * through aclfvCreateRepoRange interface.
 * @param repoRange [IN/OUT]     fv repo range.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclfvCreateRepoRange
 */
ACL_FUNC_VISIBILITY aclError aclfvDestroyRepoRange(aclfvRepoRange *repoRange);

/* *
 * @ingroup AscendCL
 * @brief Create query table.
 * @param queryCnt [IN]: Number of tables, the maximum number is 6
 * @param tableLen [IN]: Single table length, table length is 32KB
 * @param tableData [IN/OUT]: Feature value list
 * @param tableDataLen [IN]: The length of memory requested by the featureData
 * pointer
 * @retval null for failed. OtherValues success
 */
ACL_FUNC_VISIBILITY aclfvQueryTable *aclfvCreateQueryTable(uint32_t queryCnt, uint32_t tableLen, uint8_t *tableData,
    uint32_t tableDataLen);

/* *
 * @ingroup AscendCL
 * @brief Destroy query table.
 *
 * @par Function
 * Can only destroy query table information created
 * through aclfvCreateQueryTable interface.
 * @param queryTable [IN/OUT]     query table.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclfvCreateQueryTable
 */
ACL_FUNC_VISIBILITY aclError aclfvDestroyQueryTable(aclfvQueryTable *queryTable);

/* *
 * @ingroup AscendCL
 * @brief Create search input.
 * @param queryTable [IN/OUT]: query table
 * @param repoRange [IN/OUT]: query repo range
 * @param topk [IN]: query topk
 * @retval null for failed. OtherValues success
 */
ACL_FUNC_VISIBILITY aclfvSearchInput *aclfvCreateSearchInput(aclfvQueryTable *queryTable, aclfvRepoRange *repoRange,
    uint32_t topk);

/* *
 * @ingroup AscendCL
 * @brief Destroy search input.
 *
 * @par Function
 * Can only destroy search input information created
 * through aclfvCreateSearchInput interface.
 * @param searchInput [IN/OUT]     search input.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclfvCreateSearchInput
 */
ACL_FUNC_VISIBILITY aclError aclfvDestroySearchInput(aclfvSearchInput *searchInput);

/* *
 * @ingroup AscendCL
 * @brief Create search result.
 * @param queryCnt [IN]: Retrieve the number of features
 * @param resultNum [IN/OUT]: The number of search results for each feature, the
 * number is queryCnt
 * @param resultNumDataLen [IN]: resultNum memory length
 * @param id0 [IN/OUT]: Level 1 library id0
 * @param id1 [IN/OUT]: Secondary library id1
 * @param resultOffset [IN/OUT]: The offset of the bottom library corresponding
 * to each feature retrieval result, total length topK * queryCnt
 * @param resultDistance [IN/OUT]: Distance, total length topK * queryCnt
 * @param dataLen [IN]: The memory size requested by
 * id0\id1\reslutOffset\resultDistance
 * @retval null for failed. OtherValues success
 */
ACL_FUNC_VISIBILITY aclfvSearchResult *aclfvCreateSearchResult(uint32_t queryCnt, uint32_t *resultNum,
    uint32_t resultNumDataLen, uint32_t *id0, uint32_t *id1, uint32_t *resultOffset, float *resultDistance,
    uint32_t dataLen);

/* *
 * @ingroup AscendCL
 * @brief Destroy search result.
 *
 * @par Function
 * Can only destroy search result information created
 * through aclfvCreateSearchResult interface.
 * @param searchResult [IN/OUT]     search result.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclfvCreateSearchResult
 */
ACL_FUNC_VISIBILITY aclError aclfvDestroySearchResult(aclfvSearchResult *searchResult);

/* *
 * @ingroup AscendCL
 * @brief fv IP initialize.
 *
 * @param fsNum [IN]     max repo num, used to apply for memory.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure.
 */
ACL_FUNC_VISIBILITY aclError aclfvInit(uint64_t fsNum);

/* *
 * @ingroup AscendCL
 * @brief release fv resources.
 *
 * @par Function
 * Can only release fv resources created
 * through aclfvInit interface.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure.
 *
 * @see aclfvInit
 */
ACL_FUNC_VISIBILITY aclError aclfvRelease();

/* *
 * @ingroup AscendCL
 * @brief fv repo add.
 * @param type [IN]: repo add type
 * @param featureInfo [IN/OUT]: add feature information
 * @param stream [IN]: stream of task execute
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure.
 */
ACL_FUNC_VISIBILITY aclError aclfvRepoAdd(aclfvSearchType type, aclfvFeatureInfo *featureInfo, aclrtStream stream);

/* *
 * @ingroup AscendCL
 * @brief fv repo del.
 * @param type [IN]: repo delete type
 * @param repoRange [IN/OUT]: repo range information
 * @param stream [IN]: stream of task execute
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure.
 */
ACL_FUNC_VISIBILITY aclError aclfvRepoDel(aclfvSearchType type, aclfvRepoRange *repoRange, aclrtStream stream);

/* *
 * @ingroup AscendCL
 * @brief fv accurate del.
 * @param featureInfo [IN/OUT]: accurate delete feature information
 * @param stream [IN]: stream of task execute
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure.
 */
ACL_FUNC_VISIBILITY aclError aclfvDel(aclfvFeatureInfo *featureInfo, aclrtStream stream);

/* *
 * @ingroup AscendCL
 * @brief fv accurate modify.
 * @param featureInfo [IN/OUT]: accurate modify feature information
 * @param stream [IN]: stream of task execute
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure.
 */
ACL_FUNC_VISIBILITY aclError aclfvModify(aclfvFeatureInfo *featureInfo, aclrtStream stream);

/* *
 * @ingroup AscendCL
 * @brief fv search.
 * @param type [IN]: search type
 * @param searchInput [IN/OUT]: search input
 * @param searchRst [IN/OUT]: search result
 * @param stream [IN]: stream of task execute
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure.
 */
ACL_FUNC_VISIBILITY aclError aclfvSearch(aclfvSearchType type, aclfvSearchInput *searchInput,
    aclfvSearchResult *searchRst, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_OPS_ACL_RETR_H_
