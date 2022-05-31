/**
* @file acl.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_H_
#define INC_EXTERNAL_ACL_ACL_H_

#include "acl_rt.h"
#include "acl_op.h"
#include "acl_mdl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Current version is 1.1.0
#define ACL_MAJOR_VERSION    1
#define ACL_MINOR_VERSION    1
#define ACL_PATCH_VERSION    0

/**
 * @ingroup AscendCL
 * @brief acl initialize
 *
 * @par Restriction
 * The aclInit interface can be called only once in a process
 * @param configPath [IN]    the config path,it can be NULL
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclInit(const char *configPath);

/**
 * @ingroup AscendCL
 * @brief acl finalize
 *
 * @par Restriction
 * Need to call aclFinalize before the process exits.
 * After calling aclFinalize,the services cannot continue to be used normally.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclFinalize();

/**
 * @ingroup AscendCL
 * @brief query ACL interface version
 *
 * @param majorVersion[OUT] ACL interface major version
 * @param minorVersion[OUT] ACL interface minor version
 * @param patchVersion[OUT] ACL interface patch version
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetVersion(int32_t *majorVersion, int32_t *minorVersion, int32_t *patchVersion);

/**
 * @ingroup AscendCL
 * @brief get recent error message
 *
 * @retval null for failed
 * @retval OtherValues success
*/
ACL_FUNC_VISIBILITY const char *aclGetRecentErrMsg();

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_H_
