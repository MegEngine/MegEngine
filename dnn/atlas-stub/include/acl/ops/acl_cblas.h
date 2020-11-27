/**
* @file acl_cblas.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef INC_EXTERNAL_ACL_OPS_ACL_CBLAS_H_
#define INC_EXTERNAL_ACL_OPS_ACL_CBLAS_H_

#include "../acl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum aclTransType {
    ACL_TRANS_N,
    ACL_TRANS_T,
    ACL_TRANS_NZ,
    ACL_TRANS_NZ_T
} aclTransType;

typedef enum aclComputeType {
    ACL_COMPUTE_HIGH_PRECISION,
    ACL_COMPUTE_LOW_PRECISION
} aclComputeType;

/**
 * @ingroup AscendCL
 * @brief perform the matrix-vector multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param m [IN]           number of rows of matrix A
 * @param n [IN]           number of columns of matrix A
 * @param alpha [IN]       pointer to scalar used for multiplication.
 *                         of same type as dataTypeC
 * @param a [IN]           pointer to matrix A
 * @param lda [IN]         leading dimension used to store the matrix A
 * @param dataTypeA [IN]   datatype of matrix A
 * @param x [IN]           pointer to vector x
 * @param incx [IN]        stride between consecutive elements of vector x
 * @param dataTypeX [IN]   datatype of vector x
 * @param beta [IN]        pointer to scalar used for multiplication.
 *                         of same type as dataTypeC If beta == 0,
 *                         then y does not have to be a valid input
 * @param y [IN|OUT]       pointer to vector y
 * @param incy [IN]        stride between consecutive elements of vector y
 * @param dataTypeY [IN]   datatype of vector y
 * @param type [IN]        computation type
 * @param stream [IN]      stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclblasGemvEx(aclTransType transA, int m, int n,
    const void *alpha, const void *a, int lda, aclDataType dataTypeA,
    const void *x, int incx, aclDataType dataTypeX,
    const void *beta, void *y, int incy, aclDataType dataTypeY,
    aclComputeType type, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create a handle for performing the matrix-vector multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param m [IN]           number of rows of matrix A
 * @param n [IN]           number of columns of matrix A
 * @param dataTypeA [IN]   datatype of matrix A
 * @param dataTypeX [IN]   datatype of vector x
 * @param dataTypeY [IN]   datatype of vector y
 * @param type [IN]        computation type
 * @param handle [OUT]     pointer to the pointer to the handle
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclblasCreateHandleForGemvEx(aclTransType transA,
                                                          int m,
                                                          int n,
                                                          aclDataType dataTypeA,
                                                          aclDataType dataTypeX,
                                                          aclDataType dataTypeY,
                                                          aclComputeType type,
                                                          aclopHandle **handle);

/**
 * @ingroup AscendCL
 * @brief perform the matrix-vector multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param m [IN]           number of rows of matrix A
 * @param n [IN]           number of columns of matrix A
 * @param alpha [IN]       pointer to scalar used for multiplication
 * @param a [IN]           pointer to matrix A
 * @param lda [IN]         leading dimension used to store the matrix A
 * @param x [IN]           pointer to vector x
 * @param incx [IN]        stride between consecutive elements of vector x
 * @param beta [IN]        pointer to scalar used for multiplication.
 *                         If beta value == 0,
 *                         then y does not have to be a valid input
 * @param y [IN|OUT]       pointer to vector y
 * @param incy [IN]        stride between consecutive elements of vector y
 * @param type [IN]        computation type
 * @param stream [IN]      stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclblasHgemv(aclTransType transA,
                                          int m,
                                          int n,
                                          const aclFloat16 *alpha,
                                          const aclFloat16 *a,
                                          int lda,
                                          const aclFloat16 *x,
                                          int incx,
                                          const aclFloat16 *beta,
                                          aclFloat16 *y,
                                          int incy,
                                          aclComputeType type,
                                          aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create a handle for performing the matrix-vector multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param m [IN]           number of rows of matrix A
 * @param n [IN]           number of columns of matrix A
 * @param type [IN]        computation type
 * @param handle [OUT]     pointer to the pointer to the handle
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclblasCreateHandleForHgemv(aclTransType transA,
                                                         int m,
                                                         int n,
                                                         aclComputeType type,
                                                         aclopHandle **handle);

/**
 * @ingroup AscendCL
 * @brief perform the matrix-vector multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param m [IN]           number of rows of matrix A
 * @param n [IN]           number of columns of matrix A
 * @param alpha [IN]       pointer to scalar used for multiplication
 * @param a [IN]           pointer to matrix A
 * @param lda [IN]         leading dimension used to store the matrix A
 * @param x [IN]           pointer to vector x
 * @param incx [IN]        stride between consecutive elements of vector x
 * @param beta [IN]        pointer to scalar used for multiplication.
 *                         If beta value == 0,
 *                         then y does not have to be a valid input
 * @param y [IN|OUT]       pointer to vector y
 * @param incy [IN]        stride between consecutive elements of vector y
 * @param type [IN]        computation type
 * @param stream [IN]      stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclblasS8gemv(aclTransType transA,
                                           int m,
                                           int n,
                                           const int32_t *alpha,
                                           const int8_t *a,
                                           int lda,
                                           const int8_t *x,
                                           int incx,
                                           const int32_t *beta,
                                           int32_t *y,
                                           int incy,
                                           aclComputeType type,
                                           aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create a handle for performing the matrix-vector multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param m [IN]           number of rows of matrix A
 * @param n [IN]           number of columns of matrix A
 * @param handle [OUT]     pointer to the pointer to the handle
 * @param type [IN]        computation type
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclblasCreateHandleForS8gemv(aclTransType transA,
                                                          int m,
                                                          int n,
                                                          aclComputeType type,
                                                          aclopHandle **handle);

/**
 * @ingroup AscendCL
 * @brief perform the matrix-matrix multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param transB [IN]      transpose type of matrix B
 * @param transC [IN]      transpose type of matrix C
 * @param m [IN]           number of rows of matrix A and matrix C
 * @param n [IN]           number of columns of matrix B and matrix C
 * @param k [IN]           number of columns of matrix A and rows of matrix B
 * @param alpha [IN]       pointer to scalar used for multiplication. of same type as dataTypeC
 * @param matrixA [IN]     pointer to matrix A
 * @param lda [IN]         leading dimension array used to store  matrix A
 * @param dataTypeA [IN]   datatype of matrix A
 * @param matrixB [IN]     pointer to matrix B
 * @param ldb [IN]         leading dimension array used to store  matrix B
 * @param dataTypeB [IN]   datatype of matrix B
 * @param beta [IN]        pointer to scalar used for multiplication.
 *                         of same type as dataTypeC If beta == 0,
 *                         then matrixC does not have to be a valid input
 * @param matrixC [IN|OUT] pointer to matrix C
 * @param ldc [IN]         leading dimension array used to store  matrix C
 * @param dataTypeC [IN]   datatype of matrix C
 * @param type [IN]        computation type
 * @param stream [IN]      stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclblasGemmEx(aclTransType transA,
                                           aclTransType transB,
                                           aclTransType transC,
                                           int m,
                                           int n,
                                           int k,
                                           const void *alpha,
                                           const void *matrixA,
                                           int lda,
                                           aclDataType dataTypeA,
                                           const void *matrixB,
                                           int ldb,
                                           aclDataType dataTypeB,
                                           const void *beta,
                                           void *matrixC,
                                           int ldc,
                                           aclDataType dataTypeC,
                                           aclComputeType type,
                                           aclrtStream stream);


/**
 * @ingroup AscendCL
 * @brief create a handle for performing the matrix-matrix multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param transB [IN]      transpose type of matrix B
 * @param transC [IN]      transpose type of matrix C
 * @param m [IN]           number of rows of matrix A and matrix C
 * @param n [IN]           number of columns of matrix B and matrix C
 * @param k [IN]           number of columns of matrix A and rows of matrix B
 * @param dataTypeA [IN]   datatype of matrix A
 * @param dataTypeB [IN]   datatype of matrix B
 * @param dataTypeC [IN]   datatype of matrix C
 * @param type [IN]        computation type
 * @param handle [OUT]     pointer to the pointer to the handle
 * @param type [IN]        computation type
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclblasCreateHandleForGemmEx(aclTransType transA,
                                                          aclTransType transB,
                                                          aclTransType transC,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          aclDataType dataTypeA,
                                                          aclDataType dataTypeB,
                                                          aclDataType dataTypeC,
                                                          aclComputeType type,
                                                          aclopHandle **handle);


/**
 * @ingroup AscendCL
 * @brief perform the matrix-matrix multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param transB [IN]      transpose type of matrix B
 * @param transC [IN]      transpose type of matrix C
 * @param m [IN]           number of rows of matrix A and matrix C
 * @param n [IN]           number of columns of matrix B and matrix C
 * @param k [IN]           number of columns of matrix A and rows of matrix B
 * @param alpha [IN]       pointer to scalar used for multiplication
 * @param matrixA [IN]     pointer to matrix A
 * @param lda [IN]         leading dimension used to store the matrix A
 * @param matrixB [IN]     pointer to matrix B
 * @param ldb [IN]         leading dimension used to store the matrix B
 * @param beta [IN]        pointer to scalar used for multiplication.
 *                         If beta value == 0,
 *                         then matrixC does not have to be a valid input
 * @param matrixC [IN|OUT] pointer to matrix C
 * @param ldc [IN]         leading dimension used to store the matrix C
 * @param type [IN]        computation type
 * @param stream [IN]      stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclblasHgemm(aclTransType transA,
                                          aclTransType transB,
                                          aclTransType transC,
                                          int m,
                                          int n,
                                          int k,
                                          const aclFloat16 *alpha,
                                          const aclFloat16 *matrixA,
                                          int lda,
                                          const aclFloat16 *matrixB,
                                          int ldb,
                                          const aclFloat16 *beta,
                                          aclFloat16 *matrixC,
                                          int ldc,
                                          aclComputeType type,
                                          aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create a handle for performing the matrix-matrix multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param transB [IN]      transpose type of matrix B
 * @param transC [IN]      transpose type of matrix C
 * @param m [IN]           number of rows of matrix A and matrix C
 * @param n [IN]           number of columns of matrix B and matrix C
 * @param k [IN]           number of columns of matrix A and rows of matrix B
 * @param type [IN]        computation type
 * @param handle [OUT]     pointer to the pointer to the handle
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclblasCreateHandleForHgemm(aclTransType transA,
                                                         aclTransType transB,
                                                         aclTransType transC,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         aclComputeType type,
                                                         aclopHandle **handle);

/**
 * @ingroup AscendCL
 * @brief perform the matrix-matrix multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param transB [IN]      transpose type of matrix B
 * @param transC [IN]      transpose type of matrix C
 * @param m [IN]           number of rows of matrix A and matrix C
 * @param n [IN]           number of columns of matrix B and matrix C
 * @param k [IN]           number of columns of matrix A and rows of matrix B
 * @param alpha [IN]       pointer to scalar used for multiplication
 * @param matrixA [IN]     pointer to matrix A
 * @param lda [IN]         leading dimension used to store the matrix A
 * @param matrixB [IN]     pointer to matrix B
 * @param ldb [IN]         leading dimension used to store the matrix B
 * @param beta [IN]        pointer to scalar used for multiplication.
 *                         If beta value == 0,
 *                         then matrixC does not have to be a valid input
 * @param matrixC [IN|OUT] pointer to matrix C
 * @param ldc [IN]         leading dimension used to store the matrix C
 * @param type [IN]        computation type
 * @param stream [IN]      stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclblasS8gemm(aclTransType transA,
                                           aclTransType transB,
                                           aclTransType transC,
                                           int m,
                                           int n,
                                           int k,
                                           const int32_t *alpha,
                                           const int8_t *matrixA,
                                           int lda,
                                           const int8_t *matrixB,
                                           int ldb,
                                           const int32_t *beta,
                                           int32_t *matrixC,
                                           int ldc,
                                           aclComputeType type,
                                           aclrtStream stream);


/**
 * @ingroup AscendCL
 * @brief create a handle for performing the matrix-matrix multiplication
 *
 * @param transA [IN]      transpose type of matrix A
 * @param transB [IN]      transpose type of matrix B
 * @param transC [IN]      transpose type of matrix C
 * @param m [IN]           number of rows of matrix A and matrix C
 * @param n [IN]           number of columns of matrix B and matrix C
 * @param k [IN]           number of columns of matrix A and rows of matrix B
 * @param type [IN]        computation type
 * @param handle [OUT]     pointer to the pointer to the handle
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclblasCreateHandleForS8gemm(aclTransType transA,
                                                          aclTransType transB,
                                                          aclTransType transC,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          aclComputeType type,
                                                          aclopHandle **handle);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_OPS_ACL_CBLAS_H_
