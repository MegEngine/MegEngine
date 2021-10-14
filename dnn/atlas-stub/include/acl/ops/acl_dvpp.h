/**
* @file acl_dvpp.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#if !defined(ENABLE_DVPP_INTERFACE)
#if defined(_MSC_VER)
#error message("if you want to use dvpp funtions ,please use the macro definition (ENABLE_DVPP_INTERFACE).")
#else
#error "if you want to use dvpp funtions ,please use the macro definition (ENABLE_DVPP_INTERFACE)."
#endif
#endif

#ifndef INC_EXTERNAL_ACL_OPS_ACL_DVPP_H_
#define INC_EXTERNAL_ACL_OPS_ACL_DVPP_H_

#include <stdint.h>
#include <stddef.h>
#include "../acl.h"
#include "../acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct acldvppPicDesc acldvppPicDesc;
typedef struct acldvppBatchPicDesc acldvppBatchPicDesc;
typedef struct acldvppRoiConfig acldvppRoiConfig;
typedef struct acldvppResizeConfig acldvppResizeConfig;
typedef struct acldvppChannelDesc acldvppChannelDesc;
typedef struct acldvppJpegeConfig acldvppJpegeConfig;
typedef struct aclvdecChannelDesc aclvdecChannelDesc;
typedef struct acldvppStreamDesc acldvppStreamDesc;
typedef struct aclvdecFrameConfig aclvdecFrameConfig;
typedef struct aclvencChannelDesc aclvencChannelDesc;
typedef struct aclvencFrameConfig aclvencFrameConfig;
typedef void (*aclvdecCallback)(acldvppStreamDesc *input, acldvppPicDesc *output, void *userData);
typedef void (*aclvencCallback)(acldvppPicDesc *input, acldvppStreamDesc *output, void *userdata);

// Supported Pixel Format
enum acldvppPixelFormat {
    PIXEL_FORMAT_YUV_400 = 0, // 0
    PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1, // 1
    PIXEL_FORMAT_YVU_SEMIPLANAR_420 = 2, // 2
    PIXEL_FORMAT_YUV_SEMIPLANAR_422 = 3, // 3
    PIXEL_FORMAT_YVU_SEMIPLANAR_422 = 4, // 4
    PIXEL_FORMAT_YUV_SEMIPLANAR_444 = 5, // 5
    PIXEL_FORMAT_YVU_SEMIPLANAR_444 = 6, // 6
    PIXEL_FORMAT_YUYV_PACKED_422 = 7, // 7
    PIXEL_FORMAT_UYVY_PACKED_422 = 8, // 8
    PIXEL_FORMAT_YVYU_PACKED_422 = 9, // 9
    PIXEL_FORMAT_VYUY_PACKED_422 = 10, // 10
    PIXEL_FORMAT_YUV_PACKED_444 = 11, // 11
    PIXEL_FORMAT_RGB_888 = 12, // 12
    PIXEL_FORMAT_BGR_888 = 13, // 13
    PIXEL_FORMAT_ARGB_8888 = 14, // 14
    PIXEL_FORMAT_ABGR_8888 = 15, // 15
    PIXEL_FORMAT_RGBA_8888 = 16, // 16
    PIXEL_FORMAT_BGRA_8888 = 17, // 17
    PIXEL_FORMAT_YUV_SEMI_PLANNER_420_10BIT = 18, // 18
    PIXEL_FORMAT_YVU_SEMI_PLANNER_420_10BIT = 19, // 19
    PIXEL_FORMAT_YVU_PLANAR_420 = 20, // 20
    PIXEL_FORMAT_YVU_PLANAR_422,
    PIXEL_FORMAT_YVU_PLANAR_444,
    PIXEL_FORMAT_RGB_444 = 23,
    PIXEL_FORMAT_BGR_444,
    PIXEL_FORMAT_ARGB_4444,
    PIXEL_FORMAT_ABGR_4444,
    PIXEL_FORMAT_RGBA_4444,
    PIXEL_FORMAT_BGRA_4444,
    PIXEL_FORMAT_RGB_555,
    PIXEL_FORMAT_BGR_555,
    PIXEL_FORMAT_RGB_565,
    PIXEL_FORMAT_BGR_565,
    PIXEL_FORMAT_ARGB_1555,
    PIXEL_FORMAT_ABGR_1555,
    PIXEL_FORMAT_RGBA_1555,
    PIXEL_FORMAT_BGRA_1555,
    PIXEL_FORMAT_ARGB_8565,
    PIXEL_FORMAT_ABGR_8565,
    PIXEL_FORMAT_RGBA_8565,
    PIXEL_FORMAT_BGRA_8565,
    PIXEL_FORMAT_RGB_BAYER_8BPP = 50,
    PIXEL_FORMAT_RGB_BAYER_10BPP,
    PIXEL_FORMAT_RGB_BAYER_12BPP,
    PIXEL_FORMAT_RGB_BAYER_14BPP,
    PIXEL_FORMAT_RGB_BAYER_16BPP,
    PIXEL_FORMAT_BGR_888_PLANAR = 70,
    PIXEL_FORMAT_HSV_888_PACKAGE,
    PIXEL_FORMAT_HSV_888_PLANAR,
    PIXEL_FORMAT_LAB_888_PACKAGE,
    PIXEL_FORMAT_LAB_888_PLANAR,
    PIXEL_FORMAT_S8C1,
    PIXEL_FORMAT_S8C2_PACKAGE,
    PIXEL_FORMAT_S8C2_PLANAR,
    PIXEL_FORMAT_S16C1,
    PIXEL_FORMAT_U8C1,
    PIXEL_FORMAT_U16C1,
    PIXEL_FORMAT_S32C1,
    PIXEL_FORMAT_U32C1,
    PIXEL_FORMAT_U64C1,
    PIXEL_FORMAT_S64C1,
    PIXEL_FORMAT_YUV_SEMIPLANAR_440 = 1000,
    PIXEL_FORMAT_YVU_SEMIPLANAR_440,
    PIXEL_FORMAT_FLOAT32,
    PIXEL_FORMAT_BUTT,
    PIXEL_FORMAT_UNKNOWN = 10000
};

// Stream Format
enum acldvppStreamFormat {
    H265_MAIN_LEVEL = 0,
    H264_BASELINE_LEVEL,
    H264_MAIN_LEVEL,
    H264_HIGH_LEVEL
};

// Supported Channel Mode
enum acldvppChannelMode {
    DVPP_CHNMODE_VPC = 1,
    DVPP_CHNMODE_JPEGD = 2,
    DVPP_CHNMODE_JPEGE = 4
};

/**
 * @ingroup AscendCL
 * @brief alloc device memory for dvpp.
 *
 * @par Function
 * @li It's mainly used for allocating memory to device media data processing.
 * The requested memory meets the data processing requirements.
 * After calling this interface to request memory,
 * you must release the memory using the acldvppFree interface.
 * @li When calling the acldvppMalloc interface to apply for memory,
 * the size entered by the user is aligned upwards to 32 integer multiples,
 * and an additional 32 bytes are applied.
 *
 * @par Restriction
 * If the user uses the acldvppMalloc interface to apply for a large block of
 * memory and divide and manage the memory by himself,
 * when applying for memory, the user needs to align up to 32 integer
 * times + 32 bytes (ALIGN_UP [len] +32 words) according to
 * the actual data size of each picture Section) to manage memory.
 * @param devPtr [IN]     memory pointer.
 * @param size [IN]       memory size.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppFree
 */
ACL_FUNC_VISIBILITY aclError acldvppMalloc(void **devPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief free device memory for dvpp.
 *
 * @par Function
 * Free the memory requested through the acldvppMalloc interface
 * @param devPtr [IN]      memory pointer to free.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppMalloc
 */
ACL_FUNC_VISIBILITY aclError acldvppFree(void *devPtr);

/**
 * @ingroup AscendCL
 * @brief create DvppChannelDesc.
 *
 * @par Function
 * Create a channel for image data processing.
 * The same channel can be reused
 * and is no longer available after destruction
 * @retval null for failed.
 * @retval OtherValues success.
 */
ACL_FUNC_VISIBILITY acldvppChannelDesc *acldvppCreateChannelDesc();

/**
 * @ingroup AscendCL
 * @brief destroy dvppChannelDesc.
 *
 * @par Function
 * Can only destroy channels created by the acldvppCreateChannel interface
 * @param channelDesc [IN]     the channel description.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateChannelDesc | acldvppDestroyChannel
 */
ACL_FUNC_VISIBILITY aclError acldvppDestroyChannelDesc(acldvppChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get dvpp channel Id.
 *
 * @par Restriction
 * Interface calling sequence:
 * acldvppCreateChannelDesc --> acldvppCreateChannel -->
 * acldvppGetChannelDescChannelId
 * @param channelDesc [IN]     the channel description.
 * @retval channel id.
 *
 * @see acldvppCreateChannelDesc | acldvppCreateChannel
 */
ACL_FUNC_VISIBILITY uint64_t acldvppGetChannelDescChannelId(const acldvppChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Create dvpp picture description.
 *
 * @retval null for failed.
 * @retval OtherValues success.
 */
ACL_FUNC_VISIBILITY acldvppPicDesc *acldvppCreatePicDesc();

/**
 * @ingroup AscendCL
 * @brief Destroy dvpp picture description.
 *
 * @par Function
 * Can only destroy picture description information created
 * through acldvppCreatePicDesc interface.
 * @param picDesc [IN]     dvpp picture description.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreatePicDesc
 */
ACL_FUNC_VISIBILITY aclError acldvppDestroyPicDesc(acldvppPicDesc *picDesc);

/**
 * @ingroup AscendCL
 * @brief Set dvpp picture description's data.
 *
 * @param picDesc [IN]    dvpp picture description.
 * @param dataDev [IN]    dvpp picture dataDev.Must be the memory
 *                        requested using the acldvppMalloc interface.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppMalloc
 */
ACL_FUNC_VISIBILITY aclError acldvppSetPicDescData(acldvppPicDesc *picDesc, void *dataDev);

/**
 * @ingroup AscendCL
 * @brief Set dvpp picture description's size.
 *
 * @param picDesc [IN]     dvpp picture description.
 * @param size dvpp [IN]     picture size.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetPicDescSize(acldvppPicDesc *picDesc, uint32_t size);

/**
 * @ingroup AscendCL
 * @brief Set dvpp picture description's format.
 *
 * @param picDesc [IN]     dvpp picture description.
 * @param format [IN]      dvpp picture format.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetPicDescFormat(acldvppPicDesc *picDesc, acldvppPixelFormat format);

/**
 * @ingroup AscendCL
 * @brief Set dvpp picture description's width.
 *
 * @param picDesc [IN]     dvpp picture description.
 * @param width [IN]      dvpp picture width.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetPicDescWidth(acldvppPicDesc *picDesc, uint32_t width);

/**
 * @ingroup AscendCL
 * @brief Set dvpp picture description's height.
 *
 * @param picDesc [IN]    dvpp picture description.
 * @param height [IN]    dvpp picture height.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetPicDescHeight(acldvppPicDesc *picDesc, uint32_t height);

/**
 * @ingroup AscendCL
 * @brief Set dvpp picture description's widthStride.
 *
 * @par Restriction
 * Width alignment requirements:
 * @li The minimum stride is 32 and the maximum is 4096 * 4
 * (that is, an image in argb format with a width of 4096);
 * @li For 8K scaling, widthStride is required to be aligned to 2;
 * @li For non 8K scaling, the calculation formula for widthStride
 * is different for different image formats:
 *   @li yuv400sp, yuv420sp, yuv422sp, yuv444sp: input image width aligned to 16
 *   @li yuv422packed: input image width * 2 and then align to 16
 *   @li yuv444packed, rgb888: input image width alignment * 3, alignment to 16
 *   @li xrgb8888: input image width * 4, align to 16
 *   @li HFBC:input image width
 * @param picDesc [IN]     dvpp picture description.
 * @param widthStride [IN]   dvpp picture widthStride.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetPicDescWidthStride(acldvppPicDesc *picDesc, uint32_t widthStride);

/**
 * @ingroup AscendCL
 * @brief Set dvpp picture description's heightStride.
 *
 * @par Restriction
 * Height alignment requirements:
 * @li The height of the input image is aligned to 2.
 * High stride minimum 6 and maximum 4096.
 * @param picDesc [IN]    dvpp picture description.
 * @param heightStride [IN]    dvpp picture heightStride.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetPicDescHeightStride(acldvppPicDesc *picDesc, uint32_t heightStride);

/**
 * @ingroup AscendCL
 * @brief Set dvpp picture description's retcode.
 *
 * @param picDesc [IN]    dvpp picture description.
 * @param retCode [IN]    dvpp picture retcode.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetPicDescRetCode(acldvppPicDesc *picDesc, uint32_t retCode);

/**
 * @ingroup AscendCL
 * @brief Get picture data.
 *
 * @param picDesc [IN]    dvpp picture description.
 * @retval picture data addr.
 * @retval default nullptr.
 */
ACL_FUNC_VISIBILITY void *acldvppGetPicDescData(const acldvppPicDesc *picDesc);

/**
 * @ingroup AscendCL
 * @brief Get picture data size.
 *
 * @param picDesc [IN]    dvpp picture description.
 * @retval picture data size.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t acldvppGetPicDescSize(const acldvppPicDesc *picDesc);

/**
 * @ingroup AscendCL
 * @brief Get dvpp picture desc's format.
 *
 * @param picDesc [IN]    dvpp picture description.
 * @retval format
 * @retval default PIXEL_FORMAT_YUV_400.
 */
ACL_FUNC_VISIBILITY acldvppPixelFormat acldvppGetPicDescFormat(const acldvppPicDesc *picDesc);

/**
 * @ingroup AscendCL
 * @brief Get dvpp picture desc's width.
 *
 * @param picDesc [IN]    dvpp picture description.
 * @retval width.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t acldvppGetPicDescWidth(const acldvppPicDesc *picDesc);

/**
 * @ingroup AscendCL
 * @brief Get dvpp picture desc's height.
 *
 * @param picDesc [IN]    dvpp picture description.
 * @retval height.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t acldvppGetPicDescHeight(const acldvppPicDesc *picDesc);

/**
 * @ingroup AscendCL
 * @brief Get dvpp picture desc's widthStride.
 *
 * @par Restriction
 * Width alignment requirements:
 * @li The minimum stride is 32 and the maximum is 4096 * 4
 * (that is, an image in argb format with a width of 4096);
 * @li For 8K scaling, widthStride is required to be aligned to 2;
 * @li For non 8K scaling, the calculation formula for widthStride
 * is different for different image formats:
 *   @li yuv400sp, yuv420sp, yuv422sp, yuv444sp: input image width aligned to 16
 *   @li yuv422packed: input image width * 2 and then align to 16
 *   @li yuv444packed, rgb888: input image width alignment * 3, alignment to 16
 *   @li xrgb8888: input image width * 4, align to 16
 *   @li HFBC:input image width
 * @param picDesc [IN]    dvpp picture description.
 * @retval stride width.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t acldvppGetPicDescWidthStride(const acldvppPicDesc *picDesc);

/**
 * @ingroup AscendCL
 * @brief Get dvpp picture desc's heightStride.
 *
 * @par Restriction
 * Height alignment requirements:
 * @li The height of the input image is aligned to 2.
 * High stride minimum 6 and maximum 4096.
 * @param picDesc [IN]    dvpp picture description.
 * @retval stride height.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t acldvppGetPicDescHeightStride(const acldvppPicDesc *picDesc);

/**
 * @ingroup AscendCL
 * @brief Get dvpp picture desc's retcode.
 *
 * @param picDesc [IN]    dvpp picture description.
 * @retval ret code.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t acldvppGetPicDescRetCode(const acldvppPicDesc *picDesc);

/**
 * @ingroup AscendCL
 * @brief Create dvpp roi config.
 *
 * @param left [IN]    the left offset, must be even
 * @param right [IN]    the right offset, must be odd
 * @param top [IN]    the top offset, must be even
 * @param bottom [IN]    the bottom offset, must be odd
 * @retval null for failed.
 * @retval other success
 */
ACL_FUNC_VISIBILITY acldvppRoiConfig *acldvppCreateRoiConfig(uint32_t left,
                                                             uint32_t right,
                                                             uint32_t top,
                                                             uint32_t bottom);

/**
 * @ingroup AscendCL
 * @brief Destroy dvpp roi config.
 *
 * @par Function
 * Destroys data created through the acldvppCreateRoiConfig interface
 * @param roiConfig [IN]    dvpp roi config.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateRoiConfig
 */
ACL_FUNC_VISIBILITY aclError acldvppDestroyRoiConfig(acldvppRoiConfig *roiConfig);

/**
 * @ingroup AscendCL
 * @brief Set left of RoiConfig.
 *
 * @param config [IN]    RoiConfig
 * @param left [IN]   left offset
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetRoiConfigLeft(acldvppRoiConfig *config, uint32_t left);

/**
 * @ingroup AscendCL
 * @brief Set right of RoiConfig.
 *
 * @param config [IN]    RoiConfig
 * @param right [IN]    right offset
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetRoiConfigRight(acldvppRoiConfig *config, uint32_t right);

/**
 * @ingroup AscendCL
 * @brief Set top of RoiConfig.
 *
 * @param config [IN]    RoiConfig
 * @param top [IN]    top offset
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetRoiConfigTop(acldvppRoiConfig *config, uint32_t top);

/**
 * @ingroup AscendCL
 * @brief Set bottom of RoiConfig.
 *
 * @param config [IN]    RoiConfig
 * @param bottom [IN]    bottom offset
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetRoiConfigBottom(acldvppRoiConfig *config, uint32_t bottom);

/**
 * @ingroup AscendCL
 * @brief Set RoiConfig.
 *
 * @param config [IN|OUT] RoiConfig
 * @param left [IN]       left offset
 * @param right [IN]      right offset
 * @param top [IN]        top offset
 * @param bottom [IN]     bottom offset
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetRoiConfig(acldvppRoiConfig *config,
                                                 uint32_t left,
                                                 uint32_t right,
                                                 uint32_t top,
                                                 uint32_t bottom);

/**
 * @ingroup AscendCL
 * @brief Create dvpp resize config.
 * The specified scaling algorithm is not supported.
 * The default scaling algorithm is "nearest neighbor interpolation".
 *
 * @retval null for failed.
 * @retval other success.
 */
ACL_FUNC_VISIBILITY acldvppResizeConfig *acldvppCreateResizeConfig();

/**
 * @ingroup AscendCL
 * @brief Destroy dvpp resize config.
 *
 * @par Function
 * Destroys the scaling configuration data created by
 * the acldvppCreateResizeConfig interface
 * @param resizeConfig [IN]    resize config.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateResizeConfig
 */
ACL_FUNC_VISIBILITY aclError acldvppDestroyResizeConfig(acldvppResizeConfig *resizeConfig);

/**
 * @ingroup AscendCL
 * @brief Create jpege config.
 *
 * @retval null for failed.
 * @retval other success.
 */
ACL_FUNC_VISIBILITY acldvppJpegeConfig *acldvppCreateJpegeConfig();

/**
 * @ingroup AscendCL
 * @brief Destroy jpege config.
 *
 * @par Function
 * Destroys the encoding configuration data created by
 * the acldvppCreateJpegeConfig interface
 * @param jpegeConfig config pointer to destroy.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateJpegeConfig
 */
ACL_FUNC_VISIBILITY aclError acldvppDestroyJpegeConfig(acldvppJpegeConfig *jpegeConfig);

/**
 * @ingroup AscendCL
 * @brief Set jpege config's level.
 *
 * @param jpegeConfig [IN|OUT]    Call the acldvppCreateJpegeConfig
 *                                interface to create acldvppJpegeConfig data
 * @param level [IN]   Encoding quality range [0, 100],
 *                     where level 0 encoding quality is similar to level 100,
 *                     and the smaller the value in [1, 100],
 *                     the worse the quality of the output picture.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetJpegeConfigLevel(acldvppJpegeConfig *jpegeConfig, uint32_t level);

/**
 * @ingroup AscendCL
 * @brief Get jpege config's level.
 *
 * @param jpegeConfig [IN]    jpege config.
 * @retval compression level.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t acldvppGetJpegeConfigLevel(const acldvppJpegeConfig *jpegeConfig);

/**
 * @ingroup AscendCL
 * @brief create vdecChannelDesc.Channel description information
 * when creating a video data processing channel.
 *
 * @retval null for failed.
 * @retval other success
 */
ACL_FUNC_VISIBILITY aclvdecChannelDesc *aclvdecCreateChannelDesc();

/**
 * @ingroup AscendCL
 * @brief destroy vdecChannelDesc.
 *
 * @par Function
 * Can only destroy aclvdecChannelDesc type created
 * through aclvdecCreateChannelDesc interface
 * @param channelDesc [IN]    channel description.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure

 * @see aclvdecCreateChannelDesc
 */
ACL_FUNC_VISIBILITY aclError aclvdecDestroyChannelDesc(aclvdecChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Set vdec channel description's channel id.
 *
 * @param channelDesc [IN]   vdec channel description.
 * @param channelId [IN]    decoding channel id: 0~15.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclvdecSetChannelDescChannelId(aclvdecChannelDesc *channelDesc, uint32_t channelId);

/**
 * @ingroup AscendCL
 * @brief Set vdec channel description's thread id.
 *
 * @param channelDesc [IN]    vdec channel description.
 * @param threadId [IN]     thread id.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclvdecSetChannelDescThreadId(aclvdecChannelDesc *channelDesc, uint64_t threadId);

/**
 * @ingroup AscendCL
 * @brief Set vdec channel description's callback function.
 *
 * @param channelDesc [IN]     vdec channel description.
 * @param callback [IN]     function callback.Function prototype:
 * void (* aclvdecCallback)
 * (acldvppStreamDesc * input, acldvppPicDesc * output, void* userdata)
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclvdecCallback
 */
ACL_FUNC_VISIBILITY aclError aclvdecSetChannelDescCallback(aclvdecChannelDesc *channelDesc, aclvdecCallback callback);

/**
 * @ingroup AscendCL
 * @brief Set vdec channel description's video encoding type.
 *
 * @param channelDesc [IN]     vdec channel description.
 * @param enType [IN]     video encoding type.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclvdecSetChannelDescEnType(aclvdecChannelDesc *channelDesc, acldvppStreamFormat enType);

/**
 * @ingroup AscendCL
 * @brief Set vdec channel description's out picture format.
 *
 * @param channelDesc [IN]     vdec channel description.
 * @param outPicFormat [IN]     out picture format (acldvppPixelFormat).
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclvdecSetChannelDescOutPicFormat(aclvdecChannelDesc *channelDesc,
                                                               acldvppPixelFormat outPicFormat);

/**
 * @ingroup AscendCL
 * @brief Set vdec channel description's out picture width.
 *
 * @param channelDesc [IN]     vdec channel description.
 * @param outPicWidth [IN]     out picture width.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclvdecSetChannelDescOutPicWidth(aclvdecChannelDesc *channelDesc, uint32_t outPicWidth);

/**
 * @ingroup AscendCL
 * @brief Set vdec channel description's out picture height.
 *
 * @param channelDesc [IN]     vdec channel description.
 * @param outPicHeight [IN]     out picture height.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclvdecSetChannelDescOutPicHeight(aclvdecChannelDesc *channelDesc, uint32_t outPicHeight);

/**
 * @ingroup AscendCL
 * @brief Set vdec channel description's reference frame num.
 *
 * @param channelDesc [IN]     vdec channel description.
 * @param refFrameNum [IN]     reference frame num.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclvdecSetChannelDescRefFrameNum(aclvdecChannelDesc *channelDesc, uint32_t refFrameNum);

/**
 * @ingroup AscendCL
 * @brief Get vdec channel description's channel id.
 *
 * @param channelDesc [IN]     vdec channel description.
 * @retval decoding channel id: 0~15.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t aclvdecGetChannelDescChannelId(const aclvdecChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get vdec channel description's thread id.
 *
 * @param channelDesc [IN]     vdec channel description.
 * @retval thread id.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint64_t aclvdecGetChannelDescThreadId(const aclvdecChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get vdec channel description's callback function.
 *
 * @param channelDesc [IN]    vdec channel description.
 * @retval function callback.Function prototype:
 * void (* aclvdecCallback)
 * (acldvppStreamDesc * input, acldvppPicDesc * output, void* userdata)
 * @retval default null.
 *
 * @see aclvdecCallback
 */
ACL_FUNC_VISIBILITY aclvdecCallback aclvdecGetChannelDescCallback(const aclvdecChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get vdec channel description's video encoding type.
 *
 * @param channelDesc [IN]    vdec channel description.
 * @retval video encoding type.
 * @retval default H265_MAIN_LEVEL.
 */
ACL_FUNC_VISIBILITY acldvppStreamFormat aclvdecGetChannelDescEnType(const aclvdecChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get vdec channel description's out picture format.
 *
 * @param channelDesc [IN]    vdec channel description.
 * @retval out picture format.
 * @retval default DVPP_OUTPUT_YUV420SP_UV.
 */
ACL_FUNC_VISIBILITY acldvppPixelFormat aclvdecGetChannelDescOutPicFormat(const aclvdecChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get vdec channel description's out picture width.
 *
 * @param channelDesc [IN]    vdec channel description.
 * @retval out picture width.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t aclvdecGetChannelDescOutPicWidth(const aclvdecChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get vdec channel description's out picture height.
 *
 * @param channelDesc [IN]    vdec channel description.
 * @retval out picture height (for vdec malloc memory).
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t aclvdecGetChannelDescOutPicHeight(const aclvdecChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get vdec channel description's reference frame num.
 *
 * @param channelDesc [IN]    vdec channel description.
 * @retval reference frame num.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t aclvdecGetChannelDescRefFrameNum(const aclvdecChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief create vencChannelDesc.
 * @return null for failed, other success
 */
ACL_FUNC_VISIBILITY aclvencChannelDesc *aclvencCreateChannelDesc();

/**
 * @ingroup AscendCL
 * @brief destroy vencChannelDesc.
 * @param channelDesc channel desc.
 * @return ACL_ERROR_NONE:success, other:failed
 */
ACL_FUNC_VISIBILITY aclError aclvencDestroyChannelDesc(aclvencChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Set decoding thread id for venc channel desc.
 * @param channelDesc[IN/OUT] venc channel desc
 * @param threadId[IN] thread id
 * @return ACL_ERROR_NONE for success, other for failure
 */
ACL_FUNC_VISIBILITY aclError aclvencSetChannelDescThreadId(aclvencChannelDesc *channelDesc, uint64_t threadId);

/**
 * @ingroup AscendCL
 * @brief Set func callback for venc channel desc.
 * @param channelDesc[IN/OUT] venc channel desc
 * @param callback[IN] func callback
 * @return ACL_ERROR_NONE for success, other for failure
 */
ACL_FUNC_VISIBILITY aclError aclvencSetChannelDescCallback(aclvencChannelDesc *channelDesc, aclvencCallback callback);

/**
 * @ingroup AscendCL
 * @brief Set video encoding type for venc channel desc.
 * @param channelDesc[IN/OUT] venc channel desc
 * @param enType[IN] video encoding type
 * @return ACL_ERROR_NONE for success, other for failure
 */
ACL_FUNC_VISIBILITY aclError aclvencSetChannelDescEnType(aclvencChannelDesc *channelDesc, acldvppStreamFormat enType);

/**
 * @ingroup AscendCL
 * @brief Set pic format for venc channel desc.
 * @param channelDesc[IN/OUT] venc channel desc
 * @param picFormat[IN] pic format
 * @return ACL_ERROR_NONE for success, other for failure
 */
ACL_FUNC_VISIBILITY aclError aclvencSetChannelDescPicFormat(aclvencChannelDesc *channelDesc,
    acldvppPixelFormat picFormat);

/**
 * @ingroup AscendCL
 * @brief Set out pic width for venc channel desc.
 * @param channelDesc[IN/OUT] venc channel desc
 * @param picWidth[IN] pic width
 * @return ACL_ERROR_NONE for success, other for failure
 */
ACL_FUNC_VISIBILITY aclError aclvencSetChannelDescPicWidth(aclvencChannelDesc *channelDesc, uint32_t picWidth);

/**
 * @ingroup AscendCL
 * @brief Set pic height for venc channel desc.
 * @param channelDesc[IN/OUT] venc channel desc
 * @param picHeight[IN] pic height
 * @return ACL_ERROR_NONE for success, other for failure
 */
ACL_FUNC_VISIBILITY aclError aclvencSetChannelDescPicHeight(aclvencChannelDesc *channelDesc, uint32_t picHeight);

/**
 * @ingroup AscendCL
 * @brief Set eference frame num for venc channel desc.
 * @param channelDesc[IN/OUT] venc channel desc
 * @param keyFrameInterval[IN] Interval of key frame
 * @return ACL_ERROR_NONE for success, other for failure
 */
ACL_FUNC_VISIBILITY aclError aclvencSetChannelDescKeyFrameInterval(aclvencChannelDesc *channelDesc,
    uint32_t keyFrameInterval);

/**
 * @ingroup AscendCL
 * @brief Get decoding channel id for venc channel desc.
 * @param channelDesc[IN] venc channel desc
 * @return decoding channel id: 0~15, default 0
 */
ACL_FUNC_VISIBILITY uint32_t aclvencGetChannelDescChannelId(const aclvencChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get decoding thread id for venc channel desc.
 * @param channelDesc[IN] venc channel desc
 * @return thread id, default 0
 */
ACL_FUNC_VISIBILITY uint64_t aclvencGetChannelDescThreadId(const aclvencChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get func callback for venc channel desc.
 * @param channelDesc[IN] venc channel desc
 * @return func callback, default null
 */
ACL_FUNC_VISIBILITY aclvencCallback aclvencGetChannelDescCallback(const aclvencChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get video encoding type for venc channel desc.
 * @param channelDesc[IN] venc channel desc
 * @return video encoding type, default H265_MAIN_LEVEL
 */
ACL_FUNC_VISIBILITY acldvppStreamFormat aclvencGetChannelDescEnType(const aclvencChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get pic format for venc channel desc.
 * @param channelDesc[IN] venc channel desc
 * @return pic format
 */
ACL_FUNC_VISIBILITY acldvppPixelFormat aclvencGetChannelDescPicFormat(const aclvencChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get pic width for venc channel desc.
 * @param channelDesc[IN] venc channel desc
 * @return pic width, default 0
 */
ACL_FUNC_VISIBILITY uint32_t aclvencGetChannelDescPicWidth(const aclvencChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get pic height for venc channel desc.
 * @param channelDesc[IN] venc channel desc
 * @return pic height, default 0
 */
ACL_FUNC_VISIBILITY uint32_t aclvencGetChannelDescPicHeight(const aclvencChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Get interval of key frame for venc channel desc.
 * @param channelDesc[IN] venc channel desc
 * @return interval of key frame, default 0
 */
ACL_FUNC_VISIBILITY uint32_t aclvencGetChannelDescKeyFrameInterval(const aclvencChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief get forced restart of I-frame interval from config
 * @param config[IN] venc frame config
 * @return 0: Not forced; 1: Forced restart of I-frame -1: error
 */
ACL_FUNC_VISIBILITY uint8_t aclvencGetFrameConfigForceIFrame(const aclvencFrameConfig *config);

/**
 * @ingroup AscendCL
 * @brief get forced restart of I-frame interval from config
 * @param config[IN] venc frame config
 * @return Whether it is the end frame: 0: no; 1: end frame
 */
ACL_FUNC_VISIBILITY uint8_t aclvencGetFrameConfigEos(const aclvencFrameConfig *config);

/**
 * @ingroup AscendCL
 * @brief set single frame encoding configuration parameters
 * @param config[IN/OUT] venc frame config
 * @param forceFrame[IN] forced restart of I-frame interval: 0: Not forced; 1: Forced restart of I-frame
 * @return ACL_ERROR_NONE for ok, others for fail
 */
ACL_FUNC_VISIBILITY aclError aclvencSetFrameConfigForceIFrame(aclvencFrameConfig *config, uint8_t forceIFrame);

/**
 * @ingroup AscendCL
 * @brief set single frame encoding configuration parameters
 * @param config[IN/OUT] venc frame config
 * @param eos[IN] Whether it is the end frame: 0: no; 1: end frame
 * @return ACL_ERROR_NONE for ok, others for fail
 */
ACL_FUNC_VISIBILITY aclError aclvencSetFrameConfigEos(aclvencFrameConfig *config, uint8_t eos);

/**
 * @ingroup AscendCL
 * @brief dvpp venc destroy frame config
 * @param config[IN/OUT] venc frame config
 * @return ACL_ERROR_NONE for ok, others for fail
 */
ACL_FUNC_VISIBILITY aclError aclvencDestroyFrameConfig(aclvencFrameConfig *config);

/**
 * @ingroup AscendCL
 * @brief Create dvpp venc frame config.
 * @return null for failed, other aclvencFrameConfig ptr
 */
ACL_FUNC_VISIBILITY aclvencFrameConfig *aclvencCreateFrameConfig();

/**
 * @ingroup AscendCL
 * @brief Create dvpp venc channel.
 * @param channelDesc[IN/OUT] venc channel desc
 * @return ACL_ERROR_NONE for ok, others for fail
 */
ACL_FUNC_VISIBILITY aclError aclvencCreateChannel(aclvencChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Destroy dvpp venc channel.
 * @param channelDesc[IN/OUT] venc channel desc
 * @return ACL_ERROR_NONE for ok, others for fail
 */
ACL_FUNC_VISIBILITY aclError aclvencDestroyChannel(aclvencChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief dvpp venc launch send frame task.
 * @param channelDesc[IN/OUT] venc channel desc
 * @param input[IN/OUT] input picture desc
 * @param reserve[IN/OUT] reserve parameter
 * @param config[IN/OUT] dvpp frame config
 * @param userdata[IN/OUT] user callback function
 * @return ACL_ERROR_NONE for ok, others for fail
 */
ACL_FUNC_VISIBILITY aclError aclvencSendFrame(aclvencChannelDesc *channelDesc, acldvppPicDesc *input, void *reserve,
    aclvencFrameConfig *config, void *userdata);

/**
 * @ingroup AscendCL
 * @brief Create dvpp stream description.
 *
 * @retval null for failed.
 * @retval other success.
 */
ACL_FUNC_VISIBILITY acldvppStreamDesc *acldvppCreateStreamDesc();

/**
 * @ingroup AscendCL
 * @brief Destroy dvpp stream description.
 *
 * @par Function
 * Can only destroy acldvppStreamDesc type created through
 * acldvppCreateStreamDesc interface.
 * @param streamDesc [IN]     dvpp stream description.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateStreamDesc
 */
ACL_FUNC_VISIBILITY aclError acldvppDestroyStreamDesc(acldvppStreamDesc *streamDesc);

/**
 * @ingroup AscendCL
 * @brief Set stream description's data addr.
 *
 * @param streamDesc [IN]    dvpp stream description.
 * @param dataDev [IN]    data addr.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetStreamDescData(acldvppStreamDesc *streamDesc, void *dataDev);

/**
 * @ingroup AscendCL
 * @brief Set stream description's data size.
 *
 * @param streamDesc [IN]     dvpp stream description.
 * @param size [IN]    data size.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetStreamDescSize(acldvppStreamDesc *streamDesc, uint32_t size);

/**
 * @ingroup AscendCL
 * @brief Set stream description's format.
 *
 * @param streamDesc [IN]    dvpp stream description.
 * @param format [IN]    stream format.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetStreamDescFormat(acldvppStreamDesc *streamDesc, acldvppStreamFormat format);

/**
 * @ingroup AscendCL
 * @brief Set stream description's timestamp.
 *
 * @param streamDesc [IN]    dvpp stream description.
 * @param timestamp [IN]    current timestamp.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetStreamDescTimestamp(acldvppStreamDesc *streamDesc, uint64_t timestamp);

/**
 * @ingroup AscendCL
 * @brief Set stream description's ret code.
 *
 * @param streamDesc [IN]     dvpp stream description.
 * @param retCode [IN]     result code.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetStreamDescRetCode(acldvppStreamDesc *streamDesc, uint32_t retCode);

/**
 * @ingroup AscendCL
 * @brief Set stream description's eos.
 *
 * @param streamDesc [IN]    dvpp stream description.
 * @param eos [IN]    end flag of sequence.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetStreamDescEos(acldvppStreamDesc *streamDesc, uint8_t eos);

/**
 * @ingroup AscendCL
 * @brief Get stream description's data addr.
 *
 * @param streamDesc [IN]     dvpp stream description.
 * @retval data addr.
 * @retval deault nullptr.
 */
ACL_FUNC_VISIBILITY void *acldvppGetStreamDescData(const acldvppStreamDesc *streamDesc);

/**
 * @ingroup AscendCL
 * @brief Get stream description's data size.
 *
 * @param streamDesc [IN]    dvpp stream description.
 * @retval data size.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t acldvppGetStreamDescSize(const acldvppStreamDesc *streamDesc);

/**
 * @ingroup AscendCL
 * @brief Get stream description's format.
 *
 * @param streamDesc [IN]    dvpp stream description.
 * @retval stream format.
 * @retval default ACL_DVPP_STREAM_H264.
 */
ACL_FUNC_VISIBILITY acldvppStreamFormat acldvppGetStreamDescFormat(const acldvppStreamDesc *streamDesc);

/**
 * @ingroup AscendCL
 * @brief Get stream description's timestamp.
 *
 * @param streamDesc [IN]    dvpp stream description.
 * @retval current timestamp.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint64_t acldvppGetStreamDescTimestamp(const acldvppStreamDesc *streamDesc);

/**
 * @ingroup AscendCL
 * @brief Get stream description's retCode.
 *
 * @param streamDesc [IN]    dvpp stream description.
 * @retval result code.
 * @retval default 0.
 */
ACL_FUNC_VISIBILITY uint32_t acldvppGetStreamDescRetCode(const acldvppStreamDesc *streamDesc);

/**
 * @ingroup AscendCL
 * @brief Get stream description's eos.
 *
 * @param streamDesc [IN]    dvpp stream description.
 * @retval end flag of sequence.
 * @retval default 0(false).
 */
ACL_FUNC_VISIBILITY uint8_t acldvppGetStreamDescEos(const acldvppStreamDesc *streamDesc);

/**
 * @ingroup AscendCL
 * @brief Create vdec frame config.
 *
 * @retval null for failed.
 * @retval other success.
 */
ACL_FUNC_VISIBILITY aclvdecFrameConfig *aclvdecCreateFrameConfig();

/**
 * @ingroup AscendCL
 * @brief Destroy vdec frame config.
 *
 * @par Function
 * Can only destroy aclvdecFrameConfig type created through
 *  aclvdecCreateFrameConfig interface
 * @param vdecFrameConfig [IN]     vdec frame config.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclvdecCreateFrameConfig
 */
ACL_FUNC_VISIBILITY aclError aclvdecDestroyFrameConfig(aclvdecFrameConfig *vdecFrameConfig);

/**
 * @ingroup AscendCL
 * @brief Get image width and height of jpeg.
 *
 * @param data [IN]          image data in host memory
 * @param size [IN]          the size of image data
 * @param width [OUT]        the width of image from image header
 * @param height [OUT]       the height of image from image header
 * @param components [OUT]   the components of image from image header
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppJpegGetImageInfo(const void *data,
                                                     uint32_t size,
                                                     uint32_t *width,
                                                     uint32_t *height,
                                                     int32_t *components);

/**
 * @ingroup AscendCL
 * @brief Predict encode size of jpeg image.
 *
 * @param inputDesc [IN]     dvpp image desc
 * @param config [IN]        jpeg encode config
 * @param size [OUT]         the size predicted of image
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppJpegPredictEncSize(const acldvppPicDesc *inputDesc,
                                                       const acldvppJpegeConfig *config,
                                                       uint32_t *size);

/**
 * @ingroup AscendCL
 * @brief Predict decode size of jpeg image.
 *
 * @param data [IN]                 origin image data in host memory
 * @param dataSize [IN]             the size of origin image data
 * @param outputPixelFormat [IN]    the pixel format jpeg decode
 * @param decSize [OUT]             the size predicted for decode image
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppJpegPredictDecSize(const void *data,
    uint32_t dataSize,
    acldvppPixelFormat outputPixelFormat,
    uint32_t *decSize);

/**
 * @ingroup AscendCL
 * @brief Create dvpp channel, the same channel can be reused
 * and is no longer available after destruction.
 *
 * @param channelDesc [IN|OUT]    the channel destruction
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateChannelDesc
 */
ACL_FUNC_VISIBILITY aclError acldvppCreateChannel(acldvppChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Destroy dvpp channel.
 *
 * @par Restriction
 * Can only destroy channel created through the acldvppCreateChannel interface
 * @param channelDesc [IN]   the channel destruction
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateChannel
 */
ACL_FUNC_VISIBILITY aclError acldvppDestroyChannel(acldvppChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief dvpp vpc resize.
 *
 * @par Restriction
 * Width alignment requirements:
 * @li The minimum stride is 32 and the maximum is 4096 * 4
 * (that is, an image in argb format with a width of 4096);
 * @li For 8K scaling, widthStride is required to be aligned to 2;
 * @li For non 8K scaling, the calculation formula for widthStride
 * is different for different image formats:
 *   @li yuv400sp, yuv420sp, yuv422sp, yuv444sp: input image width aligned to 16
 *   @li yuv422packed: input image width * 2 and then align to 16
 *   @li yuv444packed, rgb888: input image width alignment * 3, alignment to 16
 *   @li xrgb8888: input image width * 4, align to 16
 *   @li HFBC:input image width
 * Height alignment requirements:
 * @li The height of the input image is aligned to 2.
 * High stride minimum 6 and maximum 4096.
 * @param channelDesc [IN]   the channel destruction
 * @param inputDesc [IN]   resize input picture destruction
 * @param outputDesc [IN|OUT]   resize output picture destruction
 * @param resizeConfig [IN]    resize config
 * @param stream [IN]    resize task stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateChannel | acldvppCreatePicDesc
 * | acldvppCreateResizeConfig
 */
ACL_FUNC_VISIBILITY aclError acldvppVpcResizeAsync(acldvppChannelDesc *channelDesc,
    acldvppPicDesc *inputDesc,
    acldvppPicDesc *outputDesc,
    acldvppResizeConfig *resizeConfig,
    aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief dvpp vpc crop.
 *
 * @par Function
 * crop the input picture according to the specified area,
 * and then store the  picture in the output memory as the output picture
 *
 * @par Restriction
 * Width alignment requirements:
 * @li The minimum stride is 32 and the maximum is 4096 * 4
 * (that is, an image in argb format with a width of 4096);
 * @li For 8K scaling, widthStride is required to be aligned to 2;
 * @li For non 8K scaling, the calculation formula for widthStride
 * is different for different image formats:
 *   @li yuv400sp, yuv420sp, yuv422sp, yuv444sp: input image width aligned to 16
 *   @li yuv422packed: input image width * 2 and then align to 16
 *   @li yuv444packed, rgb888: input image width alignment * 3, alignment to 16
 *   @li xrgb8888: input image width * 4, align to 16
 *   @li HFBC:input image width
 * Height alignment requirements:
 * @li The height of the input image is aligned to 2.
 * High stride minimum 6 and maximum 4096.
 * @param channelDesc [IN]    the channel destruction
 * @param inputDesc [IN]    crop input picture destruction
 * @param outputDesc [IN|OUT]    crop output picture destruction
 * @param cropArea [IN]    crop area config
 * @param stream [IN]    crop task stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppVpcCropAsync(acldvppChannelDesc *channelDesc,
    acldvppPicDesc *inputDesc,
    acldvppPicDesc *outputDesc,
    acldvppRoiConfig *cropArea,
    aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief dvpp vpc batch crop.
 *
 * @par Function
 * crop the input batch picture according to the specified area
 * as the output batch pictures
 * @param channelDesc [IN]    the channel destruction
 * @param srcBatchPicDescs [IN|OUT]    crop input batch picture destruction
 * @param roiNums [IN]    roi config numbers
 * @param size [IN]    roiNum size
 * @param dstBatchPicDescs [IN|OUT]    crop output batch picture destruction
 * @param cropAreas [IN]    crop area configs
 * @param stream [IN]    crop batch task stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateChannel | acldvppCreateBatchPicDesc | acldvppCreateRoiConfig
 */
ACL_FUNC_VISIBILITY aclError acldvppVpcBatchCropAsync(acldvppChannelDesc *channelDesc,
    acldvppBatchPicDesc *srcBatchPicDescs,
    uint32_t *roiNums,
    uint32_t size,
    acldvppBatchPicDesc *dstBatchPicDescs,
    acldvppRoiConfig *cropAreas[],
    aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief dvpp vpc crop and paste.
 *
 * @par Function
 * crop the input picture according to the specified area,
 * and paste the picture to the specified position of the target picture
 * as the output picture
 * @param channelDesc [IN]    thechannel destruction
 * @param inputDesc [IN]    crop and paste input picture destruction
 * @param outputDesc [IN|OUT]    crop and paste output picture destruction
 * @param cropArea [IN]    crop area config
 * @param pasteArea [IN]    paste area config
 * @param stream [IN]    crop and paste task stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateChannel | acldvppCreatePicDesc | acldvppCreateRoiConfig
 */
ACL_FUNC_VISIBILITY aclError acldvppVpcCropAndPasteAsync(acldvppChannelDesc *channelDesc,
    acldvppPicDesc *inputDesc,
    acldvppPicDesc *outputDesc,
    acldvppRoiConfig *cropArea,
    acldvppRoiConfig *pasteArea,
    aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief dvpp vpc batch crop and paste.
 *
 * @par Function
 * crop the input batch picture according to the specified area,
 * and paste the pictures to the specified position of the target pictures
 * as the output batch pictures
 * @param channelDesc [IN]    the channel destruction
 * @param srcBatchPicDescs [IN|OUT]    crop input batch picture destruction
 * @param roiNums [IN]    roi config numbers
 * @param size [IN]    roiNum size
 * @param dstBatchPicDescs [IN|OUT]    crop output batch picture destruction
 * @param cropAreas [IN]    crop area configs
 * @param pasteAreas [IN]    paste area configs
 * @param stream [IN]    crop batch task stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateChannel | acldvppCreateBatchPicDesc | acldvppCreateRoiConfig
 */
 ACL_FUNC_VISIBILITY aclError acldvppVpcBatchCropAndPasteAsync(acldvppChannelDesc *channelDesc,
     acldvppBatchPicDesc *srcBatchPicDescs,
     uint32_t *roiNums,
     uint32_t size,
     acldvppBatchPicDesc *dstBatchPicDescs,
     acldvppRoiConfig *cropAreas[],
     acldvppRoiConfig *pasteAreas[],
     aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief dvpp vpc jpeg decode.
 *
 * @par Function
 * For different source picture formats, after decoding,
 * output pictures in the following format:
 * @li jpeg(444) -> YUV444SP:V is front U is back,
 * YUV420 SP V is front U is back, YUV420SP U is front V is back;
 * @li jpeg(422) -> YUV422SP:V is in front U is behind,
 * YUV420SP V is in front U is behind, YUV420SP U is in front V is behind;
 * @li jpeg(420) -> YUV420SP:
 * V is front U is back, YUV420SP U is front V is back;
 * @li jpeg(400) -> YUV420SP:UV data is filled with 0 x 80.
 * @param channelDesc [IN]    the channel destruction
 * @param data [IN]    decode input picture destruction's data
 * @param size [IN]    decode input picture destruction's size
 * @param outputDesc [IN|OUT]    decode output picture destruction
 * @param stream [IN]    decode task stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateChannel | acldvppCreatePicDesc
 */
ACL_FUNC_VISIBILITY aclError acldvppJpegDecodeAsync(acldvppChannelDesc *channelDesc,
    const void *data,
    uint32_t size,
    acldvppPicDesc *outputDesc,
    aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief dvpp vpc jpeg encode.
 *
 * @param channelDesc [IN]    the channel destruction
 * @param inputDesc [IN]    encode input picture destruction
 * @param data [IN]    encode output picture destruction's data
 * @param size [IN]    encode output picture destruction's size
 * @param config [IN]    jpeg encode config
 * @param stream [IN]    encode task stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateChannel | acldvppCreateJpegeConfig
 */
ACL_FUNC_VISIBILITY aclError acldvppJpegEncodeAsync(acldvppChannelDesc *channelDesc,
    acldvppPicDesc *inputDesc,
    const void *data,
    uint32_t *size,
    acldvppJpegeConfig *config,
    aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Create vdec channel.
 *
 * @par Function
 * Create a channel for video data processing,
 * the same channel can be reused,
 * and is no longer available after destruction
 * @param channelDesc [IN]    the channel destruction
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclvdecCreateChannelDesc
 */
ACL_FUNC_VISIBILITY aclError aclvdecCreateChannel(aclvdecChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief Destroy vdec channel.
 *
 * @par Function
 * Can only destroy channels created by the aclvdecCreateChannel interface
 * @param channelDesc [IN]    the channel destruction
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclvdecCreateChannel
 */
ACL_FUNC_VISIBILITY aclError aclvdecDestroyChannel(aclvdecChannelDesc *channelDesc);

/**
 * @ingroup AscendCL
 * @brief dvpp vdec send frame.
 *
 * @par Function
 * Pass the input memory to be decoded
 * and the decoded output memory to the decoder for decoding
 * @param channelDesc [IN]    vdec channel destruction
 * @param input [IN]    input stream destruction
 * @param output [IN]    output picture destruction
 * @param config [IN]     vdec frame config
 * @param userData [IN]    user data for callback function
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclvdecCreateChannel | acldvppCreateStreamDesc | acldvppCreatePicDesc
 */
ACL_FUNC_VISIBILITY aclError aclvdecSendFrame(aclvdecChannelDesc *channelDesc,
    acldvppStreamDesc *input,
    acldvppPicDesc *output,
    aclvdecFrameConfig *config,
    void* userData);

/**
 * @ingroup AscendCL
 * @brief dvpp vpc convert color.
 *
 * @par Restriction
 * @li outputDesc:Width height stride, No changes are allowed. Just configure 0
 * @par Function
 * Convert color gamut
 * @param channelDesc [IN|OUT]   the channel destruction
 * @param inputDesc [IN|OUT] convert color input picture destruction
 * @param outputDesc [IN|OUT] convert color output picture destruction
 * @param stream [IN]  convert color task stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateChannel | acldvppCreatePicDesc
 */
ACL_FUNC_VISIBILITY aclError acldvppVpcConvertColorAsync(acldvppChannelDesc *channelDesc,
    acldvppPicDesc *inputDesc,
    acldvppPicDesc *outputDesc,
    aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief dvpp vpc pyramid down.
 *
 * @par Restriction
 * @li outputDesc:format only supported YUV400
 * @par Function
 * Image pyramid down
 * @param channelDesc [IN|OUT]   the channel destruction
 * @param inputDesc [IN|OUT] pyr down input picture destruction
 * @param outputDesc [IN|OUT] pyr down output picture destruction
 * @param reserve [IN|OUT] reserved param , must be nullptr
 * @param stream [IN] pyr down task stream
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateChannel | acldvppCreatePicDesc
 */
ACL_FUNC_VISIBILITY aclError acldvppVpcPyrDownAsync(acldvppChannelDesc *channelDesc,
    acldvppPicDesc *inputDesc,
    acldvppPicDesc *outputDesc,
    void *reserve,
    aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Set dvpp channel mode.
 *
 * @param channelDesc [IN|OUT] the channel destruction
 * @param mode [IN] channel mode
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetChannelDescMode(acldvppChannelDesc *channelDesc,
    uint32_t mode);

/**
 * @ingroup AscendCL
 * @brief Set resize config interpolation.
 *
 * @param resizeConfig [IN|OUT] the resize config
 * @param interpolation [IN] interpolation
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acldvppSetResizeConfigInterpolation(acldvppResizeConfig *resizeConfig,
    uint32_t interpolation);

/**
 * @ingroup AscendCL
 * @brief Get resize config interpolation.
 * @param resizeConfig [IN] the resize config
 * @retval Interpolation of resize config.
 */
ACL_FUNC_VISIBILITY uint32_t acldvppGetResizeConfigInterpolation(const acldvppResizeConfig *resizeConfig);

/**
 * @ingroup AscendCL
 * @brief Set vdec channel out mode.
 *
 * @param channelDesc [IN|OUT] the channel destruction
 * @param outMode [IN] channel out mode
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclvdecSetChannelDescOutMode(aclvdecChannelDesc *channelDesc,
    uint32_t outMode);

/**
 * @ingroup AscendCL
 * @brief Create dvpp batch picture description.
 *
 * @param batchSize [IN]    batch size
 * @retval null for failed.
 * @retval OtherValues success.
 */
ACL_FUNC_VISIBILITY acldvppBatchPicDesc *acldvppCreateBatchPicDesc(uint32_t batchSize);

/**
 * @ingroup AscendCL
 * @brief Get dvpp picture description.
 *
 * @param batchPicDesc [IN]    dvpp batch picture description.
 * @param index [IN]    index of batch
 * @retval null for failed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateBatchPicDesc
 */
ACL_FUNC_VISIBILITY acldvppPicDesc *acldvppGetPicDesc(acldvppBatchPicDesc *batchPicDesc, uint32_t index);

/**
 * @ingroup AscendCL
 * @brief Destroy dvpp batch picture description.
 *
 * @par Function
 * Can only destroy batch picture description information created
 * through acldvppCreateBatchPicDesc interface.
 * @param batchPicDesc [IN]     dvpp batch picture description.
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acldvppCreateBatchPicDesc
 */
ACL_FUNC_VISIBILITY aclError acldvppDestroyBatchPicDesc(acldvppBatchPicDesc *batchPicDesc);


#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_OPS_ACL_DVPP_H_
