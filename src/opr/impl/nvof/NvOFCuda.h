/*
* Copyright 2018-2019 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/**
 * \file src/opr/impl/nvof/NvOFCuda.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain_build_config.h"

#if MGB_CUDA
#pragma once
#include <memory>
#include "cuda.h"
#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"
#include "NvOF.h"

#define CUDA_DRVAPI_CALL(call)                                               \
    do {                                                                     \
        CUresult err__ = call;                                               \
        if (err__ != CUDA_SUCCESS) {                                         \
            const char* szErrName = NULL;                                    \
            cuGetErrorName(err__, &szErrName);                               \
            std::ostringstream errorLog;                                     \
            errorLog << "CUDA driver API error " << szErrName;               \
            std::cout << "Exception: " << __FILE__ << ":" << __LINE__ << ":" \
                      << errorLog.str() << std::endl;                        \
            mgb_throw(MegBrainError, "CUDA_DRVAPI_CALL ERROR");              \
        }                                                                    \
    } while (0)

class NvOFCudaAPI : public NvOFAPI {
public:
    NvOFCudaAPI(CUcontext cuContext, CUstream inputStream = nullptr, CUstream outputStream = nullptr);
    ~NvOFCudaAPI();

    NV_OF_CUDA_API_FUNCTION_LIST* GetAPI()
    {
        std::lock_guard<std::mutex> lock(m_lock);
        return  m_ofAPI.get();
    }

    CUcontext GetCudaContext() { return m_cuContext; }
    NvOFHandle GetHandle() { return m_hOF; }
    CUstream GetCudaStream(NV_OF_BUFFER_USAGE usage);
private:
    CUstream m_inputStream;
    CUstream m_outputStream;
    NvOFHandle m_hOF;
    std::unique_ptr<NV_OF_CUDA_API_FUNCTION_LIST> m_ofAPI;
    CUcontext m_cuContext;
};

/**
 * @brief Optical Flow for the CUDA interface
 */
class NvOFCuda : public NvOF
{
public:
    static NvOFObj Create(CUcontext cuContext, uint32_t nWidth, uint32_t nHeight,
        NV_OF_BUFFER_FORMAT eInBufFmt,
        NV_OF_CUDA_BUFFER_TYPE eInBufType,
        NV_OF_CUDA_BUFFER_TYPE eOutBufType,
        NV_OF_MODE eMode,
        NV_OF_PERF_LEVEL preset,
        CUstream inputStream = nullptr,
        CUstream outputStream = nullptr);
    ~NvOFCuda() {};

private:
    NvOFCuda(CUcontext cuContext,
        uint32_t nWidth,
        uint32_t nHeight,
        NV_OF_BUFFER_FORMAT eInBufFmt,
        NV_OF_CUDA_BUFFER_TYPE eInBufType,
        NV_OF_CUDA_BUFFER_TYPE eOutBufType, 
        NV_OF_MODE eMode,
        NV_OF_PERF_LEVEL preset,
        CUstream inputStream = nullptr,
        CUstream outputStream = nullptr);
    /**
    *  @brief This function is used to retrieve supported grid size for output.
    *  This function is an override of pure virtual function NvOF::DoGetOutputGridSizes().
    */
    virtual void DoGetOutputGridSizes(uint32_t* vals, uint32_t* size) override;

    /**
    *  @brief This function is used to initialize the OF engine.
    *  This function is an override of pure virtual function NvOF::DoInit().
    */
    virtual void DoInit(const NV_OF_INIT_PARAMS& initParams) override;

    /**
    *  @brief This function is used to estimate the optical flow between 2 images.
    *  This function is an override of pure virtual function NvOF::DoExecute().
    */
    virtual void DoExecute(const NV_OF_EXECUTE_INPUT_PARAMS& executeInParams, NV_OF_EXECUTE_OUTPUT_PARAMS& executeOutParams) override;

    /**
    *  @brief This function is used to allocate buffers used for optical flow estimation 
    *  using the cuda interface. This function is an override of pure virtual function
    *  NvOF::DoAllocBuffers().
    */
    virtual std::vector<NvOFBufferObj> DoAllocBuffers(NV_OF_BUFFER_DESCRIPTOR ofBufferDesc,
                uint32_t elementSize, uint32_t numBuffers) override;

    /**
    *  @brief This a helper function for allocating NvOFBuffer objects using the cuda
    *  interface.
    */
    std::unique_ptr<NvOFBuffer> CreateOFBufferObject(const NV_OF_BUFFER_DESCRIPTOR& desc, uint32_t elementSize, NV_OF_CUDA_BUFFER_TYPE bufferType);
    NV_OF_CUDA_BUFFER_TYPE GetBufferType(NV_OF_BUFFER_USAGE usage);

private:
    CUcontext m_cuContext;
    std::shared_ptr<NvOFCudaAPI> m_NvOFAPI;
    NV_OF_CUDA_BUFFER_TYPE   m_eInBufType;
    NV_OF_CUDA_BUFFER_TYPE   m_eOutBufType;
};

/*
 * A wrapper over an NvOFGPUBufferHandle which has been created with buffer
 * type NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR.
 */
class NvOFBufferCudaDevicePtr : public NvOFBuffer
{
public:
    ~NvOFBufferCudaDevicePtr();
    CUdeviceptr getCudaDevicePtr() { return m_devPtr; }
    virtual void UploadData(const void* pData, CUmemorytype mem_type) override;
    virtual void DownloadData(void* pData, CUmemorytype mem_type) override;
    NV_OF_CUDA_BUFFER_STRIDE_INFO getStrideInfo() { return m_strideInfo; }
private:
    NvOFBufferCudaDevicePtr(std::shared_ptr<NvOFCudaAPI> ofAPI, const NV_OF_BUFFER_DESCRIPTOR& desc, uint32_t elementSize);
    CUdeviceptr m_devPtr;
    CUcontext m_cuContext;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_strideInfo;
    std::shared_ptr<NvOFCudaAPI> m_NvOFAPI;
    friend class NvOFCuda;
};

/*
 * A wrapper over an NvOFGPUBufferHandle which has been created with buffer
 * type NV_OF_CUDA_BUFFER_TYPE_CUARRAY.
 */
class NvOFBufferCudaArray : public NvOFBuffer
{
public:
    ~NvOFBufferCudaArray();
    virtual void UploadData(const void* pData, CUmemorytype mem_type) override;
    virtual void DownloadData(void* pData, CUmemorytype mem_type) override;
    CUarray getCudaArray() { return m_cuArray; }
private:
    NvOFBufferCudaArray(std::shared_ptr<NvOFCudaAPI> ofAPI, const NV_OF_BUFFER_DESCRIPTOR& desc, uint32_t elementSize);
    CUarray m_cuArray;
    CUcontext m_cuContext;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_strideInfo;
    std::shared_ptr<NvOFCudaAPI> m_NvOFAPI;
    friend class NvOFCuda;
};

#endif
