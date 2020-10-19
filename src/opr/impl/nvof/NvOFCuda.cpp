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
 * \file src/opr/impl/nvof/NvOFCuda.cpp
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
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include "megbrain/common.h"
#include "NvOFCuda.h"

NvOFCudaAPI::NvOFCudaAPI(CUcontext cuContext, CUstream inputStream, CUstream outputStream)
    : m_inputStream(inputStream), m_outputStream(outputStream), m_cuContext(cuContext)
{
    typedef NV_OF_STATUS(NVOFAPI *PFNNvOFAPICreateInstanceCuda)(uint32_t apiVer, NV_OF_CUDA_API_FUNCTION_LIST* cudaOf);
#if defined(_WIN32)
    PFNNvOFAPICreateInstanceCuda NvOFAPICreateInstanceCuda = (PFNNvOFAPICreateInstanceCuda)GetProcAddress(m_hModule, "NvOFAPICreateInstanceCuda");
#else
    PFNNvOFAPICreateInstanceCuda NvOFAPICreateInstanceCuda = (PFNNvOFAPICreateInstanceCuda)dlsym(m_hModule, "NvOFAPICreateInstanceCuda");
#endif
    if (!NvOFAPICreateInstanceCuda)
    {
        mgb_throw(MegBrainError,
                  "NVOF: Cannot find NvOFAPICreateInstanceCuda() entry in NVOF "
                  "library err type: NV_OF_ERR_OF_NOT_AVAILABLE");
    }

    m_ofAPI.reset(new NV_OF_CUDA_API_FUNCTION_LIST());

    NVOF_API_CALL(NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, m_ofAPI.get()));
    NVOF_API_CALL(m_ofAPI->nvCreateOpticalFlowCuda(m_cuContext, &m_hOF));
    NVOF_API_CALL(m_ofAPI->nvOFSetIOCudaStreams(m_hOF, m_inputStream, m_outputStream));
}

NvOFCudaAPI::~NvOFCudaAPI()
{
    if (m_ofAPI)
    {
        m_ofAPI->nvOFDestroy(m_hOF);
    }
}

CUstream NvOFCudaAPI::GetCudaStream(NV_OF_BUFFER_USAGE usage)
{
    CUstream stream = 0;
    if (usage == NV_OF_BUFFER_USAGE_INPUT)
    {
        stream = m_inputStream;
    }
    else if ((usage == NV_OF_BUFFER_USAGE_OUTPUT) ||
        (usage == NV_OF_BUFFER_USAGE_COST) ||
        (usage == NV_OF_BUFFER_USAGE_HINT))
    {
        stream = m_outputStream;
    }
    return stream;
}

NvOFObj NvOFCuda::Create(CUcontext cuContext, uint32_t nWidth, uint32_t nHeight,
    NV_OF_BUFFER_FORMAT eInBufFmt,
    NV_OF_CUDA_BUFFER_TYPE eInBufType,
    NV_OF_CUDA_BUFFER_TYPE eOutBufType,
    NV_OF_MODE eMode,
    NV_OF_PERF_LEVEL preset,
    CUstream inputStream,
    CUstream outputStream)
{
    std::unique_ptr<NvOF> ofObj(new NvOFCuda(cuContext,
        nWidth,
        nHeight,
        eInBufFmt,
        eInBufType,
        eOutBufType,
        eMode,
        preset,
        inputStream,
        outputStream));
    return ofObj;
}

NvOFCuda::NvOFCuda(CUcontext cuContext,
    uint32_t nWidth,
    uint32_t nHeight,
    NV_OF_BUFFER_FORMAT eInBufFmt,
    NV_OF_CUDA_BUFFER_TYPE eInBufType,
    NV_OF_CUDA_BUFFER_TYPE eOutBufType,
    NV_OF_MODE eMode,
    NV_OF_PERF_LEVEL preset,
    CUstream inputStream,
    CUstream outputStream)
: NvOF(nWidth, nHeight, eInBufFmt, eMode, preset),
  m_cuContext(cuContext),
  m_eInBufType(eInBufType),
  m_eOutBufType(eOutBufType)
{
    m_NvOFAPI = std::make_shared<NvOFCudaAPI>(m_cuContext, inputStream, outputStream);
}

void NvOFCuda::DoGetOutputGridSizes(uint32_t* vals, uint32_t* size)
{
    NVOF_API_CALL(m_NvOFAPI->GetAPI()->nvOFGetCaps(m_NvOFAPI->GetHandle(), NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, vals, size));
}

void NvOFCuda::DoExecute(const NV_OF_EXECUTE_INPUT_PARAMS& executeInParams,
    NV_OF_EXECUTE_OUTPUT_PARAMS& executeOutParams)
{
    NVOF_API_CALL(m_NvOFAPI->GetAPI()->nvOFExecute(m_NvOFAPI->GetHandle(), &executeInParams, &executeOutParams));
}

void NvOFCuda::DoInit(const NV_OF_INIT_PARAMS& initParams)
{
    NVOF_API_CALL(m_NvOFAPI->GetAPI()->nvOFInit(m_NvOFAPI->GetHandle(), &initParams));
}

NV_OF_CUDA_BUFFER_TYPE NvOFCuda::GetBufferType(NV_OF_BUFFER_USAGE usage)
{
    NV_OF_CUDA_BUFFER_TYPE bufferType = NV_OF_CUDA_BUFFER_TYPE_UNDEFINED;
    if (usage == NV_OF_BUFFER_USAGE_INPUT)
    {
        bufferType = m_eInBufType;
    }
    else if ((usage  == NV_OF_BUFFER_USAGE_OUTPUT) || 
            (usage == NV_OF_BUFFER_USAGE_COST)   ||
            (usage == NV_OF_BUFFER_USAGE_HINT))
    {
        bufferType = m_eOutBufType;
    }

    return bufferType;
}

std::vector<NvOFBufferObj>
NvOFCuda::DoAllocBuffers(NV_OF_BUFFER_DESCRIPTOR ofBufferDesc,
    uint32_t elementSize, uint32_t numBuffers)
{
    std::vector<NvOFBufferObj> ofBuffers;
    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        NV_OF_CUDA_BUFFER_TYPE bufferType = GetBufferType(ofBufferDesc.bufferUsage);
        ofBuffers.emplace_back(CreateOFBufferObject(ofBufferDesc, elementSize, bufferType).release());
    }
    return ofBuffers;
}

std::unique_ptr<NvOFBuffer> NvOFCuda::CreateOFBufferObject(const NV_OF_BUFFER_DESCRIPTOR& desc, uint32_t elementSize, NV_OF_CUDA_BUFFER_TYPE bufferType)
{
    std::unique_ptr<NvOFBuffer> pBuffer;
    if (bufferType == NV_OF_CUDA_BUFFER_TYPE_CUARRAY)
    {
        pBuffer.reset(new NvOFBufferCudaArray(m_NvOFAPI, desc, elementSize));
    }
    else
    {
        pBuffer.reset(new NvOFBufferCudaDevicePtr(m_NvOFAPI, desc, elementSize));
    }
    return pBuffer;
}

NvOFBufferCudaDevicePtr::NvOFBufferCudaDevicePtr(std::shared_ptr<NvOFCudaAPI> ofAPI, const NV_OF_BUFFER_DESCRIPTOR& desc, uint32_t elementSize) :
    NvOFBuffer(desc, elementSize), m_devPtr(0), m_NvOFAPI(ofAPI)
{
    m_cuContext = m_NvOFAPI->GetCudaContext();
    NVOF_API_CALL(m_NvOFAPI->GetAPI()->nvOFCreateGPUBufferCuda(m_NvOFAPI->GetHandle(),
        &desc,
        NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
        &m_hGPUBuffer));
    m_devPtr = m_NvOFAPI->GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hGPUBuffer);
    NVOF_API_CALL(m_NvOFAPI->GetAPI()->nvOFGPUBufferGetStrideInfo(m_hGPUBuffer, &m_strideInfo));
}

NvOFBufferCudaDevicePtr::~NvOFBufferCudaDevicePtr()
{
    m_NvOFAPI->GetAPI()->nvOFDestroyGPUBufferCuda(m_hGPUBuffer);
}

void NvOFBufferCudaDevicePtr::UploadData(const void* pData,
                                         CUmemorytype mem_type) {
    CUstream stream = m_NvOFAPI->GetCudaStream(getBufferUsage());
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
    CUDA_MEMCPY2D cuCopy2d;
    memset(&cuCopy2d, 0, sizeof(cuCopy2d));
    cuCopy2d.WidthInBytes = getWidth()* getElementSize();
    mgb_assert(
            CU_MEMORYTYPE_HOST == mem_type || CU_MEMORYTYPE_DEVICE == mem_type,
            "do not imp mem type!!!");
    cuCopy2d.srcMemoryType = mem_type;
    if (CU_MEMORYTYPE_HOST == mem_type) {
        cuCopy2d.srcHost = pData;
    } else if (CU_MEMORYTYPE_DEVICE == mem_type) {
        cuCopy2d.srcDevice = (CUdeviceptr)pData;
    }
    cuCopy2d.srcPitch = cuCopy2d.WidthInBytes;
    cuCopy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cuCopy2d.dstDevice = getCudaDevicePtr();
    cuCopy2d.dstPitch = m_strideInfo.strideInfo[0].strideXInBytes;
    cuCopy2d.Height   = getHeight();
    CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));

    if (getBufferFormat() == NV_OF_BUFFER_FORMAT_NV12)
    {
        cuCopy2d.Height   = (getHeight() + 1)/2;
        cuCopy2d.srcHost  = ((const uint8_t *)pData + (cuCopy2d.srcPitch * cuCopy2d.Height));
        cuCopy2d.dstY     = m_strideInfo.strideInfo[0].strideYInBytes;
        CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));
    }
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(&m_cuContext));
}

void NvOFBufferCudaDevicePtr::DownloadData(void* pData, CUmemorytype mem_type) {
    CUstream stream = m_NvOFAPI->GetCudaStream(getBufferUsage());
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
    CUDA_MEMCPY2D cuCopy2d;
    memset(&cuCopy2d, 0, sizeof(cuCopy2d));
    cuCopy2d.WidthInBytes = getWidth() * getElementSize();

    mgb_assert(
            CU_MEMORYTYPE_HOST == mem_type || CU_MEMORYTYPE_DEVICE == mem_type,
            "do not imp mem type!!!");
    cuCopy2d.dstMemoryType = mem_type;
    if (CU_MEMORYTYPE_HOST == mem_type) {
        cuCopy2d.dstHost = pData;
    } else if (CU_MEMORYTYPE_DEVICE == mem_type) {
        cuCopy2d.dstDevice = (CUdeviceptr)pData;
    }
    cuCopy2d.dstPitch = cuCopy2d.WidthInBytes;
    cuCopy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cuCopy2d.srcDevice = getCudaDevicePtr();
    cuCopy2d.srcPitch = m_strideInfo.strideInfo[0].strideXInBytes;
    cuCopy2d.Height = getHeight();
    CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));
    if (getBufferFormat() == NV_OF_BUFFER_FORMAT_NV12)
    {
        cuCopy2d.Height = (getHeight() + 1) / 2;
        cuCopy2d.dstHost = ((uint8_t *)pData + (cuCopy2d.dstPitch * cuCopy2d.Height));
        cuCopy2d.srcY = m_strideInfo.strideInfo[0].strideYInBytes;
        CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));
    }
    CUDA_DRVAPI_CALL(cuStreamSynchronize(stream));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(&m_cuContext));
}

NvOFBufferCudaArray::NvOFBufferCudaArray(std::shared_ptr<NvOFCudaAPI> ofAPI, const NV_OF_BUFFER_DESCRIPTOR& desc, uint32_t elementSize) :
    NvOFBuffer(desc, elementSize), m_cuArray(0), m_NvOFAPI(ofAPI)
{
    m_cuContext = m_NvOFAPI->GetCudaContext();
    NVOF_API_CALL(m_NvOFAPI->GetAPI()->nvOFCreateGPUBufferCuda(m_NvOFAPI->GetHandle(),
        &desc,
        NV_OF_CUDA_BUFFER_TYPE_CUARRAY,
        &m_hGPUBuffer));
    m_cuArray = m_NvOFAPI->GetAPI()->nvOFGPUBufferGetCUarray(m_hGPUBuffer);
    NVOF_API_CALL(m_NvOFAPI->GetAPI()->nvOFGPUBufferGetStrideInfo(m_hGPUBuffer, &m_strideInfo));
}

NvOFBufferCudaArray::~NvOFBufferCudaArray()
{
    m_NvOFAPI->GetAPI()->nvOFDestroyGPUBufferCuda(m_hGPUBuffer);
}

void NvOFBufferCudaArray::UploadData(const void* pData, CUmemorytype mem_type) {
    CUstream stream = m_NvOFAPI->GetCudaStream(getBufferUsage());
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
    CUDA_MEMCPY2D cuCopy2d;
    memset(&cuCopy2d, 0, sizeof(cuCopy2d));
    cuCopy2d.WidthInBytes = getWidth() * getElementSize();
    mgb_assert(
            CU_MEMORYTYPE_HOST == mem_type || CU_MEMORYTYPE_DEVICE == mem_type,
            "do not imp mem type!!!");
    cuCopy2d.srcMemoryType = mem_type;
    if (CU_MEMORYTYPE_HOST == mem_type) {
        cuCopy2d.srcHost = pData;
    } else if (CU_MEMORYTYPE_DEVICE == mem_type) {
        cuCopy2d.srcDevice = (CUdeviceptr)pData;
    }
    cuCopy2d.srcPitch = cuCopy2d.WidthInBytes;
    cuCopy2d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    cuCopy2d.dstArray= getCudaArray();
    cuCopy2d.Height = getHeight();
    CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));

    if (getBufferFormat() == NV_OF_BUFFER_FORMAT_NV12)
    {
        cuCopy2d.Height = (getHeight() + 1) / 2;
        cuCopy2d.srcHost = ((const uint8_t *)pData + (cuCopy2d.srcPitch * cuCopy2d.Height));
        cuCopy2d.dstY = m_strideInfo.strideInfo[0].strideYInBytes;
        CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));
    }
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(&m_cuContext));
}

void NvOFBufferCudaArray::DownloadData(void* pData, CUmemorytype mem_type) {
    CUstream stream = m_NvOFAPI->GetCudaStream(getBufferUsage());
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
    CUDA_MEMCPY2D cuCopy2d;
    memset(&cuCopy2d, 0, sizeof(cuCopy2d));
    cuCopy2d.WidthInBytes = getWidth() * getElementSize();

    mgb_assert(
            CU_MEMORYTYPE_HOST == mem_type || CU_MEMORYTYPE_DEVICE == mem_type,
            "do not imp mem type!!!");
    cuCopy2d.dstMemoryType = mem_type;
    if (CU_MEMORYTYPE_HOST == mem_type) {
        cuCopy2d.dstHost = pData;
    } else if (CU_MEMORYTYPE_DEVICE == mem_type) {
        cuCopy2d.dstDevice = (CUdeviceptr)pData;
    }
    cuCopy2d.dstPitch = cuCopy2d.WidthInBytes;
    cuCopy2d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    cuCopy2d.srcArray = getCudaArray();
    cuCopy2d.Height = getHeight();
    CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));
    if (getBufferFormat() == NV_OF_BUFFER_FORMAT_NV12)
    {
        cuCopy2d.Height = (getHeight() + 1) / 2;
        cuCopy2d.dstHost = ((uint8_t *)pData + (cuCopy2d.dstPitch * cuCopy2d.Height));
        cuCopy2d.srcY = m_strideInfo.strideInfo[0].strideYInBytes;
        CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));
    }
    CUDA_DRVAPI_CALL(cuStreamSynchronize(stream));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(&m_cuContext));
}

#endif
