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
 * \file src/opr/impl/nvof/NvOF.cpp
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
#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include "NvOF.h"

NvOF::NvOF(uint32_t nWidth, uint32_t nHeight, NV_OF_BUFFER_FORMAT eInBufFmt, NV_OF_MODE eMode, 
    NV_OF_PERF_LEVEL preset) :
    m_nOutGridSize(NV_OF_OUTPUT_VECTOR_GRID_SIZE_MAX),
    m_ePreset(preset),
    m_ofMode(eMode)
{
    m_inputElementSize = 1;
    if (eInBufFmt == NV_OF_BUFFER_FORMAT_ABGR8)
        m_inputElementSize = 4;


    memset(&m_inputBufferDesc, 0, sizeof(m_inputBufferDesc));
    m_inputBufferDesc.width = nWidth;
    m_inputBufferDesc.height = nHeight;
    m_inputBufferDesc.bufferFormat = eInBufFmt;
    m_inputBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;

}

bool NvOF::CheckGridSize(uint32_t nOutGridSize)
{
    uint32_t size;
    DoGetOutputGridSizes(nullptr, &size);

    std::unique_ptr<uint32_t[]> val(new uint32_t[size]);
    DoGetOutputGridSizes(val.get(), &size);

    for (uint32_t i = 0; i < size; i++)
    {
        if (nOutGridSize == val[i])
        {
            return true;
        }
    }
    return false;
}

bool NvOF::GetNextMinGridSize(uint32_t nOutGridSize, uint32_t& nextMinOutGridSize)
{
    uint32_t size;
    DoGetOutputGridSizes(nullptr, &size);

    std::unique_ptr<uint32_t[]> val(new uint32_t[size]);
    DoGetOutputGridSizes(val.get(), &size);

    nextMinOutGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_MAX;
    for (uint32_t i = 0; i < size; i++)
    {
        if (nOutGridSize == val[i])
        {
            nextMinOutGridSize = nOutGridSize;
            return true;
        }
        if (nOutGridSize < val[i] && val[i] < nextMinOutGridSize)
        {
            nextMinOutGridSize = val[i];
        }
    }
    return (nextMinOutGridSize >= NV_OF_OUTPUT_VECTOR_GRID_SIZE_MAX) ? false : true;
}

void NvOF::Init(uint32_t nOutGridSize)
{
    m_nOutGridSize = nOutGridSize;

    auto nOutWidth = (m_inputBufferDesc.width + m_nOutGridSize - 1) / m_nOutGridSize;
    auto nOutHeight = (m_inputBufferDesc.height + m_nOutGridSize - 1) / m_nOutGridSize;

    auto outBufFmt = NV_OF_BUFFER_FORMAT_SHORT2;
    if (m_ofMode == NV_OF_MODE_OPTICALFLOW)
    {
        outBufFmt = NV_OF_BUFFER_FORMAT_SHORT2;
        m_outputElementSize = sizeof(NV_OF_FLOW_VECTOR);
    }
    else if (m_ofMode == NV_OF_MODE_STEREODISPARITY)
    {
        outBufFmt = NV_OF_BUFFER_FORMAT_SHORT;
        m_outputElementSize = sizeof(NV_OF_STEREO_DISPARITY);
    }
    else
    {
        mgb_throw(MegBrainError, "NVOF: Unsupported OF mode err type: NV_OF_ERR_INVALID_PARAM");
    }

    memset(&m_outputBufferDesc, 0, sizeof(m_outputBufferDesc));
    m_outputBufferDesc.width = nOutWidth;
    m_outputBufferDesc.height = nOutHeight;
    m_outputBufferDesc.bufferFormat = outBufFmt;
    m_outputBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;

    memset(&m_costBufferDesc, 0, sizeof(m_costBufferDesc));
    m_costBufferDesc.width = nOutWidth;
    m_costBufferDesc.height = nOutHeight;
    m_costBufferDesc.bufferFormat = NV_OF_BUFFER_FORMAT_UINT;
    m_costBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_COST;
    m_costBufElementSize = sizeof(uint32_t);

    memset(&m_hintBufferDesc, 0, sizeof(m_hintBufferDesc));
    m_hintBufferDesc.width = nOutWidth;
    m_hintBufferDesc.height = nOutHeight;
    m_hintBufferDesc.bufferFormat = outBufFmt;
    m_hintBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_HINT;
    m_hintBufElementSize = m_outputElementSize;

    memset(&m_initParams, 0, sizeof(m_initParams));
    m_initParams.width = m_inputBufferDesc.width;
    m_initParams.height = m_inputBufferDesc.height;
    m_initParams.enableExternalHints = NV_OF_FALSE;
    m_initParams.enableOutputCost = NV_OF_FALSE;
    m_initParams.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    m_initParams.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)m_nOutGridSize;
    m_initParams.mode = m_ofMode;
    m_initParams.perfLevel = m_ePreset;
    DoInit(m_initParams);
}

void NvOF::Execute(NvOFBuffer* image1,
    NvOFBuffer* image2,
    NvOFBuffer* outputBuffer,
    NvOFBuffer* hintBuffer,
    NvOFBuffer* costBuffer)
{
    NV_OF_EXECUTE_INPUT_PARAMS exeInParams;
    NV_OF_EXECUTE_OUTPUT_PARAMS exeOutParams;

    memset(&exeInParams, 0, sizeof(exeInParams));
    exeInParams.inputFrame = image1->getOFBufferHandle();
    exeInParams.referenceFrame = image2->getOFBufferHandle();
    exeInParams.disableTemporalHints = NV_OF_FALSE;
    exeInParams.externalHints = m_initParams.enableExternalHints == NV_OF_TRUE ? hintBuffer->getOFBufferHandle() : nullptr;

    memset(&exeOutParams, 0, sizeof(exeOutParams));
    exeOutParams.outputBuffer = outputBuffer->getOFBufferHandle();
    exeOutParams.outputCostBuffer = m_initParams.enableOutputCost == NV_OF_TRUE ? costBuffer->getOFBufferHandle() : nullptr;
    DoExecute(exeInParams, exeOutParams);
}


std::vector<std::unique_ptr<NvOFBuffer>>
NvOF::CreateBuffers(NV_OF_BUFFER_USAGE usage, uint32_t numBuffers)
{
    std::vector<std::unique_ptr<NvOFBuffer>> ofBuffers;

    if (usage == NV_OF_BUFFER_USAGE_INPUT)
    {
        ofBuffers = DoAllocBuffers(m_inputBufferDesc, m_inputElementSize, numBuffers);
    }
    else if (usage == NV_OF_BUFFER_USAGE_OUTPUT)
    {
        ofBuffers = DoAllocBuffers(m_outputBufferDesc, m_outputElementSize, numBuffers);
    }
    else if (usage == NV_OF_BUFFER_USAGE_COST)
    {
        ofBuffers = DoAllocBuffers(m_costBufferDesc, m_costBufElementSize, numBuffers);
    }
    else if (usage == NV_OF_BUFFER_USAGE_HINT)
    {
        ofBuffers = DoAllocBuffers(m_hintBufferDesc, m_hintBufElementSize, numBuffers);
    }
    else
    {
        mgb_throw(MegBrainError, "NVOF: Invalid parameter err type: NV_OF_ERR_GENERIC");
    }

    return ofBuffers;
}

std::vector<std::unique_ptr<NvOFBuffer>>
NvOF::CreateBuffers(uint32_t nWidth, uint32_t nHeight, NV_OF_BUFFER_USAGE usage, uint32_t numBuffers)
{
    std::vector<std::unique_ptr<NvOFBuffer>> ofBuffers;

    NV_OF_BUFFER_DESCRIPTOR bufferDesc;

    if (usage == NV_OF_BUFFER_USAGE_OUTPUT)
    {
        bufferDesc.width = nWidth;
        bufferDesc.height = nHeight;
        bufferDesc.bufferFormat = m_outputBufferDesc.bufferFormat;
        bufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;

        ofBuffers = DoAllocBuffers(bufferDesc, m_outputElementSize, numBuffers);
    }
    else
    {
        mgb_throw(MegBrainError, "NVOF: Invalid parameter err type: NV_OF_ERR_GENERIC");
    }

    return ofBuffers;
}

void NvOFAPI::LoadNvOFAPI()
{
#if defined(_WIN32)
#if defined(_WIN64)
    HMODULE hModule = LoadLibrary(TEXT("nvofapi64.dll"));
#else
    HMODULE hModule = LoadLibrary(TEXT("nvofapi.dll"));
#endif
#else
    void *hModule = dlopen("libnvidia-opticalflow.so.1", RTLD_LAZY);
#endif
    if (hModule == NULL)
    {
        mgb_throw(
                MegBrainError,
                "NVOF: NVOF library file not found. Please ensure that the "
                "NVIDIA driver is installed type: NV_OF_ERR_OF_NOT_AVAILABLE");
    }

    m_hModule = hModule;
}

#endif
