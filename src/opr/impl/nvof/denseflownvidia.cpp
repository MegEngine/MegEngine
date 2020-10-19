/**
 * \file src/opr/impl/nvof/denseflownvidia.cpp
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
#include <mutex>
#include <vector>
#include "megbrain/common.h"
#include "denseflownvidia.h"

NVFlowExtractor::NVFlowExtractor(int device_id, std::vector<size_t>& shape,
                                 uint32_t preset, bool use_cuda_stream,
                                 bool debug) {
    batch_size = shape[0];
    m_width = shape[3];
    m_height = shape[2];
    debug_flag = debug;
    m_temporal_size = shape[1];
    m_use_cuda_stream = use_cuda_stream;
    out_width = (m_width + m_out_grid_size - 1) / m_out_grid_size;
    out_height = (m_height + m_out_grid_size - 1) / m_out_grid_size;
    m_width_in_blocks = (m_width + m_blockSizeX - 1) / m_blockSizeX;
    m_height_in_blocks = (m_height + m_blockSizeY - 1) / m_blockSizeY;
    out_size = out_width * out_height * 2;
    m_device_id = device_id;

    std::unordered_map<uint32_t, NV_OF_PERF_LEVEL> preset_map = {
            {0, NV_OF_PERF_LEVEL_SLOW},
            {1, NV_OF_PERF_LEVEL_MEDIUM},
            {2, NV_OF_PERF_LEVEL_FAST}};

    _preset = preset;
    auto search = preset_map.find(_preset);
    if (search == preset_map.end()) {
        mgb_throw(MegBrainError, "NVOF: invalid preset level! err type: NV_OF_ERR_INVALID_PARAM");
    }
    perf_preset = search->second;
}

void NVFlowExtractor::create_nvof_instances(int height, int width) {
    nv_optical_flow = NvOFCuda::Create(cu_context, width, height, buffer_format,
                                       input_buffer_type, output_buffer_type,
                                       NV_OF_MODE_OPTICALFLOW, perf_preset,
                                       input_stream, output_stream);
    nv_optical_flow->Init(m_out_grid_size);
    input_buffers = nv_optical_flow->CreateBuffers(
            NV_OF_BUFFER_USAGE_INPUT, buffer_pool_size * batch_size);
    output_buffers = nv_optical_flow->CreateBuffers(
            NV_OF_BUFFER_USAGE_OUTPUT, (buffer_pool_size - 1) * batch_size);
}

void NVFlowExtractor::init_nvof_engine() {
    std::lock_guard<std::mutex> lock(m_lock);
    if (init_flag == false) {
        set_device(m_device_id);
        if (cuCtxCreate(&cu_context, 0, cu_device)) {
            mgb_log_warn(
                    "nvof: create ctx failed, fallback to get current ctx");
            CUDA_DRVAPI_CALL(cuCtxGetCurrent(&cu_context));
        }

        if (m_use_cuda_stream) {
            CUDA_DRVAPI_CALL(cuStreamCreate(&input_stream, CU_STREAM_DEFAULT));
            CUDA_DRVAPI_CALL(cuStreamCreate(&output_stream, CU_STREAM_DEFAULT));
        }
        create_nvof_instances(m_height, m_width);
        init_flag = true;
    }
}

NVFlowExtractor::~NVFlowExtractor() {
    if (debug_flag) {
        mgb_log_debug("%s: %d start", __FUNCTION__, __LINE__);
    }

    if (m_use_cuda_stream) {
        cuStreamDestroy(output_stream);
        output_stream = nullptr;
        cuStreamDestroy(input_stream);
        input_stream = nullptr;
    }

    if (debug_flag) {
        mgb_log_debug("%s: %d end", __FUNCTION__, __LINE__);
    }
}

void NVFlowExtractor::set_device(int dev_id) {
    int nGpu = 0;

    if (debug_flag) {
        mgb_log_warn("config nvof gpu device id: %d", dev_id);
    }

    CUDA_DRVAPI_CALL(cuInit(0));
    CUDA_DRVAPI_CALL(cuDeviceGetCount(&nGpu));
    if (dev_id < 0 || dev_id >= nGpu) {
        mgb_log_warn("GPU ordinal out of range. Should be with in [0, %d]",
                     nGpu - 1);
        mgb_throw(MegBrainError, "NVOF: GPU Setting Error! err type: NV_OF_ERR_GENERIC");
    }
    CUDA_DRVAPI_CALL(cuDeviceGet(&cu_device, dev_id));
}

CUmemorytype NVFlowExtractor::get_mem_type(CUdeviceptr p) {
    unsigned int mem_type;
    auto ret = cuPointerGetAttribute(&mem_type,
                                     CU_POINTER_ATTRIBUTE_MEMORY_TYPE, p);

    if (CUDA_SUCCESS == ret) {
        mgb_assert(
                CU_MEMORYTYPE_DEVICE == mem_type ||
                        CU_MEMORYTYPE_HOST == mem_type,
                "only imp CU_MEMORYTYPE_HOST or CU_MEMORYTYPE_DEVICE mem type");
    } else {
        mgb_log_warn(
                "nvof call cuPointerGetAttribute err!!, may init nvof opr on "
                "cpu comp_node, force set mem type to CU_MEMORYTYPE_HOST");
        mem_type = CU_MEMORYTYPE_HOST;
    }

    return static_cast<CUmemorytype_enum>(mem_type);
}

void NVFlowExtractor::extract_flow(unsigned char* frames,
                                   std::vector<size_t>& shape,
                                   int16_t* result_out_ptr) {
    auto batch_size = shape[0];
    auto temporal_size = shape[1];
    auto height = shape[2];
    auto width = shape[3];
    auto channel = shape[4];
    auto temporal_len = height * width * channel;
    auto batch_len = temporal_size * height * width * channel;

    init_nvof_engine();

    auto src_mem_type = get_mem_type(reinterpret_cast<CUdeviceptr>(frames));
    auto out_mem_type =
            get_mem_type(reinterpret_cast<CUdeviceptr>(result_out_ptr));

    if ((height != m_height || width != m_width) ||
        (m_temporal_size != temporal_size)) {
        mgb_log_warn("We do not support dynamic shape at mgb side");
        mgb_throw(MegBrainError, "NVOF: Nvof err shap!!!! err type: NV_OF_ERR_GENERIC");
    }

    for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        auto input_buffer_batch_offsect = buffer_pool_size * batch_idx;
        auto output_buffer_batch_offsect = (buffer_pool_size - 1) * batch_idx;
        input_buffers[input_buffer_batch_offsect]->UploadData(
                (unsigned char*)(frames + batch_idx * batch_len), src_mem_type);

        for (size_t temporal_idx = 1; temporal_idx < temporal_size;
             temporal_idx++) {
            input_buffers[input_buffer_batch_offsect +
                          temporal_idx % buffer_pool_size]
                    ->UploadData(
                            (unsigned char*)(frames + batch_idx * batch_len +
                                             temporal_idx * temporal_len),
                            src_mem_type);

            nv_optical_flow->Execute(
                    input_buffers[input_buffer_batch_offsect +
                                  (temporal_idx - 1) % buffer_pool_size]
                            .get(),
                    input_buffers[input_buffer_batch_offsect +
                                  temporal_idx % buffer_pool_size]
                            .get(),
                    output_buffers[output_buffer_batch_offsect +
                                   (temporal_idx - 1) % (buffer_pool_size - 1)]
                            .get(),
                    nullptr, nullptr);

            output_buffers[output_buffer_batch_offsect +
                           (temporal_idx - 1) % (buffer_pool_size - 1)]
                    ->DownloadData(
                            result_out_ptr +
                                    batch_idx * (temporal_size - 1) * out_size +
                                    (temporal_idx - 1) * out_size,
                            out_mem_type);
        }
    }

    CUDA_DRVAPI_CALL(cuCtxSynchronize());
}

float NVFlowExtractor::get_precision() {
    return m_precision;
}

#endif
