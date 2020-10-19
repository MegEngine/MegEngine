/**
 * \file src/opr/impl/nvof/denseflownvidia.h
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
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "NvOFCuda.h"

class NVFlowExtractor {
public:
    NVFlowExtractor(int device_id, std::vector<size_t>& shape,
                    uint32_t preset, bool use_cuda_stream, bool debug);
    void create_nvof_instances(int height, int width);
    ~NVFlowExtractor();
    void set_device(int dev_id);
    void init_memory(int batch_size, int temporal_size);
    void extract_flow(unsigned char* frames, std::vector<size_t>&, int16_t*);
    CUmemorytype get_mem_type(CUdeviceptr);
    float get_precision();
    void init_nvof_engine();

private:
    int buffer_pool_size = 6;
    bool debug_flag = false;
    bool m_use_cuda_stream = false;
    bool init_flag = false;
    size_t m_device_id = 0;
    float m_precision = 32.0f;
    uint32_t _preset = 1;
    size_t batch_size = 0;
    size_t out_size = 0;
    size_t m_width = 0;
    size_t m_height = 0;
    size_t m_temporal_size = 0;
    size_t out_width = 0;
    size_t out_height = 0;
    size_t m_width_in_blocks = 0;
    size_t m_height_in_blocks = 0;
    size_t m_blockSizeX = 4;
    size_t m_blockSizeY = 4;

    NV_OF_PERF_LEVEL perf_preset = NV_OF_PERF_LEVEL_MEDIUM;
    NV_OF_BUFFER_FORMAT buffer_format = NV_OF_BUFFER_FORMAT_ABGR8;
    NV_OF_CUDA_BUFFER_TYPE input_buffer_type =
            NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR;
    NV_OF_CUDA_BUFFER_TYPE output_buffer_type =
            NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR;
    NV_OF_OUTPUT_VECTOR_GRID_SIZE m_out_grid_size =
            NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;

    NvOFObj nv_optical_flow;
    CUdevice cu_device = 0;
    CUcontext cu_context = nullptr;
    CUstream input_stream = nullptr;
    CUstream output_stream = nullptr;
    std::vector<NvOFBufferObj> input_buffers;
    std::vector<NvOFBufferObj> output_buffers;

protected:
    std::mutex m_lock;
};

#endif
