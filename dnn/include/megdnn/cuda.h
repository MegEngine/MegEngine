/**
 * \file dnn/include/megdnn/cuda.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/basic_types.h"

#include <cuda_runtime_api.h>
#include <memory>

#include "megdnn/internal/visibility_prologue.h"
namespace megdnn {

std::unique_ptr<Handle> make_cuda_handle_with_stream(cudaStream_t stream,
        int device_id = -1);
cudaStream_t get_cuda_stream(Handle *handle);

} // namespace megdnn
#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
