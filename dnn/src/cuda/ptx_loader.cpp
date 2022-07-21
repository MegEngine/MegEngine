/**
 * \file dnn/src/cuda/ptx_loader.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/ptx_loader.h"
using namespace megdnn;
using namespace cuda;

// ******************* PTXKernelLoader *********************
const std::unordered_map<std::string, PTXKernelLoader::kernel> PTXKernelLoader::KERNEL_MAP =
        {{"ampere_conv_bias_uint4_int4_imma8832_ldg16_256x64_relu",
          ptx::run_ampere_conv_bias_uint4_int4_imma8832_ldg16_256x64_relu},
         {"ampere_conv_bias_uint4_int4_imma8832_ldg16_128x128_relu",
          ptx::run_ampere_conv_bias_uint4_int4_imma8832_ldgsts16_128x128_relu},
         {"ampere_conv_bias_uint4_int4_imma8832_ldg16_128x256_relu",
          ptx::run_ampere_conv_bias_uint4_int4_imma8832_ldg16_128x256_relu},
         {"ampere_conv_bias_uint4_int4_fuse_z_imma8832_ldg16_256x64_relu",
          ptx::run_ampere_conv_bias_uint4_int4_fuse_z_imma8832_ldg16_256x64_relu},
         {"ampere_conv_bias_uint4_int4_fuse_z_imma8832_ldg16_128x128_relu",
          ptx::run_ampere_conv_bias_uint4_int4_fuse_z_imma8832_ldgsts16_128x128_relu},
         {"ampere_conv_bias_uint4_int4_fuse_z_imma8832_ldg16_128x256_relu",
          ptx::run_ampere_conv_bias_uint4_int4_fuse_z_imma8832_ldg16_128x256_relu}};

PTXKernelLoader& PTXKernelLoader::instance() {
    static PTXKernelLoader ins;
    return ins;
}

const PTXKernelLoader::kernel PTXKernelLoader::get_kernel(
        const std::string& kernel_name) {
    decltype(KERNEL_MAP.begin()) kernel_iter;
    kernel_iter = KERNEL_MAP.find(kernel_name);
    megdnn_throw_if(
            kernel_iter == KERNEL_MAP.end(), megdnn_error,
            ssprintf("kernel name %s not found in KERNEL_MAP", kernel_name.c_str())
                    .c_str());

    return kernel_iter->second;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
