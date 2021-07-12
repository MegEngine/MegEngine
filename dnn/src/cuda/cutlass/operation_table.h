/***************************************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file dnn/src/cuda/cutlass/operation_table.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <unordered_map>

#include "src/common/hash_ct.h"
#include "src/cuda/cutlass/manifest.h"
#include "src/cuda/cutlass/util.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

class Hash {
public:
    Hash() : m_val(0) {}

    Hash& update(const void* ptr, size_t len) {
        m_val += megdnn::XXHash64CT::hash((const char*)ptr, len, 123456);
        return *this;
    }

    uint64_t digest() const { return m_val; }

private:
    uint64_t m_val;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//                          Data Structures for GemmOperationMap
/////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmKey {
    NumericTypeID element_A;
    LayoutTypeID layout_A;
    NumericTypeID element_B;
    LayoutTypeID layout_B;
    NumericTypeID element_C;
    LayoutTypeID layout_C;

    int threadblock_shape_m;
    int threadblock_shape_n;
    int threadblock_shape_k;

    int warp_shape_m;
    int warp_shape_n;
    int warp_shape_k;

    int instruction_shape_m;
    int instruction_shape_n;
    int instruction_shape_k;

    int stages;
    SplitKMode split_k_mode;

    inline bool operator==(GemmKey const& rhs) const {
        return (element_A == rhs.element_A) && (layout_A == rhs.layout_A) &&
               (element_B == rhs.element_B) && (layout_B == rhs.layout_B) &&
               (element_C == rhs.element_C) && (layout_C == rhs.layout_C) &&
               (threadblock_shape_m == rhs.threadblock_shape_m) &&
               (threadblock_shape_n == rhs.threadblock_shape_n) &&
               (threadblock_shape_k == rhs.threadblock_shape_k) &&
               (warp_shape_m == rhs.warp_shape_m) &&
               (warp_shape_n == rhs.warp_shape_n) &&
               (warp_shape_k == rhs.warp_shape_k) &&
               (instruction_shape_m == rhs.instruction_shape_m) &&
               (instruction_shape_n == rhs.instruction_shape_n) &&
               (instruction_shape_k == rhs.instruction_shape_k) &&
               (stages == rhs.stages) && (split_k_mode == rhs.split_k_mode);
    }

    inline bool operator!=(GemmKey const& rhs) const { return !(*this == rhs); }

    inline std::string str() const {
        auto tuple_to_str = [](int m, int n, int k) -> std::string {
            return std::to_string(m) + " x " + std::to_string(n) + " x " +
                   std::to_string(k);
        };

        std::string threadblock_shape_str = tuple_to_str(
                threadblock_shape_m, threadblock_shape_n, threadblock_shape_k);
        std::string warp_shape_str =
                tuple_to_str(warp_shape_m, warp_shape_n, warp_shape_k);
        std::string instruction_shape_str = tuple_to_str(
                instruction_shape_m, instruction_shape_n, instruction_shape_k);

        return std::string("{") + "\n    element_A: " + to_string(element_A) +
               "\n    layout_A: " + to_string(layout_A) +
               "\n    element_B: " + to_string(element_B) +
               "\n    layout_B: " + to_string(layout_B) +
               "\n    element_C: " + to_string(element_C) +
               "\n    layout_C: " + to_string(layout_C) +
               "\n    threadblock_shape: " + threadblock_shape_str +
               "\n    warp_shape: " + warp_shape_str +
               "\n    instruction_shape: " + instruction_shape_str +
               "\n    stages: " + std::to_string(stages) +
               "\n    split_k_mode: " + to_string(split_k_mode) + "\n}";
    }
};

struct GemmKeyHasher {
    inline size_t operator()(GemmKey const& key) const {
        return Hash()
                .update(&key.element_A, sizeof(key.element_A))
                .update(&key.layout_A, sizeof(key.layout_A))
                .update(&key.element_B, sizeof(key.element_B))
                .update(&key.layout_B, sizeof(key.layout_B))
                .update(&key.element_C, sizeof(key.element_C))
                .update(&key.layout_C, sizeof(key.layout_C))
                .update(&key.threadblock_shape_m,
                        sizeof(key.threadblock_shape_m))
                .update(&key.threadblock_shape_n,
                        sizeof(key.threadblock_shape_n))
                .update(&key.threadblock_shape_k,
                        sizeof(key.threadblock_shape_k))
                .update(&key.warp_shape_m, sizeof(key.warp_shape_m))
                .update(&key.warp_shape_n, sizeof(key.warp_shape_n))
                .update(&key.warp_shape_k, sizeof(key.warp_shape_k))
                .update(&key.stages, sizeof(key.stages))
                .update(&key.split_k_mode, sizeof(key.split_k_mode))
                .digest();
    }
};

using GemmOperationMap =
        std::unordered_map<GemmKey, std::vector<Operation const*>,
                           GemmKeyHasher>;

/////////////////////////////////////////////////////////////////////////////////////////////////
//                          Data Structures for ConvolutionOperationMap
/////////////////////////////////////////////////////////////////////////////////////////////////

struct ConvolutionKey {
    conv::Operator conv_op;

    library::NumericTypeID element_src;
    library::LayoutTypeID layout_src;
    library::NumericTypeID element_filter;
    library::LayoutTypeID layout_filter;
    library::NumericTypeID element_dst;
    library::LayoutTypeID layout_dst;
    library::NumericTypeID element_bias;
    library::LayoutTypeID layout_bias;

    conv::ConvType convolution_type;

    int threadblock_shape_m;
    int threadblock_shape_n;
    int threadblock_shape_k;

    int warp_shape_m;
    int warp_shape_n;
    int warp_shape_k;

    int instruction_shape_m;
    int instruction_shape_n;
    int instruction_shape_k;

    epilogue::EpilogueType epilogue_type;
    int stages;
    bool need_load_from_const_mem;
    bool without_shared_load;

    inline bool operator==(ConvolutionKey const& rhs) const {
        return (conv_op == rhs.conv_op) && (element_src == rhs.element_src) &&
               (layout_src == rhs.layout_src) &&
               (element_filter == rhs.element_filter) &&
               (layout_filter == rhs.layout_filter) &&
               (element_dst == rhs.element_dst) &&
               (layout_dst == rhs.layout_dst) &&
               (element_bias == rhs.element_bias) &&
               (layout_bias == rhs.layout_bias) &&
               (convolution_type == rhs.convolution_type) &&
               (threadblock_shape_m == rhs.threadblock_shape_m) &&
               (threadblock_shape_n == rhs.threadblock_shape_n) &&
               (threadblock_shape_k == rhs.threadblock_shape_k) &&
               (warp_shape_m == rhs.warp_shape_m) &&
               (warp_shape_n == rhs.warp_shape_n) &&
               (warp_shape_k == rhs.warp_shape_k) &&
               (instruction_shape_m == rhs.instruction_shape_m) &&
               (instruction_shape_n == rhs.instruction_shape_n) &&
               (instruction_shape_k == rhs.instruction_shape_k) &&
               (epilogue_type == rhs.epilogue_type) && (stages == rhs.stages) &&
               (need_load_from_const_mem == rhs.need_load_from_const_mem) &&
               (without_shared_load == rhs.without_shared_load);
    }

    inline bool operator!=(ConvolutionKey const& rhs) const {
        return !(*this == rhs);
    }

    inline std::string str() const {
        auto tuple_to_str = [](int m, int n, int k) -> std::string {
            return std::to_string(m) + " x " + std::to_string(n) + " x " +
                   std::to_string(k);
        };

        std::string threadblock_shape_str = tuple_to_str(
                threadblock_shape_m, threadblock_shape_n, threadblock_shape_k);
        std::string warp_shape_str =
                tuple_to_str(warp_shape_m, warp_shape_n, warp_shape_k);
        std::string instruction_shape_str = tuple_to_str(
                instruction_shape_m, instruction_shape_n, instruction_shape_k);

        return std::string("{") + "\n    conv_op: " + to_string(conv_op) +
               "\n    element_src: " + to_string(element_src) +
               "\n    layout_src: " + to_string(layout_src) +
               "\n    element_filter: " + to_string(element_filter) +
               "\n    layout_filter: " + to_string(layout_filter) +
               "\n    element_dst: " + to_string(element_dst) +
               "\n    layout_dst: " + to_string(layout_dst) +
               "\n    element_bias: " + to_string(element_bias) +
               "\n    layout_bias: " + to_string(layout_bias) +
               "\n    convolution_type: " + to_string(convolution_type) +
               "\n    threadblock_shape: " + threadblock_shape_str +
               "\n    warp_shape: " + warp_shape_str +
               "\n    instruction_shape: " + instruction_shape_str +
               "\n    epilogue_type: " + to_string(epilogue_type) +
               "\n    stages: " + std::to_string(stages) +
               "\n    need_load_from_const_mem: " +
               to_string(need_load_from_const_mem) +
               "\n    without_shared_load: " + to_string(without_shared_load) +
               "\n}";
    }
};

struct ConvolutionKeyHasher {
    inline size_t operator()(ConvolutionKey const& key) const {
        return Hash()
                .update(&key.conv_op, sizeof(key.conv_op))
                .update(&key.conv_op, sizeof(key.conv_op))
                .update(&key.element_src, sizeof(key.element_src))
                .update(&key.layout_src, sizeof(key.layout_src))
                .update(&key.element_filter, sizeof(key.element_filter))
                .update(&key.layout_filter, sizeof(key.layout_filter))
                .update(&key.element_dst, sizeof(key.element_dst))
                .update(&key.layout_dst, sizeof(key.layout_dst))
                .update(&key.element_bias, sizeof(key.element_bias))
                .update(&key.layout_bias, sizeof(key.layout_bias))
                .update(&key.convolution_type, sizeof(key.convolution_type))
                .update(&key.threadblock_shape_m,
                        sizeof(key.threadblock_shape_m))
                .update(&key.threadblock_shape_n,
                        sizeof(key.threadblock_shape_n))
                .update(&key.threadblock_shape_k,
                        sizeof(key.threadblock_shape_k))
                .update(&key.warp_shape_m, sizeof(key.warp_shape_m))
                .update(&key.warp_shape_n, sizeof(key.warp_shape_n))
                .update(&key.warp_shape_k, sizeof(key.warp_shape_k))
                .update(&key.instruction_shape_m,
                        sizeof(key.instruction_shape_m))
                .update(&key.instruction_shape_n,
                        sizeof(key.instruction_shape_n))
                .update(&key.instruction_shape_k,
                        sizeof(key.instruction_shape_k))
                .update(&key.epilogue_type, sizeof(key.epilogue_type))
                .update(&key.stages, sizeof(key.stages))
                .update(&key.need_load_from_const_mem,
                        sizeof(key.need_load_from_const_mem))
                .update(&key.without_shared_load,
                        sizeof(key.without_shared_load))
                .digest();
    }
};

using ConvolutionOperationMap =
        std::unordered_map<ConvolutionKey, std::vector<Operation const*>,
                           ConvolutionKeyHasher>;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Table of cutlass::library::Operation instances
class OperationTable {
public:
    /// Map of all operations of type kGemm
    GemmOperationMap gemm_operations;

    /// Map of all operations of type kConvolution
    ConvolutionOperationMap convolution_operations;

public:
    void append(Manifest const& manifest);

    Operation const* find_op(GemmKey const& key) const;

    Operation const* find_op(ConvolutionKey const& key) const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace library
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
