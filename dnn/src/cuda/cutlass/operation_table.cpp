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
#include "src/cuda/cutlass/operation_table.h"
#include "src/common/utils.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

GemmKey get_gemm_key_from_desc(const GemmDescription& desc) {
    GemmKey key;

    key.element_A = desc.A.element;
    key.layout_A = desc.A.layout;
    key.element_B = desc.B.element;
    key.layout_B = desc.B.layout;
    key.element_C = desc.C.element;
    key.layout_C = desc.C.layout;
    key.element_accumulator =
            desc.tile_description.math_instruction.element_accumulator;

    key.threadblock_shape_m = desc.tile_description.threadblock_shape.m();
    key.threadblock_shape_n = desc.tile_description.threadblock_shape.n();
    key.threadblock_shape_k = desc.tile_description.threadblock_shape.k();

    key.warp_shape_m = desc.tile_description.threadblock_shape.m() /
                       desc.tile_description.warp_count.m();
    key.warp_shape_n = desc.tile_description.threadblock_shape.n() /
                       desc.tile_description.warp_count.n();
    key.warp_shape_k = desc.tile_description.threadblock_shape.k() /
                       desc.tile_description.warp_count.k();

    key.instruction_shape_m =
            desc.tile_description.math_instruction.instruction_shape.m();
    key.instruction_shape_n =
            desc.tile_description.math_instruction.instruction_shape.n();
    key.instruction_shape_k =
            desc.tile_description.math_instruction.instruction_shape.k();

    key.stages = desc.stages;
    key.alignment_A = desc.A.alignment;
    key.alignment_B = desc.B.alignment;
    key.split_k_mode = desc.split_k_mode;

    return key;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

ConvolutionKey get_convolution_key_from_desc(const ConvolutionDescription& desc) {
    ConvolutionKey key;

    key.conv_op = desc.conv_op;

    key.element_src = desc.src.element;
    key.layout_src = desc.src.layout;
    key.element_filter = desc.filter.element;
    key.layout_filter = desc.filter.layout;
    key.element_dst = desc.dst.element;
    key.layout_dst = desc.dst.layout;
    key.element_bias = desc.bias.element;
    key.layout_bias = desc.bias.layout;
    key.element_accumulator =
            desc.tile_description.math_instruction.element_accumulator;

    key.convolution_type = desc.convolution_type;

    key.threadblock_shape_m = desc.tile_description.threadblock_shape.m();
    key.threadblock_shape_n = desc.tile_description.threadblock_shape.n();
    key.threadblock_shape_k = desc.tile_description.threadblock_shape.k();

    key.warp_shape_m = desc.tile_description.threadblock_shape.m() /
                       desc.tile_description.warp_count.m();
    key.warp_shape_n = desc.tile_description.threadblock_shape.n() /
                       desc.tile_description.warp_count.n();
    key.warp_shape_k = desc.tile_description.threadblock_shape.k() /
                       desc.tile_description.warp_count.k();

    key.instruction_shape_m =
            desc.tile_description.math_instruction.instruction_shape.m();
    key.instruction_shape_n =
            desc.tile_description.math_instruction.instruction_shape.n();
    key.instruction_shape_k =
            desc.tile_description.math_instruction.instruction_shape.k();

    key.epilogue_type = desc.epilogue_type;

    key.stages = desc.tile_description.threadblock_stages;
    key.special_optimization = desc.special_optimization;
    key.alignment_src = desc.src.alignment;
    key.alignment_filter = desc.filter.alignment;
    key.without_shared_load = desc.without_shared_load;

    key.element_rin = desc.rin.element;
    key.layout_rin = desc.rin.layout;
    key.element_rout = desc.rout.element;
    key.layout_rout = desc.rout.layout;

    return key;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

void OperationTable::append(Manifest const& manifest) {
    // Insert operations into appropriate data structure
    for (auto const& operation : manifest) {
        OperationDescription const& desc = operation->description();

        // insert all gemm operations into operation table
        if (desc.kind == OperationKind::kGemm) {
            GemmKey key =
                    get_gemm_key_from_desc(static_cast<GemmDescription const&>(desc));
            gemm_operations[key].push_back(operation.get());
        }

        // insert all conv operations into operation table
        if (desc.kind == OperationKind::kConvolution) {
            ConvolutionKey key = get_convolution_key_from_desc(
                    static_cast<ConvolutionDescription const&>(desc));
            convolution_operations[key].push_back(operation.get());
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Operation const* OperationTable::find_op(GemmKey const& key) const {
    if (gemm_operations.count(key)) {
        auto const& ops = gemm_operations.at(key);
        megdnn_assert(
                ops.size() == 1, "exactly one kernel expected, got %zu", ops.size());
        return ops[0];
    }
    return nullptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Operation const* OperationTable::find_op(ConvolutionKey const& key) const {
    if (convolution_operations.count(key) > 0) {
        auto const& ops = convolution_operations.at(key);
        megdnn_assert(
                ops.size() == 1, "exactly one kernel expected, got %zu", ops.size());
        return ops[0];
    }
    return nullptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace library
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
