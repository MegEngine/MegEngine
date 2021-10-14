/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
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
 * \file dnn/src/cuda/cutlass/gemm_operation.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cutlass/gemm/device/gemm.h"
#include "src/cuda/cutlass/library_internal.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Check whether Operator has member ReductionKernel using SFINAE (Substitution
/// Failure Is Not An Error)
template <typename Operator>
struct split_k_mode {
    template <typename T>
    static char check(typename T::ReductionKernel*);

    template <typename T>
    static int check(...);

    SplitKMode operator()() {
        if (sizeof(check<Operator>(0)) == sizeof(char)) {
            // cutlass::gemm::device::GemmSplitKParallel
            return SplitKMode::kParallel;
        } else {
            // cutlass::gemm::device::Gemm
            return SplitKMode::kNone;
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmOperationBase : public Operation {
public:
    using Operator = Operator_;
    using ElementA = typename Operator::ElementA;
    using LayoutA = typename Operator::LayoutA;
    using ElementB = typename Operator::ElementB;
    using LayoutB = typename Operator::LayoutB;
    using ElementC = typename Operator::ElementC;
    using LayoutC = typename Operator::LayoutC;
    using ElementAccumulator = typename Operator::ElementAccumulator;

    GemmOperationBase(char const* name = "unknown_gemm") {
        m_description.name = name;
        m_description.provider = Provider::kCUTLASS;
        m_description.kind = OperationKind::kGemm;
        m_description.gemm_kind = GemmKind::kGemm;

        m_description.tile_description.threadblock_shape = make_Coord(
                Operator::ThreadblockShape::kM, Operator::ThreadblockShape::kN,
                Operator::ThreadblockShape::kK);

        m_description.tile_description.threadblock_stages = Operator::kStages;

        m_description.tile_description.warp_count = make_Coord(
                Operator::GemmKernel::WarpCount::kM,
                Operator::GemmKernel::WarpCount::kN,
                Operator::GemmKernel::WarpCount::kK);

        m_description.tile_description.math_instruction.instruction_shape = make_Coord(
                Operator::InstructionShape::kM, Operator::InstructionShape::kN,
                Operator::InstructionShape::kK);

        m_description.tile_description.math_instruction.element_accumulator =
                NumericTypeMap<ElementAccumulator>::kId;

        m_description.tile_description.math_instruction.opcode_class =
                OpcodeClassMap<typename Operator::OperatorClass>::kId;

        m_description.tile_description.math_instruction.math_operation =
                MathOperationMap<typename Operator::Operator>::kId;

        m_description.tile_description.minimum_compute_capability =
                ArchMap<typename Operator::ArchTag,
                        typename Operator::OperatorClass>::kMin;

        m_description.tile_description.maximum_compute_capability =
                ArchMap<typename Operator::ArchTag,
                        typename Operator::OperatorClass>::kMax;

        m_description.A =
                make_TensorDescription<ElementA, LayoutA>(Operator::kAlignmentA);
        m_description.B =
                make_TensorDescription<ElementB, LayoutB>(Operator::kAlignmentB);
        m_description.C =
                make_TensorDescription<ElementC, LayoutC>(Operator::kAlignmentC);

        m_description.stages = Operator::kStages;

        split_k_mode<Operator> mode;
        m_description.split_k_mode = mode();
    }

    virtual OperationDescription const& description() const { return m_description; }

protected:
    GemmDescription m_description;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmOperation : public GemmOperationBase<Operator_> {
public:
    using Operator = Operator_;
    using ElementA = typename Operator::ElementA;
    using LayoutA = typename Operator::LayoutA;
    using ElementB = typename Operator::ElementB;
    using LayoutB = typename Operator::LayoutB;
    using ElementC = typename Operator::ElementC;
    using LayoutC = typename Operator::LayoutC;
    using ElementAccumulator = typename Operator::ElementAccumulator;
    using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

    using OperatorArguments = typename Operator::Arguments;

    GemmOperation(char const* name = "unknown_gemm")
            : GemmOperationBase<Operator_>(name) {}

    virtual Status run(
            void const* arguments_ptr, void* device_workspace = nullptr,
            cudaStream_t stream = nullptr) const {
        GemmArguments const* gemm_args =
                reinterpret_cast<GemmArguments const*>(arguments_ptr);

        OperatorArguments args;
        args.problem_size = gemm_args->problem_size;
        args.ref_A = {static_cast<ElementA const*>(gemm_args->A), int(gemm_args->lda)};
        args.ref_B = {static_cast<ElementB const*>(gemm_args->B), int(gemm_args->ldb)};
        args.ref_C = {static_cast<ElementC const*>(gemm_args->C), int(gemm_args->ldc)};
        args.ref_D = {static_cast<ElementC*>(gemm_args->D), int(gemm_args->ldd)};
        args.split_k_slices = gemm_args->split_k_slices;

        args.epilogue = {
                *static_cast<ElementCompute const*>(gemm_args->alpha),
                *static_cast<ElementCompute const*>(gemm_args->beta)};

        Operator op;
        Status status = op.initialize(args, device_workspace);

        if (status != Status::kSuccess) {
            return status;
        }

        return op.run(stream);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace library
}  // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
