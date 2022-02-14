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
 * \file dnn/src/cuda/cutlass/convolution_operation.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cutlass/convolution/device/convolution.h"
#include "src/cuda/cutlass/library_internal.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class ConvolutionOperationBase : public Operation {
public:
    using Operator = Operator_;
    using ElementSrc = typename Operator::ElementSrc;
    using LayoutSrc = typename Operator::LayoutSrc;
    using ElementFilter = typename Operator::ElementFilter;
    using LayoutFilter = typename Operator::LayoutFilter;
    using ElementDst = typename Operator::ElementDst;
    using LayoutDst = typename Operator::LayoutDst;
    using ElementBias = typename Operator::ElementBias;
    using LayoutBias = typename Operator::LayoutBias;
    using ElementAccumulator = typename Operator::ElementAccumulator;

    ConvolutionOperationBase(char const* name = "unknown_convolution") {
        m_description.name = name;
        m_description.provider = Provider::kCUTLASS;
        m_description.kind = OperationKind::kConvolution;
        m_description.conv_op = Operator::kConvolutionalOperator;

        m_description.tile_description.threadblock_shape = make_Coord(
                Operator::ThreadblockShape::kM, Operator::ThreadblockShape::kN,
                Operator::ThreadblockShape::kK);

        m_description.tile_description.threadblock_stages = Operator::kStages;

        m_description.tile_description.warp_count = make_Coord(
                Operator::ConvolutionKernel::WarpCount::kM,
                Operator::ConvolutionKernel::WarpCount::kN,
                Operator::ConvolutionKernel::WarpCount::kK);

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

        m_description.src =
                make_TensorDescription<ElementSrc, LayoutSrc>(Operator::kAlignmentSrc);
        m_description.filter = make_TensorDescription<ElementFilter, LayoutFilter>(
                Operator::kAlignmentFilter);
        m_description.dst =
                make_TensorDescription<ElementDst, LayoutDst>(Operator::kAlignmentDst);
        m_description.bias = make_TensorDescription<ElementBias, LayoutBias>(
                Operator::kAlignmentDst);

        m_description.convolution_type = Operator::kConvolutionType;
        m_description.arch_tag = ArchTagMap<typename Operator::ArchTag>::kId;

        m_description.epilogue_type = Operator::EpilogueOutputOp::kType;
        m_description.epilogue_count = Operator::EpilogueOutputOp::kCount;

        m_description.threadblock_swizzle =
                ThreadblockSwizzleMap<typename Operator::ThreadblockSwizzle>::kId;

        m_description.special_optimization = Operator::kSpecialOpt;
        m_description.gemm_mode = Operator::kGemmMode;
        m_description.without_shared_load = Operator::kWithoutSharedLoad;
    }

    virtual OperationDescription const& description() const { return m_description; }

protected:
    ConvolutionDescription m_description;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename EpilogueOp, epilogue::EpilogueType type>
struct init_epilogue_param_;

template <typename EpilogueOp>
struct init_epilogue_param_<EpilogueOp, epilogue::EpilogueType::kLinearCombination> {
    using ElementCompute = typename EpilogueOp::ElementCompute;
    typename EpilogueOp::Params get(ConvolutionArguments const* conv_args) {
        return {*static_cast<ElementCompute const*>(conv_args->alpha),
                *static_cast<ElementCompute const*>(conv_args->beta)};
    }
};

template <typename EpilogueOp>
struct init_epilogue_param_<
        EpilogueOp, epilogue::EpilogueType::kBiasAddLinearCombination> {
    using ElementCompute = typename EpilogueOp::ElementCompute;
    typename EpilogueOp::Params get(ConvolutionArguments const* conv_args) {
        return {*static_cast<ElementCompute const*>(conv_args->alpha),
                *static_cast<ElementCompute const*>(conv_args->beta),
                *static_cast<ElementCompute const*>(conv_args->gamma),
                *static_cast<ElementCompute const*>(conv_args->delta)};
    }
};

template <typename EpilogueOp>
struct init_epilogue_param_<
        EpilogueOp, epilogue::EpilogueType::kBiasAddLinearCombinationClamp> {
    using ElementCompute = typename EpilogueOp::ElementCompute;
    typename EpilogueOp::Params get(ConvolutionArguments const* conv_args) {
        return {*static_cast<ElementCompute const*>(conv_args->alpha),
                *static_cast<ElementCompute const*>(conv_args->beta),
                *static_cast<ElementCompute const*>(conv_args->gamma),
                *static_cast<ElementCompute const*>(conv_args->delta)};
    }
};

template <typename EpilogueOp>
struct init_epilogue_param_<
        EpilogueOp, epilogue::EpilogueType::kBiasAddLinearCombinationRelu> {
    using ElementCompute = typename EpilogueOp::ElementCompute;
    typename EpilogueOp::Params get(ConvolutionArguments const* conv_args) {
        return {*static_cast<ElementCompute const*>(conv_args->alpha),
                *static_cast<ElementCompute const*>(conv_args->beta),
                *static_cast<ElementCompute const*>(conv_args->gamma),
                *static_cast<ElementCompute const*>(conv_args->threshold),
                *static_cast<ElementCompute const*>(conv_args->delta),
                *static_cast<ElementCompute const*>(conv_args->theta)};
    }
};

template <typename EpilogueOp>
struct init_epilogue_param_<
        EpilogueOp, epilogue::EpilogueType::kBiasAddLinearCombinationReluClamp> {
    using ElementCompute = typename EpilogueOp::ElementCompute;
    typename EpilogueOp::Params get(ConvolutionArguments const* conv_args) {
        return {*static_cast<ElementCompute const*>(conv_args->alpha),
                *static_cast<ElementCompute const*>(conv_args->beta),
                *static_cast<ElementCompute const*>(conv_args->gamma),
                *static_cast<ElementCompute const*>(conv_args->threshold),
                *static_cast<ElementCompute const*>(conv_args->delta),
                *static_cast<ElementCompute const*>(conv_args->theta)};
    }
};

template <typename EpilogueOp>
struct init_epilogue_param_<
        EpilogueOp, epilogue::EpilogueType::kBiasAddLinearCombinationHSwish> {
    using ElementCompute = typename EpilogueOp::ElementCompute;
    typename EpilogueOp::Params get(ConvolutionArguments const* conv_args) {
        return {*static_cast<ElementCompute const*>(conv_args->alpha),
                *static_cast<ElementCompute const*>(conv_args->beta),
                *static_cast<ElementCompute const*>(conv_args->gamma),
                *static_cast<ElementCompute const*>(conv_args->scale),
                *static_cast<ElementCompute const*>(conv_args->delta),
                *static_cast<ElementCompute const*>(conv_args->theta)};
    }
};

template <typename EpilogueOp>
struct init_epilogue_param_<
        EpilogueOp, epilogue::EpilogueType::kBiasAddLinearCombinationHSwishClamp> {
    using ElementCompute = typename EpilogueOp::ElementCompute;
    typename EpilogueOp::Params get(ConvolutionArguments const* conv_args) {
        return {*static_cast<ElementCompute const*>(conv_args->alpha),
                *static_cast<ElementCompute const*>(conv_args->beta),
                *static_cast<ElementCompute const*>(conv_args->gamma),
                *static_cast<ElementCompute const*>(conv_args->scale),
                *static_cast<ElementCompute const*>(conv_args->delta),
                *static_cast<ElementCompute const*>(conv_args->theta)};
    }
};

}  // namespace detail

template <typename EpilogueOp>
struct init_epilogue_param
        : public detail::init_epilogue_param_<EpilogueOp, EpilogueOp::kType> {};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class ConvolutionOperation : public ConvolutionOperationBase<Operator_> {
public:
    using Operator = Operator_;
    using ElementSrc = typename Operator::ElementSrc;
    using LayoutSrc = typename Operator::LayoutSrc;
    using ElementFilter = typename Operator::ElementFilter;
    using LayoutFilter = typename Operator::LayoutFilter;
    using ElementBias = typename Operator::ElementBias;
    using LayoutBias = typename Operator::LayoutBias;
    using ElementDst = typename Operator::ElementDst;
    using LayoutDst = typename Operator::LayoutDst;
    using ElementAccumulator = typename Operator::ElementAccumulator;
    using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

    using OperatorArguments = typename Operator::Arguments;

    ConvolutionOperation(char const* name = "unknown_gemm")
            : ConvolutionOperationBase<Operator_>(name) {}

    virtual Status run(
            void const* arguments_ptr, void* device_workspace = nullptr,
            cudaStream_t stream = nullptr) const {
        cutlass::conv::Operator conv_op = this->m_description.conv_op;
        ConvolutionArguments const* conv_args =
                reinterpret_cast<ConvolutionArguments const*>(arguments_ptr);
        const auto& ps = conv_args->problem_size;

        OperatorArguments args;
        args.problem_size = ps;
        args.ref_src = {
                static_cast<ElementSrc*>(const_cast<void*>(conv_args->src)),
                LayoutSrc::packed(implicit_gemm_tensor_a_extent(conv_op, ps))};
        args.ref_filter = {
                static_cast<ElementFilter*>(const_cast<void*>(conv_args->filter)),
                LayoutFilter::packed(implicit_gemm_tensor_b_extent(conv_op, ps))};
        args.ref_bias = {
                static_cast<ElementBias*>(const_cast<void*>(conv_args->bias)),
                LayoutBias::packed(implicit_gemm_tensor_bias_extent(conv_op, ps))};
        args.ref_z = {
                static_cast<ElementDst*>(const_cast<void*>(conv_args->z)),
                LayoutDst::packed(implicit_gemm_tensor_c_extent(conv_op, ps))};
        args.ref_dst = {
                static_cast<ElementDst*>(conv_args->dst),
                LayoutDst::packed(implicit_gemm_tensor_c_extent(conv_op, ps))};

        args.output_op = init_epilogue_param<typename Operator::EpilogueOutputOp>().get(
                conv_args);

        if (conv_args->extra_param) {
            args.extra_param = *reinterpret_cast<typename Operator::ExtraParam const*>(
                    conv_args->extra_param);
        }

        Operator op;
        Status status = op.initialize(args, device_workspace);

        if (status != Status::kSuccess) {
            return status;
        }

        return op.run(stream);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// We add a new template class to handle convolution backward filter operation, because
/// the device-level convolution operator of backward filter is different from the
/// others (convolution forward and convolution backward data).
/// But the description object is reused in this wrapper of convolution backward filter.
/// The reason is that we do not want to introduce an another unnecessary structure.
/// TODO: Maybe the device-level operator in cutlass for convoluton forward, backward
/// data and backward filter should be combined.
template <typename Operator_>
class ConvolutionBackwardFilterOperationBase : public Operation {
public:
    using Operator = Operator_;
    using ElementSrc = typename Operator::ElementSrc;
    using LayoutSrc = typename Operator::LayoutSrc;
    using ElementDiff = typename Operator::ElementDiff;
    using LayoutDiff = typename Operator::LayoutDiff;
    using ElementGrad = typename Operator::ElementGrad;
    using LayoutGrad = typename Operator::LayoutGrad;
    using ElementAccumulator = typename Operator::ElementAccumulator;

    ConvolutionBackwardFilterOperationBase(char const* name = "unknown_convolution") {
        m_description.name = name;
        m_description.provider = Provider::kCUTLASS;
        m_description.kind = OperationKind::kConvolution;
        m_description.conv_op = Operator::kConvolutionalOperator;

        m_description.tile_description.threadblock_shape = make_Coord(
                Operator::ThreadblockShape::kM, Operator::ThreadblockShape::kN,
                Operator::ThreadblockShape::kK);

        m_description.tile_description.threadblock_stages = Operator::kStages;

        m_description.tile_description.warp_count = make_Coord(
                Operator::ConvolutionKernel::WarpCount::kM,
                Operator::ConvolutionKernel::WarpCount::kN,
                Operator::ConvolutionKernel::WarpCount::kK);

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

        /// src in description -> src in C++ template
        m_description.src =
                make_TensorDescription<ElementSrc, LayoutSrc>(Operator::kAlignmentSrc);
        /// filter in description -> diff in C++ template
        m_description.filter = make_TensorDescription<ElementDiff, LayoutDiff>(
                Operator::kAlignmentDiff);
        /// dst in description -> grad in C++ template
        m_description.dst = make_TensorDescription<ElementGrad, LayoutGrad>(
                Operator::kAlignmentGrad);
        /// because bias tensor is not used in ConvolutionBackwardFilter operation, the
        /// following tensor description is a dummy arguments
        m_description.bias = make_TensorDescription<ElementGrad, LayoutGrad>(
                Operator::kAlignmentGrad);

        m_description.convolution_type = Operator::kConvolutionType;
        m_description.arch_tag = ArchTagMap<typename Operator::ArchTag>::kId;

        m_description.epilogue_type = Operator::EpilogueOutputOp::kType;
        m_description.epilogue_count = Operator::EpilogueOutputOp::kCount;

        m_description.threadblock_swizzle =
                ThreadblockSwizzleMap<typename Operator::ThreadblockSwizzle>::kId;

        m_description.special_optimization = Operator::kSpecialOpt;
        m_description.gemm_mode = Operator::kGemmMode;
        /// ConvolutionBackwardFilter operation is only used for depthwise convolution,
        /// so the option without_shared_load is always true
        m_description.without_shared_load = true;
    }

    virtual OperationDescription const& description() const { return m_description; }

protected:
    ConvolutionDescription m_description;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class ConvolutionBackwardFilterOperation
        : public ConvolutionBackwardFilterOperationBase<Operator_> {
public:
    using Operator = Operator_;
    using ElementSrc = typename Operator::ElementSrc;
    using LayoutSrc = typename Operator::LayoutSrc;
    using ElementDiff = typename Operator::ElementDiff;
    using LayoutDiff = typename Operator::LayoutDiff;
    using ElementGrad = typename Operator::ElementGrad;
    using LayoutGrad = typename Operator::LayoutGrad;
    using ElementAccumulator = typename Operator::ElementAccumulator;
    using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

    using OperatorArguments = typename Operator::Arguments;

    ConvolutionBackwardFilterOperation(char const* name = "unknown_gemm")
            : ConvolutionBackwardFilterOperationBase<Operator_>(name) {}

    virtual Status run(
            void const* arguments_ptr, void* device_workspace = nullptr,
            cudaStream_t stream = nullptr) const {
        cutlass::conv::Operator conv_op = this->m_description.conv_op;
        ConvolutionArguments const* conv_args =
                reinterpret_cast<ConvolutionArguments const*>(arguments_ptr);
        const auto& ps = conv_args->problem_size;

        OperatorArguments args;
        args.problem_size = ps;
        /// src in convolution arguments -> ref_src
        args.ref_src = {
                static_cast<ElementSrc*>(const_cast<void*>(conv_args->src)),
                LayoutSrc::packed(implicit_gemm_tensor_b_extent(conv_op, ps))};
        /// filter in convolution arguments -> ref_diff
        args.ref_diff = {
                static_cast<ElementDiff*>(const_cast<void*>(conv_args->filter)),
                LayoutDiff::packed(implicit_gemm_tensor_a_extent(conv_op, ps))};
        /// dst in convolution arguments -> ref_grad
        args.ref_grad = {
                static_cast<ElementGrad*>(conv_args->dst),
                LayoutGrad::packed(implicit_gemm_tensor_c_extent(conv_op, ps))};

        args.output_op = init_epilogue_param<typename Operator::EpilogueOutputOp>().get(
                conv_args);

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
