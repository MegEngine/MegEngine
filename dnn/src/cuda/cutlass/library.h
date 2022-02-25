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
#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/tensor_coord.h"

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/epilogue/epilogue.h"
#include "cutlass/gemm/gemm.h"

#pragma GCC diagnostic pop

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Layout type identifier
enum class LayoutTypeID {
    kUnknown,
    kColumnMajor,
    kRowMajor,
    kColumnMajorInterleavedK2,
    kRowMajorInterleavedK2,
    kColumnMajorInterleavedK4,
    kRowMajorInterleavedK4,
    kColumnMajorInterleavedK16,
    kRowMajorInterleavedK16,
    kColumnMajorInterleavedK32,
    kRowMajorInterleavedK32,
    kColumnMajorInterleavedK64,
    kRowMajorInterleavedK64,
    kTensorNCHW,
    kTensorNCDHW,
    kTensorNHWC,
    kTensorNDHWC,
    kTensorNC4HW4,
    kTensorC4RSK4,
    kTensorNC8HW8,
    kTensorC8RSK8,
    kTensorNC16HW16,
    kTensorC16RSK16,
    kTensorNC32HW32,
    kTensorC32RSK32,
    kTensorNC64HW64,
    kTensorC64RSK64,
    kTensorK4RSC4,
    kTensorCK4RS4,
    kTensorCK8RS8,
    kTensorCK16RS16,
    kInvalid
};

/// Numeric data type
enum class NumericTypeID {
    kUnknown,
    kVoid,
    kB1,
    kU2,
    kU4,
    kU8,
    kU16,
    kU32,
    kU64,
    kS2,
    kS4,
    kS8,
    kS16,
    kS32,
    kS64,
    kF16,
    kBF16,
    kTF32,
    kF32,
    kF64,
    kCF16,
    kCBF16,
    kCF32,
    kCTF32,
    kCF64,
    kCS2,
    kCS4,
    kCS8,
    kCS16,
    kCS32,
    kCS64,
    kCU2,
    kCU4,
    kCU8,
    kCU16,
    kCU32,
    kCU64,
    kInvalid
};

/// Enumerated type describing a transformation on a complex value.
enum class ComplexTransform { kNone, kConjugate, kInvalid };

/// Providers
enum class Provider {
    kNone,
    kCUTLASS,
    kReferenceHost,
    kReferenceDevice,
    kCUBLAS,
    kCUDNN,
    kInvalid
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumeration indicating the kind of operation
enum class OperationKind {
    kGemm,
    kConv2d,
    kConv3d,
    kConvolution,
    kEqGemm,
    kSparseGemm,
    kReduction,
    kInvalid
};

/// Enumeration indicating whether scalars are in host or device memory
enum class ScalarPointerMode { kHost, kDevice, kInvalid };

/// Describes how reductions are performed across threadblocks
enum class SplitKMode { kNone, kSerial, kParallel, kParallelSerial, kInvalid };

/// Indicates the classificaition of the math instruction
enum class OpcodeClassID { kSimt, kTensorOp, kWmmaTensorOp, kSparseTensorOp, kInvalid };

enum class ArchTagID {
    kSm50,
    kSm60,
    kSm61,
    kSm70,
    kSm72,
    kSm75,
    kSm80,
    kSm86,
    kInvalid
};

enum class MathOperationID {
    kAdd,
    kMultiplyAdd,
    kMultiplyAddSaturate,
    kMultiplyAddFastBF16,
    kMultiplyAddFastF16,
    kMultiplyAddComplex,
    kMultiplyAddGaussianComplex,
    kXorPopc,
    kInvalid
};

enum class ThreadblockSwizzleID {
    kGemmIdentity,
    kGemmHorizontal,
    kGemmBatchedIdentity,
    kGemmSplitKIdentity,
    kGemmSplitKHorizontal,
    kGemvBatchedStridedDefault,
    kGemvBatchedStridedReduction,
    kConvolutionFpropCxRSKx,
    kConvolutionDgradCxRSKx,
    kConvolutionFpropNCxHWx,
    kConvolutionFpropTrans,
    kConvolutionDgradNCxHWx,
    kConvolutionDgradTrans,
    kDepthwiseConvolutionFprop,
    kDepthwiseConvolutionDgrad,
    kDepthwiseConvolutionWgrad,
    kInvalid
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumeration indicating what kind of GEMM operation to perform
enum class GemmKind {
    kGemm,
    kSparse,
    kUniversal,
    kPlanarComplex,
    kPlanarComplexArray,
    kInvalid
};

/// Mode of Universal GEMM
using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

/// Enumeration indicating what kind of Conv2d operation to perform
enum class ConvKind { kUnknown, kFprop, kDgrad, kWgrad, kInvalid };

enum class ConvModeID { kCrossCorrelation, kConvolution, kInvalid };

// Iterator algorithm enum in order of general performance-efficiency
enum class IteratorAlgorithmID { kNone, kAnalytic, kOptimized, kInvalid };

enum class EpilogueKind {
    kUnknown,
    kBiasAddLinearCombination,
    kBiasAddLinearCombinationClamp,
    kBiasAddLInearCombinationHSwish,
    kBiasAddLInearCombinationHSwishClamp,
    kBiasAddLInearCombinationRelu,
    kBiasAddLInearCombinationReluClamp,
    kConversion,
    kLinearCombination,
    kLinearCombinationClamp,
    kLinearCombinationPlanarComplex,
    kLinearCombinationRelu,
    kLinearCombinationSigmoid,
    kInvalid
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct MathInstructionDescription {
    /// Shape of the target math instruction
    cutlass::gemm::GemmCoord instruction_shape;

    /// Describes the data type of the internal accumulator
    NumericTypeID element_accumulator;

    /// Classification of math instruction
    OpcodeClassID opcode_class;

    /// Type of math operation performed
    MathOperationID math_operation;

    //
    // Methods
    //

    MathInstructionDescription(
            cutlass::gemm::GemmCoord instruction_shape = cutlass::gemm::GemmCoord(),
            NumericTypeID element_accumulator = NumericTypeID::kInvalid,
            OpcodeClassID opcode_class = OpcodeClassID::kInvalid,
            MathOperationID math_operation = MathOperationID::kMultiplyAdd)
            : instruction_shape(instruction_shape),
              element_accumulator(element_accumulator),
              opcode_class(opcode_class),
              math_operation(math_operation) {}

    // Equality operator
    inline bool operator==(MathInstructionDescription const& rhs) const {
        return ((instruction_shape == rhs.instruction_shape) &&
                (element_accumulator == rhs.element_accumulator) &&
                (opcode_class == rhs.opcode_class) &&
                (math_operation == rhs.math_operation));
    }

    // Inequality operator
    inline bool operator!=(MathInstructionDescription const& rhs) const {
        return !(*this == rhs);
    }
};

/// Structure describing the tiled structure of a GEMM-like computation
struct TileDescription {
    /// Describes the shape of a threadblock (in elements)
    cutlass::gemm::GemmCoord threadblock_shape;

    /// Describes the number of pipeline stages in the threadblock-scoped
    /// mainloop
    int threadblock_stages;

    /// Number of warps in each logical dimension
    cutlass::gemm::GemmCoord warp_count;

    /// Core math instruction
    MathInstructionDescription math_instruction;

    /// Minimum compute capability (e.g. 70, 75) of a device eligible to run the
    /// operation.
    int minimum_compute_capability;

    /// Minimum compute capability (e.g. 70, 75) of a device eligible to run the
    /// operation.
    int maximum_compute_capability;

    //
    // Methods
    //

    TileDescription(
            cutlass::gemm::GemmCoord threadblock_shape = cutlass::gemm::GemmCoord(),
            int threadblock_stages = 0,
            cutlass::gemm::GemmCoord warp_count = cutlass::gemm::GemmCoord(),
            MathInstructionDescription math_instruction = MathInstructionDescription(),
            int minimum_compute_capability = 0, int maximum_compute_capability = 0)
            : threadblock_shape(threadblock_shape),
              threadblock_stages(threadblock_stages),
              warp_count(warp_count),
              math_instruction(math_instruction),
              minimum_compute_capability(minimum_compute_capability),
              maximum_compute_capability(maximum_compute_capability) {}

    // Equality operator
    inline bool operator==(TileDescription const& rhs) const {
        return ((threadblock_shape == rhs.threadblock_shape) &&
                (threadblock_stages == rhs.threadblock_stages) &&
                (warp_count == rhs.warp_count) &&
                (math_instruction == rhs.math_instruction) &&
                (minimum_compute_capability == rhs.minimum_compute_capability) &&
                (maximum_compute_capability == rhs.maximum_compute_capability));
    }

    // Inequality operator
    inline bool operator!=(TileDescription const& rhs) const { return !(*this == rhs); }
};

/// High-level description of an operation
struct OperationDescription {
    /// Unique identifier describing the operation
    char const* name;

    /// Operation provider
    Provider provider;

    /// Kind of operation
    OperationKind kind;

    /// Describes the tiled structure of a GEMM-like computation
    TileDescription tile_description;

    //
    // Methods
    //
    OperationDescription(
            char const* name = "unknown", OperationKind kind = OperationKind::kInvalid,
            TileDescription const& tile_description = TileDescription())
            : name(name), kind(kind), tile_description(tile_description) {}
};

/// Structure describing the properties of a tensor
struct TensorDescription {
    /// Numeric type of an individual element
    NumericTypeID element;

    /// Enumerant identifying the layout function for the tensor
    LayoutTypeID layout;

    /// Alignment restriction on pointers, strides, and extents
    int alignment;

    /// log2() of the maximum extent of each dimension
    int log_extent_range;

    /// log2() of the maximum value each relevant stride may have
    int log_stride_range;

    //
    // Methods
    //

    TensorDescription(
            NumericTypeID element = NumericTypeID::kInvalid,
            LayoutTypeID layout = LayoutTypeID::kInvalid, int alignment = 1,
            int log_extent_range = 24, int log_stride_range = 24)
            : element(element),
              layout(layout),
              alignment(alignment),
              log_extent_range(log_extent_range),
              log_stride_range(log_stride_range) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmDescription : public OperationDescription {
    GemmKind gemm_kind;

    TensorDescription A;
    TensorDescription B;
    TensorDescription C;

    int stages;
    SplitKMode split_k_mode;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmArguments {
    /// GEMM problem size
    gemm::GemmCoord problem_size;

    /// Device pointers to input and output matrices
    void const* A;
    void const* B;
    void const* C;
    void* D;

    /// Leading dimensions of input and output matrices
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldd;

    /// Number of partitions of K dimension
    int split_k_slices;

    /// Host or device pointers to epilogue scalars, note that these pointers
    /// will be interpreted as ElementCompute* in method `op->run(args)`, a
    /// different dtype here results in undefined epilogue behaviors
    void const* alpha;
    void const* beta;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct ConvolutionDescription : public OperationDescription {
    conv::Operator conv_op;

    TensorDescription src;
    TensorDescription filter;
    TensorDescription dst;
    TensorDescription bias;

    conv::ConvType convolution_type;
    ArchTagID arch_tag;

    epilogue::EpilogueType epilogue_type;
    int epilogue_count;

    ThreadblockSwizzleID threadblock_swizzle;

    conv::SpecialOptimizeDesc special_optimization;
    conv::ImplicitGemmMode gemm_mode;
    bool without_shared_load;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct ConvolutionArguments {
    /// Problem size
    conv::Conv2dProblemSize problem_size;

    /// Device pointers to input and output tensors
    void const* src;
    void const* filter;
    void const* bias;
    void const* z;
    void* dst;

    /// Host or device pointers to epilogue scalars, note that these pointers
    /// will be interpreted as ElementCompute* in method `op->run(args)`, a
    /// different dtype here results in undefined epilogue behaviors
    void const* alpha;
    void const* beta;
    void const* gamma;
    void const* delta;
    void const* theta;
    void const* threshold;
    void const* scale;

    /// Host pointer to extra param struct
    void const* extra_param;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Base class for all operations
class Operation {
public:
    virtual ~Operation() {}

    virtual OperationDescription const& description() const = 0;

    virtual Status run(
            void const* arguments, void* device_workspace = nullptr,
            cudaStream_t stream = nullptr) const = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace library
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
