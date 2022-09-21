#
# \file generator.py
#
# \brief Generates the CUTLASS Library's instances
#

import argparse
import enum
import os.path
import platform
import string

from library import *
from manifest import *

###################################################################################################

#
def CudaToolkitVersionSatisfies(semantic_ver_string, major, minor, patch=0):

    # by default, use the latest CUDA Toolkit version
    cuda_version = [11, 0, 132]

    # Update cuda_version based on parsed string
    if semantic_ver_string != "":
        for i, x in enumerate([int(x) for x in semantic_ver_string.split(".")]):
            if i < len(cuda_version):
                cuda_version[i] = x
            else:
                cuda_version.append(x)
    return cuda_version >= [major, minor, patch]


###################################################################################################
###################################################################################################

#
def CreateGemmOperator(
    manifest,
    layouts,
    tile_descriptions,
    data_type,
    alignment_constraints,
    complex_transforms=None,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity8,
):

    if complex_transforms is None:
        complex_transforms = [(ComplexTransform.none, ComplexTransform.none)]

    element_a, element_b, element_c, element_epilogue = data_type

    operations = []

    # by default, only generate the largest tile and largest alignment
    if manifest.args.kernels == "":
        tile_descriptions = [tile_descriptions[0]]
        alignment_constraints = [alignment_constraints[0]]

    for layout in layouts:
        for tile_description in tile_descriptions:
            for alignment in alignment_constraints:
                for complex_transform in complex_transforms:

                    alignment_c = min(8, alignment)

                    A = TensorDescription(
                        element_a, layout[0], alignment, complex_transform[0]
                    )
                    B = TensorDescription(
                        element_b, layout[1], alignment, complex_transform[1]
                    )
                    C = TensorDescription(element_c, layout[2], alignment_c)

                    new_operation = GemmOperation(
                        GemmKind.Universal,
                        tile_description.minimum_compute_capability,
                        tile_description,
                        A,
                        B,
                        C,
                        element_epilogue,
                        epilogue_functor,
                        swizzling_functor,
                    )

                    manifest.append(new_operation)
                    operations.append(new_operation)

    return operations


###########################################################################################################
#   ConvolutionOperator support variations
#        ____________________________________________________________________
#         ConvolutionalOperator |        Analytic      |      Optimized
#        ____________________________________________________________________
#        |       Fprop          |     (strided)        |    (strided)
#        |       Dgrad          |   (strided, unity*)  |     (unity)
#        |       Wgrad          |     (strided)        |    (strided)
#        ____________________________________________________________________
#
# Note :  Operator marked (*) are supported but not generated to keep the instantiated kernel count low
###########################################################################################################
# Convolution for 2D operations
def CreateConv2dOperator(
    manifest,
    layout,
    tile_descriptions,
    data_type,
    alignment,
    conv_kinds=[ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad],
    epilogue_functor=EpilogueFunctor.LinearCombination,
):

    element_a, element_b, element_c, element_epilogue = data_type

    # one exceptional case
    alignment_c = min(8, alignment)

    # iterator algorithm (analytic and optimized)
    iterator_algorithms = [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]

    # by default, only generate the largest tile size
    if manifest.args.kernels == "":
        tile_descriptions = [tile_descriptions[0]]

    operations = []

    for tile in tile_descriptions:
        for conv_kind in conv_kinds:
            for iterator_algorithm in iterator_algorithms:
                A = TensorDescription(element_a, layout[0], alignment)
                B = TensorDescription(element_b, layout[1], alignment)
                C = TensorDescription(element_c, layout[2], alignment_c)

                # unity stride only for Optimized Dgrad
                if (iterator_algorithm == IteratorAlgorithm.Optimized) and (
                    conv_kind == ConvKind.Dgrad
                ):
                    new_operation = Conv2dOperation(
                        conv_kind,
                        iterator_algorithm,
                        tile.minimum_compute_capability,
                        tile,
                        A,
                        B,
                        C,
                        element_epilogue,
                        StrideSupport.Unity,
                        epilogue_functor,
                    )

                    manifest.append(new_operation)
                    operations.append(new_operation)

                # strided dgrad is not supported by Optimized Dgrad
                if (iterator_algorithm == IteratorAlgorithm.Optimized) and (
                    conv_kind == ConvKind.Dgrad
                ):
                    continue

                # strided support for Fprop (Analytic/Optimized), Dgrad (Analytic), and Wgrad (Analytic)
                new_operation = Conv2dOperation(
                    conv_kind,
                    iterator_algorithm,
                    tile.minimum_compute_capability,
                    tile,
                    A,
                    B,
                    C,
                    element_epilogue,
                    StrideSupport.Strided,
                    epilogue_functor,
                )

                manifest.append(new_operation)
                operations.append(new_operation)

    return operations


###################################################################################################
###################################################################################################


def GenerateConv2d_Simt(args):
    operations = []

    layouts = [(LayoutType.TensorNC4HW4, LayoutType.TensorC4RSK4)]

    math_instructions = [
        MathInstruction(
            [1, 1, 4],
            DataType.s8,
            DataType.s8,
            DataType.s32,
            OpcodeClass.Simt,
            MathOperation.multiply_add,
        )
    ]

    dst_layouts = [
        LayoutType.TensorNC4HW4,
        LayoutType.TensorNC32HW32,
        LayoutType.TensorNHWC,
        LayoutType.TensorNHWC,
        LayoutType.TensorNCHW,
    ]

    dst_types = [DataType.s8, DataType.s8, DataType.u4, DataType.s4, DataType.f32]

    max_cc = 1024

    for math_inst in math_instructions:
        for layout in layouts:
            for dst_type, dst_layout in zip(dst_types, dst_layouts):
                if dst_type == DataType.s4 or dst_type == DataType.u4:
                    min_cc = 75
                    use_special_optimization = SpecialOptimizeDesc.NoneSpecialOpt
                else:
                    min_cc = 61
                    use_special_optimization = SpecialOptimizeDesc.ConvFilterUnity
                tile_descriptions = [
                    TileDescription(
                        [128, 128, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [64, 128, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 32, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [32, 128, 32], 2, [1, 2, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [32, 64, 32], 2, [1, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [64, 32, 32], 2, [1, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [16, 128, 16], 1, [1, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [16, 64, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc
                    ),
                ]
                for tile in tile_descriptions:
                    if (
                        dst_layout == LayoutType.TensorNC32HW32
                        and tile.threadblock_shape[0] > 32
                    ):
                        continue
                    if (
                        dst_layout == LayoutType.TensorNCHW
                        or dst_layout == LayoutType.TensorNHWC
                    ) and tile.threadblock_shape[0] > 16:
                        continue
                    operations += GenerateConv2d(
                        ConvType.Convolution,
                        ConvKind.Fprop,
                        [tile],
                        layout[0],
                        layout[1],
                        dst_layout,
                        dst_type,
                        min_cc,
                        32,
                        32,
                        32,
                        use_special_optimization,
                    )
    return operations


def GenerateConv2d_TensorOp_8816(args):
    operations = []

    layouts = [(LayoutType.TensorNC32HW32, LayoutType.TensorC32RSK32)]

    math_instructions = [
        MathInstruction(
            [8, 8, 16],
            DataType.s8,
            DataType.s8,
            DataType.s32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add_saturate,
        )
    ]

    dst_layouts = [LayoutType.TensorNC32HW32, LayoutType.TensorNC4HW4]

    dst_types = [DataType.s8, DataType.s8]

    use_special_optimization = SpecialOptimizeDesc.ConvFilterUnity

    min_cc = 75
    max_cc = 1024

    cuda_major = 10
    cuda_minor = 2

    for math_inst in math_instructions:
        for layout in layouts:
            for dst_type, dst_layout in zip(dst_types, dst_layouts):
                if dst_layout == LayoutType.TensorNC32HW32:
                    tile_descriptions = [
                        TileDescription(
                            [128, 256, 64], 2, [2, 4, 1], math_inst, min_cc, max_cc
                        ),
                        TileDescription(
                            [256, 128, 64], 2, [4, 2, 1], math_inst, min_cc, max_cc
                        ),
                        TileDescription(
                            [128, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc
                        ),
                        TileDescription(
                            [128, 64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc
                        ),
                        TileDescription(
                            [64, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc
                        ),
                        TileDescription(
                            [128, 64, 32], 1, [2, 2, 1], math_inst, min_cc, max_cc
                        ),
                        TileDescription(
                            [128, 32, 32], 1, [2, 1, 1], math_inst, min_cc, max_cc
                        ),
                    ]
                    operations += GenerateConv2d(
                        ConvType.Convolution,
                        ConvKind.Fprop,
                        tile_descriptions,
                        layout[0],
                        layout[1],
                        dst_layout,
                        dst_type,
                        min_cc,
                        128,
                        128,
                        64,
                        use_special_optimization,
                        ImplicitGemmMode.GemmTN,
                        True,
                        cuda_major,
                        cuda_minor,
                    )
                else:
                    assert dst_layout == LayoutType.TensorNC4HW4
                    tile_descriptions = [
                        TileDescription(
                            [64, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc
                        ),
                        TileDescription(
                            [32, 128, 32], 1, [1, 2, 1], math_inst, min_cc, max_cc
                        ),
                    ]
                    operations += GenerateConv2d(
                        ConvType.Convolution,
                        ConvKind.Fprop,
                        tile_descriptions,
                        layout[0],
                        layout[1],
                        dst_layout,
                        dst_type,
                        min_cc,
                        128,
                        128,
                        64,
                        use_special_optimization,
                        ImplicitGemmMode.GemmNT,
                        False,
                        cuda_major,
                        cuda_minor,
                    )

    layouts_nhwc = [
        (LayoutType.TensorNHWC, LayoutType.TensorNC4HW4, 32),
        (LayoutType.TensorNHWC, LayoutType.TensorNC8HW8, 64),
        (LayoutType.TensorNHWC, LayoutType.TensorNC16HW16, 128),
    ]

    dst_layouts_nhwc = [LayoutType.TensorNHWC]

    for math_inst in math_instructions:
        for layout in layouts_nhwc:
            for dst_layout in dst_layouts_nhwc:
                dst_type = math_inst.element_b
                tile_descriptions = [
                    TileDescription(
                        [128, 32, 32], 1, [2, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [64, 16, 32], 2, [1, 1, 1], math_inst, min_cc, max_cc
                    ),
                ]
                for tile in tile_descriptions:
                    dst_align = 32 if tile.threadblock_shape[1] == 16 else 64
                    operations += GenerateConv2d(
                        ConvType.Convolution,
                        ConvKind.Fprop,
                        [tile],
                        layout[0],
                        layout[1],
                        dst_layout,
                        dst_type,
                        min_cc,
                        layout[2],
                        layout[2],
                        dst_align,
                        use_special_optimization,
                        ImplicitGemmMode.GemmTN,
                        False,
                        cuda_major,
                        cuda_minor,
                    )
                    if (
                        tile.threadblock_shape[1] == 16
                        or tile.threadblock_shape[1] == 32
                    ):
                        operations += GenerateConv2d(
                            ConvType.Convolution,
                            ConvKind.Fprop,
                            [tile],
                            layout[0],
                            layout[1],
                            dst_layout,
                            dst_type,
                            min_cc,
                            layout[2],
                            layout[2],
                            dst_align,
                            use_special_optimization,
                            ImplicitGemmMode.GemmTN,
                            True,
                            cuda_major,
                            cuda_minor,
                        )

    out_dtypes = [DataType.s4, DataType.u4, DataType.f32]

    # INT8x8x4 and INT8x8x32
    for math_inst in math_instructions:
        for layout in layouts_nhwc:
            for dst_layout in dst_layouts_nhwc:
                for out_dtype in out_dtypes:
                    tile_descriptions = [
                        TileDescription(
                            [128, 32, 32], 1, [2, 1, 1], math_inst, min_cc, max_cc
                        ),
                        TileDescription(
                            [64, 16, 32], 2, [1, 1, 1], math_inst, min_cc, max_cc
                        ),
                    ]
                    for tile in tile_descriptions:
                        dst_align = (
                            4 * DataTypeSize[out_dtype]
                            if tile.threadblock_shape[1] == 16
                            or out_dtype == DataType.f32
                            else 8 * DataTypeSize[out_dtype]
                        )
                        operations += GenerateConv2d(
                            ConvType.Convolution,
                            ConvKind.Fprop,
                            [tile],
                            layout[0],
                            layout[1],
                            dst_layout,
                            out_dtype,
                            min_cc,
                            layout[2],
                            layout[2],
                            dst_align,
                            use_special_optimization,
                            ImplicitGemmMode.GemmTN,
                            False,
                            cuda_major,
                            cuda_minor,
                        )
                        if tile.threadblock_shape[1] == 16 or (
                            tile.threadblock_shape[1] == 32
                            and out_dtype != DataType.f32
                        ):
                            operations += GenerateConv2d(
                                ConvType.Convolution,
                                ConvKind.Fprop,
                                [tile],
                                layout[0],
                                layout[1],
                                dst_layout,
                                out_dtype,
                                min_cc,
                                layout[2],
                                layout[2],
                                dst_align,
                                use_special_optimization,
                                ImplicitGemmMode.GemmTN,
                                True,
                                cuda_major,
                                cuda_minor,
                            )

    return operations


def GenerateConv2d_TensorOp_8832(args):
    operations = []

    layouts = [(LayoutType.TensorNC64HW64, LayoutType.TensorC64RSK64)]

    math_instructions = [
        MathInstruction(
            [8, 8, 32],
            DataType.s4,
            DataType.s4,
            DataType.s32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add_saturate,
        ),
        MathInstruction(
            [8, 8, 32],
            DataType.s4,
            DataType.u4,
            DataType.s32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add_saturate,
        ),
    ]

    dst_layouts = [LayoutType.TensorNC64HW64]

    use_special_optimization = SpecialOptimizeDesc.ConvFilterUnity

    min_cc = 75
    max_cc = 1024

    cuda_major = 10
    cuda_minor = 2

    for math_inst in math_instructions:
        for layout in layouts:
            for dst_layout in dst_layouts:
                dst_type = math_inst.element_b
                tile_descriptions = [
                    TileDescription(
                        [128, 256, 128], 2, [2, 4, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 64, 128], 2, [2, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 64, 64], 1, [2, 1, 1], math_inst, min_cc, max_cc
                    ),
                ]
                operations += GenerateConv2d(
                    ConvType.Convolution,
                    ConvKind.Fprop,
                    tile_descriptions,
                    layout[0],
                    layout[1],
                    dst_layout,
                    dst_type,
                    min_cc,
                    128,
                    128,
                    64,
                    use_special_optimization,
                    ImplicitGemmMode.GemmTN,
                    True,
                    cuda_major,
                    cuda_minor,
                )

    layouts_nhwc = [
        (LayoutType.TensorNHWC, LayoutType.TensorNC8HW8, 32),
        (LayoutType.TensorNHWC, LayoutType.TensorNC16HW16, 64),
        (LayoutType.TensorNHWC, LayoutType.TensorNC32HW32, 128),
    ]

    dst_layouts_nhwc = [LayoutType.TensorNHWC]

    for math_inst in math_instructions:
        for layout in layouts_nhwc:
            for dst_layout in dst_layouts_nhwc:
                dst_type = math_inst.element_b
                tile_descriptions = [
                    TileDescription(
                        [128, 16, 64], 2, [1, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 32, 64], 1, [2, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 64, 64], 1, [2, 1, 1], math_inst, min_cc, max_cc
                    ),
                ]
                for tile in tile_descriptions:
                    dst_align = 16 if tile.threadblock_shape[1] == 16 else 32
                    operations += GenerateConv2d(
                        ConvType.Convolution,
                        ConvKind.Fprop,
                        [tile],
                        layout[0],
                        layout[1],
                        dst_layout,
                        dst_type,
                        min_cc,
                        layout[2],
                        layout[2],
                        dst_align,
                        use_special_optimization,
                        ImplicitGemmMode.GemmTN,
                        False,
                        cuda_major,
                        cuda_minor,
                    )
                    if (
                        tile.threadblock_shape[1] == 32
                        or tile.threadblock_shape[1] == 64
                    ):
                        dst_align = 32 if tile.threadblock_shape[1] == 32 else 64
                        operations += GenerateConv2d(
                            ConvType.Convolution,
                            ConvKind.Fprop,
                            [tile],
                            layout[0],
                            layout[1],
                            dst_layout,
                            dst_type,
                            min_cc,
                            layout[2],
                            layout[2],
                            dst_align,
                            use_special_optimization,
                            ImplicitGemmMode.GemmTN,
                            True,
                            cuda_major,
                            cuda_minor,
                        )
    # INT4x4x8
    for math_inst in math_instructions:
        for layout in layouts_nhwc:
            for dst_layout in dst_layouts_nhwc:
                tile_descriptions = [
                    TileDescription(
                        [128, 16, 64], 2, [1, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 32, 64], 1, [2, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 64, 64], 1, [2, 1, 1], math_inst, min_cc, max_cc
                    ),
                ]
                for tile in tile_descriptions:
                    dst_align = 32 if tile.threadblock_shape[1] == 16 else 64
                    operations += GenerateConv2d(
                        ConvType.Convolution,
                        ConvKind.Fprop,
                        [tile],
                        layout[0],
                        layout[1],
                        dst_layout,
                        DataType.s8,
                        min_cc,
                        layout[2],
                        layout[2],
                        dst_align,
                        use_special_optimization,
                        ImplicitGemmMode.GemmTN,
                        False,
                        cuda_major,
                        cuda_minor,
                    )
                    if (
                        tile.threadblock_shape[1] == 32
                        or tile.threadblock_shape[1] == 64
                    ):
                        dst_align = 64 if tile.threadblock_shape[1] == 32 else 128
                        operations += GenerateConv2d(
                            ConvType.Convolution,
                            ConvKind.Fprop,
                            [tile],
                            layout[0],
                            layout[1],
                            dst_layout,
                            DataType.s8,
                            min_cc,
                            layout[2],
                            layout[2],
                            dst_align,
                            use_special_optimization,
                            ImplicitGemmMode.GemmTN,
                            True,
                            cuda_major,
                            cuda_minor,
                        )

    return operations


def GenerateDeconv_Simt(args):
    operations = []

    layouts = [(LayoutType.TensorNC4HW4, LayoutType.TensorK4RSC4)]

    math_instructions = [
        MathInstruction(
            [1, 1, 4],
            DataType.s8,
            DataType.s8,
            DataType.s32,
            OpcodeClass.Simt,
            MathOperation.multiply_add,
        )
    ]

    dst_layouts = [LayoutType.TensorNC4HW4]

    dst_types = [DataType.s8]

    use_special_optimization = SpecialOptimizeDesc.DeconvDoubleUpsampling

    min_cc = 61
    max_cc = 1024

    for math_inst in math_instructions:
        for layout in layouts:
            for dst_type, dst_layout in zip(dst_types, dst_layouts):
                tile_descriptions = [
                    TileDescription(
                        [32, 128, 32], 2, [1, 2, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [16, 128, 16], 2, [1, 2, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [16, 128, 16], 1, [1, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [16, 64, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc
                    ),
                ]
                operations += GenerateConv2d(
                    ConvType.Convolution,
                    ConvKind.Dgrad,
                    tile_descriptions,
                    layout[0],
                    layout[1],
                    dst_layout,
                    dst_type,
                    min_cc,
                    32,
                    32,
                    32,
                    use_special_optimization,
                )
    return operations


def GenerateDeconv_TensorOp_8816(args):
    operations = []

    layouts = [
        (LayoutType.TensorNHWC, LayoutType.TensorCK4RS4, 32),
        (LayoutType.TensorNHWC, LayoutType.TensorCK8RS8, 64),
        (LayoutType.TensorNHWC, LayoutType.TensorCK16RS16, 128),
    ]

    math_instructions = [
        MathInstruction(
            [8, 8, 16],
            DataType.s8,
            DataType.s8,
            DataType.s32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add_saturate,
        )
    ]

    dst_layouts = [LayoutType.TensorNHWC]

    dst_types = [DataType.s8]

    use_special_optimization = SpecialOptimizeDesc.DeconvDoubleUpsampling

    min_cc = 75
    max_cc = 1024

    cuda_major = 10
    cuda_minor = 2

    for math_inst in math_instructions:
        for layout in layouts:
            for dst_type, dst_layout in zip(dst_types, dst_layouts):
                tile_descriptions = [
                    TileDescription(
                        [128, 32, 32], 1, [2, 1, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [64, 16, 32], 2, [1, 1, 1], math_inst, min_cc, max_cc
                    ),
                ]
                for tile in tile_descriptions:
                    dst_align = 32 if tile.threadblock_shape[1] == 16 else 64
                    operations += GenerateConv2d(
                        ConvType.Convolution,
                        ConvKind.Dgrad,
                        [tile],
                        layout[0],
                        layout[1],
                        dst_layout,
                        dst_type,
                        min_cc,
                        layout[2],
                        layout[2],
                        dst_align,
                        use_special_optimization,
                        ImplicitGemmMode.GemmTN,
                        False,
                        cuda_major,
                        cuda_minor,
                    )
    return operations


################################################################################
# parameters
# Edge - for tiles, the edges represent the length of one side
# Ratio - the maximum ratio between 2 edges, limits the skinnyness of tiles
# MaxEdge - maximum length of each edge
# Min/Max - minimum/maximum of the product of edge lengths
################################################################################

warpsPerThreadblockEdge = [1, 2, 4, 8, 16]
warpsPerThreadblockRatio = 2
warpsPerThreadblockMax = 16
# NOTE 1x32 and 2x16 warp tile shapes fail validation for ~10% of cases

warpShapeEdges = [8, 16, 32, 64, 128, 256]
warpShapeRatio = 4
warpShapeMax = 64 * 64
warpShapeMin = 8 * 8

threadblockEdgeMax = 256

#   char,         type             bits/elem, max tile,    L0 threadblock tiles
precisions = {
    "c": ["cutlass::complex<float>", 64, 64 * 128, [[64, 128], [64, 32]]],
    "d": ["double", 64, 64 * 64, [[64, 64], [32, 32]]],
    "h": ["cutlass::half_t", 16, 128 * 256, [[256, 128], [64, 128], [64, 32]]],
    "i": ["int", 32, 128 * 128, [[128, 64], [16, 32]]],
    "s": ["float", 32, 128 * 128, [[128, 256], [128, 128], [64, 64]]],
    "z": ["cutlass::complex<double>", 128, 64 * 64, [[32, 64], [16, 32]]],
}
# L1 will have a single kernel for every unique shape
# L2 will have everything else
def GenerateGemm_Simt(args):
    ################################################################################
    # warps per threadblock
    ################################################################################
    warpsPerThreadblocks = []
    for warpsPerThreadblock0 in warpsPerThreadblockEdge:
        for warpsPerThreadblock1 in warpsPerThreadblockEdge:
            if (
                warpsPerThreadblock0 / warpsPerThreadblock1 <= warpsPerThreadblockRatio
                and warpsPerThreadblock1 / warpsPerThreadblock0
                <= warpsPerThreadblockRatio
                and warpsPerThreadblock0 * warpsPerThreadblock1
                <= warpsPerThreadblockMax
            ):
                warpsPerThreadblocks.append(
                    [warpsPerThreadblock0, warpsPerThreadblock1]
                )

    ################################################################################
    # warp shapes
    ################################################################################
    warpNumThreads = 32
    warpShapes = []
    for warp0 in warpShapeEdges:
        for warp1 in warpShapeEdges:
            if (
                warp0 / warp1 <= warpShapeRatio
                and warp1 / warp0 <= warpShapeRatio
                and warp0 * warp1 <= warpShapeMax
                and warp0 * warp1 > warpShapeMin
            ):
                warpShapes.append([warp0, warp1])

    # sgemm
    (
        precisionType,
        precisionBits,
        threadblockMaxElements,
        threadblockTilesL0,
    ) = precisions["s"]

    layouts = [
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),  # nn
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),  # nt
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),  # tn
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),  # tt
    ]

    math_instructions = [
        MathInstruction(
            [1, 1, 1],
            DataType.f32,
            DataType.f32,
            DataType.f32,
            OpcodeClass.Simt,
            MathOperation.multiply_add,
        )
    ]

    min_cc = 50
    max_cc = 1024

    operations = []
    for math_inst in math_instructions:
        for layout in layouts:
            data_type = [
                math_inst.element_a,
                math_inst.element_b,
                math_inst.element_accumulator,
                math_inst.element_accumulator,
            ]
            tile_descriptions = [
                TileDescription([64, 256, 8], 2, [2, 4, 1], math_inst, min_cc, max_cc),
                TileDescription([256, 64, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([32, 256, 8], 2, [2, 4, 1], math_inst, min_cc, max_cc),
                TileDescription([256, 32, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([128, 64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([128, 32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([32, 64, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 32, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([32, 32, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([8, 32, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([16, 32, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([16, 64, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([16, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
            ]
            for warpsPerThreadblock in warpsPerThreadblocks:
                for warpShape in warpShapes:
                    warpThreadsM = 0
                    if warpShape[0] > warpShape[1]:
                        warpThreadsM = 8
                    else:
                        warpThreadsM = 4
                    warpThreadsN = warpNumThreads / warpThreadsM

                    # skip shapes with conflicting rectangularity
                    # they are unlikely to be fastest
                    blockG = warpsPerThreadblock[0] > warpsPerThreadblock[1]
                    blockL = warpsPerThreadblock[0] < warpsPerThreadblock[1]
                    warpG = warpShape[0] > warpShape[1]
                    warpL = warpShape[0] < warpShape[1]

                    blockG2 = warpsPerThreadblock[0] > warpsPerThreadblock[1] * 2
                    blockL2 = warpsPerThreadblock[0] * 2 < warpsPerThreadblock[1]
                    warpG2 = warpShape[0] > warpShape[1] * 2
                    warpL2 = warpShape[0] * 2 < warpShape[1]

                    if blockG2 and warpL:
                        continue
                    if blockL2 and warpG:
                        continue
                    if warpG2 and blockL:
                        continue
                    if warpL2 and blockG:
                        continue

                    # check threadblock ratios and max
                    threadblockTile = [
                        warpShape[0] * warpsPerThreadblock[0],
                        warpShape[1] * warpsPerThreadblock[1],
                    ]
                    if threadblockTile[0] * threadblockTile[1] > threadblockMaxElements:
                        continue
                    if threadblockTile[0] > threadblockEdgeMax:
                        continue
                    if threadblockTile[1] > threadblockEdgeMax:
                        continue
                    totalThreads = (
                        warpNumThreads * warpsPerThreadblock[0] * warpsPerThreadblock[1]
                    )

                    # calculate unroll
                    # ensure that every iteration at least a full load of A,B are done
                    unrollMin = 8
                    unrollMin0 = totalThreads // threadblockTile[0]
                    unrollMin1 = totalThreads // threadblockTile[1]
                    unroll = max(unrollMin, unrollMin0, unrollMin1)

                    threadTileM = warpShape[0] // warpThreadsM
                    threadTileN = warpShape[1] // warpThreadsN
                    if threadTileM < 2 or threadTileN < 2:
                        continue
                    if threadTileM * threadTileN * precisionBits > 8 * 8 * 32:
                        continue

                    # epilogue currently only supports N < WarpNumThreads
                    if threadblockTile[1] < warpNumThreads:
                        continue

                    # limit smem
                    smemBitsA = threadblockTile[0] * unroll * 2 * precisionBits
                    smemBitsB = threadblockTile[1] * unroll * 2 * precisionBits
                    smemKBytes = (smemBitsA + smemBitsB) / 8 / 1024
                    if smemKBytes > 48:
                        continue

                    tile = TileDescription(
                        [threadblockTile[0], threadblockTile[1], unroll],
                        2,
                        [
                            threadblockTile[0] // warpShape[0],
                            threadblockTile[1] // warpShape[1],
                            1,
                        ],
                        math_inst,
                        min_cc,
                        max_cc,
                    )

                    def filter(t: TileDescription) -> bool:
                        nonlocal tile
                        return (
                            t.threadblock_shape[0] == tile.threadblock_shape[0]
                            and t.threadblock_shape[1] == tile.threadblock_shape[1]
                            and t.threadblock_shape[2] == tile.threadblock_shape[2]
                            and t.warp_count[0] == tile.warp_count[0]
                            and t.warp_count[1] == tile.warp_count[1]
                            and t.warp_count[2] == tile.warp_count[2]
                            and t.stages == tile.stages
                        )

                    if not any(t for t in tile_descriptions if filter(t)):
                        continue

                    operations += GeneratesGemm(
                        tile, data_type, layout[0], layout[1], layout[2], min_cc
                    )
    return operations


#
def GenerateDwconv2d_Simt(args, conv_kind):
    ################################################################################
    # warps per threadblock
    ################################################################################
    warpsPerThreadblocks = []
    for warpsPerThreadblock0 in warpsPerThreadblockEdge:
        for warpsPerThreadblock1 in warpsPerThreadblockEdge:
            if (
                warpsPerThreadblock0 / warpsPerThreadblock1 <= warpsPerThreadblockRatio
                and warpsPerThreadblock1 / warpsPerThreadblock0
                <= warpsPerThreadblockRatio
                and warpsPerThreadblock0 * warpsPerThreadblock1
                <= warpsPerThreadblockMax
            ):
                warpsPerThreadblocks.append(
                    [warpsPerThreadblock0, warpsPerThreadblock1]
                )

    ################################################################################
    # warp shapes
    ################################################################################
    warpNumThreads = 32
    warpShapes = []
    for warp0 in warpShapeEdges:
        for warp1 in warpShapeEdges:
            if (
                warp0 / warp1 <= warpShapeRatio
                and warp1 / warp0 <= warpShapeRatio
                and warp0 * warp1 <= warpShapeMax
                and warp0 * warp1 > warpShapeMin
            ):
                warpShapes.append([warp0, warp1])

    # sgemm
    (
        precisionType,
        precisionBits,
        threadblockMaxElements,
        threadblockTilesL0,
    ) = precisions["s"]

    layouts = [(LayoutType.TensorNCHW, LayoutType.TensorNCHW)]

    math_instructions = [
        MathInstruction(
            [1, 1, 1],
            DataType.f32,
            DataType.f32,
            DataType.f32,
            OpcodeClass.Simt,
            MathOperation.multiply_add,
        )
    ]

    min_cc = 50
    max_cc = 1024

    dst_layouts = [LayoutType.TensorNCHW]

    dst_types = [DataType.f32]

    if conv_kind == ConvKind.Wgrad:
        alignment_constraints = [32]
    else:
        alignment_constraints = [128, 32]

    operations = []
    for math_inst in math_instructions:
        tile_descriptions = [
            TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([32, 64, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 32, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([32, 32, 8], 2, [1, 1, 1], math_inst, min_cc, max_cc),
        ]
        for warpsPerThreadblock in warpsPerThreadblocks:
            for warpShape in warpShapes:
                warpThreadsM = 0
                if warpShape[0] > warpShape[1]:
                    warpThreadsM = 8
                else:
                    warpThreadsM = 4
                warpThreadsN = warpNumThreads / warpThreadsM

                # skip shapes with conflicting rectangularity
                # they are unlikely to be fastest
                blockG = warpsPerThreadblock[0] > warpsPerThreadblock[1]
                blockL = warpsPerThreadblock[0] < warpsPerThreadblock[1]
                warpG = warpShape[0] > warpShape[1]
                warpL = warpShape[0] < warpShape[1]

                blockG2 = warpsPerThreadblock[0] > warpsPerThreadblock[1] * 2
                blockL2 = warpsPerThreadblock[0] * 2 < warpsPerThreadblock[1]
                warpG2 = warpShape[0] > warpShape[1] * 2
                warpL2 = warpShape[0] * 2 < warpShape[1]

                if blockG2 and warpL:
                    continue
                if blockL2 and warpG:
                    continue
                if warpG2 and blockL:
                    continue
                if warpL2 and blockG:
                    continue

                # check threadblock ratios and max
                threadblockTile = [
                    warpShape[0] * warpsPerThreadblock[0],
                    warpShape[1] * warpsPerThreadblock[1],
                ]
                if threadblockTile[0] * threadblockTile[1] > threadblockMaxElements:
                    continue
                if threadblockTile[0] > threadblockEdgeMax:
                    continue
                if threadblockTile[1] > threadblockEdgeMax:
                    continue
                totalThreads = (
                    warpNumThreads * warpsPerThreadblock[0] * warpsPerThreadblock[1]
                )

                # calculate unroll
                # ensure that every iteration at least a full load of A,B are done
                unrollMin = 8
                unrollMin0 = totalThreads // threadblockTile[0]
                unrollMin1 = totalThreads // threadblockTile[1]
                unroll = max(unrollMin, unrollMin0, unrollMin1)

                threadTileM = warpShape[0] // warpThreadsM
                threadTileN = warpShape[1] // warpThreadsN
                if threadTileM < 2 or threadTileN < 2:
                    continue
                if threadTileM * threadTileN * precisionBits > 8 * 8 * 32:
                    continue

                # epilogue currently only supports N < WarpNumThreads
                if threadblockTile[1] < warpNumThreads:
                    continue

                # limit smem
                smemBitsA = threadblockTile[0] * unroll * 2 * precisionBits
                smemBitsB = threadblockTile[1] * unroll * 2 * precisionBits
                smemKBytes = (smemBitsA + smemBitsB) / 8 / 1024
                if smemKBytes > 48:
                    continue

                tile = TileDescription(
                    [threadblockTile[0], threadblockTile[1], unroll],
                    2,
                    [
                        threadblockTile[0] // warpShape[0],
                        threadblockTile[1] // warpShape[1],
                        1,
                    ],
                    math_inst,
                    min_cc,
                    max_cc,
                )

                def filter(t: TileDescription) -> bool:
                    nonlocal tile
                    return (
                        t.threadblock_shape[0] == tile.threadblock_shape[0]
                        and t.threadblock_shape[1] == tile.threadblock_shape[1]
                        and t.threadblock_shape[2] == tile.threadblock_shape[2]
                        and t.warp_count[0] == tile.warp_count[0]
                        and t.warp_count[1] == tile.warp_count[1]
                        and t.warp_count[2] == tile.warp_count[2]
                        and t.stages == tile.stages
                    )

                if not any(t for t in tile_descriptions if filter(t)):
                    continue

                for layout in layouts:
                    for dst_type, dst_layout in zip(dst_types, dst_layouts):
                        for alignment_src in alignment_constraints:
                            operations += GenerateConv2d(
                                ConvType.DepthwiseConvolution,
                                conv_kind,
                                [tile],
                                layout[0],
                                layout[1],
                                dst_layout,
                                dst_type,
                                min_cc,
                                alignment_src,
                                32,
                                32,
                                SpecialOptimizeDesc.NoneSpecialOpt,
                                ImplicitGemmMode.GemmNT
                                if conv_kind == ConvKind.Wgrad
                                else ImplicitGemmMode.GemmTN,
                            )
    return operations


def GenerateRegionRestrictedconv2d_Simt(args, conv_kind):
    ################################################################################
    # warps per threadblock
    ################################################################################
    warpsPerThreadblocks = []
    for warpsPerThreadblock0 in warpsPerThreadblockEdge:
        for warpsPerThreadblock1 in warpsPerThreadblockEdge:
            if (
                warpsPerThreadblock0 / warpsPerThreadblock1 <= warpsPerThreadblockRatio
                and warpsPerThreadblock1 / warpsPerThreadblock0
                <= warpsPerThreadblockRatio
                and warpsPerThreadblock0 * warpsPerThreadblock1
                <= warpsPerThreadblockMax
            ):
                warpsPerThreadblocks.append(
                    [warpsPerThreadblock0, warpsPerThreadblock1]
                )

    ################################################################################
    # warp shapes
    ################################################################################
    warpNumThreads = 32
    warpShapes = []
    for warp0 in warpShapeEdges:
        for warp1 in warpShapeEdges:
            if (
                warp0 / warp1 <= warpShapeRatio
                and warp1 / warp0 <= warpShapeRatio
                and warp0 * warp1 <= warpShapeMax
                and warp0 * warp1 > warpShapeMin
            ):
                warpShapes.append([warp0, warp1])

    # sgemm
    (
        precisionType,
        precisionBits,
        threadblockMaxElements,
        threadblockTilesL0,
    ) = precisions["s"]

    layouts = [(LayoutType.TensorNCHW, LayoutType.TensorNCHW)]

    math_instructions = [
        MathInstruction(
            [1, 1, 1],
            DataType.f32,
            DataType.f32,
            DataType.f32,
            OpcodeClass.Simt,
            MathOperation.multiply_add,
            DataType.s32,
            DataType.s32,
        ),
        MathInstruction(
            [1, 1, 1],
            DataType.f32,
            DataType.f32,
            DataType.f32,
            OpcodeClass.Simt,
            MathOperation.multiply_add,
            DataType.s8,
            DataType.s8,
        ),
    ]

    min_cc = 50
    max_cc = 1024

    dst_layouts = [LayoutType.TensorNCHW]

    dst_types = [DataType.f32]

    if conv_kind == ConvKind.Wgrad:
        alignment_constraints = [32]
    else:
        alignment_constraints = [128, 32]

    operations = []
    for math_inst in math_instructions:
        tile_descriptions = [
            TileDescription([128, 128, 8], 1, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 64, 8], 1, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 8], 1, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 32, 8], 1, [2, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([32, 128, 8], 1, [1, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 64, 8], 1, [2, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([32, 64, 8], 1, [1, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 32, 8], 1, [1, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([32, 32, 8], 1, [1, 1, 1], math_inst, min_cc, max_cc),
        ]
        for warpsPerThreadblock in warpsPerThreadblocks:
            for warpShape in warpShapes:
                warpThreadsM = 0
                if warpShape[0] > warpShape[1]:
                    warpThreadsM = 8
                else:
                    warpThreadsM = 4
                warpThreadsN = warpNumThreads / warpThreadsM

                # skip shapes with conflicting rectangularity
                # they are unlikely to be fastest
                blockG = warpsPerThreadblock[0] > warpsPerThreadblock[1]
                blockL = warpsPerThreadblock[0] < warpsPerThreadblock[1]
                warpG = warpShape[0] > warpShape[1]
                warpL = warpShape[0] < warpShape[1]

                blockG2 = warpsPerThreadblock[0] > warpsPerThreadblock[1] * 2
                blockL2 = warpsPerThreadblock[0] * 2 < warpsPerThreadblock[1]
                warpG2 = warpShape[0] > warpShape[1] * 2
                warpL2 = warpShape[0] * 2 < warpShape[1]

                if blockG2 and warpL:
                    continue
                if blockL2 and warpG:
                    continue
                if warpG2 and blockL:
                    continue
                if warpL2 and blockG:
                    continue

                # check threadblock ratios and max
                threadblockTile = [
                    warpShape[0] * warpsPerThreadblock[0],
                    warpShape[1] * warpsPerThreadblock[1],
                ]
                if threadblockTile[0] * threadblockTile[1] > threadblockMaxElements:
                    continue
                if threadblockTile[0] > threadblockEdgeMax:
                    continue
                if threadblockTile[1] > threadblockEdgeMax:
                    continue
                totalThreads = (
                    warpNumThreads * warpsPerThreadblock[0] * warpsPerThreadblock[1]
                )

                # calculate unroll
                # ensure that every iteration at least a full load of A,B are done
                unrollMin = 8
                unrollMin0 = totalThreads // threadblockTile[0]
                unrollMin1 = totalThreads // threadblockTile[1]
                unroll = max(unrollMin, unrollMin0, unrollMin1)

                threadTileM = warpShape[0] // warpThreadsM
                threadTileN = warpShape[1] // warpThreadsN
                if threadTileM < 2 or threadTileN < 2:
                    continue
                if threadTileM * threadTileN * precisionBits > 8 * 8 * 32:
                    continue

                # epilogue currently only supports N < WarpNumThreads
                if threadblockTile[1] < warpNumThreads:
                    continue

                # limit smem
                smemBitsA = threadblockTile[0] * unroll * 2 * precisionBits
                smemBitsB = threadblockTile[1] * unroll * 2 * precisionBits
                smemKBytes = (smemBitsA + smemBitsB) / 8 / 1024
                if smemKBytes > 48:
                    continue

                tile = TileDescription(
                    [threadblockTile[0], threadblockTile[1], unroll],
                    1,
                    [
                        threadblockTile[0] // warpShape[0],
                        threadblockTile[1] // warpShape[1],
                        1,
                    ],
                    math_inst,
                    min_cc,
                    max_cc,
                )

                def filter(t: TileDescription) -> bool:
                    nonlocal tile
                    return (
                        t.threadblock_shape[0] == tile.threadblock_shape[0]
                        and t.threadblock_shape[1] == tile.threadblock_shape[1]
                        and t.threadblock_shape[2] == tile.threadblock_shape[2]
                        and t.warp_count[0] == tile.warp_count[0]
                        and t.warp_count[1] == tile.warp_count[1]
                        and t.warp_count[2] == tile.warp_count[2]
                        and t.stages == tile.stages
                    )

                if not any(t for t in tile_descriptions if filter(t)):
                    continue

                for layout in layouts:
                    for dst_type, dst_layout in zip(dst_types, dst_layouts):
                        for alignment_src in alignment_constraints:
                            operations += GenerateConv2d(
                                ConvType.RegionRestrictedConvolution,
                                conv_kind,
                                [tile],
                                layout[0],
                                layout[1],
                                dst_layout,
                                dst_type,
                                min_cc,
                                alignment_src,
                                32,
                                32,
                                SpecialOptimizeDesc.NoneSpecialOpt,
                                ImplicitGemmMode.GemmNT
                                if conv_kind == ConvKind.Wgrad
                                else ImplicitGemmMode.GemmTN,
                            )
    return operations


#
def GenerateDwconv2d_TensorOp_884(args, conv_kind):
    layouts = [(LayoutType.TensorNCHW, LayoutType.TensorNCHW)]

    math_instructions = [
        MathInstruction(
            [8, 8, 4],
            DataType.f16,
            DataType.f16,
            DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
        MathInstruction(
            [8, 8, 4],
            DataType.f16,
            DataType.f16,
            DataType.f16,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
    ]

    min_cc = 70
    max_cc = 75

    dst_layouts = [LayoutType.TensorNCHW]

    if conv_kind == ConvKind.Wgrad:
        dst_types = [DataType.f32]
    else:
        dst_types = [DataType.f16]

    alignment_constraints = [128, 32, 16]
    cuda_major = 10
    cuda_minor = 1

    operations = []
    for math_inst in math_instructions:
        tile_descriptions = [
            TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 128, 32], 2, [4, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 64, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
        ]
        for layout in layouts:
            for dst_type, dst_layout in zip(dst_types, dst_layouts):
                for alignment_src in alignment_constraints:
                    if conv_kind == ConvKind.Wgrad:
                        # skip io16xc16
                        if math_inst.element_accumulator == DataType.f16:
                            continue
                        for alignment_diff in alignment_constraints:
                            operations += GenerateConv2d(
                                ConvType.DepthwiseConvolution,
                                conv_kind,
                                tile_descriptions,
                                layout[0],
                                layout[1],
                                dst_layout,
                                dst_type,
                                min_cc,
                                alignment_src,
                                alignment_diff,
                                32,  # always f32 output
                                SpecialOptimizeDesc.NoneSpecialOpt,
                                ImplicitGemmMode.GemmNT,
                                False,
                                cuda_major,
                                cuda_minor,
                            )
                    else:
                        operations += GenerateConv2d(
                            ConvType.DepthwiseConvolution,
                            conv_kind,
                            tile_descriptions,
                            layout[0],
                            layout[1],
                            dst_layout,
                            dst_type,
                            min_cc,
                            alignment_src,
                            16,
                            16,
                            SpecialOptimizeDesc.NoneSpecialOpt,
                            ImplicitGemmMode.GemmTN,
                            False,
                            cuda_major,
                            cuda_minor,
                        )

    return operations


#
def GenerateGemv_Simt(args):
    threadBlockShape_N = [128, 64, 32]
    ldgBits_A = [128, 64, 32]
    ldgBits_B = [128, 64, 32]

    layouts = [(LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor)]

    math_instructions = [
        MathInstruction(
            [1, 1, 1],
            DataType.f32,
            DataType.f32,
            DataType.f32,
            OpcodeClass.Simt,
            MathOperation.multiply_add,
        )
    ]

    min_cc = 50

    operations = []
    for math_inst in math_instructions:
        for layout in layouts:
            data_type = [
                math_inst.element_a,
                math_inst.element_b,
                math_inst.element_accumulator,
                math_inst.element_accumulator,
            ]
            for threadblock_shape_n in threadBlockShape_N:
                for align_a in ldgBits_A:
                    for align_b in ldgBits_B:
                        ldg_elements_a = align_a // DataTypeSize[math_inst.element_a]
                        ldg_elements_b = align_b // DataTypeSize[math_inst.element_b]
                        threadblock_shape_k = (256 * ldg_elements_a) // (
                            threadblock_shape_n // ldg_elements_b
                        )
                        threadblock_shape = [
                            1,
                            threadblock_shape_n,
                            threadblock_shape_k,
                        ]
                        thread_shape = [1, ldg_elements_b, ldg_elements_a]

                        operations.append(
                            GeneratesGemv(
                                math_inst,
                                threadblock_shape,
                                thread_shape,
                                data_type,
                                layout[0],
                                layout[1],
                                layout[2],
                                min_cc,
                                align_a,
                                align_b,
                            )
                        )
    return operations


#
def GeneratesGemm_TensorOp_1688(args):
    layouts = [
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),  # nn
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),  # nt
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),  # tn
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),  # tt
    ]

    math_instructions = [
        MathInstruction(
            [16, 8, 8],
            DataType.f16,
            DataType.f16,
            DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
        MathInstruction(
            [16, 8, 8],
            DataType.f16,
            DataType.f16,
            DataType.f16,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
    ]

    min_cc = 75
    max_cc = 1024

    alignment_constraints = [
        8,
        4,
        2,
        # 1
    ]
    cuda_major = 10
    cuda_minor = 2

    operations = []
    for math_inst in math_instructions:
        for layout in layouts:
            for align in alignment_constraints:
                tile_descriptions = [
                    TileDescription(
                        [256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc
                    ),
                    ## comment some configuration to reduce compilation time and binary size
                    #         TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                    #         TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                    #         TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                ]

                data_type = [
                    math_inst.element_a,
                    math_inst.element_b,
                    math_inst.element_a,
                    math_inst.element_accumulator,
                ]

                for tile in tile_descriptions:
                    operations += GeneratesGemm(
                        tile,
                        data_type,
                        layout[0],
                        layout[1],
                        layout[2],
                        min_cc,
                        align * 16,
                        align * 16,
                        align * 16,
                        cuda_major,
                        cuda_minor,
                    )
    return operations


#
def GeneratesGemm_TensorOp_884(args):
    layouts = [
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),  # nn
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),  # nt
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),  # tn
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),  # tt
    ]

    math_instructions = [
        MathInstruction(
            [8, 8, 4],
            DataType.f16,
            DataType.f16,
            DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
        MathInstruction(
            [8, 8, 4],
            DataType.f16,
            DataType.f16,
            DataType.f16,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
    ]

    min_cc = 70
    max_cc = 75

    alignment_constraints = [
        8,
        4,
        2,
        # 1
    ]
    cuda_major = 10
    cuda_minor = 1

    operations = []
    for math_inst in math_instructions:
        for layout in layouts:
            for align in alignment_constraints:
                tile_descriptions = [
                    TileDescription(
                        [256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc
                    ),
                    TileDescription(
                        [128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc
                    ),
                    ## comment some configuration to reduce compilation time and binary size
                    #         TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                    #         TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                    #         TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                ]

                data_type = [
                    math_inst.element_a,
                    math_inst.element_b,
                    math_inst.element_a,
                    math_inst.element_accumulator,
                ]

                for tile in tile_descriptions:
                    operations += GeneratesGemm(
                        tile,
                        data_type,
                        layout[0],
                        layout[1],
                        layout[2],
                        min_cc,
                        align * 16,
                        align * 16,
                        align * 16,
                        cuda_major,
                        cuda_minor,
                    )

    return operations


#
def GenerateConv2dOperations(args):
    if args.type == "simt":
        return GenerateConv2d_Simt(args)
    elif args.type == "tensorop8816":
        return GenerateConv2d_TensorOp_8816(args)
    else:
        assert args.type == "tensorop8832", (
            "operation conv2d only support"
            "simt, tensorop8816 and tensorop8832. (got:{})".format(args.type)
        )
        return GenerateConv2d_TensorOp_8832(args)


def GenerateDeconvOperations(args):
    if args.type == "simt":
        return GenerateDeconv_Simt(args)
    else:
        assert args.type == "tensorop8816", (
            "operation deconv only support"
            "simt and tensorop8816. (got:{})".format(args.type)
        )
        return GenerateDeconv_TensorOp_8816(args)


def GenerateDwconv2dFpropOperations(args):
    if args.type == "simt":
        return GenerateDwconv2d_Simt(args, ConvKind.Fprop)
    else:
        assert args.type == "tensorop884", (
            "operation dwconv2d fprop only support"
            "simt, tensorop884. (got:{})".format(args.type)
        )
        return GenerateDwconv2d_TensorOp_884(args, ConvKind.Fprop)


def GenerateDwconv2dDgradOperations(args):
    if args.type == "simt":
        return GenerateDwconv2d_Simt(args, ConvKind.Dgrad)
    else:
        assert args.type == "tensorop884", (
            "operation dwconv2d fprop only support"
            "simt, tensorop884. (got:{})".format(args.type)
        )
        return GenerateDwconv2d_TensorOp_884(args, ConvKind.Dgrad)


def GenerateDwconv2dWgradOperations(args):
    if args.type == "simt":
        return GenerateDwconv2d_Simt(args, ConvKind.Wgrad)
    else:
        assert args.type == "tensorop884", (
            "operation dwconv2d fprop only support"
            "simt, tensorop884. (got:{})".format(args.type)
        )
        return GenerateDwconv2d_TensorOp_884(args, ConvKind.Wgrad)


def GenerateRegionRestrictedconv2dWgradOperations(args):
    assert args.type == "simt", (
        "operation RegionRestrictedconv2d wgrad only support"
        "simt. (got:{})".format(args.type)
    )
    return GenerateRegionRestrictedconv2d_Simt(args, ConvKind.Wgrad)


def GenerateGemmOperations(args):
    if args.type == "tensorop884":
        return GeneratesGemm_TensorOp_884(args)
    elif args.type == "tensorop1688":
        return GeneratesGemm_TensorOp_1688(args)
    else:
        assert (
            args.type == "simt"
        ), "operation gemm only support" "simt. (got:{})".format(args.type)
        return GenerateGemm_Simt(args)


def GenerateGemvOperations(args):
    assert args.type == "simt", "operation gemv only support" "simt. (got:{})".format(
        args.type
    )
    return GenerateGemv_Simt(args)


################################################################################
# parameters
# split_number - the concated file will be divided into split_number parts
# file_path - the path of file, which is need to be concated
# operations - args.operations
# type - args.type
# head - the head in the file
# required_cuda_ver_major - required cuda major
# required_cuda_ver_minor - required cuda minjor
# epilogue - the epilogue in the file
# wrapper_path - wrapper path
################################################################################
def ConcatFile(
    split_number: int,
    file_path: str,
    operations: str,
    type: str,
    head: str,
    required_cuda_ver_major: str,
    required_cuda_ver_minor: str,
    epilogue: str,
    wrapper_path=None,
):
    import os

    meragefiledir = file_path
    filenames = os.listdir(meragefiledir)
    # filter file
    if "tensorop" in type:
        sub_string_1 = "tensorop"
        sub_string_2 = type[8:]
    else:
        sub_string_1 = sub_string_2 = "simt"
    if "dwconv2d_" in operations:
        filtered_operations = operations[:2] + operations[9:]
    if "rrconv2d_" in operations:
        filtered_operations = operations[:2] + operations[9:]
    elif ("conv2d" in operations) or ("deconv" in operations):
        filtered_operations = "cutlass"
    else:
        filtered_operations = operations
    # get the file list number
    file_list = {}
    file_list[operations + type] = 0
    for filename in filenames:
        if (
            (filtered_operations in filename)
            and (sub_string_1 in filename)
            and (sub_string_2 in filename)
            and ("all_" not in filename)
        ):
            file_list[operations + type] += 1
    # concat file for linux
    flag_1 = 0
    flag_2 = 0
    for filename in filenames:
        if (
            (filtered_operations in filename)
            and (sub_string_1 in filename)
            and (sub_string_2 in filename)
            and ("all_" not in filename)
        ):
            flag_1 += 1
            filepath = meragefiledir + "/" + filename
            if (flag_1 >= flag_2 * (file_list[operations + type] / split_number)) and (
                flag_1 <= (flag_2 + 1) * (file_list[operations + type] / split_number)
            ):
                file = open(
                    file_path + "/{}_{}_{}.cu".format(operations, type, flag_2), "a"
                )
                # write Template at the head
                if wrapper_path is None:
                    file.write(
                        SubstituteTemplate(
                            head,
                            {
                                "required_cuda_ver_major": str(required_cuda_ver_major),
                                "required_cuda_ver_minor": str(required_cuda_ver_minor),
                            },
                        )
                    )
                else:
                    file.write(
                        SubstituteTemplate(
                            head,
                            {
                                "wrapper_path": wrapper_path,
                                "required_cuda_ver_major": str(required_cuda_ver_major),
                                "required_cuda_ver_minor": str(required_cuda_ver_minor),
                            },
                        )
                    )
                # concat all the remaining files
                if flag_2 == (split_number - 1):
                    for line in open(filepath):
                        file.writelines(line)
                    os.remove(filepath)
                    file.write("\n")
                    file.write(epilogue)
                    continue
                for line in open(filepath):
                    file.writelines(line)
                os.remove(filepath)
                file.write("\n")
                file.write(epilogue)
            else:
                # write Template at the head
                if wrapper_path is None:
                    file.write(
                        SubstituteTemplate(
                            head,
                            {
                                "required_cuda_ver_major": str(required_cuda_ver_major),
                                "required_cuda_ver_minor": str(required_cuda_ver_minor),
                            },
                        )
                    )
                else:
                    file.write(
                        SubstituteTemplate(
                            head,
                            {
                                "wrapper_path": wrapper_path,
                                "required_cuda_ver_major": str(required_cuda_ver_major),
                                "required_cuda_ver_minor": str(required_cuda_ver_minor),
                            },
                        )
                    )
                for line in open(filepath):
                    file.writelines(line)
                os.remove(filepath)
                file.write("\n")
                file.write(epilogue)
                file.close()
                flag_2 += 1

        # concat file for windows
        elif filename[0].isdigit() and ("all_" not in filename):
            flag_1 += 1
            filepath = meragefiledir + "/" + filename
            if (flag_1 >= flag_2 * (len(filenames) / split_number)) and (
                flag_1 <= (flag_2 + 1) * (len(filenames) / split_number)
            ):
                file = open(
                    file_path + "/{}_{}_{}.cu".format(operations, type, flag_2), "a"
                )
                # write Template at the head
                if wrapper_path is None:
                    file.write(
                        SubstituteTemplate(
                            head,
                            {
                                "required_cuda_ver_major": str(required_cuda_ver_major),
                                "required_cuda_ver_minor": str(required_cuda_ver_minor),
                            },
                        )
                    )
                else:
                    file.write(
                        SubstituteTemplate(
                            head,
                            {
                                "wrapper_path": wrapper_path,
                                "required_cuda_ver_major": str(required_cuda_ver_major),
                                "required_cuda_ver_minor": str(required_cuda_ver_minor),
                            },
                        )
                    )
                # concat all the remaining files
                if flag_2 == (split_number - 1):
                    for line in open(filepath):
                        file.writelines(line)
                    os.remove(filepath)
                    file.write("\n")
                    file.write(epilogue)
                    continue
                for line in open(filepath):
                    file.writelines(line)
                os.remove(filepath)
                file.write("\n")
                file.write(epilogue)
            else:
                # write Template at the head
                if wrapper_path is None:
                    file.write(
                        SubstituteTemplate(
                            head,
                            {
                                "required_cuda_ver_major": str(required_cuda_ver_major),
                                "required_cuda_ver_minor": str(required_cuda_ver_minor),
                            },
                        )
                    )
                else:
                    file.write(
                        SubstituteTemplate(
                            head,
                            {
                                "wrapper_path": wrapper_path,
                                "required_cuda_ver_major": str(required_cuda_ver_major),
                                "required_cuda_ver_minor": str(required_cuda_ver_minor),
                            },
                        )
                    )
                for line in open(filepath):
                    file.writelines(line)
                os.remove(filepath)
                file.write("\n")
                file.write(epilogue)
                file.close()
                flag_2 += 1


###################################################################################################
###################################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generates device kernel registration code for CUTLASS Kernels"
    )
    parser.add_argument(
        "--operations",
        type=str,
        choices=[
            "gemm",
            "gemv",
            "conv2d",
            "deconv",
            "dwconv2d_fprop",
            "dwconv2d_dgrad",
            "dwconv2d_wgrad",
            "rrconv2d_wgrad",
        ],
        required=True,
        help="Specifies the operation to generate (gemm, gemv, conv2d, deconv, dwconv2d_fprop, dwconv2d_dgrad, dwconv2d_wgrad)",
    )
    parser.add_argument(
        "output", type=str, help="output directory for CUTLASS kernel files"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["simt", "tensorop8816", "tensorop8832", "tensorop884", "tensorop1688"],
        default="simt",
        help="kernel type of CUTLASS kernel generator",
    )

    gemv_wrapper_path = (
        "src/cuda/matrix_mul/cutlass_matrix_mul_wrapper_batched_gemv_strided.cuinl"
    )
    short_path = (
        platform.system() == "Windows" or platform.system().find("NT") >= 0
    ) and ("true" != os.getenv("CUTLASS_WITH_LONG_PATH", default="False").lower())
    args = parser.parse_args()

    if args.operations == "gemm":
        operations = GenerateGemmOperations(args)
    elif args.operations == "gemv":
        operations = GenerateGemvOperations(args)
    elif args.operations == "conv2d":
        operations = GenerateConv2dOperations(args)
    elif args.operations == "deconv":
        operations = GenerateDeconvOperations(args)
    elif args.operations == "dwconv2d_fprop":
        operations = GenerateDwconv2dFpropOperations(args)
    elif args.operations == "dwconv2d_dgrad":
        operations = GenerateDwconv2dDgradOperations(args)
    elif args.operations == "dwconv2d_wgrad":
        operations = GenerateDwconv2dWgradOperations(args)
    else:
        assert args.operations == "rrconv2d_wgrad", "invalid operation"
        operations = GenerateRegionRestrictedconv2dWgradOperations(args)

    if (
        args.operations == "conv2d"
        or args.operations == "deconv"
        or args.operations == "dwconv2d_fprop"
        or args.operations == "dwconv2d_dgrad"
        or args.operations == "dwconv2d_wgrad"
    ):
        for operation in operations:
            with EmitConvSingleKernelWrapper(
                args.output, operation, short_path
            ) as emitter:
                emitter.emit()
        head = EmitConvSingleKernelWrapper(
            args.output, operations[0], short_path
        ).header_template
        required_cuda_ver_major = operations[0].required_cuda_ver_major
        required_cuda_ver_minor = operations[0].required_cuda_ver_minor
        epilogue = EmitConvSingleKernelWrapper(
            args.output, operations[0], short_path
        ).epilogue_template
        if "tensorop" in args.type:
            ConcatFile(
                4,
                args.output,
                args.operations,
                args.type,
                head,
                required_cuda_ver_major,
                required_cuda_ver_minor,
                epilogue,
            )
        else:
            ConcatFile(
                2,
                args.output,
                args.operations,
                args.type,
                head,
                required_cuda_ver_major,
                required_cuda_ver_minor,
                epilogue,
            )
    elif args.operations == "rrconv2d_wgrad":
        for operation in operations:
            with EmitRegionRestrictedConvSingleKernelWrapper(
                args.output, operation, short_path
            ) as emitter:
                emitter.emit()
        head = EmitRegionRestrictedConvSingleKernelWrapper(
            args.output, operations[0], short_path
        ).header_template
        required_cuda_ver_major = operations[0].required_cuda_ver_major
        required_cuda_ver_minor = operations[0].required_cuda_ver_minor
        epilogue = EmitRegionRestrictedConvSingleKernelWrapper(
            args.output, operations[0], short_path
        ).epilogue_template
        if "tensorop" in args.type:
            ConcatFile(
                4,
                args.output,
                args.operations,
                args.type,
                head,
                required_cuda_ver_major,
                required_cuda_ver_minor,
                epilogue,
            )
        else:
            ConcatFile(
                2,
                args.output,
                args.operations,
                args.type,
                head,
                required_cuda_ver_major,
                required_cuda_ver_minor,
                epilogue,
            )
    elif args.operations == "gemm":
        for operation in operations:
            with EmitGemmSingleKernelWrapper(
                args.output, operation, short_path
            ) as emitter:
                emitter.emit()
        head = EmitGemmSingleKernelWrapper(
            args.output, operations[0], short_path
        ).header_template
        required_cuda_ver_major = operations[0].required_cuda_ver_major
        required_cuda_ver_minor = operations[0].required_cuda_ver_minor
        epilogue = EmitGemmSingleKernelWrapper(
            args.output, operations[0], short_path
        ).epilogue_template
        if args.type == "tensorop884":
            ConcatFile(
                30,
                args.output,
                args.operations,
                args.type,
                head,
                required_cuda_ver_major,
                required_cuda_ver_minor,
                epilogue,
            )
        else:
            ConcatFile(
                2,
                args.output,
                args.operations,
                args.type,
                head,
                required_cuda_ver_major,
                required_cuda_ver_minor,
                epilogue,
            )
    elif args.operations == "gemv":
        for operation in operations:
            with EmitGemvSingleKernelWrapper(
                args.output, operation, gemv_wrapper_path, short_path
            ) as emitter:
                emitter.emit()
        head = EmitGemvSingleKernelWrapper(
            args.output, operations[0], gemv_wrapper_path, short_path
        ).header_template
        required_cuda_ver_major = operations[0].required_cuda_ver_major
        required_cuda_ver_minor = operations[0].required_cuda_ver_minor
        epilogue = EmitGemvSingleKernelWrapper(
            args.output, operations[0], gemv_wrapper_path, short_path
        ).epilogue_template
        ConcatFile(
            2,
            args.output,
            args.operations,
            args.type,
            head,
            required_cuda_ver_major,
            required_cuda_ver_minor,
            epilogue,
            wrapper_path=gemv_wrapper_path,
        )

    if args.operations != "gemv":
        GenerateManifest(args, operations, args.output)

#
###################################################################################################
