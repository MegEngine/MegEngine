#
# \file generator.py
#
# \brief Generates the CUTLASS Library's instances
#

import enum
import functools
import operator
import os.path
import shutil

from library import *

###################################################################################################
#
# Data structure modeling a GEMM operation
#
###################################################################################################

#
class GemmOperation:
    #
    def __init__(
        self,
        gemm_kind,
        arch,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        epilogue_functor=EpilogueFunctor.LinearCombination,
        swizzling_functor=SwizzlingFunctor.Identity8,
        required_cuda_ver_major=9,
        required_cuda_ver_minor=2,
    ):

        self.operation_kind = OperationKind.Gemm
        self.arch = arch
        self.tile_description = tile_description
        self.gemm_kind = gemm_kind
        self.A = A
        self.B = B
        self.C = C
        self.element_epilogue = element_epilogue
        self.epilogue_functor = epilogue_functor
        self.swizzling_functor = swizzling_functor
        self.required_cuda_ver_major = required_cuda_ver_major
        self.required_cuda_ver_minor = required_cuda_ver_minor

    #
    def is_complex(self):
        complex_operators = [
            MathOperation.multiply_add_complex,
            MathOperation.multiply_add_complex_gaussian,
        ]
        return (
            self.tile_description.math_instruction.math_operation in complex_operators
        )

    #
    def is_split_k_parallel(self):
        return self.gemm_kind == GemmKind.SplitKParallel

    #
    def is_planar_complex(self):
        return self.gemm_kind in (GemmKind.PlanarComplex, GemmKind.PlanarComplexArray)

    #
    def accumulator_type(self):
        accum = self.tile_description.math_instruction.element_accumulator

        if self.is_complex():
            return get_complex_from_real(accum)

        return accum

    #
    def short_math_name(self):
        if (
            self.tile_description.math_instruction.math_operation
            == MathOperation.multiply_add_complex_gaussian
        ):
            return "g%s" % ShortDataTypeNames[self.accumulator_type()]
        return ShortDataTypeNames[self.accumulator_type()]

    #
    def core_name(self):
        """ The basic operation kind is prefixed with a letter indicating the accumulation type. """

        inst_shape = ""
        inst_operation = ""
        intermediate_type = ""

        math_operations_map = {MathOperation.xor_popc: "xor"}

        if (
            self.tile_description.math_instruction.opcode_class == OpcodeClass.TensorOp
            or self.tile_description.math_instruction.opcode_class
            == OpcodeClass.WmmaTensorOp
        ):

            math_op = self.tile_description.math_instruction.math_operation
            math_op_string = (
                math_operations_map[math_op]
                if math_op in math_operations_map.keys()
                else ""
            )

            inst_shape = "%d%d%d" % tuple(
                self.tile_description.math_instruction.instruction_shape
            )
            inst_shape += math_op_string

            if (
                self.tile_description.math_instruction.element_a != self.A.element
                and self.tile_description.math_instruction.element_a
                != self.tile_description.math_instruction.element_accumulator
            ):
                intermediate_type = DataTypeNames[
                    self.tile_description.math_instruction.element_a
                ]

        return "%s%s%s%s" % (
            self.short_math_name(),
            inst_shape,
            intermediate_type,
            GemmKindNames[self.gemm_kind],
        )

    #
    def extended_name(self):
        """ Append data types if they differ from compute type. """
        if self.is_complex():
            extended_name = "${core_name}"
        else:
            if (
                self.C.element
                != self.tile_description.math_instruction.element_accumulator
                and self.A.element
                != self.tile_description.math_instruction.element_accumulator
            ):
                extended_name = "${element_c}_${core_name}_${element_a}"
            elif (
                self.C.element
                == self.tile_description.math_instruction.element_accumulator
                and self.A.element
                != self.tile_description.math_instruction.element_accumulator
            ):
                extended_name = "${core_name}_${element_a}"
            else:
                extended_name = "${core_name}"

        extended_name = SubstituteTemplate(
            extended_name,
            {
                "element_a": DataTypeNames[self.A.element],
                "element_c": DataTypeNames[self.C.element],
                "core_name": self.core_name(),
            },
        )

        return extended_name

    #
    def layout_name(self):
        if self.is_complex() or self.is_planar_complex():
            return "%s%s" % (
                ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)],
                ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)],
            )
        return "%s%s" % (
            ShortLayoutTypeNames[self.A.layout],
            ShortLayoutTypeNames[self.B.layout],
        )

    #
    def procedural_name(self):
        """ The full procedural name indicates architecture, extended name, tile size, and layout. """
        threadblock = self.tile_description.procedural_name()

        opcode_class_name = OpcodeClassNames[
            self.tile_description.math_instruction.opcode_class
        ]

        alignment = max([self.A.alignment, self.B.alignment, self.C.alignment])

        return SubstituteTemplate(
            "cutlass_${opcode_class}_${extended_name}_${threadblock}_${layout}_align${alignment}",
            {
                "opcode_class": opcode_class_name,
                "extended_name": self.extended_name(),
                "threadblock": threadblock,
                "layout": self.layout_name(),
                "alignment": "%d" % self.A.alignment,
            },
        )

    #
    def configuration_name(self):
        """ The full procedural name indicates architecture, extended name, tile size, and layout. """
        return self.procedural_name()


###################################################################################################
#
# Data structure modeling a GEMV Batched Strided operation
#
###################################################################################################

#
class GemvBatchedStridedOperation:
    #
    def __init__(
        self,
        gemm_kind,
        arch,
        math_inst,
        threadblock_shape,
        thread_shape,
        A,
        B,
        C,
        required_cuda_ver_major=9,
        required_cuda_ver_minor=2,
    ):

        self.operation_kind = OperationKind.Gemm
        self.arch = arch
        self.gemm_kind = gemm_kind
        self.math_instruction = math_inst
        self.threadblock_shape = threadblock_shape
        self.thread_shape = thread_shape
        self.A = A
        self.B = B
        self.C = C
        self.required_cuda_ver_major = required_cuda_ver_major
        self.required_cuda_ver_minor = required_cuda_ver_minor

    #
    def accumulator_type(self):
        accum = self.math_instruction.element_accumulator

        return accum

    #
    def short_math_name(self):
        return ShortDataTypeNames[self.accumulator_type()]

    #
    def core_name(self):
        """ The basic operation kind is prefixed with a letter indicating the accumulation type. """

        return "%s%s" % (self.short_math_name(), GemmKindNames[self.gemm_kind])

    #
    def extended_name(self):
        """ Append data types if they differ from compute type. """
        if (
            self.C.element != self.math_instruction.element_accumulator
            and self.A.element != self.math_instruction.element_accumulator
        ):
            extended_name = "${element_c}_${core_name}_${element_a}"
        elif (
            self.C.element == self.math_instruction.element_accumulator
            and self.A.element != self.math_instruction.element_accumulator
        ):
            extended_name = "${core_name}_${element_a}"
        else:
            extended_name = "${core_name}"

        extended_name = SubstituteTemplate(
            extended_name,
            {
                "element_a": DataTypeNames[self.A.element],
                "element_c": DataTypeNames[self.C.element],
                "core_name": self.core_name(),
            },
        )

        return extended_name

    #
    def layout_name(self):
        return "%s%s" % (
            ShortLayoutTypeNames[self.A.layout],
            ShortLayoutTypeNames[self.B.layout],
        )

    #
    def procedural_name(self):
        """ The full procedural name indicates architecture, extended name, tile size, and layout. """
        threadblock = "%dx%d_%d" % (
            self.threadblock_shape[0],
            self.threadblock_shape[1],
            self.threadblock_shape[2],
        )

        opcode_class_name = OpcodeClassNames[self.math_instruction.opcode_class]

        alignment_a = self.A.alignment
        alignment_b = self.B.alignment

        return SubstituteTemplate(
            "cutlass_${opcode_class}_${extended_name}_${threadblock}_${layout}_align${alignment_a}x${alignment_b}",
            {
                "opcode_class": opcode_class_name,
                "extended_name": self.extended_name(),
                "threadblock": threadblock,
                "layout": self.layout_name(),
                "alignment_a": "%d" % alignment_a,
                "alignment_b": "%d" % alignment_b,
            },
        )

    #
    def configuration_name(self):
        """ The full procedural name indicates architecture, extended name, tile size, and layout. """
        return self.procedural_name()


#
def GeneratesGemm(
    tile,
    data_type,
    layout_a,
    layout_b,
    layout_c,
    min_cc,
    align_a=32,
    align_b=32,
    align_c=32,
    required_cuda_ver_major=9,
    required_cuda_ver_minor=2,
):
    operations = []
    swizzling_functor = SwizzlingFunctor.Identity1

    element_a, element_b, element_c, element_epilogue = data_type

    if tile.math_instruction.element_accumulator == DataType.s32:
        epilogues = [EpilogueFunctor.LinearCombinationClamp]
    else:
        assert (
            tile.math_instruction.element_accumulator == DataType.f32
            or tile.math_instruction.element_accumulator == DataType.f16
        )
        epilogues = [EpilogueFunctor.LinearCombination]

    for epilogue in epilogues:
        A = TensorDescription(
            element_a, layout_a, int(align_a // DataTypeSize[element_a])
        )
        B = TensorDescription(
            element_b, layout_b, int(align_b // DataTypeSize[element_b])
        )
        C = TensorDescription(
            element_c, layout_c, int(align_c // DataTypeSize[element_c])
        )
        operations.append(
            GemmOperation(
                GemmKind.Gemm,
                min_cc,
                tile,
                A,
                B,
                C,
                element_epilogue,
                epilogue,
                swizzling_functor,
                required_cuda_ver_major,
                required_cuda_ver_minor,
            )
        )
        operations.append(
            GemmOperation(
                GemmKind.SplitKParallel,
                min_cc,
                tile,
                A,
                B,
                C,
                element_epilogue,
                epilogue,
                swizzling_functor,
                required_cuda_ver_major,
                required_cuda_ver_minor,
            )
        )
    return operations


def GeneratesGemv(
    math_inst,
    threadblock_shape,
    thread_shape,
    data_type,
    layout_a,
    layout_b,
    layout_c,
    min_cc,
    align_a=32,
    align_b=32,
    align_c=32,
    required_cuda_ver_major=9,
    required_cuda_ver_minor=2,
):
    element_a, element_b, element_c, element_epilogue = data_type

    A = TensorDescription(element_a, layout_a, int(align_a // DataTypeSize[element_a]))
    B = TensorDescription(element_b, layout_b, int(align_b // DataTypeSize[element_b]))
    C = TensorDescription(element_c, layout_c, int(align_c // DataTypeSize[element_c]))
    return GemvBatchedStridedOperation(
        GemmKind.GemvBatchedStrided,
        min_cc,
        math_inst,
        threadblock_shape,
        thread_shape,
        A,
        B,
        C,
        required_cuda_ver_major,
        required_cuda_ver_minor,
    )


###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

#
class EmitGemmInstance:
    """ Responsible for emitting a CUTLASS template definition"""

    def __init__(self):
        self.gemm_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::Gemm<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    false,
    ${math_operation}
    ${residual}
  >;
"""
        self.gemm_complex_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::GemmComplex<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${transform_a},
    ${transform_b},
    ${math_operation}
    ${residual}
  >;
"""

    def emit(self, operation):

        warp_shape = [
            operation.tile_description.threadblock_shape[idx]
            // operation.tile_description.warp_count[idx]
            for idx in range(3)
        ]

        epilogue_vector_length = int(
            min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
            / DataTypeSize[operation.C.element]
        )

        residual = ""

        values = {
            "operation_name": operation.procedural_name(),
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[operation.A.layout],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[operation.B.layout],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[operation.C.layout],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            "instruction_shape_n": str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            "instruction_shape_k": str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            "epilogue_vector_length": str(epilogue_vector_length),
            "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
            "epilogue_functor": EpilogueFunctorTag[operation.epilogue_functor],
            "swizzling_functor": SwizzlingFunctorTag[operation.swizzling_functor],
            "stages": str(operation.tile_description.stages),
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
            "transform_a": ComplexTransformTag[operation.A.complex_transform],
            "transform_b": ComplexTransformTag[operation.B.complex_transform],
            "math_operation": MathOperationTag[
                operation.tile_description.math_instruction.math_operation
            ],
            "residual": residual,
        }

        template = (
            self.gemm_complex_template if operation.is_complex() else self.gemm_template
        )

        return SubstituteTemplate(template, values)


#
class EmitGemvBatchedStridedInstance:
    """ Responsible for emitting a CUTLASS template definition"""

    def __init__(self):
        self.template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::kernel::DefaultGemv<
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>, 
    cutlass::gemm::GemmShape<${thread_shape_m}, ${thread_shape_n}, ${thread_shape_k}>, 
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c}
  >;
"""

    def emit(self, operation):

        values = {
            "operation_name": operation.procedural_name(),
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[operation.A.layout],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[operation.B.layout],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[operation.C.layout],
            "threadblock_shape_m": str(operation.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.threadblock_shape[2]),
            "thread_shape_m": str(operation.thread_shape[0]),
            "thread_shape_n": str(operation.thread_shape[1]),
            "thread_shape_k": str(operation.thread_shape[2]),
        }

        return SubstituteTemplate(self.template, values)


###################################################################################################


class EmitSparseGemmInstance:
    """ Responsible for emitting a CUTLASS template definition"""

    def __init__(self):
        self.gemm_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::SparseGemm<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    false,
    ${math_operation}
    ${residual}
  >;
"""

    def emit(self, operation):

        warp_shape = [
            operation.tile_description.threadblock_shape[idx]
            // operation.tile_description.warp_count[idx]
            for idx in range(3)
        ]

        epilogue_vector_length = int(
            min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
            / DataTypeSize[operation.C.element]
        )

        residual = ""

        values = {
            "operation_name": operation.procedural_name(),
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[operation.A.layout],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[operation.B.layout],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[operation.C.layout],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            "instruction_shape_n": str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            "instruction_shape_k": str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            "epilogue_vector_length": str(epilogue_vector_length),
            "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
            "epilogue_functor": EpilogueFunctorTag[operation.epilogue_functor],
            "swizzling_functor": SwizzlingFunctorTag[operation.swizzling_functor],
            "stages": str(operation.tile_description.stages),
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
            "transform_a": ComplexTransformTag[operation.A.complex_transform],
            "transform_b": ComplexTransformTag[operation.B.complex_transform],
            "math_operation": MathOperationTag[
                operation.tile_description.math_instruction.math_operation
            ],
            "residual": residual,
        }

        template = self.gemm_template

        return SubstituteTemplate(template, values)


###################################################################################################


#
class EmitGemmUniversalInstance:
    """ Responsible for emitting a CUTLASS template definition"""

    def __init__(self):
        self.gemm_template = """
// Gemm operator ${operation_name}
using ${operation_name}_base = 
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},    // transposed B operand
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},    // transposed A operand
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name} : 
  public ${operation_name}_base { };
"""
        self.gemm_template_interleaved = """
// Gemm operator ${operation_name}
using ${operation_name}_base = 
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name} : 
  public ${operation_name}_base { };
"""

    def emit(self, operation):

        threadblock_shape = operation.tile_description.threadblock_shape
        warp_count = operation.tile_description.warp_count

        warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

        epilogue_vector_length = int(
            min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
            / DataTypeSize[operation.C.element]
        )

        transpose_layouts = {
            LayoutType.ColumnMajor: LayoutType.RowMajor,
            LayoutType.RowMajor: LayoutType.ColumnMajor,
        }

        if (
            operation.A.layout in transpose_layouts.keys()
            and operation.B.layout in transpose_layouts.keys()
            and operation.C.layout in transpose_layouts.keys()
        ):

            instance_layout_A = transpose_layouts[operation.A.layout]
            instance_layout_B = transpose_layouts[operation.B.layout]
            instance_layout_C = transpose_layouts[operation.C.layout]

            gemm_template = self.gemm_template
        else:
            instance_layout_A, instance_layout_B, instance_layout_C = (
                operation.A.layout,
                operation.B.layout,
                operation.C.layout,
            )

            gemm_template = self.gemm_template_interleaved
        #

        values = {
            "operation_name": operation.procedural_name(),
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[instance_layout_A],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[instance_layout_B],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[instance_layout_C],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            "instruction_shape_n": str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            "instruction_shape_k": str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            "epilogue_vector_length": str(epilogue_vector_length),
            "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
            "epilogue_functor": EpilogueFunctorTag[operation.epilogue_functor],
            "swizzling_functor": SwizzlingFunctorTag[operation.swizzling_functor],
            "stages": str(operation.tile_description.stages),
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
            "transform_a": ComplexTransformTag[operation.A.complex_transform],
            "transform_b": ComplexTransformTag[operation.B.complex_transform],
            "math_operation": MathOperationTag[
                operation.tile_description.math_instruction.math_operation
            ],
        }

        return SubstituteTemplate(gemm_template, values)


###################################################################################################

#
class EmitGemmPlanarComplexInstance:
    """ Responsible for emitting a CUTLASS template definition"""

    def __init__(self):
        self.template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${alignment_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${alignment_b},
    ${element_c}, cutlass::layout::RowMajor,
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    cutlass::epilogue::thread::LinearCombinationPlanarComplex<
      ${element_c},
      ${alignment_c},
      ${element_accumulator},
      ${element_epilogue}
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    ${stages},
    ${math_operator}
  >::GemmKernel;

  struct ${operation_name} : 
    public Operation_${operation_name} { };
"""

    def emit(self, operation):

        warp_shape = [
            operation.tile_description.threadblock_shape[idx]
            // operation.tile_description.warp_count[idx]
            for idx in range(3)
        ]

        # exchange and transpose A and B types, layouts, and complex transforms since the C layout is row-major
        transposed_layout_A = TransposedLayout[operation.A.layout]
        transposed_layout_B = TransposedLayout[operation.B.layout]

        values = {
            "operation_name": operation.procedural_name(),
            "element_a": DataTypeTag[operation.B.element],
            "layout_a": LayoutTag[transposed_layout_B],
            "transform_a": ComplexTransformTag[operation.B.complex_transform],
            "alignment_a": str(operation.B.alignment),
            "element_b": DataTypeTag[operation.A.element],
            "layout_b": LayoutTag[transposed_layout_A],
            "transform_b": ComplexTransformTag[operation.A.complex_transform],
            "alignment_b": str(operation.A.alignment),
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[operation.C.layout],
            "element_accumulator": DataTypeTag[
                operation.tile_description.math_instruction.element_accumulator
            ],
            "opcode_class": OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            "instruction_shape_n": str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            "instruction_shape_k": str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            "alignment_c": str(operation.C.alignment),
            "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
            "stages": str(operation.tile_description.stages),
            "math_operator": "cutlass::arch::OpMultiplyAdd",
        }

        return SubstituteTemplate(self.template, values)


###################################################################################################

#
class EmitGemmPlanarComplexArrayInstance:
    """ Responsible for emitting a CUTLASS template definition"""

    def __init__(self):
        self.template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${alignment_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${alignment_b},
    ${element_c}, cutlass::layout::RowMajor,
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    cutlass::epilogue::thread::LinearCombinationPlanarComplex<
      ${element_c},
      ${alignment_c},
      ${element_accumulator},
      ${element_epilogue}
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    ${stages},
    ${math_operator}
  >::GemmArrayKernel;

  struct ${operation_name} : public Operation_${operation_name} { };
"""

    def emit(self, operation):

        warp_shape = [
            operation.tile_description.threadblock_shape[idx]
            // operation.tile_description.warp_count[idx]
            for idx in range(3)
        ]

        # exchange and transpose A and B types, layouts, and complex transforms since the C layout is row-major
        transposed_layout_A = TransposedLayout[operation.A.layout]
        transposed_layout_B = TransposedLayout[operation.B.layout]

        values = {
            "operation_name": operation.procedural_name(),
            "element_a": DataTypeTag[operation.B.element],
            "layout_a": LayoutTag[transposed_layout_B],
            "transform_a": ComplexTransformTag[operation.B.complex_transform],
            "alignment_a": str(operation.B.alignment),
            "element_b": DataTypeTag[operation.A.element],
            "layout_b": LayoutTag[transposed_layout_A],
            "transform_b": ComplexTransformTag[operation.A.complex_transform],
            "alignment_b": str(operation.A.alignment),
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[operation.C.layout],
            "element_accumulator": DataTypeTag[
                operation.tile_description.math_instruction.element_accumulator
            ],
            "opcode_class": OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            "instruction_shape_n": str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            "instruction_shape_k": str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            "alignment_c": str(operation.C.alignment),
            "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
            "stages": str(operation.tile_description.stages),
            "math_operator": "cutlass::arch::OpMultiplyAdd",
        }

        return SubstituteTemplate(self.template, values)


#
class EmitGemmSplitKParallelInstance:
    """ Responsible for emitting a CUTLASS template definition"""

    def __init__(self):
        self.template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::GemmSplitKParallel<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >, 
    cutlass::epilogue::thread::Convert<
      ${element_accumulator}, 
      ${epilogue_vector_length}, 
      ${element_accumulator}
    >, 
    cutlass::reduction::thread::ReduceAdd<
      ${element_accumulator}, 
      ${element_accumulator}, 
      ${epilogue_vector_length}
    >, 
    cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,
    ${stages}, 
    ${align_a}, 
    ${align_b}, 
    ${math_operation}
  >;
"""

    def emit(self, operation):

        warp_shape = [
            operation.tile_description.threadblock_shape[idx]
            // operation.tile_description.warp_count[idx]
            for idx in range(3)
        ]

        epilogue_vector_length = int(
            min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
            / DataTypeSize[operation.C.element]
        )

        values = {
            "operation_name": operation.procedural_name(),
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[operation.A.layout],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[operation.B.layout],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[operation.C.layout],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            "instruction_shape_n": str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            "instruction_shape_k": str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            "epilogue_vector_length": str(epilogue_vector_length),
            "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
            "epilogue_functor": EpilogueFunctorTag[operation.epilogue_functor],
            "stages": str(operation.tile_description.stages),
            "math_operation": MathOperationTag[
                operation.tile_description.math_instruction.math_operation
            ],
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
        }

        return SubstituteTemplate(self.template, values)


###################################################################################################


###################################################################################################
#
# Emitters functions for all targets
#
###################################################################################################


class EmitGemmConfigurationLibrary:
    def __init__(self, operation_path, configuration_name):
        self.configuration_name = configuration_name
        self.configuration_path = os.path.join(
            operation_path, "%s.cu" % configuration_name
        ).replace("\\", "/")

        self.instance_emitter = {
            GemmKind.Gemm: EmitGemmInstance,
            GemmKind.Sparse: EmitSparseGemmInstance,
            GemmKind.Universal: EmitGemmUniversalInstance,
            GemmKind.PlanarComplex: EmitGemmPlanarComplexInstance,
            GemmKind.PlanarComplexArray: EmitGemmPlanarComplexArrayInstance,
        }

        self.gemm_kind_wrappers = {
            GemmKind.Gemm: "GemmOperation",
            GemmKind.Sparse: "GemmSparseOperation",
            GemmKind.Universal: "GemmUniversalOperation",
            GemmKind.PlanarComplex: "GemmPlanarComplexOperation",
            GemmKind.PlanarComplexArray: "GemmPlanarComplexArrayOperation",
        }

        self.wmma_guard_start = "#if defined(CUTLASS_ARCH_WMMA_SM${sm_number}_ENABLED)"

        self.instance_template = {
            GemmKind.Gemm: """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<Operation_${operation_name}>("${operation_name}"));
${compile_guard_end}
""",
            GemmKind.Sparse: """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<Operation_${operation_name}>("${operation_name}"));
${compile_guard_end}
""",
            GemmKind.Universal: """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
      cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
    >("${operation_name}"));
${compile_guard_end}
""",
            GemmKind.PlanarComplex: """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
    cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
  >("${operation_name}"));
${compile_guard_end}
""",
            GemmKind.PlanarComplexArray: """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
    cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
  >("${operation_name}"));
${compile_guard_end}
""",
        }

        self.header_template = """
/*
  Generated by gemm_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "gemm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

        self.initialize_function_template = """

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_${configuration_name}(Manifest &manifest) {

"""
        self.epilogue_template = """

}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace library
}  // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

    def __enter__(self):
        self.configuration_file = open(self.configuration_path, "w")
        self.configuration_file.write(self.header_template)

        self.instance_definitions = []
        self.instance_wrappers = []

        self.operations = []
        return self

    def emit(self, operation):
        emitter = self.instance_emitter[operation.gemm_kind]()

        self.operations.append(operation)

        self.instance_definitions.append(emitter.emit(operation))

        self.instance_wrappers.append(
            SubstituteTemplate(
                self.instance_template[operation.gemm_kind],
                {
                    "configuration_name": self.configuration_name,
                    "operation_name": operation.procedural_name(),
                    "gemm_kind": self.gemm_kind_wrappers[operation.gemm_kind],
                    "compile_guard_start": SubstituteTemplate(
                        self.wmma_guard_start, {"sm_number": str(operation.arch)}
                    )
                    if operation.tile_description.math_instruction.opcode_class
                    == OpcodeClass.WmmaTensorOp
                    else "",
                    "compile_guard_end": "#endif"
                    if operation.tile_description.math_instruction.opcode_class
                    == OpcodeClass.WmmaTensorOp
                    else "",
                },
            )
        )

    def __exit__(self, exception_type, exception_value, traceback):

        # Write instance definitions in top-level namespace
        for instance_definition in self.instance_definitions:
            self.configuration_file.write(instance_definition)

        # Add wrapper objects within initialize() function
        self.configuration_file.write(
            SubstituteTemplate(
                self.initialize_function_template,
                {"configuration_name": self.configuration_name},
            )
        )

        for instance_wrapper in self.instance_wrappers:
            self.configuration_file.write(instance_wrapper)

        self.configuration_file.write(self.epilogue_template)
        self.configuration_file.close()


###################################################################################################
###################################################################################################


class EmitGemmSingleKernelWrapper:
    def __init__(self, kernel_path, gemm_operation, short_path=False):
        self.short_path = short_path
        self.kernel_path = kernel_path
        self.operation = gemm_operation

        instance_emitters = {
            GemmKind.Gemm: EmitGemmInstance(),
            GemmKind.SplitKParallel: EmitGemmSplitKParallelInstance(),
        }
        self.instance_emitter = instance_emitters[self.operation.gemm_kind]

        self.header_template = """
#if __CUDACC_VER_MAJOR__ > ${required_cuda_ver_major} || (__CUDACC_VER_MAJOR__ == ${required_cuda_ver_major} && __CUDACC_VER_MINOR__ >= ${required_cuda_ver_minor})                 
// ignore warning of cutlass
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"

#include "src/cuda/cutlass/manifest.h"
#include "src/cuda/cutlass/gemm_operation.h"
"""
        self.instance_template = """
${operation_instance}
"""

        self.manifest_template = """
namespace cutlass {
namespace library {

void initialize_${operation_name}(Manifest &manifest) {
  manifest.append(new GemmOperation<
      Operation_${operation_name}
    >("${operation_name}"));
}

}  // namespace library
}  // namespace cutlass
"""

        self.epilogue_template = """
#pragma GCC diagnostic pop
#endif
"""

    #
    def __enter__(self):
        if self.short_path:
            self.kernel_path = os.path.join(self.kernel_path, "%s.cu" % GlobalCnt.cnt)
            GlobalCnt.cnt += 1
        else:
            self.kernel_path = os.path.join(
                self.kernel_path, "%s.cu" % self.operation.procedural_name()
            )
        self.kernel_file = open(self.kernel_path, "w")
        return self

    #
    def emit(self):
        self.kernel_file.write(
            SubstituteTemplate(
                self.instance_template,
                {"operation_instance": self.instance_emitter.emit(self.operation)},
            )
        )

        # emit manifest helper
        manifest = SubstituteTemplate(
            self.manifest_template, {"operation_name": self.operation.procedural_name()}
        )
        self.kernel_file.write(manifest)

    #
    def __exit__(self, exception_type, exception_value, traceback):
        self.kernel_file.close()


###################################################################################################
###################################################################################################


class EmitGemvSingleKernelWrapper:
    def __init__(self, kernel_path, gemm_operation, wrapper_path, short_path=False):
        self.kernel_path = kernel_path
        self.wrapper_path = wrapper_path
        self.operation = gemm_operation
        self.short_path = short_path

        self.wrapper_template = """
template void megdnn::cuda::cutlass_wrapper::
  cutlass_vector_matrix_mul_batched_strided_wrapper<Operation_${operation_name}>(
      BatchedGemmCoord const& problem_size,
      const typename Operation_${operation_name}::ElementA* d_A, size_t lda, size_t batch_stride_a,
      const typename Operation_${operation_name}::ElementB* d_B, size_t ldb, size_t batch_stride_b,
      typename Operation_${operation_name}::ElementCD* d_C, size_t ldc, size_t batch_stride_c,
      cudaStream_t stream);
"""

        self.instance_emitter = EmitGemvBatchedStridedInstance()

        self.header_template = """
#if __CUDACC_VER_MAJOR__ > ${required_cuda_ver_major} || (__CUDACC_VER_MAJOR__ == ${required_cuda_ver_major} && __CUDACC_VER_MINOR__ >= ${required_cuda_ver_minor})
// ignore warning of cutlass
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "${wrapper_path}"
"""
        self.instance_template = """
${operation_instance}
"""

        self.epilogue_template = """
#pragma GCC diagnostic pop
#endif
"""

    #
    def __enter__(self):
        if self.short_path:
            self.kernel_path = os.path.join(self.kernel_path, "%s.cu" % GlobalCnt.cnt)
            GlobalCnt.cnt += 1
        else:
            self.kernel_path = os.path.join(
                self.kernel_path, "%s.cu" % self.operation.procedural_name()
            )
        self.kernel_file = open(self.kernel_path, "w")
        return self

    #
    def emit(self):
        self.kernel_file.write(
            SubstituteTemplate(
                self.instance_template,
                {"operation_instance": self.instance_emitter.emit(self.operation)},
            )
        )

        # emit wrapper
        wrapper = SubstituteTemplate(
            self.wrapper_template, {"operation_name": self.operation.procedural_name()}
        )
        self.kernel_file.write(wrapper)

    #
    def __exit__(self, exception_type, exception_value, traceback):
        self.kernel_file.close()


###################################################################################################
###################################################################################################
