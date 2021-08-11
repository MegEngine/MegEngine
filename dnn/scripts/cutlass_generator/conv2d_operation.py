#
# \file generator.py
#
# \brief Generates the CUTLASS Library's instances
#
#

import enum
import os.path
import shutil
from typing import Tuple, List

from library import *

###################################################################################################

#
class Conv2dOperation:
  #
  def __init__(self, conv_kind, conv_type, arch, tile_description, src, flt, bias, dst, element_epilogue, \
    epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4, \
    special_optimization = SpecialOptimizeDesc.NoneSpecialOpt, implicit_gemm_mode = ImplicitGemmMode.GemmNT, \
    without_shared_load = False, required_cuda_ver_major = 9, required_cuda_ver_minor = 2):

    self.operation_kind = OperationKind.Conv2d
    self.conv_kind = conv_kind
    self.arch = arch
    self.tile_description = tile_description
    self.conv_type = conv_type
    self.src = src
    self.flt = flt
    self.bias = bias
    self.dst = dst
    self.element_epilogue = element_epilogue
    self.epilogue_functor = epilogue_functor
    self.swizzling_functor = swizzling_functor
    self.special_optimization = special_optimization  
    self.implicit_gemm_mode = implicit_gemm_mode
    self.without_shared_load = without_shared_load
    self.required_cuda_ver_major = required_cuda_ver_major
    self.required_cuda_ver_minor = required_cuda_ver_minor

  #
  def accumulator_type(self):
    accum = self.tile_description.math_instruction.element_accumulator

    return accum

  #
  def core_name(self):
    ''' The basic operation kind is prefixed with a letter indicating the accumulation type. '''

    intermediate_type = ''

    if self.tile_description.math_instruction.opcode_class == OpcodeClass.TensorOp:
      inst_shape = "%d%d%d" % tuple(self.tile_description.math_instruction.instruction_shape)
      if self.tile_description.math_instruction.element_a != self.flt.element and \
        self.tile_description.math_instruction.element_a != self.accumulator_type():
        intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]
    else:
      inst_shape = ''

    special_opt = ''
    if self.special_optimization == SpecialOptimizeDesc.ConvFilterUnity:
      special_opt = '_1x1'
    elif self.special_optimization == SpecialOptimizeDesc.DeconvDoubleUpsampling:
       special_opt = '_s2'

    reorder_k = ''
    if self.without_shared_load:
      reorder_k = '_roc'

    return "%s%s%s%s%s%s_%s" % (ShortDataTypeNames[self.accumulator_type()], \
      inst_shape, intermediate_type, ConvKindNames[self.conv_kind], special_opt, \
      reorder_k, ShortEpilogueNames[self.epilogue_functor])

  #
  def extended_name(self):
    if self.dst.element != self.tile_description.math_instruction.element_accumulator:
      if self.src.element != self.flt.element:
        extended_name = "${element_dst}_${core_name}_${element_src}_${element_flt}"
      elif self.src.element == self.flt.element:
        extended_name = "${element_dst}_${core_name}_${element_src}"
    else:
      if self.src.element != self.flt.element:
        extended_name = "${core_name}_${element_src}_${element_flt}"
      elif self.src.element == self.flt.element:
        extended_name = "${core_name}_${element_src}"
 
    extended_name = SubstituteTemplate(extended_name, {
      'element_src': DataTypeNames[self.src.element], 
      'element_flt': DataTypeNames[self.flt.element], 
      'element_dst': DataTypeNames[self.dst.element],
      'core_name': self.core_name()
      })

    return extended_name

  #
  def layout_name(self):
    if self.src.layout == self.dst.layout:
      layout_name = "${src_layout}_${flt_layout}"
    else:
      layout_name = "${src_layout}_${flt_layout}_${dst_layout}"

    layout_name = SubstituteTemplate(layout_name, {
      'src_layout': ShortLayoutTypeNames[self.src.layout], 
      'flt_layout': ShortLayoutTypeNames[self.flt.layout], 
      'dst_layout': ShortLayoutTypeNames[self.dst.layout], 
    })
    
    return layout_name

#
  def configuration_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''

    opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]
    
    warp_shape = [int(self.tile_description.threadblock_shape[idx] / self.tile_description.warp_count[idx]) for idx in range(3)]


    threadblock = "%dx%dx%d_%dx%dx%d_%d" % (
      self.tile_description.threadblock_shape[0],
      self.tile_description.threadblock_shape[1],
      self.tile_description.threadblock_shape[2],
      warp_shape[0], 
      warp_shape[1], 
      warp_shape[2], 
      self.tile_description.stages, 
    )

    configuration_name = "cutlass_${opcode_class}_${extended_name}_${threadblock}_${layout}"

    return SubstituteTemplate(
      configuration_name,
      {
        'opcode_class': opcode_class_name,
        'extended_name': self.extended_name(),
        'threadblock': threadblock,
        'layout': self.layout_name(),
      }
    )

  #
  def procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    return self.configuration_name()

###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

class EmitConv2dInstance:
  def __init__(self):
    self.template = """
// kernel instance "${operation_name}" generated by cutlass generator
using Convolution = 
  typename cutlass::conv::device::Convolution<
    ${element_src}, 
    ${layout_src},
    ${element_flt}, 
    ${layout_flt},
    ${element_dst}, 
    ${layout_dst},
    ${element_bias}, 
    ${layout_bias}, 
    ${element_accumulator}, 
    ${conv_type},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_dst},
      ${epilogue_vector_length},
      ${element_accumulator}, 
      ${element_bias}, 
      ${element_epilogue}
    >,
    ${swizzling_functor},     
    ${stages},
    ${alignment_src}, 
    ${alignment_filter}, 
    ${special_optimization}, 
    ${math_operator},
    ${implicit_gemm_mode}, 
    ${without_shared_load}>;
"""


  def emit(self, operation):

    warp_shape = [int(operation.tile_description.threadblock_shape[idx] / operation.tile_description.warp_count[idx]) for idx in range(3)]

    epilogue_vector_length = int(min(operation.dst.alignment * DataTypeSize[operation.dst.element], 128) / DataTypeSize[operation.dst.element])

    values = {
      'operation_name': operation.procedural_name(),
      'conv_type': ConvTypeTag[operation.conv_type], 
      'element_src': DataTypeTag[operation.src.element],
      'layout_src': LayoutTag[operation.src.layout],
      'element_flt': DataTypeTag[operation.flt.element],
      'layout_flt': LayoutTag[operation.flt.layout],
      'element_dst': DataTypeTag[operation.dst.element],
      'layout_dst': LayoutTag[operation.dst.layout],
      'element_bias': DataTypeTag[operation.bias.element],
      'layout_bias': LayoutTag[operation.bias.layout],
      'element_accumulator': DataTypeTag[operation.accumulator_type()], 
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_vector_length': str(epilogue_vector_length),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'alignment_src': str(operation.src.alignment), 
      'alignment_filter': str(operation.flt.alignment), 
      'special_optimization': SpecialOptimizeDescTag[operation.special_optimization],  
      'math_operator': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'implicit_gemm_mode': ImplicitGemmModeTag[operation.implicit_gemm_mode],
      'without_shared_load': str(operation.without_shared_load).lower()
    }

    return SubstituteTemplate(self.template, values)

class EmitDeconvInstance:
  def __init__(self):
    self.template = """
// kernel instance "${operation_name}" generated by cutlass generator
using Deconvolution = 
  typename cutlass::conv::device::Deconvolution<
    ${element_src}, 
    ${layout_src},
    ${element_flt}, 
    ${layout_flt},
    ${element_dst}, 
    ${layout_dst},
    ${element_bias}, 
    ${layout_bias}, 
    ${element_accumulator}, 
    ${conv_type},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_dst},
      ${epilogue_vector_length},
      ${element_accumulator}, 
      ${element_bias}, 
      ${element_epilogue}
    >,
    ${swizzling_functor},     
    ${stages},
    ${alignment_src}, 
    ${alignment_filter}, 
    ${special_optimization}, 
    ${math_operator},
    ${implicit_gemm_mode}>;
"""


  def emit(self, operation):

    warp_shape = [int(operation.tile_description.threadblock_shape[idx] / operation.tile_description.warp_count[idx]) for idx in range(3)]

    epilogue_vector_length = int(min(operation.dst.alignment * DataTypeSize[operation.dst.element], 128) / DataTypeSize[operation.dst.element])

    values = {
      'operation_name': operation.procedural_name(),
      'conv_type': ConvTypeTag[operation.conv_type],
      'element_src': DataTypeTag[operation.src.element],
      'layout_src': LayoutTag[operation.src.layout],
      'element_flt': DataTypeTag[operation.flt.element],
      'layout_flt': LayoutTag[operation.flt.layout],
      'element_dst': DataTypeTag[operation.dst.element],
      'layout_dst': LayoutTag[operation.dst.layout],
      'element_bias': DataTypeTag[operation.bias.element],
      'layout_bias': LayoutTag[operation.bias.layout],
      'element_accumulator': DataTypeTag[operation.accumulator_type()], 
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_vector_length': str(epilogue_vector_length),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'alignment_src': str(operation.src.alignment), 
      'alignment_filter': str(operation.flt.alignment), 
      'special_optimization': SpecialOptimizeDescTag[operation.special_optimization],
      'math_operator': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'implicit_gemm_mode': ImplicitGemmModeTag[operation.implicit_gemm_mode]
    }

    return SubstituteTemplate(self.template, values)


###################################################################################################
#
# Generator functions for all layouts
#
###################################################################################################

#
def GenerateConv2d(conv_kind, tile_descriptions, src_layout, flt_layout, dst_layout, dst_type, min_cc, src_align = 32, flt_align = 32, dst_align = 32, \
  use_special_optimization = SpecialOptimizeDesc.NoneSpecialOpt, implicit_gemm_mode = ImplicitGemmMode.GemmNT, without_shared_load = False, \
  required_cuda_ver_major = 9, required_cuda_ver_minor = 2):
  operations = []

  element_epilogue = DataType.f32 
  if conv_kind == ConvKind.Fprop:
    if implicit_gemm_mode == ImplicitGemmMode.GemmTN:
      swizzling_functor = SwizzlingFunctor.ConvFpropTrans
    else:
      swizzling_functor = SwizzlingFunctor.ConvFpropNCxHWx
  else:
    swizzling_functor = SwizzlingFunctor.ConvDgradNCxHWx

  # skip rule
  def filter_tile_with_layout(tile: TileDescription, layout: LayoutType) -> bool:
    return layout == LayoutType.TensorNC32HW32 and \
           tile.threadblock_shape[0] % 32 != 0
  
  # rule for bias_type and epilogues
  def get_bias_type_and_epilogues(tile: TileDescription, \
                                  out_dtype: DataType) -> Tuple[DataType, List[EpilogueFunctor]]:
    if tile.math_instruction.element_accumulator == DataType.s32 and \
        out_dtype != DataType.f32:
      bias_type = DataType.s32
      if tile.math_instruction.element_b == DataType.u4:
        epilogues = [EpilogueFunctor.BiasAddLinearCombinationClamp, EpilogueFunctor.BiasAddLinearCombinationReluClamp]
      else:
        epilogues = [EpilogueFunctor.BiasAddLinearCombinationClamp, EpilogueFunctor.BiasAddLinearCombinationReluClamp, \
                     EpilogueFunctor.BiasAddLinearCombinationHSwishClamp]
    elif tile.math_instruction.element_accumulator == DataType.f32 or \
        out_dtype == DataType.f32:
      bias_type = DataType.f32
      epilogues = [EpilogueFunctor.BiasAddLinearCombination, EpilogueFunctor.BiasAddLinearCombinationRelu, \
                   EpilogueFunctor.BiasAddLinearCombinationHSwish]
    return bias_type, epilogues

  # rule for filter alignment
  def get_flt_align(tile: TileDescription) -> int:
    nonlocal flt_align
    if tile.math_instruction.opcode_class == OpcodeClass.Simt \
            and tile.math_instruction.element_accumulator == DataType.s32:
      thread_num = tile.warp_count[0] * tile.warp_count[1] * tile.warp_count[2] * 32
      flt_block = tile.threadblock_shape[0] * tile.threadblock_shape[2] \
                                            * DataTypeSize[tile.math_instruction.element_a]
      load_per_thread = flt_block//thread_num 
      if load_per_thread >= 128:
        flt_align = 128
      elif load_per_thread >= 64:
        flt_align = 64
      else:
        assert load_per_thread >= 32
        flt_align = 32
    return flt_align

  def get_dst_align(tile: TileDescription, out_layout: LayoutType) -> int:
    nonlocal dst_align
    if tile.math_instruction.opcode_class == OpcodeClass.TensorOp \
      and dst_layout == LayoutType.TensorNC4HW4:
      dst_align = 32
    return dst_align

  def filter_epilogue_with_conv_kind(epilogue: EpilogueFunctor, conv_kind: ConvKind) -> bool:
    return conv_kind == ConvKind.Dgrad \
        and epilogue != EpilogueFunctor.BiasAddLinearCombinationClamp

  # loop over all tile descriptions
  for tile in tile_descriptions:
    if filter_tile_with_layout(tile, dst_layout):
      continue

    bias_type, epilogues = get_bias_type_and_epilogues(tile, dst_type)

    flt_align = get_flt_align(tile)

    dst_align = get_dst_align(tile, dst_layout)

    for epilogue in epilogues:
      if filter_epilogue_with_conv_kind(epilogue, conv_kind):
        continue

      if dst_type == DataType.f32:
        bias_type = DataType.f32
      #
      src = TensorDescription(tile.math_instruction.element_b, src_layout, int(src_align / DataTypeSize[tile.math_instruction.element_b]))
      flt = TensorDescription(tile.math_instruction.element_a, flt_layout, int(flt_align / DataTypeSize[tile.math_instruction.element_a]))
      bias = TensorDescription(bias_type, dst_layout, max(1, int(32 / DataTypeSize[bias_type])))
      dst = TensorDescription(dst_type, dst_layout, int(dst_align / DataTypeSize[dst_type])) 

      new_operation = Conv2dOperation(conv_kind, ConvType.Convolution, min_cc, tile, src, flt, bias, dst, element_epilogue, epilogue, swizzling_functor, SpecialOptimizeDesc.NoneSpecialOpt, implicit_gemm_mode, without_shared_load, required_cuda_ver_major, required_cuda_ver_minor)
      operations.append(new_operation)
      if use_special_optimization != SpecialOptimizeDesc.NoneSpecialOpt:
        new_operation = Conv2dOperation(conv_kind, ConvType.Convolution, min_cc, tile, src, flt, bias, dst, element_epilogue, epilogue, swizzling_functor, use_special_optimization , implicit_gemm_mode, without_shared_load, required_cuda_ver_major, required_cuda_ver_minor)
        operations.append(new_operation)
  return operations

###################################################################################################
#
# Emitters functions for all targets
#
###################################################################################################

class EmitConv2dConfigurationLibrary:
  def __init__(self, operation_path, configuration_name):
    self.configuration_name = configuration_name
    self.configuration_path = os.path.join(operation_path, "%s.cu" % configuration_name)

    self.instance_emitter = EmitConv2dInstance()

    self.instance_template = """
${operation_instance}

// Derived class
struct ${operation_name} : 
  public ${operation_name}_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////

"""
    self.header_template = """
/*
  Generated by conv2d_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
"""

    self.configuration_header = """

namespace cutlass {
namespace library {

// Initialize all instances
void initialize_${configuration_name}(Manifest &manifest) {

"""

    self.configuration_instance = """
  using Operation_${operation_name} = cutlass::conv::device::ImplicitGemmConvolution<
    ${operation_name}>;

  manifest.append(new cutlass::library::Conv2dOperation<
    Operation_${operation_name}>(
      "${operation_name}"));

"""

    self.configuration_epilogue = """
}
"""
    self.epilogue_template = """

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

  #
  def __enter__(self):
    self.configuration_file = open(self.configuration_path, "w")
    self.configuration_file.write(SubstituteTemplate(self.header_template, {
      'configuration_name': self.configuration_name
      }))
    self.operations = []
    return self

  #
  def emit(self, operation):
    self.operations.append(operation)
    self.configuration_file.write(SubstituteTemplate(self.instance_template, {
      'configuration_name': self.configuration_name,
      'operation_name': operation.procedural_name(),
      'operation_instance': self.instance_emitter.emit(operation)
      }))

  #
  def __exit__(self, exception_type, exception_value, traceback):

    self.configuration_file.write(SubstituteTemplate(self.configuration_header, {
      'configuration_name': self.configuration_name
      }))

    for operation in self.operations:
      self.configuration_file.write(SubstituteTemplate(self.configuration_instance, {
        'configuration_name': self.configuration_name,
        'operation_name': operation.procedural_name()  
      }))

    self.configuration_file.write(self.configuration_epilogue)
    self.configuration_file.write(self.epilogue_template)
    self.configuration_file.close()

###################################################################################################
###################################################################################################

# Emitters for Conv Kernel Wrapper
#
###################################################################################################

class EmitConvSingleKernelWrapper():
  def __init__(self, kernel_path, operation, short_path=False):
    self.kernel_path = kernel_path
    self.operation = operation
    self.short_path = short_path

    if self.operation.conv_kind == ConvKind.Fprop:
      self.instance_emitter = EmitConv2dInstance()
      self.convolution_name = "Convolution"
    else:
      assert self.operation.conv_kind == ConvKind.Dgrad
      self.instance_emitter = EmitDeconvInstance()
      self.convolution_name = "Deconvolution"

    self.header_template = """
#if __CUDACC_VER_MAJOR__ > ${required_cuda_ver_major} || (__CUDACC_VER_MAJOR__ == ${required_cuda_ver_major} && __CUDACC_VER_MINOR__ >= ${required_cuda_ver_minor})
// ignore warning of cutlass
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include "cutlass/convolution/device/convolution.h"

#include "src/cuda/cutlass/manifest.h"
#include "src/cuda/cutlass/convolution_operation.h"
"""
    self.instance_template = """
${operation_instance}
"""

    self.manifest_template = """
namespace cutlass {
namespace library {

void initialize_${operation_name}(Manifest &manifest) {
  manifest.append(new ConvolutionOperation<${convolution_name}>(
    "${operation_name}"
  ));
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
      self.kernel_path = os.path.join(self.kernel_path, "%s.cu" % self.operation.procedural_name())
    self.kernel_file = open(self.kernel_path, "w")
    self.kernel_file.write(SubstituteTemplate(self.header_template, {
      'required_cuda_ver_major': str(self.operation.required_cuda_ver_major),
      'required_cuda_ver_minor': str(self.operation.required_cuda_ver_minor),
    }))
    return self

  #
  def emit(self):
    self.kernel_file.write(SubstituteTemplate(self.instance_template, {
      'operation_instance': self.instance_emitter.emit(self.operation),
    }))

    # emit manifest helper
    manifest = SubstituteTemplate(self.manifest_template, {
      'operation_name': self.operation.procedural_name(),
      'convolution_name': self.convolution_name
    })
    self.kernel_file.write(manifest)

  #
  def __exit__(self, exception_type, exception_value, traceback):
    self.kernel_file.write(self.epilogue_template)
    self.kernel_file.close()


###################################################################################################
###################################################################################################

