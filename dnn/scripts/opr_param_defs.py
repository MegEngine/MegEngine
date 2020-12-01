pdef('Empty')

pdef('Axis').add_fields('int32', 'axis', 0)

(pdef('Convolution', version=0, is_legacy=True).
 add_enum('Mode', 'CROSS_CORRELATION', 'CONVOLUTION').
 add_fields(
     'uint32',
     Doc('pad_h', 'padding on one side on the first dimension'), 0,
     Doc('pad_w', 'padding on one side on the second dimension'), 0,
     Doc('stride_h', 'kernel stride on the first dimension'), 1,
     Doc('stride_w', 'kernel stride on the second dimension'), 1,
     Doc('dilate_h', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1,
     Doc('dilate_w', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1
 ).
 add_enum('DataType',
          Doc('FLOAT', 'input/output both float32/float16'),
          'INT8x8x16',
          'INT8x8x32',
          Doc('FLOAT_IO16xC32', 'input/output both float16, the internal '
              'compute is float32'),
          Doc('QUINT8x8x32', 'input QuantizedAsymm8, output QuantizedS32'),
          Doc('INT8x8xX', 'input int8, output specified by tensor DType'),
          Doc('QUINT4x4x32', 'input QuantizedAsymm4, output QuantizedS32'),
          name_field='data_type').
 add_enum('Sparse',
          Doc('DENSE', 'dense convolution: filter shape should be '
              '[oc, ic, spatial...] if format is NCHW, '
              '[oc, spatial..., ic] if format is NHWC'),
          Doc('GROUP', 'group convolution: filter shape should be '
              '[group, oc_per_group, ic_per_group, spatial...] if format is NCHW, '
              '[group, oc_per_group, spatial..., ic_per_group] if format is NHWC')
          ).
 add_enum(Doc('Format', 'convolution data/filter/output format; see '
              ':class:`RelayoutFormat` for more details'),
          'NCHW', 'NHWC', 'NHWCD4', 'NCHW4', 'NCHW8', 'NCHW32', 'NCHW88',
          'NCHW44','NCHW44_DOT',
          Doc('NCHW_WINOGRAD', 'NCHW layout with weights tranformed by winograd'),
          Doc('NCHW88_WINOGRAD', 'NCHW88 layout with weights tranformed by winograd'),
          Doc('NCHW44_WINOGRAD', 'NCHW44 layout with weights tranformed by winograd'), 
          Doc('NCHW4_NCHW32', 'NCHW4_NCHW32 means input tensors are nchw4 layout, output tensor is nchw32 layout'), 
          Doc('NCHW32_NCHW4', 'NCHW32_NCHW4 means input tensors are nchw32 layout, output tensor is nchw4 layout'), 
          Doc('NCHW4_NCHW', 'NCHW4_NCHW means input tensors are nchw4 layout, output tensor is nchw layout'), 
          Doc('CHWN4', 'CHWN4 is currently only used on Nvidia platform for fast implementation '
              'of convolution using CUDA/SASS. The channels are splitted to groups of 4 channels.'))
 )

(pdef('Convolution', version=1).
 add_enum_alias('Mode', 'ConvolutionV0').
 add_fields(
     'uint32',
     Doc('pad_h', 'padding on one side on the first dimension'), 0,
     Doc('pad_w', 'padding on one side on the second dimension'), 0,
     Doc('stride_h', 'kernel stride on the first dimension'), 1,
     Doc('stride_w', 'kernel stride on the second dimension'), 1,
     Doc('dilate_h', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1,
     Doc('dilate_w', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1
 ).
 add_enum_alias('Sparse', 'ConvolutionV0').
 add_enum_alias('Format', 'ConvolutionV0').
 add_enum(Doc('ComputeMode', 'Specifies special computation modes, e.g. '
                             'different combinations of intermediate result '
                             'data types.'),
          Doc('DEFAULT', 'No special requirements on the precision of '
                         'intermediate results.'),
          Doc('FLOAT32', 'Use Float32 accumulator and intermediate result. '
                         'Only supported when input and output is Float16.'),
          name_field='compute_mode')
 )

(pdef('MaskPropagate').
 add_fields(
     'uint32',
     Doc('pad_h', 'padding on one side on the first dimension'), 0,
     Doc('pad_w', 'padding on one side on the second dimension'), 0,
     Doc('stride_h', 'kernel stride on the first dimension'), 1,
     Doc('stride_w', 'kernel stride on the second dimension'), 1,
     Doc('kernel_h', 'kernel height'), 1,
     Doc('kernel_w', 'kernel width'), 1,
     Doc('dilate_h', 'dilate height'), 1,
     Doc('dilate_w', 'dilate width'), 1)
 )

(pdef('ConvPooling').
 add_enum('Method', 'WITH_TEXTURE_OBJ', 'WITH_SHARED_MEM').
 add_enum_alias('ConvMode', 'ConvolutionV0', 'Mode').
 add_enum('PoolMode', 'AVERAGE', 'MAX').
 add_enum('NonlineMode', 'IDENTITY', 'RELU', 'SIGMOID').
 add_fields('uint32', 'pool_shape_h', 1, 'pool_shape_w', 1, 'pool_stride_h', 1, 'pool_stride_w', 1, \
  'pool_pad_h', 0, 'pool_pad_w', 0, 'conv_stride_h', 1, 'conv_stride_w', 1, 'conv_pad_h', 0, 'conv_pad_w', 0))

(pdef('ConvBias', 'legacy conv_bias', version=0, is_legacy=True).
 add_enum('NonlineMode', 'IDENTITY', 'RELU', 'SIGMOID', 'H_SWISH').
 add_enum_alias('Mode', 'ConvolutionV0').
 add_fields('uint32', 'pad_h', 0, 'pad_w', 0, 'stride_h', 1, 'stride_w', 1))

(pdef('ConvBias', 'active(conv(x, w) + bias)', version=1, is_legacy=True).
 add_enum_alias('NonlineMode', 'ConvBiasV0').
 add_enum_alias('Mode', 'ConvolutionV0').
 add_enum_alias('DataType', 'ConvolutionV0', name_field='data_type').
 add_enum_alias('Sparse', 'ConvolutionV0').
 add_enum_alias('Format', 'ConvolutionV0').
 add_fields(
     'uint32',
     Doc('pad_h', 'padding on one side on the first dimension'), 0,
     Doc('pad_w', 'padding on one side on the second dimension'), 0,
     Doc('stride_h', 'kernel stride on the first dimension'), 1,
     Doc('stride_w', 'kernel stride on the second dimension'), 1,
     Doc('dilate_h', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1,
     Doc('dilate_w', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1)
 )

(pdef('ConvBias', 'active(conv(x, w) + bias)', version=2, is_legacy=True).
 add_enum_alias('NonlineMode', 'ConvBiasV0').
 add_enum_alias('Mode', 'ConvolutionV0').
 add_enum_alias('Sparse', 'ConvolutionV0').
 add_enum_alias('Format', 'ConvolutionV0').
 add_fields(
     'uint32',
     Doc('pad_h', 'padding on one side on the first dimension'), 0,
     Doc('pad_w', 'padding on one side on the second dimension'), 0,
     Doc('stride_h', 'kernel stride on the first dimension'), 1,
     Doc('stride_w', 'kernel stride on the second dimension'), 1,
     Doc('dilate_h', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1,
     Doc('dilate_w', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1).
 add_enum_alias('ComputeMode', 'Convolution', name_field='compute_mode')
 )

(pdef('ConvBias', 'active(conv(x, w) + bias)', version=3).
 add_enum_alias('NonlineMode', 'ConvBiasV0').
 add_enum_alias('Mode', 'ConvolutionV0').
 add_enum_alias('Sparse', 'ConvolutionV0').
 add_enum_alias('Format', 'ConvolutionV0').
 add_fields(
     'uint32',
     Doc('pad_h', 'padding on one side on the first dimension'), 0,
     Doc('pad_w', 'padding on one side on the second dimension'), 0,
     Doc('stride_h', 'kernel stride on the first dimension'), 1,
     Doc('stride_w', 'kernel stride on the second dimension'), 1,
     Doc('dilate_h', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1,
     Doc('dilate_w', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1,
     Doc('output_block_size', 'detail meaning \see winograd in conv bias'), 0).
 add_enum_alias('ComputeMode', 'Convolution', name_field='compute_mode')
 )

(pdef('SeparableConv').
 add_enum_alias('Mode', 'ConvolutionV0').
 add_enum('BorderMode', 'BORDER_REPLICATE', 'BORDER_REFLECT',
          'BORDER_REFLECT_101','BORDER_WRAP',
          'BORDER_CONSTANT', 'BORDER_TRANSPARENT','BORDER_ISOLATED').
 add_fields('bool', 'is_symm_kernel', 'true').
 add_fields('uint32', 'pad_h', 0, 'pad_w', 0, 'stride_h', 1, 'stride_w', 1,
            'ksize_h', 3, 'ksize_w', 3, 'anchor_h', 1, 'anchor_w', 1))

(pdef('Images2Neibs').
 add_fields('uint32', 'pad_h', 0, 'pad_w', 0, 'stride_h', 1, 'stride_w', 1,
            'window_h', 3, 'window_w', 3))

(pdef('Pooling').
 add_enum(
     'Mode',
     Doc('MAX', 'maximum value inside pooling window'),
     Doc('AVERAGE',
         'arithmetic mean of all values inside pooling window. Padding values '
         'are taken into account and are viewed as zero'),
     Doc('AVERAGE_COUNT_EXCLUDE_PADDING',
         'arithmetic mean of all values inside pooling window. No padding is'
         'used.')
 ).
 add_fields('uint32', 'pad_h', 0, 'pad_w', 0, 'stride_h', 2, 'stride_w', 2,
            'window_h', 2, 'window_w', 2).
 add_enum_alias('Format', 'ConvolutionV0')
 )

(pdef('AdaptivePooling').
 add_enum_alias('Mode', 'Pooling').
 add_enum_alias('Format', 'ConvolutionV0')
 )

(pdef('LRN',
      'see ImageNet Classification with Deep Convolutional Neural Networks for'
      ' meaning of the fields').
 add_fields('uint32', Doc('n', 'must be odd'), 5).
 add_fields('float32', 'k', '2.f', 'alpha', '1e-4f', 'beta', '0.75f')
)

(pdef('BN').
 add_enum(
     'ParamDim',
     Doc('DIM_11HW', 'Dim of params (Sigma, Mu) is 1 x 1 x H x W'),
     Doc('DIM_1CHW', 'Dim of params (Sigma, Mu) is 1 x C x H x W'),
     Doc('DIM_1C11', 'Dim of params (Sigma, Mu) is 1 x C x 1 x 1'),
     name_field='param_dim'
 ).
 add_enum(
     'FwdMode',
     Doc('TRAINING', 'Training phase.'),
     Doc('INFERENCE', 'Inference phase.'),
     name_field='fwd_mode'
 ).
 add_fields('float64', 'epsilon', '1e-4f').
 add_fields('float64', 'avg_factor', '1.f').
 add_fields('float32', 'scale', '1.f').
 add_fields('float32', 'bias', '0.f')
)

(pdef('ROIPooling').
 add_enum(
     'Mode',
     Doc('MAX', 'maximum value inside pooling window; pooling result would '
         'be 0 if pooling window is empty'),
     Doc('AVERAGE',
         'arithmetic mean of all values inside pooling window; pooling result '
         'would be 0 if pooling window is empty')
 ).
 add_fields('float32', 'scale', '1.f'))

INTERP_MODES = ['NEAREST', 'LINEAR', 'AREA', 'CUBIC', 'LANCZOS4']
BORDER_MODES = [Doc('REPLICATE', 'aaaaaa|abcdefgh|hhhhhhh'),
                Doc('REFLECT', 'fedcba|abcdefgh|hgfedcb'),
                Doc('REFLECT_101', 'gfedcb|abcdefgh|gfedcba'),
                Doc('WRAP', 'cdefgh|abcdefgh|abcdefg'),
                Doc('CONSTANT', 'iiiiii|abcdefgh|iiiiiii'),
                Doc('TRANSPARENT', ''),
                Doc('ISOLATED', '')]
(pdef('WarpPerspective', version=1).
 add_enum('InterpolationMode', *INTERP_MODES,
          name_field='imode', default=1,
          member_alias=[(i, 'INTER_{}'.format(i)) for i in INTERP_MODES]
          ).
 add_enum('BorderMode', *BORDER_MODES,
          name_field='bmode',
          member_alias=[(i, 'BORDER_{}'.format(i)) for i in BORDER_MODES]
          ).
 add_enum_alias('Format', 'ConvolutionV0').
 add_fields('float32', Doc('border_val', 'used for CONSTANT bmode'), '.0f'))

pdef('SpatialTfGridGenerator').add_enum('Mode', 'AFFINE')
pdef('SpatialTfSampler').add_enum('Mode', 'BILINEAR')

pdef('AddUpdate').add_fields(
    'float32', 'alpha', '1.f', 'beta', '1.f', 'bias', '0.f')

pdef('Elemwise').add_enum(
    'Mode',
    Doc('RELU', 'unary: max(x, 0)'),
    Doc('ABS', 'unary: abs(x)'),
    Doc('ACOS', 'unary: acos(x)'),
    Doc('ASIN', 'unary: asin(x)'),
    Doc('CEIL', 'unary: ceil(x)'),
    Doc('COS', 'unary: cos(x)'),
    Doc('EXP', 'unary: exp(x)'),
    Doc('EXPM1', 'unary: numerically stable exp(x)-1'),
    Doc('FLOOR', 'unary: floor(x)'),
    Doc('LOG', 'unary: natural logarithm, log(x)'),
    Doc('LOG1P', 'unary: numerically stable log(x+1)'),
    Doc('NEGATE', 'unary: -x'),
    Doc('SIGMOID', 'unary: 1/(1+exp(-x))'),
    Doc('SIN', 'unary: sin(x)'),
    Doc('TANH', 'unary: tanh(x)'),

    Doc('ABS_GRAD', 'binary: x > 0 ? y : -y'),
    Doc('ADD', 'binary: x + y'),
    Doc('FLOOR_DIV', 'binary: floor(x / y)'),
    Doc('MAX', 'binary: max(x, y)'),
    Doc('MIN', 'binary: min(x, y)'),
    Doc('MOD', 'binary: x % y or fmodf(x, y)'),
    Doc('MUL', 'binary: x * y'),
    Doc('POW', 'binary: pow(x, y)'),
    Doc('SIGMOID_GRAD', 'binary: x * (1 - x) * y'),
    Doc('SUB', 'binary: x - y'),
    Doc('SWITCH_GT0', 'binary: (x > 0) * y'),
    Doc('TANH_GRAD', 'binary: (1 - x * x) * y'),
    Doc('TRUE_DIV', 'binary: x / y'),
    Doc('LOG_SUM_EXP', 'binary: numerically stable log(exp(x) + exp(y))'),

    Doc('LT', 'binary: x < y'),
    Doc('LEQ', 'binary: x <= y'),
    Doc('EQ', 'binary: x == y'),

    Doc('SHL', 'bitwise binary: x << y. '
        'Note that result is undefined if y < 0 or y >= bitwidth. Logical '
        'shift is performed for unsigned intergers, and arithmetic shift for '
        'signed ones.'),
    Doc('SHR', 'bitwise binary: x >> y; see SHL mode for more details'),

    Doc('COND_LEQ_MOV', 'ternary: x <= y ? z : 0'),
    Doc('FUSE_MUL_ADD3',
        'compute ``a * b + c`` where c must either have same layout as '
        'a or b, or be a scalar'),

    Doc('FUSE_MUL_ADD4',
        'compute ``a * A + b * B`` where a and b must have equal layout, '
        'and A and B must have equal layout. In the inputs ``b`` and ``B`` '
        'can be swapped'),

    Doc('FUSE_ADD_RELU', 'binary: max(x+y, 0)'),
    Doc('FUSE_ADD_SIGMOID', 'binary: 1/(1+exp(-(x+y)))'),
    Doc('FUSE_ADD_TANH', 'binary: tanh(x+y)'),
    Doc('FAST_TANH', 'unary: rational approximation of tanh(x)'),
    Doc('FAST_TANH_GRAD', 'binary: grad of the rational approximation of tanh(x)'),

    Doc('ROUND', 'unary: round(x), the nearest integer value to x, rounding '
                 'halfway cases away from zero. Float only.'),
    Doc('RMULH', 'binary: rounded higher l bits of x * y, where l is the bit '
                'length of x.'),

    Doc('ATAN2','binary: atan2(y,x)'),
    Doc('ERF', 'unary: erf(x)'),
    Doc('ERFINV', 'unary: inverse function of erf(x)'),
    Doc('ERFC', 'unary: erfc(x)'),
    Doc('ERFCINV', 'unary: inverse function of erfc(x)'),
    Doc('H_SWISH', 'unary: x * clip(x + 3, 0, 6) / 6'),
    Doc('H_SWISH_GRAD', 'binary: x < -3 ? 0 : (x > 3 ? y : (2 * x + 3) / 6 * y)'),
    Doc('FUSE_ADD_H_SWISH', 'binary: hswish(x+y)'),

    Doc('NOT', 'unary: !x'),
    Doc('AND', 'binary: x && y'),
    Doc('OR', 'binary: x || y'),
    Doc('XOR', 'binary: x ^ y')
)

pdef('ElemwiseMultiType').add_enum(
    'Mode',
    Doc('FUSE_MUL_ADD3_INT16x32x32x32',
        'compute ``a * b + c`` requiring that ``a`` be int16 and ``b`` and '
        '``c``  int32, and the result is int32. This mode is optimized for '
        'the channel-broadacsted case, i.e. ``a`` has shape (A, B, C) and '
        '``b`` and ``c`` have shape (1, C, 1)'),
    Doc('FUSE_MUL_ADD3_IXxF32xF32xI8',
        'compuate ``a * b + c`` where the inputs ``a`` is an integer type '
        '``b`` and ``c`` are both ``float32``, the result is '
        '``int8``. This is currently only optimized for ``(1, x)`` '
        'broadcast for ``b`` and ``c``. Computation is carried in floating '
        'points and results are rounded towards zero with saturated cast to '
        'int.'),
    Doc('ROUND_SHR_SATURATE_IXxI8xI8',
        'Compute ``a >> b``, round the result according to lower ``b`` bits '
        'of ``a``` and make a saturating conversion to int8. Where ``a`` should'
        ' be an integer tensor and ``b`` should be an int8 scalar.'),
    Doc('FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8',
        'Fused operation of an int16 elemwise add, an int16 rounding multiply '
        'high and an int16 to int8 rounding right shift with saturation.'),
    Doc('FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8',
        'Fused operation of an int32 elemwise add, an int32 rounding multiply '
        'high and an int32 to int8 rounding right shift with saturation.'),
    Doc('ROUND_SHR_SATURATE_IXxI8xI16',
        'Compute ``a >> b``, round the result according to lower ``b`` bits of '
        '``a``` and make a saturating conversion to int16. Where ``a`` should'
        ' be an integer tensor and ``b`` should be an int8 scalar.'),
    Doc('QADD', 'Fused elemwise add two quantized int8 with specified'
        'output quantized dtype'),
    Doc('QFUSE_ADD_RELU', 'Fused elemwise add two quantized int8 followed'
         ' by ReLU and typecvt to specified dtype'),
    Doc('QMUL', 'Fused elemwise multiply two quantized int8 with specified'
        'output quantized dtype'),
    Doc('QMIN', 'Fused elemwise min two quantized int8 with specified'
        'output quantized dtype'),
    Doc('QMAX', 'quantized: max(x, y), with specified output quantized dtype'),
    Doc('QSUB', 'quantized: x - y'),
    Doc('QTRUE_DIV', 'quantized: x / y'),
    Doc('QFUSE_ADD_SIGMOID', 'quantized: sigmoid(x + y)'),
    Doc('QFUSE_ADD_TANH', 'quantized: tanh(x + y)'),
    Doc('QRELU', 'quantized: x > 0 ? x : 0'),
    Doc('QABS', 'quantized: x > 0 ? x : -x'),
    Doc('QSIGMOID', 'quantized: sigmoid(x)'),
    Doc('QEXP', 'quantized: exp(x)'),
    Doc('QTANH', 'quantized: tanh(x)'),
    Doc('QFUSE_MUL_ADD3', 'quantized: x * y + z'),
    Doc('QFAST_TANH', 'quantized: fast_tanh(x)'),
    Doc('QNEGATE', 'quantized: -x'),
    Doc('QACOS', 'quantized: acos(x)'),
    Doc('QASIN', 'quantized: asin(x)'),
    Doc('QCEIL', 'quantized: ceil(x)'),
    Doc('QCOS', 'quantized: cos(x)'),
    Doc('QEXPM1', 'quantized: expm1(x)'),
    Doc('QFLOOR', 'quantized: floor(x)'),
    Doc('QLOG', 'quantized: log(x)'),
    Doc('QLOG1P', 'quantized: log1p(x)'),
    Doc('QSIN', 'quantized: sin(x)'),
    Doc('QROUND', 'quantized: round(x)'),
    Doc('QERF', 'quantized: erf(x)'),
    Doc('QERFINV', 'quantized: erfinv(x)'),
    Doc('QERFC', 'quantized: erfc(x)'),
    Doc('QERFCINV', 'quantized: erfcinv(x)'),
    Doc('QABS_GRAD', 'quantized: abs_grad'),
    Doc('QFLOOR_DIV', 'quantized floor_div'),
    Doc('QMOD', 'quantized mod'),
    Doc('QSIGMOID_GRAD', 'quantized sigmoid_grad'),
    Doc('QSWITCH_GT0', 'quantized switch_gt0'),
    Doc('QTANH_GRAD', 'quantized tanh_grad'),
    Doc('QLT', 'quantized lt'),
    Doc('QLEQ', 'quantized leq'),
    Doc('QEQ', 'quantized eq'),
    Doc('QPOW', 'quantized pow'),
    Doc('QLOG_SUM_EXP', 'quantized log_sum_exp'),
    Doc('QFAST_TANH_GRAD', 'quantized fast_tanh_grad'),
    Doc('QATAN2', 'quantized atan2'),
    Doc('QCOND_LEQ_MOV', 'quantized cond_leq_mov'),
    Doc('QH_SWISH', 'quantized h_swish'),
    Doc('QFUSE_ADD_H_SWISH', 'quantized h_swish(x+y)'),
    Doc('QH_SWISH_GRAD', 'quantized h_swish_grad')
)

pdef('PowC', 'power with constant exponent').add_fields('float32', 'exp', 0)

(pdef('DctChannelSelect', '2d discrete cosine transform').add_enum_alias('Format', 'ConvolutionV0').
 add_enum('FastImpl', 'NONE', 'FIX_32_MASK').add_fields('int32', 'dct_block_size', 8))

(pdef('MatrixMul', version=0, is_legacy=True).
 add_fields('bool', 'transposeA', 'false', 'transposeB', 'false').
 add_enum('DataType',
     Doc('FLOAT', 'input/output both float32/float16'),
     'INT8x8x16',
     'INT8x8x32',
     Doc('FLOAT_IO16xC32', 'input/output both float16, the internal compute is '
         'float32'),
     Doc('QUINT8x8x32', 'input QuantizedAsymm8, output QuantizedS32'),
     Doc('QUINT4x4x32', 'input QuantizedAsymm4, output QuantizedS32'),
     name_field='data_type'))

(pdef('MatrixMul', version=1, is_legacy=True).
 add_fields('bool', 'transposeA', 'false', 'transposeB', 'false').
 add_enum(Doc('ComputeMode', 'Specifies special computation modes, e.g. '
                             'different combinations of intermediate result '
                             'data types.'),
          Doc('DEFAULT', 'No special requirements on the precision of '
                         'intermediate results.'),
          Doc('FLOAT32', 'Use Float32 accumulator and intermediate result. '
                         'Only supported when input and output is Float16.'),
          name_field='compute_mode'))

(pdef('MatrixMul', version=2).
 add_fields('bool', 'transposeA', 'false', 'transposeB', 'false').
 add_enum_alias('ComputeMode', 'MatrixMulV1', name_field='compute_mode').
 add_enum('Format',
          Doc('DEFAULT', 'Normal matrix mul: (M, K) x (K, N) = (M, N)'),
          Doc('MK4', 'Split 4 from M and K, better for neon compute:'
              '(M/4, K/4, 4(k), 4(m)) x (K/4, N, 4(k)). if transposeA the '
              'layout is (K/4, M/4, 4(k), 4(m)) x (K/4, N, 4(k))'),
          Doc('MK8', 'Split 8 from M and K, better for neon compute:'
              '(M/8, K/8, 8(k), 8(m)) x (K/8, N, 8(k)). if transposeA the '
              'layout is (K/8, M/8, 8(k), 8(m)) x (K/8, N, 8(k))'),
          Doc('MK4_DOT', 'Split 4 from M and K, better for neon dotprod:'
              'M/4, K/4, 4(m), 4(k)) x (K/4, N, 4(k)). if transposeA the '
              'layout is (K/4, M/4, 4(m), 4(k)) x (K/4, N, 4(k))'))
 )

(pdef('Winograd', 'winograd param used in convbias').
  add_fields(
      'uint32',
      Doc('output_block_size', 'output block size, detail meaning see winograd '
          'in convbias, equals to the meaning of m in F(m, r)'), 0).
  add_enum_alias('Format', 'MatrixMul').
  add_enum_alias('ComputeMode', 'Convolution', name_field='compute_mode')
 )

(pdef('SVD').
 add_fields('bool',
            Doc('full_matrices',
                'Whether to compute the full-sized u and v or only the leading'
                ' min(m, n) singular vectors. Ignored if compute_uv is '
                'false.'),
            'false',
            Doc('compute_uv',
                'Whether the left (u) and right (v) singular vectors will be '
                'computed and outputted.'),
            'true'))

(pdef('Reduce', 'legacy reduce', version=0, is_legacy=True).
 add_enum('Mode',
          'SUM',
          Doc('SUM_SQR', 'sum of x * x for each element x'),
          'PRODUCT', 'MIN', 'MAX').
 add_fields('int32',
            Doc('axis',
                'axis along which reduction is performed; if -1 is given, '
                'reduce to given target shape (only used in megbrain)'),
            -1))

(pdef('Reduce', 'reduce along given axis', version=1, is_legacy=True).
 add_enum('Mode',
          'SUM',
          Doc('SUM_SQR', 'sum of x * x for each element x'),
          'PRODUCT', 'MIN', 'MAX', 'MEAN').
 add_fields('int32',
            Doc('axis',
                'axis along which reduction is performed; if -1 is given, '
                'reduce to given target shape (only used in megbrain)'),
            -1).
 add_enum('DataType',
          Doc('DEFAULT',
'''
input/output are the same data type, and the internal computation type would be chosen by the input/output dtypes and the reduction mode.
Currently, ```DEFAULT``` mode means:

+--------------------+-----------------------------------+-------------------+
| Input/Output DType | Mode                              | Computation DType |
+====================+===================================+===================+
| FLOAT32            | MIN/MAX/MEAN/SUM/SUM_SQR/PRODUCT  | FLOAT32           |
+--------------------+-----------------------------------+-------------------+
| FLOAT16            | MIN/MAX/MEAN/SUM/SUM_SQR/PRODUCT  | FLOAT16           |
+--------------------+-----------------------------------+-------------------+
| INT32              | MIN/MAX/MEAN/SUM/SUM_SQR/PRODUCT  | INT32             |
+--------------------+-----------------------------------+-------------------+
| INT8               | MIN/MAX/MEAN/SUM/SUM_SQR/PRODUCT  | INT8              |
+--------------------+-----------------------------------+-------------------+
| QuantizedS8        | MIN/MAX                           | QuantizedS8       |
+--------------------+-----------------------------------+-------------------+
| QuantizedS8        | MEAN/SUM                          | QuantizedS32      |
+--------------------+-----------------------------------+-------------------+
| Quantized8Asymm    | MIN/MAX                           | Quantized8Asymm   |
+--------------------+-----------------------------------+-------------------+
| Quantized8Asymm    | MEAN/SUM                          | QuantizedS32      |
+--------------------+-----------------------------------+-------------------+

'''
),
          Doc('FLOAT_IO16xC32', 'Deprecated. This was replaced by '
              'FLOAT_O16xC32, and input\'s dtype decided by actual input tensor.'),
          Doc('FLOAT_O32xC32', 'compute/output both are float32'),
          Doc('FLOAT_O16xC32', 'compute are float32, output float16'),
          Doc('QUINT_I8xO32', 'input quint8, compute and output are qint32'),
          Doc('QINT_I8xO32', 'input qint8, compute and output are qint32'),
     name_field='data_type'))

(pdef('Reduce', 'reduce along given axis', version=2).
 add_enum('Mode',
          'SUM',
          Doc('SUM_SQR', 'sum of x * x for each element x'),
          'PRODUCT', 'MIN', 'MAX', 'MEAN').
 add_fields('int32',
            Doc('axis',
                'axis along which reduction is performed; if INT_MAX is given, '
                'reduce to given target shape (only used in megbrain)'),
            (1<<31)-1).
 add_enum('DataType',
          Doc('DEFAULT',
'''
input/output are the same data type, and the internal computation type would be chosen by the input/output dtypes and the reduction mode.
Currently, ```DEFAULT``` mode means:

+--------------------+-----------------------------------+-------------------+
| Input/Output DType | Mode                              | Computation DType |
+====================+===================================+===================+
| FLOAT32            | MIN/MAX/MEAN/SUM/SUM_SQR/PRODUCT  | FLOAT32           |
+--------------------+-----------------------------------+-------------------+
| FLOAT16            | MIN/MAX/MEAN/SUM/SUM_SQR/PRODUCT  | FLOAT16           |
+--------------------+-----------------------------------+-------------------+
| INT32              | MIN/MAX/MEAN/SUM/SUM_SQR/PRODUCT  | INT32             |
+--------------------+-----------------------------------+-------------------+
| INT8               | MIN/MAX/MEAN/SUM/SUM_SQR/PRODUCT  | INT8              |
+--------------------+-----------------------------------+-------------------+
| QuantizedS8        | MIN/MAX                           | QuantizedS8       |
+--------------------+-----------------------------------+-------------------+
| QuantizedS8        | MEAN/SUM                          | QuantizedS32      |
+--------------------+-----------------------------------+-------------------+
| Quantized8Asymm    | MIN/MAX                           | Quantized8Asymm   |
+--------------------+-----------------------------------+-------------------+
| Quantized8Asymm    | MEAN/SUM                          | QuantizedS32      |
+--------------------+-----------------------------------+-------------------+

'''
),
          Doc('FLOAT_IO16xC32', 'Deprecated. This was replaced by '
              'FLOAT_O16xC32, and input\'s dtype decided by actual input tensor.'),
          Doc('FLOAT_O32xC32', 'compute/output both are float32'),
          Doc('FLOAT_O16xC32', 'compute are float32, output float16'),
          Doc('QUINT_I8xO32', 'input quint8, compute and output are qint32'),
          Doc('QINT_I8xO32', 'input qint8, compute and output are qint32'),
     name_field='data_type'))

(pdef('Cumsum', 'calculate accumulated sum along given axis', version=0, is_legacy=True).
 add_fields('int32',
          Doc('axis',
              'axis along which cumsum is performed'),
          -1).
 add_fields('bool',
          Doc('exclusive',
              'whether the current element is taken into account'),
          'true').
 add_fields('bool',
          Doc('reverse',
              'whether the cumsum is forward or backward'),
          'false'))

(pdef('Cumsum', 'calculate accumulated sum along given axis', version=1).
 add_fields('int32',
          Doc('axis',
              'axis along which cumsum is performed, default with INT_MAX'),
          (1<<31)-1).
 add_fields('bool',
          Doc('exclusive',
              'whether the current element is taken into account'),
          'true').
 add_fields('bool',
          Doc('reverse',
              'whether the cumsum is forward or backward'),
          'false'))

(pdef('CondTake').
 add_enum('Mode',
          Doc('EQ', 'take if ``abs(data-val)<eps``'),
          Doc('NEQ', 'take if ``abs(data-val)>=eps``'),
          Doc('LT', 'take if ``data<val``'),
          Doc('LEQ', 'take if ``data<=val``'),
          Doc('GT', 'take if ``data>val``'),
          Doc('GEQ', 'take if ``data>=val``')).
 add_fields('float32',
            Doc('val', 'the value to be compared with; note that for integer '
                'data, val is also converted to int'), 0).
 add_fields('float32', Doc('eps', 'used for float equality comparison'),
            1e-6))


pdef('Argsort').add_enum('Order', 'ASCENDING', 'DESCENDING')

(pdef('IndexingRemap').
 add_fields('bool',
            Doc('is_non_overlapping',
                'Whether no two dst element maps to the same src element. '
                'Enabling this option can accelerate gradient operator since'
                ' atomic adding operations could be avoided.'),
            'false'))

pdef('Sleep').add_fields('float32', Doc('time', 'time to sleep in seconds'), 0)

(pdef('Linspace').
 add_fields('bool',
            Doc('endpoint',
                'Whether stop is included in the generated tensor'),
            'true'))

(pdef('LinspaceFull').
 add_fields('float64',
            Doc('start', 'The first val.'),
            0).
 add_fields('float64',
            Doc('stop', 'The last val.'),
            1).
 add_fields('bool',
            Doc('endpoint',
                'Whether stop is included in the generated tensor'),
            'true'))

(pdef('Eye').
 add_fields(
     'int32',
     Doc('k', 'Index of the diagonal: 0 (the default) refers to the main '
         'diagonal, a positive value refers to an upper diagonal, and a '
         'negative value to a lower diagonal.'),
     0).
 add_fields(
     'dtype', Doc('dtype', 'data type of output value'),
     'DTypeEnum::Float32'))

pdef('UniformRNG').add_fields('uint64', 'seed', 0)

(pdef('GaussianRNG').
 add_fields('uint64', 'seed', 0).
 add_fields('float32', 'mean', 0, 'std', 1))

(pdef('Flip').
 add_fields('bool', 'vertical', 'false', 'horizontal', 'false'))

(pdef('Rotate')
 .add_fields('bool', 'clockwise', 'true'))

(pdef('ROICopy')
 .add_fields('uint32', 'row_from', 0, 'row_to', 0, 'col_from', 0, 'col_to', 0))

(pdef('CvtColor')
 .add_enum('Mode', 'RGB2GRAY', 'RGB2YUV', 'YUV2RGB', 'GRAY2RGB', 'RGBA2RGB',
    'RGBA2BGR', 'RGBA2GRAY', 'RGB2BGR', 'BGR2GRAY', 'BGR2RGB',
    Doc('YUV2GRAY_NV21', 'For historical reasons, referred to as YCC by opencv'),
    'YUV2RGB_NV21', 'YUV2BGR_NV21', 'YUV2GRAY_NV12', 'YUV2RGB_NV12',
    'YUV2BGR_NV12', 'YUV2GRAY_YV12', 'YUV2RGB_YV12', 'YUV2BGR_YV12',
    'YUV2GRAY_YU12', 'YUV2RGB_YU12', 'YUV2BGR_YU12',
    'YCrCb2RGB', 'YCrCb2BGR',
    Doc('BT601_YUV2RGB_NV21', 'BT601 yuv format, referred to as YUV by opencv'),
    'BT601_YUV2BGR_NV21', 'BT601_YUV2RGB_NV12', 'BT601_YUV2BGR_NV12',
    'BT601_YUV2RGB_YV12', 'BT601_YUV2BGR_YV12', 'BT601_YUV2RGB_YU12',
    'BT601_YUV2BGR_YU12',
    member_alias=[('YUV2GRAY_NV21', 'BT601_YUV2GRAY_NV21'),
                  ('YUV2GRAY_NV12', 'BT601_YUV2GRAY_NV12'),
                  ('YUV2GRAY_YV12', 'BT601_YUV2GRAY_YV12'),
                  ('YUV2GRAY_YU12', 'BT601_YUV2GRAY_YU12')],
    name_field = 'mode'))

(pdef('WarpAffine', version=0, is_legacy=True)
 .add_enum_alias('InterpolationMode', 'WarpPerspective', name_field='imode')
 .add_enum_alias('BorderMode', 'WarpPerspective', name_field='border_mode')
 .add_fields('float32', Doc('border_val', 'used for CONSTANT bmode'), '.0f'))

(pdef('WarpAffine', version=1)
 .add_enum_alias('InterpolationMode', 'WarpPerspective', name_field='imode')
 .add_enum_alias('BorderMode', 'WarpPerspective', name_field='border_mode')
 .add_fields('float32', Doc('border_val', 'used for CONSTANT bmode'), '.0f')
 .add_enum_alias('Format', 'ConvolutionV0', default=1))

(pdef('GaussianBlur')
 .add_enum_alias('BorderMode', 'WarpPerspective', name_field='border_mode')
 .add_fields('uint32', 'kernel_height', 0, 'kernel_width', 0)
 .add_fields('float32','sigma_x', '0.f', 'sigma_y', '0.f'))

(pdef('Resize', version=0, is_legacy=True)
 .add_enum_alias('InterpolationMode', 'WarpPerspective', name_field='imode'))

(pdef('Resize', version=1)
 .add_enum_alias('InterpolationMode', 'WarpPerspective', name_field='imode')
 .add_enum_alias('Format', 'ConvolutionV0', default=1))

(pdef('Remap', version=0)
 .add_enum_alias('InterpolationMode', 'WarpPerspective', name_field='imode')
 .add_enum_alias('BorderMode', 'WarpPerspective', name_field='border_type')
 .add_enum_alias('Format', 'ConvolutionV0', default=1)
 .add_fields('float32', 'scalar', '0.f'))

(pdef('Convolution3D').
 add_enum('Mode', 'CROSS_CORRELATION', 'CONVOLUTION').
 add_fields(
     'uint32',
     Doc('pad_d', 'padding on one side on the first dimension'), 0,
     Doc('pad_h', 'padding on one side on the second dimension'), 0,
     Doc('pad_w', 'padding on one side on the third dimension'), 0,
     Doc('stride_d', 'kernel stride on the first dimension'), 1,
     Doc('stride_h', 'kernel stride on the second dimension'), 1,
     Doc('stride_w', 'kernel stride on the third dimension'), 1,
     Doc('dilate_d', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the first dimension'), 1,
     Doc('dilate_h', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1,
     Doc('dilate_w', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the third dimension'), 1
 ).
 add_enum('Sparse',
          Doc('DENSE', 'dense convolution: filter shape should be '
              '[oc, ic, spatial...] if format is NCDHW, '
              '[oc, spatial..., ic] if format is NDHWC'),
          Doc('GROUP', 'group convolution: filter shape should be '
              '[group, oc_per_group, ic_per_group, spatial...] if format is NCDHW, '
              '[group, oc_per_group, spatial..., ic_per_group] if format is NDHWC')
          ).
 add_enum('DataType',
          Doc('FLOAT', 'input/output both float32/float16'),
          Doc('FLOAT_IO16xC32', 'input/output both float16, the internal '
              'compute is float32'),
          name_field='data_type').
 add_enum('Format', 'NCDHW', 'NDHWC')
 )

(pdef('Conv3DBias').
 add_enum('NonlineMode', 'IDENTITY', 'RELU', 'SIGMOID').
 add_enum_alias('Mode', 'Convolution3D').
 add_fields('uint32', 'pad_d', 0, 'pad_h', 0, 'pad_w', 0,
                'stride_d', 1, 'stride_h', 1, 'stride_w', 0))

(pdef('SeparableConv3D').
 add_enum_alias('Mode', 'Convolution3D').
 add_enum('BorderMode', 'BORDER_REPLICATE', 'BORDER_REFLECT',
          'BORDER_REFLECT_101','BORDER_WRAP',
          'BORDER_CONSTANT', 'BORDER_TRANSPARENT','BORDER_ISOLATED').
 add_fields('bool', 'is_symm_kernel', 'true').
 add_fields('uint32', 'pad_d', 0, 'pad_h', 0, 'pad_w', 0,
            'stride_d', 0, 'stride_h', 1, 'stride_w', 1,
            'ksize_d', 0, 'ksize_h', 3, 'ksize_w', 3,
            'anchor_d', 0, 'anchor_h', 1, 'anchor_w', 1))

(pdef('TopK').
 add_enum(
     'Mode',
     Doc('KTH_ONLY', "only the value of the k'th element would be computed"),
     Doc('VALUE_IDX_NOSORT',
         'all the top-k values and corresponding indices would be computed; '
         'no order is guaranteed'),
     Doc('VALUE_IDX_SORTED',
         'all the top-k values and corresponding indices sorted'))
 )

RELAYOUT_FORMAT_MODE_DOC = """
Relayout mode.

**Naming conventions**

1. ``A_B`` means change from layout format ``A`` to ``B``.
2. ``INTER_WEIGHT_xx`` means relayout the weight for faster processing by
   :attr:`Convolution.Format.NHWCD4` convolutions.
3. A suffix of ``I`` means ``Image2DPack4TensorFormat`` tensor format is used
   for faster processing on GPUs.

**Layout definitions**

* ``NCHW`` layout: ``{N, C, H, W}``
* ``NHWC`` layout: ``{N, H, W, C}``
* ``NHWCD4`` layout: ``{N, H, (C + 3) / 4, W, 4}``
* ``NHWCD4I`` layout: with ``align_axis = 2``
* ``NCHW4`` layout: ``{N, C/4, H, W, 4}``
* ``NCHW88`` layout: ``{N, C/8, H, W, 8}``
* ``CHWN4`` layout: ``{C/4, H, W, N, 4}``

**Float weight transformation definitions**

+---------------+---------------------------------+--------------------+--------------------------------------+------+
| Sparsity Type | Input Layout                    | Input Req          | Output Layout                        | Axis |
+===============+=================================+====================+======================================+======+
| DENSE         | ``{OC, IC, FH, FW}``            | ``OC % 4 == 0``    | ``{OC/4, FH, FW, IC, 4}``            | 3    |
+---------------+---------------------------------+--------------------+--------------------------------------+------+
| GROUP         | ``{GROUP, OCPG, ICPG, FH, FW}`` | ``OCPG % 4 == 0``  | ``{GROUP, OCPG/4, FH, FW, ICPG, 4}`` | 4    |
|               |                                 | ``ICPG % 4 == 0``  |                                      |      |
+---------------+---------------------------------+--------------------+--------------------------------------+------+
| CHAN          | ``{GROUP, 1, 1, FH, FW}``       | ``GROUP % 4 == 0`` | ``{GROUP / 4, 1, FH ,FW, 4}``        | 1    |
+---------------+---------------------------------+--------------------+--------------------------------------+------+

**Float weight transformation nchw88 definitions**

+---------------+---------------------------------+--------------------+--------------------------------------+
| Sparsity Type | Input Layout                    | Input Req          | Output Layout                        |
+===============+=================================+====================+======================================+
| DENSE         | ``{OC, IC, FH, FW}``            | ``OC % 8 == 0``    |``{OC/8, IC/8 ,FH, FW, 8(IC), 8(OC)}``|
|               |                                 | ``IC % 8 == 0``    |                                      |
+---------------+---------------------------------+--------------------+--------------------------------------+
| GROUP         | ``{GROUP, OCPG, ICPG, FH, FW}`` | ``OCPG % 8 == 0``  | ``{GROUP, OCPG/8, ICPG/8 FH, FW,     |
|               |                                 | ``ICPG % 8 == 0``  |  8(ICPG), 8(OCPG)} ``                |
+---------------+---------------------------------+--------------------+--------------------------------------+
| CHAN          | ``{GROUP, 1, 1, FH, FW}``       | ``GROUP % 8 == 0`` | ``{GROUP / 8, 1, FH ,FW, 8}``        |
+---------------+---------------------------------+--------------------+--------------------------------------+

**Int8(DOT) weight transformation definitions**

+---------------+---------------------------------+--------------------+------------------------------------------+------+
| Sparsity Type | Input Layout                    | Input Req          | Output Layout                            | Axis |
+===============+=================================+====================+==========================================+======+
| DENSE         | ``{OC, IC, FH, FW}``            | ``OC % 4 == 0``    | ``{OC/4, FH, FW, IC/4, 4, 4}`            | 3    |
+---------------+---------------------------------+--------------------+------------------------------------------+------+
| GROUP         | ``{GROUP, OCPG, ICPG, FH, FW}`` | ``OCPG % 4 == 0``  | ``{GROUP, OCPG/4, FH, FW, ICPG/4, 4, 4}``| 4    |
|               |                                 | ``ICPG % 4 == 0``  |                                          |      |
+---------------+---------------------------------+--------------------+------------------------------------------+------+

Note: the axis column means the corresponding ``align_axis`` for image format
when the ``I`` suffix is present.

"""
(pdef('RelayoutFormat', 'Change the tensor layout format').
 add_enum(
     Doc('Mode', RELAYOUT_FORMAT_MODE_DOC),
     'NHWC_NHWCD4',
     'NHWCD4_NHWC',
     'NHWC_NHWCD4I',
     'NCHW_NHWCD4',
     'NCHW_NHWCD4I',
     'NHWCD4I_NCHW',
     'NHWCD4_NCHW',
     'INTER_WEIGHT_DENSE',
     'INTER_WEIGHT_DENSEI',
     'INTER_WEIGHT_GROUP',
     'INTER_WEIGHT_GROUPI',
     'INTER_WEIGHT_CHAN',
     'INTER_WEIGHT_CHANI',
     'INTER_WEIGHT_DENSEI_DOT',
     'INTER_WEIGHT_GROUPI_DOT',
     'NCHW4_CHWN4',
     'CHWN4_NCHW4',
     'NCHW_NCHW88_CONV_DENSE_WEIGHT',
     'NCHW_NCHW88_CONV_CHAN_WEIGHT',
     'NCHW_NCHW88_CONV_GROUP_WEIGHT',
     'NCHW_NCHW88',
     'NCHW88_NCHW',
     'NCHW_NCHW4_IC_SMALL',
     'NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT',
     'NCHW_NCHW4',
     )
 )


(pdef('SeparableFilter').
 add_enum_alias('Format', 'ConvolutionV0').
 add_enum_alias('BorderMode', 'WarpPerspective').
 add_fields('bool', 'is_symm_kernel', 'true').
 add_fields('uint32', 'ksize_h', 3, 'ksize_w', 3, 'anchor_h', 1, 'anchor_w', 1))

(pdef('LocalShare', 'Local share convolution').
 add_enum_alias('Mode', 'ConvolutionV0').
 add_fields(
     'uint32',
     Doc('pad_h', 'padding on one side on the first dimension'), 0,
     Doc('pad_w', 'padding on one side on the second dimension'), 0,
     Doc('stride_h', 'kernel stride on the first dimension'), 1,
     Doc('stride_w', 'kernel stride on the second dimension'), 1,
     Doc('dilate_h', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1,
     Doc('dilate_w', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1,
     Doc('spatial_groups_h', 'spatial groups on the first dimension'), 1,
     Doc('spatial_groups_w', 'spatial groups on the second dimension'), 1
 ).
 add_enum_alias('Sparse', 'ConvolutionV0').
 add_enum_alias('Format', 'ConvolutionV0').
 add_enum_alias('ComputeMode', 'Convolution')
 )

(pdef('ROIAlign').
 add_enum('Mode', 'MAX', 'AVERAGE', name_field='mode').
 add_enum_alias('Format', 'ConvolutionV0').
 add_fields('float32', 'spatial_scale', '1.0').
 add_fields('float32', 'offset', '0.0').
 add_fields('uint32',
            'pooled_height', '1',
            'pooled_width', '1',
            'sample_height', '2',
            'sample_width', '2')
 )
(pdef('DeformablePSROIPooling').
 add_fields('bool', 'no_trans', 'true').
 add_fields('float32', 'spatial_scale', 1,
     'trans_std', 1).
 add_fields('uint32',
    Doc('pooled_h', 'height of pooling output'), 1,
    Doc('pooled_w', 'width of pooling output'), 1,
    Doc('part_size', 'size of each deformable part'), 1,
    Doc('sample_per_part', 'sample count of each bbox'), 1))

(pdef('BatchConvBias', 'Batch convolution (unshare weights on the batch dimension)').
 add_enum_alias('NonlineMode', 'ConvBiasV0').
 add_enum_alias('Mode', 'ConvolutionV0').
 add_fields(
     'uint32',
     Doc('pad_h', 'padding on one side on the first dimension'), 0,
     Doc('pad_w', 'padding on one side on the second dimension'), 0,
     Doc('stride_h', 'kernel stride on the first dimension'), 1,
     Doc('stride_w', 'kernel stride on the second dimension'), 1,
     Doc('dilate_h', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1,
     Doc('dilate_w', 'dilation (i.e. size of each zero-padded kernel block) '
         'on the second dimension'), 1,
 ).
 add_enum_alias('Sparse', 'ConvolutionV0').
 add_enum_alias('Format', 'ConvolutionV0').
 add_enum_alias('ComputeMode', 'Convolution', name_field="compute_mode")
 )
(pdef('FakeQuant').
 add_fields('int32','qmin','-2147483648').
 add_fields('int32','qmax','2147483647')
 )


