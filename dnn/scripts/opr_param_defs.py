pdef('Empty')

pdef('Axis').add_fields('int32', 'axis', 0)

(pdef('Convolution', version=0, is_legacy=True).
 add_enum('Mode', 'CROSS_CORRELATION = 0', 'CONVOLUTION = 1').
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
          Doc('FLOAT = 0', 'input/output both float32/float16'),
          'INT8x8x16 = 1',
          'INT8x8x32 = 2',
          Doc('FLOAT_IO16xC32 = 3', 'input/output both float16, the internal '
              'compute is float32'),
          Doc('QUINT8x8x32 = 4', 'input QuantizedAsymm8, output QuantizedS32'),
          Doc('INT8x8xX = 5', 'input int8, output specified by tensor DType'),
          Doc('QUINT4x4x32 = 6', 'input QuantizedAsymm4, output QuantizedS32'),
          name_field='data_type').
 add_enum('Sparse',
          Doc('DENSE = 0', 'dense convolution: filter shape should be '
              '[oc, ic, spatial...] if format is NCHW, '
              '[oc, spatial..., ic] if format is NHWC'),
          Doc('GROUP = 1', 'group convolution: filter shape should be '
              '[group, oc_per_group, ic_per_group, spatial...] if format is NCHW, '
              '[group, oc_per_group, spatial..., ic_per_group] if format is NHWC')
          ).
 add_enum(Doc('Format', 'convolution data/filter/output format; see '
              ':class:`RelayoutFormat` for more details'),
          'NCHW = 0', 'NHWC = 1', 'NHWCD4 = 2', 'NCHW4 = 3', 'NCHW8 = 4', 'NCHW32 = 5', 'NCHW88 = 6',
          'NCHW44 = 7','NCHW44_DOT = 8',
          Doc('NCHW_WINOGRAD = 9', 'NCHW layout with weights tranformed by winograd'),
          Doc('NCHW88_WINOGRAD = 10', 'NCHW88 layout with weights tranformed by winograd'),
          Doc('NCHW44_WINOGRAD = 11', 'NCHW44 layout with weights tranformed by winograd'),
          Doc('NCHW4_NCHW32 = 12', 'NCHW4_NCHW32 means input tensors are nchw4 layout, output tensor is nchw32 layout'),
          Doc('NCHW32_NCHW4 = 13', 'NCHW32_NCHW4 means input tensors are nchw32 layout, output tensor is nchw4 layout'),
          Doc('NCHW4_NCHW = 14', 'NCHW4_NCHW means input tensors are nchw4 layout, output tensor is nchw layout'),
          Doc('NHWC_NCHW = 15', 'NHWC_NCHW means input tensors are nhwc layout, '
              'output tensor is nchw layout'),
          Doc('NHWC_NCHW4_IC_SMALL = 16', 'NHWC_NCHW4_IC_SMALL means input tensors are nhwc(c < 4) layout, '
              'output tensor is nchw4 layout, padding c=4'),
          Doc('NCHW_NCHW4_IC_SMALL = 17', 'NCHW_NCHW4_IC_SMALL means input tensors are nchw(c < 4) layout, '
              'output tensor is nchw4 layout, padding c=4'),
          Doc('CHWN4 = 18', 'CHWN4 is currently only used on Nvidia platform for fast implementation '
              'of convolution using CUDA/SASS. The channels are splitted to groups of 4 channels.'),
          Doc('NCHW4_NHWC = 19', 'NCHW4_NHWC means input tensors are nchw4 layout, output tensor is nhwc layout'))
 )

(pdef('Convolution', version=1, is_legacy=True).
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
          Doc('DEFAULT = 0', 'No special requirements on the precision of '
                         'intermediate results.'),
          Doc('FLOAT32 = 1', 'Use Float32 accumulator and intermediate result. '
                         'Only supported when input and output is Float16.'),
          name_field='compute_mode')
 )

(pdef('Convolution', version=2).
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
 add_enum(Doc('Format', 'convolution data/filter/output format; see '
              ':class:`RelayoutFormat` for more details'),
          'NCHW = 0', 'NHWC = 1', 'NHWCD4 = 2', 'NCHW4 = 3', 'NCHW8 = 4', 'NCHW32 = 5', 'NCHW88 = 6',
          'NCHW44 = 7','NCHW44_DOT = 8',
          Doc('NCHW4_NCHW32 = 9', 'NCHW4_NCHW32 means input tensors are nchw4 layout, output tensor is nchw32 layout'),
          Doc('NCHW32_NCHW4 = 10', 'NCHW32_NCHW4 means input tensors are nchw32 layout, output tensor is nchw4 layout'),
          Doc('NCHW4_NCHW = 11', 'NCHW4_NCHW means input tensors are nchw4 layout, output tensor is nchw layout'),
          Doc('NHWC_NCHW = 12', 'NHWC_NCHW means input tensors are nhwc layout, '
              'output tensor is nchw layout'),
          Doc('NHWC_NCHW4_IC_SMALL = 13', 'NHWC_NCHW4_IC_SMALL means input tensors are nhwc(c < 4) layout, '
              'output tensor is nchw4 layout, padding c=4'),
          Doc('NCHW_NCHW4_IC_SMALL = 14', 'NCHW_NCHW4_IC_SMALL means input tensors are nchw(c < 4) layout, '
              'output tensor is nchw4 layout, padding c=4'),
          Doc('CHWN4 = 15', 'CHWN4 is currently only used on Nvidia platform for fast implementation '
              'of convolution using CUDA/SASS. The channels are splitted to groups of 4 channels.'),
          Doc('NCHW64 = 16', 'NCHW64 is designed for convolution implementation to utilizing TensorCore '
              'instructions for 4-bit integers on Nvidia platforms'),
          Doc('NCHW4_NHWC = 17', 'NCHW4_NHWC means input tensors are nchw4 layout, output tensor is nhwc layout')).
 add_enum_alias('ComputeMode', 'ConvolutionV1',name_field='compute_mode')
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
 add_enum('Method', 'WITH_TEXTURE_OBJ = 0', 'WITH_SHARED_MEM = 1').
 add_enum_alias('ConvMode', 'ConvolutionV0', 'Mode').
 add_enum('PoolMode', 'AVERAGE = 0', 'MAX = 1').
 add_enum('NonlineMode', 'IDENTITY = 0', 'RELU = 1', 'SIGMOID = 2').
 add_fields('uint32', 'pool_shape_h', 1, 'pool_shape_w', 1, 'pool_stride_h', 1, 'pool_stride_w', 1, \
  'pool_pad_h', 0, 'pool_pad_w', 0, 'conv_stride_h', 1, 'conv_stride_w', 1, 'conv_pad_h', 0, 'conv_pad_w', 0))

(pdef('ConvBias', 'legacy conv_bias', version=0, is_legacy=True).
 add_enum('NonlineMode', 'IDENTITY = 0', 'RELU = 1', 'SIGMOID = 2', 'H_SWISH = 3').
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
 add_enum_alias('ComputeMode', 'ConvolutionV1', name_field='compute_mode')
 )

(pdef('ConvBias', 'active(conv(x, w) + bias)', version=3, is_legacy=True).
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
 add_enum_alias('ComputeMode', 'ConvolutionV1', name_field='compute_mode')
 )

(pdef('ConvBias', 'active(conv(x, w) + bias)', version=4).
 add_enum_alias('NonlineMode', 'ConvBiasV0').
 add_enum_alias('Mode', 'ConvolutionV0').
 add_enum_alias('Sparse', 'ConvolutionV0').
 add_enum_alias('Format', 'Convolution').
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
 add_enum_alias('ComputeMode', 'ConvolutionV1', name_field='compute_mode')
 )
(pdef('SeparableConv').
 add_enum_alias('Mode', 'ConvolutionV0').
 add_enum('BorderMode', 'BORDER_REPLICATE = 0', 'BORDER_REFLECT = 1',
          'BORDER_REFLECT_101 = 2','BORDER_WRAP = 3',
          'BORDER_CONSTANT = 4', 'BORDER_TRANSPARENT = 5','BORDER_ISOLATED = 6').
 add_fields('bool', 'is_symm_kernel', 'true').
 add_fields('uint32', 'pad_h', 0, 'pad_w', 0, 'stride_h', 1, 'stride_w', 1,
            'ksize_h', 3, 'ksize_w', 3, 'anchor_h', 1, 'anchor_w', 1))

(pdef('Images2Neibs').
 add_fields('uint32', 'pad_h', 0, 'pad_w', 0, 'stride_h', 1, 'stride_w', 1,
            'dilate_h', 1, 'dilate_w', 1, 'window_h', 3, 'window_w', 3))

(pdef('SlidingWindowTranspose').
 add_fields('uint32', 'out_h', 0, 'out_w', 0, 'pad_h', 0, 'pad_w', 0, 'stride_h', 1, 'stride_w', 1,
            'dilate_h', 1, 'dilate_w', 1, 'window_h', 3, 'window_w', 3))

(pdef('Pooling', version=0, is_legacy=True).
 add_enum(
     'Mode',
     Doc('MAX = 0', 'maximum value inside pooling window'),
     Doc('AVERAGE = 1',
         'arithmetic mean of all values inside pooling window. Padding values '
         'are taken into account and are viewed as zero'),
     Doc('AVERAGE_COUNT_EXCLUDE_PADDING = 2',
         'arithmetic mean of all values inside pooling window. No padding is'
         'used.')
 ).
 add_fields('uint32', 'pad_h', 0, 'pad_w', 0, 'stride_h', 2, 'stride_w', 2,
            'window_h', 2, 'window_w', 2).
 add_enum_alias('Format', 'ConvolutionV0')
 )

(pdef('Pooling', version=1).
 add_enum_alias('Mode','PoolingV0').
 add_fields('uint32', 'pad_h', 0, 'pad_w', 0, 'stride_h', 2, 'stride_w', 2,
            'window_h', 2, 'window_w', 2).
 add_enum_alias('Format', 'Convolution')
 )

(pdef('Softmax').
 add_fields('int32', 'axis', -1)
)

(pdef('AdaptivePooling', version=0, is_legacy=True).
 add_enum_alias('Mode', 'PoolingV0').
 add_enum_alias('Format', 'ConvolutionV0')
 )

(pdef('AdaptivePooling', version=1).
 add_enum_alias('Mode', 'PoolingV0').
 add_enum_alias('Format', 'Convolution')
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
     Doc('DIM_11HW = 0', 'Dim of params (Sigma, Mu) is 1 x 1 x H x W'),
     Doc('DIM_1CHW = 1', 'Dim of params (Sigma, Mu) is 1 x C x H x W'),
     Doc('DIM_1C11 = 2', 'Dim of params (Sigma, Mu) is 1 x C x 1 x 1'),
     Doc('DIM_111C = 3', 'Dim of params (Sigma, Mu) is 1 x 1 x 1 x C'),
     name_field='param_dim'
 ).
 add_enum(
     'FwdMode',
     Doc('TRAINING = 0', 'Training phase.'),
     Doc('INFERENCE = 1', 'Inference phase.'),
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
     Doc('MAX = 0', 'maximum value inside pooling window; pooling result would '
         'be 0 if pooling window is empty'),
     Doc('AVERAGE = 1',
         'arithmetic mean of all values inside pooling window; pooling result '
         'would be 0 if pooling window is empty')
 ).
 add_fields('float32', 'scale', '1.f'))

INTERP_MODES = ['NEAREST = 0', 'LINEAR = 1', 'AREA = 2', 'CUBIC = 3', 'LANCZOS4 = 4']
BORDER_MODES = [Doc('REPLICATE = 0', 'aaaaaa|abcdefgh|hhhhhhh'),
                Doc('REFLECT = 1', 'fedcba|abcdefgh|hgfedcb'),
                Doc('REFLECT_101 = 2', 'gfedcb|abcdefgh|gfedcba'),
                Doc('WRAP = 3', 'cdefgh|abcdefgh|abcdefg'),
                Doc('CONSTANT = 4', 'iiiiii|abcdefgh|iiiiiii'),
                Doc('TRANSPARENT = 5', ''),
                Doc('ISOLATED = 6', '')]
(pdef('WarpPerspective', version=1, is_legacy=True).
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

(pdef('WarpPerspective', version=2).
 add_enum_alias('InterpolationMode','WarpPerspectiveV1',name_field="imode").
 add_enum_alias('BorderMode','WarpPerspectiveV1',name_field="bmode").
 add_enum_alias('Format', 'Convolution').
 add_fields('float32', Doc('border_val', 'used for CONSTANT bmode'), '.0f'))


pdef('SpatialTfGridGenerator').add_enum('Mode', 'AFFINE = 0')
pdef('SpatialTfSampler').add_enum('Mode', 'BILINEAR = 0')

pdef('AddUpdate').add_fields(
    'float32', 'alpha', '1.f', 'beta', '1.f', 'bias', '0.f')

pdef('Elemwise').add_enum(
    'Mode',
    Doc('RELU = 0', 'unary: max(x, 0)'),
    Doc('ABS = 1', 'unary: abs(x)'),
    Doc('ACOS = 2', 'unary: acos(x)'),
    Doc('ASIN = 3', 'unary: asin(x)'),
    Doc('CEIL = 4', 'unary: ceil(x)'),
    Doc('COS = 5', 'unary: cos(x)'),
    Doc('EXP = 6', 'unary: exp(x)'),
    Doc('EXPM1 = 7', 'unary: numerically stable exp(x)-1'),
    Doc('FLOOR = 8', 'unary: floor(x)'),
    Doc('LOG = 9', 'unary: natural logarithm, log(x)'),
    Doc('LOG1P = 10', 'unary: numerically stable log(x+1)'),
    Doc('NEGATE = 11', 'unary: -x'),
    Doc('SIGMOID = 12', 'unary: 1/(1+exp(-x))'),
    Doc('SIN = 13', 'unary: sin(x)'),
    Doc('TANH = 14', 'unary: tanh(x)'),

    Doc('ABS_GRAD = 15', 'binary: x > 0 ? y : -y'),
    Doc('ADD = 16', 'binary: x + y'),
    Doc('FLOOR_DIV = 17', 'binary: floor(x / y)'),
    Doc('MAX = 18', 'binary: max(x, y)'),
    Doc('MIN = 19', 'binary: min(x, y)'),
    Doc('MOD = 20', 'binary: x % y or fmodf(x, y)'),
    Doc('MUL = 21', 'binary: x * y'),
    Doc('POW = 22', 'binary: pow(x, y)'),
    Doc('SIGMOID_GRAD = 23', 'binary: x * (1 - x) * y'),
    Doc('SUB = 24', 'binary: x - y'),
    Doc('SWITCH_GT0 = 25', 'binary: (x > 0) * y'),
    Doc('TANH_GRAD = 26', 'binary: (1 - x * x) * y'),
    Doc('TRUE_DIV = 27', 'binary: x / y'),
    Doc('LOG_SUM_EXP = 28', 'binary: numerically stable log(exp(x) + exp(y))'),

    Doc('LT = 29', 'binary: x < y'),
    Doc('LEQ = 30', 'binary: x <= y'),
    Doc('EQ = 31', 'binary: x == y'),

    Doc('SHL = 32', 'bitwise binary: x << y. '
        'Note that result is undefined if y < 0 or y >= bitwidth. Logical '
        'shift is performed for unsigned intergers, and arithmetic shift for '
        'signed ones.'),
    Doc('SHR = 33', 'bitwise binary: x >> y; see SHL mode for more details'),

    Doc('COND_LEQ_MOV = 34', 'ternary: x <= y ? z : 0'),
    Doc('FUSE_MUL_ADD3 = 35',
        'compute ``a * b + c`` where c must either have same layout as '
        'a or b, or be a scalar'),

    Doc('FUSE_MUL_ADD4 = 36',
        'compute ``a * A + b * B`` where a and b must have equal layout, '
        'and A and B must have equal layout. In the inputs ``b`` and ``B`` '
        'can be swapped'),

    Doc('FUSE_ADD_RELU = 37', 'binary: max(x+y, 0)'),
    Doc('FUSE_ADD_SIGMOID = 38', 'binary: 1/(1+exp(-(x+y)))'),
    Doc('FUSE_ADD_TANH = 39', 'binary: tanh(x+y)'),
    Doc('FAST_TANH = 40', 'unary: rational approximation of tanh(x)'),
    Doc('FAST_TANH_GRAD = 41', 'binary: grad of the rational approximation of tanh(x)'),

    Doc('ROUND = 42', 'unary: round(x), the nearest integer value to x, rounding '
                 'halfway cases away from zero. Float only.'),
    Doc('RMULH = 43', 'binary: rounded higher l bits of x * y, where l is the bit '
                'length of x.'),

    Doc('ATAN2 = 44','binary: atan2(y,x)'),
    Doc('ERF = 45', 'unary: erf(x)'),
    Doc('ERFINV = 46', 'unary: inverse function of erf(x)'),
    Doc('ERFC = 47', 'unary: erfc(x)'),
    Doc('ERFCINV = 48', 'unary: inverse function of erfc(x)'),
    Doc('H_SWISH = 49', 'unary: x * clip(x + 3, 0, 6) / 6'),
    Doc('H_SWISH_GRAD = 50', 'binary: x < -3 ? 0 : (x > 3 ? y : (2 * x + 3) / 6 * y)'),
    Doc('FUSE_ADD_H_SWISH = 51', 'binary: hswish(x+y)'),

    Doc('NOT = 52', 'unary: !x'),
    Doc('AND = 53', 'binary: x && y'),
    Doc('OR = 54', 'binary: x || y'),
    Doc('XOR = 55', 'binary: x ^ y'),
    Doc('SILU = 56', 'unary: x / (1 + exp(-x))'),
    Doc('SILU_GRAD = 57', 'binary: grad(x / (1 + exp(-x))'),
    Doc('GELU = 58', 'unary: x Phi(x)'),
    Doc('GELU_GRAD = 59', 'binary: grad(x Phi(x))'),
    Doc('COND_LT_MOV = 60', 'ternary: x < y ? z : 0'),
    Doc('NEQ = 61', 'binary: x != y'),
    Doc('ISNAN = 62', 'unary: isnan(x)'),
    Doc('ISINF = 63', 'unary: isinf(x)'),
    Doc('SINH = 64', 'unary: sinh(x)'),
    Doc('COSH = 65', 'unary: cosh(x)'),
    Doc('ASINH = 66', 'unary: asinh(x)'),
    Doc('ACOSH = 67', 'unary: acosh(x)'),
    Doc('ATANH = 68', 'unary: atanh(x)'),
    Doc('TAN = 69', 'unary: tan(x)'),
    Doc('ASINH_GRAD = 70', 'binary: y / sqrt(x^2 + 1)'),
    Doc('ACOSH_GRAD = 71', 'binary: y / sqrt(x^2 - 1) (x > 1)'),
    Doc('ATANH_GRAD = 72', 'binary: y / (1 - x^2) (|x| < 1)'),
    Doc('PRELU = 73', 'binary: x > 0 ? x : x * y'),
    Doc('CLIP = 74', 'ternary: x <= y ? y : (x <= z ? x : z)'),
    Doc('PRELU_GRAD = 75', 'ternary: x > 0 ? y : y * z'),
    Doc('SOFTPLUS = 76', 'unary: log(1 + e^x)'),
    Doc('SOFTPLUS_GRAD = 77', 'binary: y * e^x / (1 + e^x)'),
    Doc('RELU6 = 78', 'unary: min(max(0, x), 6)'),
    Doc('RELU6_GRAD = 79', 'binary: x < 0 ? 0 : (x > 6 ? 0 : y)'),
    Doc('HSIGMOID = 80', 'unary: relu6(x + 3) / 6'),
    Doc('HSIGMOID_GRAD = 81', 'binary: x < -3 ? 0 : (x > 3 ? 0 : y / 6)'),
    Doc('LOGSIGMOID = 82', 'unary: -log(1 + e^(-x))'),
    Doc('SQRT = 83', 'unary: x^(1/2)'),
    Doc('SQUARE = 84', 'unary: x^2'),
    Doc('SIGN = 85', 'unary: sgn(x)'),
)

pdef('ElemwiseMultiType').add_enum(
    'Mode',
    Doc('FUSE_MUL_ADD3_INT16x32x32x32 = 0',
        'compute ``a * b + c`` requiring that ``a`` be int16 and ``b`` and '
        '``c``  int32, and the result is int32. This mode is optimized for '
        'the channel-broadacsted case, i.e. ``a`` has shape (A, B, C) and '
        '``b`` and ``c`` have shape (1, C, 1)'),
    Doc('FUSE_MUL_ADD3_IXxF32xF32xI8 = 1',
        'compuate ``a * b + c`` where the inputs ``a`` is an integer type '
        '``b`` and ``c`` are both ``float32``, the result is '
        '``int8``. This is currently only optimized for ``(1, x)`` '
        'broadcast for ``b`` and ``c``. Computation is carried in floating '
        'points and results are rounded towards zero with saturated cast to '
        'int.'),
    Doc('ROUND_SHR_SATURATE_IXxI8xI8 = 2',
        'Compute ``a >> b``, round the result according to lower ``b`` bits '
        'of ``a``` and make a saturating conversion to int8. Where ``a`` should'
        ' be an integer tensor and ``b`` should be an int8 scalar.'),
    Doc('FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8 = 3',
        'Fused operation of an int16 elemwise add, an int16 rounding multiply '
        'high and an int16 to int8 rounding right shift with saturation.'),
    Doc('FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8 = 4',
        'Fused operation of an int32 elemwise add, an int32 rounding multiply '
        'high and an int32 to int8 rounding right shift with saturation.'),
    Doc('ROUND_SHR_SATURATE_IXxI8xI16 = 5',
        'Compute ``a >> b``, round the result according to lower ``b`` bits of '
        '``a``` and make a saturating conversion to int16. Where ``a`` should'
        ' be an integer tensor and ``b`` should be an int8 scalar.'),
    Doc('QADD = 6', 'Fused elemwise add two quantized int8 with specified'
        'output quantized dtype'),
    Doc('QFUSE_ADD_RELU = 7', 'Fused elemwise add two quantized int8 followed'
         ' by ReLU and typecvt to specified dtype'),
    Doc('QMUL = 8', 'Fused elemwise multiply two quantized int8 with specified'
        'output quantized dtype'),
    Doc('QMIN = 9', 'Fused elemwise min two quantized int8 with specified'
        'output quantized dtype'),
    Doc('QMAX = 10', 'quantized: max(x, y), with specified output quantized dtype'),
    Doc('QSUB = 11', 'quantized: x - y'),
    Doc('QTRUE_DIV = 12', 'quantized: x / y'),
    Doc('QFUSE_ADD_SIGMOID = 13', 'quantized: sigmoid(x + y)'),
    Doc('QFUSE_ADD_TANH = 14', 'quantized: tanh(x + y)'),
    Doc('QRELU = 15', 'quantized: x > 0 ? x : 0'),
    Doc('QABS = 16', 'quantized: x > 0 ? x : -x'),
    Doc('QSIGMOID = 17', 'quantized: sigmoid(x)'),
    Doc('QEXP = 18', 'quantized: exp(x)'),
    Doc('QTANH = 19', 'quantized: tanh(x)'),
    Doc('QFUSE_MUL_ADD3 = 20', 'quantized: x * y + z'),
    Doc('QFAST_TANH = 21', 'quantized: fast_tanh(x)'),
    Doc('QNEGATE = 22', 'quantized: -x'),
    Doc('QACOS = 23', 'quantized: acos(x)'),
    Doc('QASIN = 24', 'quantized: asin(x)'),
    Doc('QCEIL = 25', 'quantized: ceil(x)'),
    Doc('QCOS = 26', 'quantized: cos(x)'),
    Doc('QEXPM1 = 27', 'quantized: expm1(x)'),
    Doc('QFLOOR = 28', 'quantized: floor(x)'),
    Doc('QLOG = 29', 'quantized: log(x)'),
    Doc('QLOG1P = 30', 'quantized: log1p(x)'),
    Doc('QSIN = 31', 'quantized: sin(x)'),
    Doc('QROUND = 32', 'quantized: round(x)'),
    Doc('QERF = 33', 'quantized: erf(x)'),
    Doc('QERFINV = 34', 'quantized: erfinv(x)'),
    Doc('QERFC = 35', 'quantized: erfc(x)'),
    Doc('QERFCINV = 36', 'quantized: erfcinv(x)'),
    Doc('QABS_GRAD = 37', 'quantized: abs_grad'),
    Doc('QFLOOR_DIV = 38', 'quantized floor_div'),
    Doc('QMOD = 39', 'quantized mod'),
    Doc('QSIGMOID_GRAD = 40', 'quantized sigmoid_grad'),
    Doc('QSWITCH_GT0 = 41', 'quantized switch_gt0'),
    Doc('QTANH_GRAD = 42', 'quantized tanh_grad'),
    Doc('QLT = 43', 'quantized lt'),
    Doc('QLEQ = 44', 'quantized leq'),
    Doc('QEQ = 45', 'quantized eq'),
    Doc('QPOW = 46', 'quantized pow'),
    Doc('QLOG_SUM_EXP = 47', 'quantized log_sum_exp'),
    Doc('QFAST_TANH_GRAD = 48', 'quantized fast_tanh_grad'),
    Doc('QATAN2 = 49', 'quantized atan2'),
    Doc('QCOND_LEQ_MOV = 50', 'quantized cond_leq_mov'),
    Doc('QH_SWISH = 51', 'quantized h_swish'),
    Doc('QFUSE_ADD_H_SWISH = 52', 'quantized h_swish(x+y)'),
    Doc('QH_SWISH_GRAD = 53', 'quantized h_swish_grad'),
    Doc('FUSE_MUL_ADD3_INT16xF32xF32xF32 = 54',
        'compute ``a * b + c`` requiring that ``a`` be int16 and ``b`` and '
        '``c``  float32, and the result is float32.'),
    Doc('MUL_INT16xF32xF32 = 55',
        'compute ``a * b `` requiring that ``a`` be int16 and ``b`` float32, '
        'and the result is float32.'),
    Doc('FUSE_MUL_ADD3_UINT8xF32xF32xF32 = 56',
        'compute ``a * b + c`` requiring that ``a`` be uint8 and ``b`` and '
        '``c``  float32, and the result is float32.'),
    Doc('QCOND_LT_MOV = 57', 'quantized cond_lt_mov'),
    Doc('EQ = 58', 'eq'),
    Doc('NEQ = 59', 'eq'),
    Doc('LT = 60', 'lt'),
    Doc('LEQ = 61', 'leq'),
    Doc('ISNAN = 62', 'isnan'),
    Doc('ISINF = 63', 'isinf')
)

pdef('PowC', 'power with constant exponent').add_fields('float32', 'exp', 0)

(pdef('DctChannelSelect', '2d discrete cosine transform', version=0, is_legacy=True).add_enum_alias('Format', 'ConvolutionV0').
 add_enum('FastImpl', 'NONE = 0', 'FIX_32_MASK = 1').add_fields('int32', 'dct_block_size', 8))

(pdef('DctChannelSelect', '2d discrete cosine transform', version=1).add_enum_alias('Format', 'Convolution').
 add_enum_alias('FastImpl', 'DctChannelSelectV0').add_fields('int32', 'dct_block_size', 8))

(pdef('MatrixMul', version=0, is_legacy=True).
 add_fields('bool', 'transposeA', 'false', 'transposeB', 'false').
 add_enum('DataType',
     Doc('FLOAT = 0', 'input/output both float32/float16'),
     'INT8x8x16 = 1',
     'INT8x8x32 = 2',
     Doc('FLOAT_IO16xC32 = 3', 'input/output both float16, the internal compute is '
         'float32'),
     Doc('QUINT8x8x32 = 4', 'input QuantizedAsymm8, output QuantizedS32'),
     Doc('QUINT4x4x32 = 5', 'input QuantizedAsymm4, output QuantizedS32'),
     name_field='data_type'))

(pdef('MatrixMul', version=1, is_legacy=True).
 add_fields('bool', 'transposeA', 'false', 'transposeB', 'false').
 add_enum(Doc('ComputeMode', 'Specifies special computation modes, e.g. '
                             'different combinations of intermediate result '
                             'data types.'),
          Doc('DEFAULT = 0', 'No special requirements on the precision of '
                         'intermediate results.'),
          Doc('FLOAT32 = 1', 'Use Float32 accumulator and intermediate result. '
                         'Only supported when input and output is Float16.'),
          name_field='compute_mode'))

(pdef('MatrixMul', version=2).
 add_fields('bool', 'transposeA', 'false', 'transposeB', 'false').
 add_enum_alias('ComputeMode', 'MatrixMulV1', name_field='compute_mode').
 add_enum('Format',
          Doc('DEFAULT = 0', 'Normal matrix mul: (M, K) x (K, N) = (M, N)'),
          Doc('MK4 = 1', 'Split 4 from M and K, better for neon compute:'
              '(M/4, K/4, 4(k), 4(m)) x (K/4, N, 4(k)). if transposeA the '
              'layout is (K/4, M/4, 4(k), 4(m)) x (K/4, N, 4(k))'),
          Doc('MK8 = 2', 'Split 8 from M and K, better for neon compute:'
              '(M/8, K/8, 8(k), 8(m)) x (K/8, N, 8(k)). if transposeA the '
              'layout is (K/8, M/8, 8(k), 8(m)) x (K/8, N, 8(k))'),
          Doc('MK4_DOT = 3', 'Split 4 from M and K, better for neon dotprod:'
              'M/4, K/4, 4(m), 4(k)) x (K/4, N, 4(k)). if transposeA the '
              'layout is (K/4, M/4, 4(m), 4(k)) x (K/4, N, 4(k))'),
          Doc('N32K4_DOT = 4', 'Split 32 from N and 4 from K, better for neon gevm dotprod:'
              'N/32, K/4, 32(n), 4(k)')
              )
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
          'SUM = 0',
          Doc('SUM_SQR = 1', 'sum of x * x for each element x'),
          'PRODUCT = 2', 'MIN = 3', 'MAX = 4').
 add_fields('int32',
            Doc('axis',
                'axis along which reduction is performed; if -1 is given, '
                'reduce to given target shape (only used in megbrain)'),
            -1))

(pdef('Reduce', 'reduce along given axis', version=1, is_legacy=True).
 add_enum('Mode',
          'SUM = 0',
          Doc('SUM_SQR = 1', 'sum of x * x for each element x'),
          'PRODUCT = 2', 'MIN = 3', 'MAX = 4', 'MEAN = 5').
 add_fields('int32',
            Doc('axis',
                'axis along which reduction is performed; if -1 is given, '
                'reduce to given target shape (only used in megbrain)'),
            -1).
 add_enum('DataType',
          Doc('DEFAULT = 0',
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
          Doc('FLOAT_IO16xC32 = 1', 'Deprecated. This was replaced by '
              'FLOAT_O16xC32, and input\'s dtype decided by actual input tensor.'),
          Doc('FLOAT_O32xC32 = 2', 'compute/output both are float32'),
          Doc('FLOAT_O16xC32 = 3', 'compute are float32, output float16'),
          Doc('QUINT_I8xO32 = 4', 'input quint8, compute and output are qint32'),
          Doc('QINT_I8xO32 = 5', 'input qint8, compute and output are qint32'),
     name_field='data_type'))

(pdef('Reduce', 'reduce along given axis', version=2).
 add_enum('Mode',
          'SUM = 0',
          Doc('SUM_SQR = 1', 'sum of x * x for each element x'),
          'PRODUCT = 2', 'MIN = 3', 'MAX = 4', 'MEAN = 5').
 add_fields('int32',
            Doc('axis',
                'axis along which reduction is performed; if INT_MAX is given, '
                'reduce to given target shape (only used in megbrain)'),
            (1<<31)-1).
 add_enum('DataType',
          Doc('DEFAULT = 0',
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
          Doc('FLOAT_IO16xC32 = 1', 'Deprecated. This was replaced by '
              'FLOAT_O16xC32, and input\'s dtype decided by actual input tensor.'),
          Doc('FLOAT_O32xC32 = 2', 'compute/output both are float32'),
          Doc('FLOAT_O16xC32 = 3', 'compute are float32, output float16'),
          Doc('QUINT_I8xO32 = 4', 'input quint8, compute and output are qint32'),
          Doc('QINT_I8xO32 = 5', 'input qint8, compute and output are qint32'),
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
          Doc('EQ = 0', 'take if ``abs(data-val)<eps``'),
          Doc('NEQ = 1', 'take if ``abs(data-val)>=eps``'),
          Doc('LT = 2', 'take if ``data<val``'),
          Doc('LEQ = 3', 'take if ``data<=val``'),
          Doc('GT = 4', 'take if ``data>val``'),
          Doc('GEQ = 5', 'take if ``data>=val``')).
 add_fields('float32',
            Doc('val', 'the value to be compared with; note that for integer '
                'data, val is also converted to int'), 0).
 add_fields('float32', Doc('eps', 'used for float equality comparison'),
            1e-6))


pdef('Argsort').add_enum('Order', 'ASCENDING = 0', 'DESCENDING = 1')

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

(pdef('Diag').
 add_fields(
     'int32',
     Doc('k', 'Index of the diagonal: 0 (the default) refers to the main '
         'diagonal, a positive value refers to an upper diagonal, and a '
         'negative value to a lower diagonal.'),
     0))

(pdef('Cross').
 add_fields(
     'int32',
     Doc('axisa', 'axis of a that defines the vector(s). By default, the last axis.'),
         '-1').
 add_fields(
     'int32',
     Doc('axisb', 'axis of b that defines the vector(s). By default, the last axis.'),
         '-1').
 add_fields(
     'int32',
     Doc('axisc', 'axis of c containing the cross product vector(s). Ignored if both '
         'input vectors have dimension 2, as the return is scalar. By default, the '
         'last axis.'),
         '-1')
)

(pdef('UniformRNG', version=0, is_legacy=True).
 add_fields('uint64', 'seed', 0))

(pdef('UniformRNG', version=1).
 add_fields('uint64', 'seed', 0).
 add_fields(
     'dtype', Doc('dtype', 'The dtype of output Tensor. Only support Float32.'),
     'DTypeEnum::Float32'))

(pdef('GaussianRNG', version=0, is_legacy=True).
 add_fields('uint64', 'seed', 0).
 add_fields('float32', 'mean', 0, 'std', 1))

(pdef('GaussianRNG', version=1).
 add_fields('uint64', 'seed', 0).
 add_fields('float32', 'mean', 0, 'std', 1).
 add_fields(
     'dtype', Doc('dtype', 'The dtype of output Tensor. Only support Float32.'),
     'DTypeEnum::Float32'))

(pdef('GammaRNG').
 add_fields('uint64', 'seed', 0))

(pdef('BetaRNG').
 add_fields('uint64', 'seed', 0))

(pdef('PoissonRNG').
 add_fields('uint64', 'seed', 0))

(pdef('PermutationRNG').
 add_fields('uint64', 'seed', 0).
 add_fields(
     'dtype', Doc('dtype', 'The dtype of output Tensor. Int32, Int16 and '
                  'Float32 are supported.'),
     'DTypeEnum::Int32'))

(pdef('ShuffleRNG').
 add_fields('uint64', 'seed', 0))

(pdef('ExponentialRNG').
 add_fields('uint64', 'seed', 0))

(pdef('Flip').
 add_fields('bool', 'vertical', 'false', 'horizontal', 'false'))

(pdef('Rotate')
 .add_fields('bool', 'clockwise', 'true'))

(pdef('ROICopy')
 .add_fields('uint32', 'row_from', 0, 'row_to', 0, 'col_from', 0, 'col_to', 0))

(pdef('CvtColor')
 .add_enum('Mode', 'RGB2GRAY = 0', 'RGB2YUV = 1', 'YUV2RGB = 2', 'GRAY2RGB = 3', 'RGBA2RGB = 4',
    'RGBA2BGR = 5', 'RGBA2GRAY = 6', 'RGB2BGR = 7', 'BGR2GRAY = 8', 'BGR2RGB = 9',
    Doc('YUV2GRAY_NV21 = 10', 'For historical reasons, referred to as YCC by opencv'),
    'YUV2RGB_NV21 = 11', 'YUV2BGR_NV21 = 12', 'YUV2GRAY_NV12 = 13', 'YUV2RGB_NV12 = 14',
    'YUV2BGR_NV12 = 15', 'YUV2GRAY_YV12 = 16', 'YUV2RGB_YV12 = 17', 'YUV2BGR_YV12 = 18',
    'YUV2GRAY_YU12 = 19', 'YUV2RGB_YU12 = 20', 'YUV2BGR_YU12 = 21',
    'YCrCb2RGB = 22', 'YCrCb2BGR = 23',
    Doc('BT601_YUV2RGB_NV21 = 24', 'BT601 yuv format, referred to as YUV by opencv'),
    'BT601_YUV2BGR_NV21 = 25', 'BT601_YUV2RGB_NV12 = 26', 'BT601_YUV2BGR_NV12 = 27',
    'BT601_YUV2RGB_YV12 = 28', 'BT601_YUV2BGR_YV12 = 29', 'BT601_YUV2RGB_YU12 = 30',
    'BT601_YUV2BGR_YU12 = 31',
    member_alias=[('YUV2GRAY_NV21', 'BT601_YUV2GRAY_NV21'),
                  ('YUV2GRAY_NV12', 'BT601_YUV2GRAY_NV12'),
                  ('YUV2GRAY_YV12', 'BT601_YUV2GRAY_YV12'),
                  ('YUV2GRAY_YU12', 'BT601_YUV2GRAY_YU12')],
    name_field = 'mode'))

(pdef('WarpAffine', version=0, is_legacy=True)
 .add_enum_alias('InterpolationMode', 'WarpPerspectiveV1', name_field='imode')
 .add_enum_alias('BorderMode', 'WarpPerspectiveV1', name_field='border_mode')
 .add_fields('float32', Doc('border_val', 'used for CONSTANT bmode'), '.0f'))

(pdef('WarpAffine', version=1, is_legacy=True)
 .add_enum_alias('InterpolationMode', 'WarpPerspectiveV1', name_field='imode')
 .add_enum_alias('BorderMode', 'WarpPerspectiveV1', name_field='border_mode')
 .add_fields('float32', Doc('border_val', 'used for CONSTANT bmode'), '.0f')
 .add_enum_alias('Format', 'ConvolutionV0', default=1))

(pdef('WarpAffine', version=2)
 .add_enum_alias('InterpolationMode', 'WarpPerspectiveV1', name_field='imode')
 .add_enum_alias('BorderMode', 'WarpPerspectiveV1', name_field='border_mode')
 .add_fields('float32', Doc('border_val', 'used for CONSTANT bmode'), '.0f')
 .add_enum_alias('Format', 'Convolution', default=1))


(pdef('GaussianBlur')
 .add_enum_alias('BorderMode', 'WarpPerspectiveV1', name_field='border_mode')
 .add_fields('uint32', 'kernel_height', 0, 'kernel_width', 0)
 .add_fields('float32','sigma_x', '0.f', 'sigma_y', '0.f'))

(pdef('Resize', version=0, is_legacy=True)
 .add_enum_alias('InterpolationMode', 'WarpPerspectiveV1', name_field='imode'))

(pdef('Resize', version=1, is_legacy=True)
 .add_enum_alias('InterpolationMode', 'WarpPerspectiveV1', name_field='imode')
 .add_enum_alias('Format', 'ConvolutionV0', default=1))

(pdef('Resize', version=2)
 .add_enum_alias('InterpolationMode', 'WarpPerspectiveV1', name_field='imode')
 .add_enum_alias('Format', 'Convolution', default=1))

(pdef('Remap', version=0,is_legacy=True)
 .add_enum_alias('InterpolationMode', 'WarpPerspectiveV1', name_field='imode')
 .add_enum_alias('BorderMode', 'WarpPerspectiveV1', name_field='border_type')
 .add_enum_alias('Format', 'ConvolutionV0', default=1)
 .add_fields('float32', 'scalar', '0.f'))

(pdef('Remap', version=1)
 .add_enum_alias('InterpolationMode', 'WarpPerspectiveV1', name_field='imode')
 .add_enum_alias('BorderMode', 'WarpPerspectiveV1', name_field='border_type')
 .add_enum_alias('Format', 'Convolution', default=1)
 .add_fields('float32', 'scalar', '0.f'))

(pdef('Convolution3D').
 add_enum('Mode', 'CROSS_CORRELATION = 0', 'CONVOLUTION = 1').
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
          Doc('DENSE = 0', 'dense convolution: filter shape should be '
              '[oc, ic, spatial...] if format is NCDHW, '
              '[oc, spatial..., ic] if format is NDHWC'),
          Doc('GROUP = 1', 'group convolution: filter shape should be '
              '[group, oc_per_group, ic_per_group, spatial...] if format is NCDHW, '
              '[group, oc_per_group, spatial..., ic_per_group] if format is NDHWC')
          ).
 add_enum('DataType',
          Doc('FLOAT = 0', 'input/output both float32/float16'),
          Doc('FLOAT_IO16xC32 = 1', 'input/output both float16, the internal '
              'compute is float32'),
          name_field='data_type').
 add_enum('Format', 'NCDHW = 0', 'NDHWC = 1')
 )

(pdef('Conv3DBias').
 add_enum('NonlineMode', 'IDENTITY = 0', 'RELU = 1', 'SIGMOID = 2').
 add_enum_alias('Mode', 'Convolution3D').
 add_fields('uint32', 'pad_d', 0, 'pad_h', 0, 'pad_w', 0,
                'stride_d', 1, 'stride_h', 1, 'stride_w', 0))

(pdef('SeparableConv3D').
 add_enum_alias('Mode', 'Convolution3D').
 add_enum('BorderMode', 'BORDER_REPLICATE = 0', 'BORDER_REFLECT = 1',
          'BORDER_REFLECT_101 = 2','BORDER_WRAP = 3',
          'BORDER_CONSTANT = 4', 'BORDER_TRANSPARENT = 5','BORDER_ISOLATED = 6').
 add_fields('bool', 'is_symm_kernel', 'true').
 add_fields('uint32', 'pad_d', 0, 'pad_h', 0, 'pad_w', 0,
            'stride_d', 0, 'stride_h', 1, 'stride_w', 1,
            'ksize_d', 0, 'ksize_h', 3, 'ksize_w', 3,
            'anchor_d', 0, 'anchor_h', 1, 'anchor_w', 1))

(pdef('Resize3D')
 .add_enum_alias('InterpolationMode', 'WarpPerspectiveV1', name_field='imode')
 .add_enum_alias('Format', 'Convolution3D', default=1)
 .add_fields('bool', 'align_corners', 'false'))

(pdef('TopK').
 add_enum(
     'Mode',
     Doc('KTH_ONLY = 0', "only the value of the k'th element would be computed"),
     Doc('VALUE_IDX_NOSORT = 1',
         'all the top-k values and corresponding indices would be computed; '
         'no order is guaranteed'),
     Doc('VALUE_IDX_SORTED = 2',
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
* ``NCHW64`` layout: ``{N, C/64, H, W, 64}``

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

Note: NCHW_NCHW4_WEIGHT will auto pad oc and ic, you should remove oc in later opr by seting group and oc param with NCHW4_NCHW
"""
(pdef('RelayoutFormat', 'Change the tensor layout format', version=0, is_legacy=True).
 add_enum(
     Doc('Mode', RELAYOUT_FORMAT_MODE_DOC),
     'NHWC_NHWCD4 = 0',
     'NHWCD4_NHWC = 1',
     'NHWC_NHWCD4I = 2',
     'NCHW_NHWCD4 = 3',
     'NCHW_NHWCD4I = 4',
     'NHWCD4I_NCHW = 5',
     'NHWCD4_NCHW = 6',
     'INTER_WEIGHT_DENSE = 7',
     'INTER_WEIGHT_DENSEI = 8',
     'INTER_WEIGHT_GROUP = 9',
     'INTER_WEIGHT_GROUPI = 10',
     'INTER_WEIGHT_CHAN = 11',
     'INTER_WEIGHT_CHANI = 12',
     'INTER_WEIGHT_DENSEI_DOT = 13',
     'INTER_WEIGHT_GROUPI_DOT = 14',
     'NCHW4_CHWN4 = 15',
     'CHWN4_NCHW4 = 16',
     'NCHW_NCHW88_CONV_DENSE_WEIGHT = 17',
     'NCHW_NCHW88_CONV_CHAN_WEIGHT = 18',
     'NCHW_NCHW88_CONV_GROUP_WEIGHT = 19',
     'NCHW_NCHW88 = 20',
     'NCHW88_NCHW = 21',
     'NCHW_NCHW4_IC_SMALL = 22',
     'NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT = 23',
     'NCHW_NCHW4 = 24',
     'NCHW4_NCHW = 25',
     'NCHW_NCHW4_WEIGHT = 26',
     'NCHW_NCHW64 = 27',
     'NCHW64_NCHW = 28',
     'NCHW_NHWC = 29',
     'NHWC_NCHW = 30',
     'NHWCD4I_NHWC = 31',
    )
 )

(pdef('RelayoutFormat', 'Change the tensor layout format', version=1).
    add_enum_alias('Mode', 'RelayoutFormatV0').
    add_fields('uint32', 'oc', '0').
    add_fields('uint32', 'group', '1')
)

(pdef('SeparableFilter', version=0, is_legacy=True).
 add_enum_alias('Format', 'ConvolutionV0').
 add_enum_alias('BorderMode', 'WarpPerspectiveV1').
 add_fields('bool', 'is_symm_kernel', 'true').
 add_fields('uint32', 'ksize_h', 3, 'ksize_w', 3, 'anchor_h', 1, 'anchor_w', 1))

(pdef('SeparableFilter', version=1).
 add_enum_alias('Format', 'Convolution').
 add_enum_alias('BorderMode', 'WarpPerspectiveV1').
 add_fields('bool', 'is_symm_kernel', 'true').
 add_fields('uint32', 'ksize_h', 3, 'ksize_w', 3, 'anchor_h', 1, 'anchor_w', 1))

(pdef('LocalShare', 'Local share convolution',version=0, is_legacy=True).
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
 add_enum_alias('ComputeMode', 'ConvolutionV1')
 )

(pdef('LocalShare', 'Local share convolution', version=1).
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
 add_enum_alias('Format', 'Convolution').
 add_enum_alias('ComputeMode', 'ConvolutionV1')
 )


(pdef('ROIAlign',version=0,is_legacy=True).
 add_enum('Mode', 'MAX = 0', 'AVERAGE = 1', name_field='mode').
 add_enum_alias('Format', 'ConvolutionV0').
 add_fields('float32', 'spatial_scale', '1.0').
 add_fields('float32', 'offset', '0.0').
 add_fields('uint32',
            'pooled_height', '1',
            'pooled_width', '1',
            'sample_height', '2',
            'sample_width', '2')
 )

(pdef('ROIAlign', version=1).
 add_enum_alias('Mode', 'ROIAlignV0', name_field='mode').
 add_enum_alias('Format', 'Convolution').
 add_fields('float32', 'spatial_scale', '1.0').
 add_fields('float32', 'offset', '0.0').
 add_fields('uint32',
            'pooled_height', '1',
            'pooled_width', '1',
            'sample_height', '2',
            'sample_width', '2')
 )

(pdef('Correlation').
 add_enum_alias('Format', 'ConvolutionV0').
 add_fields('uint32', 'kernel_size', '1').
 add_fields('uint32', 'max_displacement', '1').
 add_fields('uint32', 'stride1', '1').
 add_fields('uint32', 'stride2', '1').
 add_fields('uint32', 'pad_size', '0').
 add_fields('bool', 'is_multiply', 'true')
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

(pdef('BatchConvBias', 'Batch convolution (unshare weights on the batch dimension)',version=0,is_legacy=True).
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
 add_enum_alias('ComputeMode', 'ConvolutionV1', name_field="compute_mode")
 )

(pdef('BatchConvBias', 'Batch convolution (unshare weights on the batch dimension)',version=1).
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
 add_enum_alias('Format', 'Convolution').
 add_enum_alias('ComputeMode', 'ConvolutionV1', name_field="compute_mode")
 )

(pdef('FakeQuant').
 add_fields('int32','qmin','-2147483648').
 add_fields('int32','qmax','2147483647')
 )
(pdef('TQT').
 add_fields('int32', 'qmin', '-2147483648').
 add_fields('int32', 'qmax', '2147483647')
 )
(pdef('LSQ').
 add_fields('int32', 'qmin', '-2147483648').
 add_fields('int32', 'qmax', '2147483647')
 )
pdef('Fill').add_fields('float32', 'value', '0')

pdef('CheckNonFinite').add_fields('float32', 'scale', '1.0')


PADDING_MODES = [Doc('REPLICATE = 0', 'aaaaaa|abcdefgh|hhhhhhh'),
                Doc('REFLECT = 1', 'fedcba|abcdefgh|hgfedcb'),
                Doc('CONSTANT = 2', 'iiiiii|abcdefgh|iiiiiii')]
(pdef('Padding').
 add_fields('uint32', Doc('front_offset_dim0','offset in dim 0'), 0).
 add_fields('uint32', Doc('front_offset_dim1','offset in dim 1'), 0).
 add_fields('uint32', Doc('front_offset_dim2','offset in dim 2'), 0).
 add_fields('uint32', Doc('front_offset_dim3','offset in dim 3'), 0).
 add_fields('uint32', Doc('front_offset_dim4','offset in dim 4'), 0).
 add_fields('uint32', Doc('front_offset_dim5','offset in dim 5'), 0).
 add_fields('uint32', Doc('front_offset_dim6','offset in dim 6'), 0).
 add_fields('uint32', Doc('back_offset_dim0', 'back offset in dim0'), 0).
 add_fields('uint32', Doc('back_offset_dim1', 'back offset in dim1'), 0).
 add_fields('uint32', Doc('back_offset_dim2', 'back offset in dim2'), 0).
 add_fields('uint32', Doc('back_offset_dim3', 'back offset in dim3'), 0).
 add_fields('uint32', Doc('back_offset_dim4', 'back offset in dim4'), 0).
 add_fields('uint32', Doc('back_offset_dim5', 'back offset in dim5'), 0).
 add_fields('uint32', Doc('back_offset_dim6', 'back offset in dim6'), 0).
 add_fields('float32', Doc('padding_val','param of padding opr'), 0).
 add_enum('PaddingMode', *PADDING_MODES,
          name_field='padding_mode', default=2,
          member_alias=[(i, 'PADDING_{}'.format(i)) for i in PADDING_MODES]
          )
)

(pdef('LayerNorm')
 .add_fields('bool', 'affine', 'true')
 .add_fields('float32', 'eps', '1e-5f')
 .add_fields('uint64', 'normalized_dim', '1')
 .add_fields('uint64', 'normalized_size', '1')
)

(pdef('GeneralNorm')
 .add_fields('bool', 'affine', 'true')
 .add_fields('float32', 'eps', '1e-5f')
 .add_fields('uint64', 'axis_start', '0')
 .add_fields('uint64', 'axis_end', '0')
 )

(pdef('Dropout')
 .add_fields('float32', 'drop_prob', '0')
 .add_fields('uint64', 'seed', '0')
)

(pdef('GroupNorm')
 .add_fields('bool', 'affine', 'true')
 .add_fields('float32', 'eps', '1e-5f')
 .add_fields('uint32', 'group', '1')
 .add_enum_alias('Format', 'Convolution')
)

(pdef('RNNCell').
 add_enum('NonlineMode', 'IDENTITY = 0', 'RELU = 1', 'TANH = 2')
 )

(pdef('RNN').
 add_fields('uint32', Doc('num_layers', 'Number of recurrent layers'), '1').
 add_fields('bool', Doc('bidirectional', 'If becomes a bidirectional RNN'), 'false').
 add_fields('bool', Doc('bias', 'If the layer use bias weights b_ih and b_hh'), 'true').
 add_fields('uint32', Doc('hidden_size', 'The number of features in the hidden state'), '128').
 add_fields('float32', Doc('dropout', 'If introduce a Dropout layer on the outputs of each RNN layer'), '0.f').
 add_enum_alias('NonlineMode', 'RNNCell').
 add_enum_alias('FwdMode', 'BN', name_field='fwd_mode')
 )

(pdef('LSTM').
 add_fields('uint32', Doc('num_layers', 'Number of recurrent layers'), '1').
 add_fields('bool', Doc('bidirectional', 'If becomes a bidirectional LSTM'), 'false').
 add_fields('bool', Doc('bias', 'If the layer use bias weights b_ih and b_hh'), 'true').
 add_fields('uint32', Doc('hidden_size', 'The number of features in the hidden state'), '128').
 add_fields('uint32', Doc('proj_size', 'If use LSTM with projections of corresponding size'), '0').
 add_fields('float32', Doc('dropout', 'If introduce a Dropout layer on the outputs of each LSTM layer'), '0.f').
 add_enum_alias('FwdMode', 'BN', name_field='fwd_mode')
 )

(pdef('LAMBUpdate').
 add_fields('float32', Doc('beta_1', 'beta_1 paramter of lamb'), '1.f').
 add_fields('float32', Doc('beta_2', 'beta_2 paramter of lamb'), '1.f').
 add_fields('float32', Doc('step', 'training step'), '1.f').
 add_fields('float32', Doc('lr', 'learning rate'), '1.f').
 add_fields('float32', Doc('weight_decay', 'weight decay to adjust learning rate'), '1.f').
 add_fields('float32', Doc('eps', 'eps to multi'), '1.f').
 add_fields('bool', Doc('bias_correction', 'whether correct bias'), 'true').
 add_fields('bool', Doc('always_adapt', 'apply adaptive lr to 0.0'), 'false')
)
(pdef("Norm").
 add_enum('Mode',
            Doc('P_NORM=0', 'calculate p-norm, parameter p would be ignored in other mode'),
            Doc('INF_NORM=1', 'infinite norm'),
            Doc('NEG_INF_NORM=2', 'negative infinite norm'), name_field="mode").
 add_fields('float32', Doc('p', 'the order of norm'), '2').
 add_fields('int32', Doc('dim', 'which dim the norm performed along'), '-1'),
 )

(pdef('MultiHeadAttn')
 .add_fields('uint32', Doc('num_heads', 'Number of parallel attention heads.'), '1')
 .add_fields('uint32', Doc('embeding_size', 'Total dimension of the model.'), '0')
 .add_fields('uint32', Doc('k_size', 'Total number of features for keys.'), '0')
 .add_fields('uint32', Doc('v_size', 'Total number of features for values.'), '0')
 .add_fields('uint32', Doc('qproj_size', 'query weight projection.'), '0')
 .add_fields('uint32', Doc('kproj_size', 'key weight projection.'), '0')
 .add_fields('uint32', Doc('vproj_size', 'value weight projection.'), '0')
 .add_fields('uint32', Doc('oproj_size', 'output weight projection.'), '0')
 .add_fields('bool', Doc('qbias', 'Whether to add query bias.'), 'false')
 .add_fields('bool', Doc('kbias', 'Whether to add key bias.'), 'false')
 .add_fields('bool', Doc('vbias', 'Whether to add value bias.'), 'false')
 .add_fields('bool', Doc('obias', 'Whether to add out bias.'), 'false')
 .add_fields('float32', Doc('sm_scaler', 'Softmax smoothing (1.0 >= smScaler >= 0.0) or sharpening (smScaler > 1.0) coefficient.'), '1.f')
 .add_fields('uint32', Doc('input_order', 'The sequence data layout, allows the user to select 3! = 6 different data layouts or permutations of BEAM, BATCH and TIME dimensions.'), '0')
 .add_enum('AttnMaskType',
           Doc('NO_MASK = 0', 'Indicates that there is no mask.'),
           Doc('DEFAULT_MASK = 1', 'Use the default mask which the upper right triangle of the mask is -inf, and the diagonal and lower left triangle are all 0.'),
           Doc('CUDNN_STYLE_MASK = 2', 'Indicates the use of a cudnn style mask.'),
           Doc('USER_DEFINED_MASK = 3', 'Use the user-defined mask.'), name_field="attn_mask_type")
 .add_enum(Doc('TensorCombinationType', 'Used to determine whether mask tensor and bias_kv tensor exist in the input. Note that bias_kv here is not m_kbias and m_vbias in the linear layer, and bias_kv here will be added to the K and V at sequence dimensions, where K and V are the matrices of key and value after projection, and K and V will be used to calculate the attention matrix.'),
           Doc('NONE = 0', 'Indicates that there are no mask tensor and bias_kv tensor in the input.'),
           Doc('ONLY_MASK = 1',
               'Indicates that there is only mask tensor in input.'),
           Doc('ONLY_BIASKV = 2', 'Indicates that there is only bias_kv tensor in input.'),
           Doc('ALL = 3', 'Indicates that there are mask tensor and bias_kv tensor in the input.'), name_field="tensor_combination_type")
 .add_fields('bool', Doc('add_zero_attn', 'Whether to add a new batch of zeros to the key and value sequences.'), 'false')
 .add_fields('bool', Doc('need_weights', 'Whether to return the attention matrix, which is the output result of softmax.'), 'false')
 .add_fields('bool', Doc('reslink', 'Whether to add input query to final output.'), 'false')
 .add_fields('bool', Doc('training', 'Whether it is in training mode.'), 'true')
 .add_fields('uint64', Doc('seed', 'Random number seed for drop'), '0')
 .add_fields('float32', Doc('attn_prob', 'Dropout probability on attention, is applied directly to the softmax output'), '0.f')
 .add_fields('float32', Doc('out_prob', 'Dropout probability on output, alters the multi-m_head attention output'), '0.f')
 )

(pdef('MultinomialRNG').
 add_fields('uint64', 'seed', '0').
 add_fields('uint64', 'num_samples', '1').
 add_fields('bool', 'replacement', 'false')
 )
