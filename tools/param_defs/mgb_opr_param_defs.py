pdef('DType').add_fields('dtype', 'dtype', 'DTypeEnum::Byte')

pdef('PersistentOutputStorage').add_fields(
    'int32', Doc(
        'share_key',
        'This is used for controlling memory sharing. Multiple '
        "``PersistentOutputStorage'' oprs with the same ``share_key'' "
        "would share underlying tensor storage. Note that the value ``-1'' is "
        'treated specially: storage of oprs with this key would be private and '
        'would not be shared with any other opr.'
    ),
    -1)

(pdef('OptionalAxis', 'optinal axis: axis == -1 means no axis').
 add_fields('int32', 'axis', -1))
(pdef('OptionalAxisV1', 'optinal axis: axis == MAX_NDIM means no axis').
 add_const('int32', 'MAX_NDIM', 7).
 add_const('int32', 'INVALID_AXIS', 'MAX_NDIM').
 add_fields('int32', 'axis', 'INVALID_AXIS'))

(pdef('ExecutionPolicy', version=0, is_legacy=True).
 add_enum('Strategy',
          Doc('HEURISTIC = 0', 'use heuristic to choose the fastest algorithm'),
          Doc('HEURISTIC_REPRODUCIBLE = 1', 'use heuristic to choose the fastest algorithm, '
              'and the chosen algorithm is reproducible'),
          Doc('PROFILE = 2',
              'run possible algorithms on real device to find the best'),
          Doc('PROFILE_REPRODUCIBLE = 3',
              'the fastest of profile result that is also reproducible'),
          Doc('PROFILE_HEURISTIC = 4',
              'use profile result and heuristic to choose the fastest algorithm')).
 add_fields('uint64',
            Doc('workspace_limit', 'workspace limit in bytes'),
            str(2**64-1)+'ull'))

(pdef('ExecutionPolicy', 'specify how to select an algorithm for an operator', version=1).
 add_bit_combination_enum('Strategy',
          Doc('HEURISTIC = 1 << 0', 'use heuristic to choose the fastest algorithm'),
          Doc('PROFILE = 1 << 1',
              'run possible algorithms on real device to find the best'),
          Doc('REPRODUCIBLE = 1 << 2',
              'when profile or heuristic algo selection it require the algos'
              'must be reproducible'),
          Doc('OPTIMIZED = 1 << 3',
              'profile require algos are optmized to achieve fast-profile'),
          default=('HEURISTIC',),
          member_alias=[(('HEURISTIC', 'REPRODUCIBLE'), 'HEURISTIC_REPRODUCIBLE'),
                        (('PROFILE', 'REPRODUCIBLE'), 'PROFILE_REPRODUCIBLE'),
                        (('PROFILE', 'HEURISTIC'), 'PROFILE_HEURISTIC'),
                        (('OPTIMIZED',), 'OPTMIZED'),
                        ]).
 add_fields('uint64',
            Doc('workspace_limit', 'workspace limit in bytes'),
            str(2**64-1)+'ull'))

(pdef('AssertEqual').
 add_fields('float32',
            Doc('maxerr', 'max allowed error; error is defined as the minimal '
                'of absolute and relative error'),
            1e-4).
 add_fields('bool', Doc('verbose', 'whether to print maxerr to stdout '
                        'during opr exec'),
            'false')
 )

(pdef('CollectiveComm', 'collective communication between multiple computing '
      'nodes on localhost')
 .add_enum(Doc('Mode', 'mode of collective communication'),
           Doc('REDUCE_SUM = 0', 'reduce by sum to output computing node'),
           Doc('BROADCAST = 1', 'copy input value to each output computing node'),
           Doc('ALL_GATHER = 2', 'each output comp node gets the concatenated '
               'value of all inputs'),
           Doc('REDUCE_SCATTER_SUM = 3',
               'reduce inputs by sum and each output gets one part of it'),
           Doc('ALL_REDUCE_SUM = 4', 'every output gets the sum of all inputs'),
           Doc('ALL_REDUCE_MAX = 5', 'every output gets the max of all inputs'),
           Doc('ALL_REDUCE_MIN = 6', 'every output gets the min of all inputs'),
           Doc('ALL_REDUCE_PROD = 7', 'every output gets the prod of all inputs'),
           Doc('GATHER = 8', 'concat inputs to one node'),
           Doc('SCATTER = 9', 'scatter input to each output computing node'),
           Doc('ALL_TO_ALL = 10', 'scatter inputs and gather them on each computing node'),
           name_field='mode'))

(pdef('FakeSerializedDType',
      'HACK: The tag of this param def is actually used for another '
      'non-generated param def SerializedDType, the sole purpose of this param '
      'def is to provide a spare tag. Do not use.'
))

(pdef('CondExecPred',
      'evaluate a predicate and branch keys to setup ExecutionMask objects '
      'with associated predicate proxy vars (PPVs)')
 .add_enum(Doc('Mode', 'how to compare predicate var with branch keys'),
           Doc('CASE = 0',
               'The outputs correspond to branch keys, '
               'and the one which equals predicate would be activated. '
               'This behaves like a case-statement in many languages.'),
           Doc('CASE_FALLBACK = 1', 'like :attr:`CASE`, but add an extra output '
               'that would be activated if no branch is matched'),
           Doc('PIECEWISE = 2', 'One more outputs would be produced than the '
               'number of branch keys, representing the interval in which the '
               'predicate var fits in. The intervals are defined as '
               r':math:`(-\\infty, k_0), [k_0, k_1), \\ldots, '
               '[k_{n-2}, k_{n-1}), [k_{n-1}, \\infty)`. '
               'The keys must be given in ascending order.')
           )
 .add_fields('float32',
             Doc('eps',
                 'threshold for checking equality of float point values'),
             1e-4)
 )

(pdef('CondExecPredLogical',
      'compute a logical function over a set of PPVs')
 .add_enum('Mode', Doc('OR = 0', 'logical or'),
           Doc('AND = 1', 'logical and'),
           Doc('XOR = 2', 'exclusive-or'),
           Doc('NOR = 3', 'not or(inputs)'),
           Doc('NAND = 4', 'not and(inputs)'),
           Doc('XNOR = 5', 'not xor(inputs)'))
 )

(pdef('CondExecMark',
      'add ExecutionMask of the input PPV to this opr and readers of the '
      'outputs of this opr')
 .add_enum(Doc('GradMode', 'mode for computing the gradient'),
           Doc('SUM = 0', 'normal gradient mode: sum all the activated components'),
           Doc('SUM_COND_OUT = 1', 'use :attr:`CondExecMerge.SUM_COND_OUT` mode so '
               'oprs that depend on the gradient opr would not be executed '
               'if the forward var is not used.'),
           name_field='grad_mode')
 .add_enum(Doc('StaticInfer',
               """static inference option. **Note:** This is a workaround: since
               currently static inference in MegBrain does not take conditional
               execution into account, this option can be used to bypass static
               inference errors. This is currently only used by automatically
               generated gradient oprs."""),
           Doc('SHAPE_VALUE = 0', 'enable both shape and value inference'),
           Doc('SHAPE_ONLY = 1',
               'only enable shape inference (disable value inference)'),
           Doc('NONE = 2', 'disable both shape and value inference'),
           name_field='static_infer')
 )

(pdef('CondExecMerge', 'merge multiple conditional execution branches')
 .add_fields('uint32', Doc('nr_output',
                           'number of output vars (i.e. vars per branch)'),
             1)
 .add_enum('Mode',
           Doc('EXACT_ONE = 0', 'copy the var whose mask is activated to the output'
               ', requiring that exactly one branch is active'),
           Doc('EXACT_ONE_SAME_SHAPE = 1', 'like :attr:`EXACT_ONE` with the '
               'requirement that all branches have the same shape, so shape '
               'inference can be easier'),
           Doc('SUM = 2', 'sum all the active branches into output var; require '
               'all branches to have the same shape. Extra shape vars are '
               'needed in this mod, so the outputs can be initialized to zero '
               'when no input is active (and their shapes are probably '
               'unknown).'),
           Doc('SUM_COND_OUT = 3', 'like :attr:`SUM` but also add an ExecutionMask'
               ' to the readers of output vars, so they would be skipped if '
               ' no branch is taken')
           )
 )

(pdef('NvOf', 'opr Implements NVIDIA Optical Flow SDK.').add_fields('uint32', 'precision', 1))
