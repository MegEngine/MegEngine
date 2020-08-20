
ARITIES = {1: 'UNARY', 2: 'BINARY', 3: 'TERNARY'}

DTYPES = {'dt_int32': ('Int32', 'INT'),
          'dt_uint8': ('Uint8', 'INT'),
          'dt_int8': ('Int8', 'INT'),
          'dt_int16': ('Int16', 'INT'),
          'dt_bool': ('Bool', 'BOOL'),
          'dt_float32': ('Float32', 'FLOAT'),
          'dt_float16': ('Float16', 'FLOAT'),
          'dt_bfloat16': ('BFloat16', 'FLOAT')
          }

MODES = {
    (1, 'INT'): ['RELU', 'ABS', 'NEGATE'],
    (2, 'INT'): ['ABS_GRAD', 'ADD', 'FLOOR_DIV', 'MAX', 'MIN', 'MOD', 'MUL',
                 'SIGMOID_GRAD', 'SUB', 'SWITCH_GT0', 'TANH_GRAD', 'LT', 'LEQ',
                 'EQ', 'FUSE_ADD_RELU', 'SHL', 'SHR', 'RMULH'],
    (3, 'INT'): ['COND_LEQ_MOV'],

    (1, 'FLOAT'): ['RELU', 'ABS', 'NEGATE', 'ACOS', 'ASIN', 'CEIL', 'COS',
                   'EXP', 'EXPM1', 'FLOOR', 'LOG', 'LOG1P', 'SIGMOID', 'SIN',
                   'TANH', 'FAST_TANH', 'ROUND', 'ERF', 'ERFINV', 'ERFC',
                   'ERFCINV', 'H_SWISH'],
    (2, 'FLOAT'): ['ABS_GRAD', 'ADD', 'FLOOR_DIV', 'MAX', 'MIN', 'MOD', 'MUL',
                   'SIGMOID_GRAD', 'SUB', 'SWITCH_GT0', 'TANH_GRAD', 'LT',
                   'LEQ', 'EQ', 'FUSE_ADD_RELU', 'TRUE_DIV', 'POW',
                   'LOG_SUM_EXP', 'FUSE_ADD_TANH', 'FAST_TANH_GRAD',
                   'FUSE_ADD_SIGMOID', 'ATAN2', 'H_SWISH_GRAD',
                   'FUSE_ADD_H_SWISH'],
    (3, 'FLOAT'): ['COND_LEQ_MOV', 'FUSE_MUL_ADD3'],
    (1, 'BOOL'): ['NOT'],
    (2, 'BOOL'): ['AND', 'OR', 'XOR', 'LT', 'LEQ', 'EQ'],
    (3, 'BOOL'): []
}
