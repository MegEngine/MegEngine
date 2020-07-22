/**
 * \file src/opr/test/basic_arith/elemwise_ternary_trait_def.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef DEF_TRAIT
#error "DEF_TRAIT must be defined"
#endif

/* ======================= ternary ======================= */
#define _CUR_ARITY 3
#define _EXPAND_PARAMS \
    ctype x = inp[0][idx]; \
    ctype y = inp[1][idx]; \
    ctype z = inp[2][idx]

#define _ALLOW_BOOL false
#define _ALLOW_FLOAT true
#define _ALLOW_INT true
DEF_TRAIT(COND_LEQ_MOV, x <= y ? z : 0)
DEF_TRAIT(FUSE_MUL_ADD3, x * y + z)
#undef _ALLOW_INT
#undef _ALLOW_FLOAT

#undef _CUR_ARITY
#undef _EXPAND_PARAMS

/* ======================= quaternary ======================= */
#define _CUR_ARITY 4
#define _EXPAND_PARAMS \
    ctype i0 = inp[0][idx]; \
    ctype i1 = inp[1][idx]; \
    ctype i2 = inp[2][idx]; \
    ctype i3 = inp[3][idx]

#define _ALLOW_FLOAT true
#define _ALLOW_INT true
DEF_TRAIT(FUSE_MUL_ADD4, i0 * i1 + i2 * i3)
#undef _ALLOW_INT
#undef _ALLOW_FLOAT

#undef _CUR_ARITY
#undef _EXPAND_PARAMS
#undef _ALLOW_BOOL

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
