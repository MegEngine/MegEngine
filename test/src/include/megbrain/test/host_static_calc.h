/**
 * \file test/src/include/megbrain/test/host_static_calc.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief static calculating on host to check opr correctness
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/opr/basic_arith.h"

namespace mgb {

    void elemwise_static_calc(opr::Elemwise::Mode mode,
            HostTensorND &dest, const std::vector<HostTensorND>& inputs);

#define EL2(_name, _mode) \
    static inline void host_##_name (HostTensorND &dest, \
            const HostTensorND &a, const HostTensorND &b) { \
        elemwise_static_calc(opr::Elemwise::Mode::_mode, dest, {a, b}); \
    }

    EL2(add, ADD)
    EL2(pow, POW);

#undef EL2
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

