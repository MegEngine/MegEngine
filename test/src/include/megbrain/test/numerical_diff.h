/**
 * \file test/src/include/megbrain/test/numerical_diff.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief numerical differentiation
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/tensor.h"

namespace mgb  {

    /*!
     * \brief estimate function differentiation by values evaluated at 2 points,
     *      using the symmetric difference quotient
     * \param input pointers to input tensors which should be read by the cost
     *      functor. Nullptrs are silently ignored.
     * \param eps value of epsilon; if empty, choose automatically and
     *      uniformly for each input
     */
    std::vector<HostTensorND> numerical_diff_pt2(
            const std::vector<HostTensorND*> &input,
            std::function<float()> cost,
            const std::vector<Maybe<float>> &eps = {});

}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

