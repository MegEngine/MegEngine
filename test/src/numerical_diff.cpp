/**
 * \file test/src/numerical_diff.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/test/numerical_diff.h"
#include "megbrain/utils/timer.h"
#include "megbrain/common.h"

#include <limits>
#include <cmath>

using namespace mgb;

std::vector<HostTensorND> mgb::numerical_diff_pt2(
        const std::vector<HostTensorND*> &input,
        std::function<float()> cost,
        const std::vector<Maybe<float>> &eps) {
    std::vector<HostTensorND> result;
    if (!eps.empty())
        mgb_assert(eps.size() == input.size());

    for (size_t cur_inp_idx = 0; cur_inp_idx < input.size(); ++ cur_inp_idx)
    {
        result.emplace_back();
        if (!input[cur_inp_idx])
            continue;
        auto &&cur_inp = input[cur_inp_idx];
        auto &&dest = result.back();
        dest.comp_node(cur_inp->comp_node()).
            dtype(cur_inp->dtype()).
            resize(cur_inp->shape());
        auto dptr = dest.ptr<float>();

        mgb_assert(cur_inp->layout().is_contiguous());
        auto cur_inp_ptr = cur_inp->ptr<float>();

        mgb::RealTimer timer;
        double prev_record = 0.0;
        for (size_t i = 0, it = cur_inp->layout().total_nr_elems();
                i < it; ++ i) {
            auto orig = cur_inp_ptr[i];
            float delta;
            if (eps.empty() || !eps[cur_inp_idx].valid()) {
                delta = std::sqrt(std::numeric_limits<float>::epsilon()) *
                    std::max<float>(std::fabs(orig), 1);
            } else {
                delta = eps[cur_inp_idx].val();
            }
            cur_inp_ptr[i] = orig - delta;
            auto c0 = cost();
            cur_inp_ptr[i] = orig + delta;
            auto c1 = cost();
            cur_inp_ptr[i] = orig;

            auto cur_time = timer.get_secs();
            if (cur_time - prev_record > 10) {
                prev_record = cur_time;
                mgb_log_warn(
                        "numerical diff running for more than %.3f secs, "
                        "consider to reduce the tensor size", cur_time);
            }

            dptr[i] = (c1 - c0) / (delta * 2);
        }
    }
    return result;
}

namespace mgb {
    // explicit inst to avoid link error for Maybe::Maybe()
    template class Maybe<float>;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
