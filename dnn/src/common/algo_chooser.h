/**
 * \file dnn/src/common/algo_chooser.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "utils.h"

namespace megdnn {

/*!
 * \brief get user-configured algorithm, or heuristic algorithm
 */
template <class Opr, typename... Args>
typename Opr::AlgoBase* get_algorithm(Opr* opr, Args&&... args) {
    typename Opr::AlgorithmInfo ret;
    auto set = opr->execution_policy().algo;
    if (set.valid()) {
        ret = set;
    } else {
        ret = opr->get_algorithm_info_heuristic(
                std::forward<Args>(args)..., std::numeric_limits<size_t>::max(),
                false);
    }
    return opr->get_algo_from_desc(ret.desc);
}

/*!
 * \brief get user-configured algorithm, or heuristic algorithm. used in opencl
 * whose algo need to be constructed each time.
 */
template <class Opr, typename... Args>
typename Opr::AlgoBase* get_algorithm_or_construct(Opr* opr, Args&&... args) {
    typename Opr::AlgorithmInfo ret;
    auto set = opr->execution_policy().algo;
    if (set.valid()) {
        return opr->algo_pack().construct_and_get_algo(set.desc);
    } else {
        ret = opr->get_algorithm_info_heuristic(
                std::forward<Args>(args)..., std::numeric_limits<size_t>::max(),
                false);
        return opr->get_algo_from_desc(ret.desc);
    }
}

/*!
 * \brief get all algorithms from algo_pack() that is available for current size
 */
template <class Opr>
std::vector<typename Opr::Algorithm*> get_all_algorithms(
        const typename Opr::AlgoBase::SizeArgs& args) {
    std::vector<typename Opr::Algorithm*> ret;
    ret.reserve(Opr::algo_pack().all_algos.size());
    for (auto i : Opr::algo_pack().all_algos) {
        if (i->is_available(args)) {
            ret.push_back(i);
        }
    }
    megdnn_assert(!ret.empty(), "no conv algorithm for %s",
                  args.to_string().c_str());
    return ret;
}

/*!
 * \brief a helper function to get a reproducible algorithm. If require a
 * reproducible algorithm, and the given algorithm is reproducible, return the
 * given algorithm. Otherwise return nullptr
 */
template <typename Opr>
typename Opr::Algorithm* get_reproducible_algo(typename Opr::AlgoBase* algo,
                                               bool reproducible) {
    if (reproducible) {
        if (algo->is_reproducible()) {
            return algo;
        }
    } else {
        return algo;
    }
    return nullptr;
}

template <typename Opr>
typename Opr::Algorithm* get_reproducible_algo(
        const std::vector<typename Opr::AlgoBase*>& algos,
        const typename Opr::AlgoBase::SizeArgs& args,
        size_t workspace_limit_in_bytes, const char* name) {
    size_t min_workspace_limit_in_bytes = std::numeric_limits<size_t>::max();
    bool available_but_limited_by_workspace = false;
    bool available_but_not_reproducible = false;
    for (auto i : algos) {
        if (i->is_available_reproducible(args, true,
                                         workspace_limit_in_bytes)) {
            return i;
        }
        if (i->is_available_reproducible(args)) {
            if (i->get_workspace_in_bytes(args) > workspace_limit_in_bytes) {
                available_but_limited_by_workspace = true;
                min_workspace_limit_in_bytes =
                        std::min(min_workspace_limit_in_bytes,
                                 i->get_workspace_in_bytes(args));
            }
        }
        if (i->is_available(args)) {
            if (!i->is_reproducible())
                available_but_not_reproducible = true;
        }
    }

    MEGDNN_MARK_USED_VAR(name);
    if (available_but_limited_by_workspace) {
        megdnn_throw(megdnn_mangle(ssprintf(
                "no reproducible %s algorithm: %s workspace limit %zu is "
                "less than mini workspace limit %zu",
                name, args.to_string().c_str(), workspace_limit_in_bytes,
                min_workspace_limit_in_bytes)));
    } else if (available_but_not_reproducible) {
        megdnn_throw(
                megdnn_mangle(ssprintf("no reproducible %s algorithm", name)));
    } else {
        megdnn_throw(megdnn_mangle(ssprintf("no usable %s algorithm", name)));
    }
}

template <typename Opr>
typename Opr::Algorithm* get_usable_algo(
        const std::vector<typename Opr::AlgoBase*>& algos,
        const typename Opr::AlgoBase::SizeArgs& args,
        size_t workspace_limit_in_bytes, const char* name) {
    size_t min_workspace_limit_in_bytes = std::numeric_limits<size_t>::max();
    bool available_but_limited_by_workspace = false;
    for (auto i : algos) {
        if (i->is_available_wk(args, workspace_limit_in_bytes)) {
            return i;
        }
        if (i->is_available(args)) {
            available_but_limited_by_workspace = true;
            min_workspace_limit_in_bytes =
                    std::min(min_workspace_limit_in_bytes,
                             i->get_workspace_in_bytes(args));
        }
    }

    MEGDNN_MARK_USED_VAR(name);
    if (available_but_limited_by_workspace) {
        megdnn_throw(megdnn_mangle(ssprintf(
                "no usable %s algorithm: %s workspace limit %zu is "
                "less than mini workspace limit %zu",
                name, args.to_string().c_str(), workspace_limit_in_bytes,
                min_workspace_limit_in_bytes)));
    } else {
        megdnn_throw(megdnn_mangle(ssprintf("no usable %s algorithm", name)));
    }
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
