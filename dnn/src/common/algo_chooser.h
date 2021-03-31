/**
 * \file dnn/src/common/algo_chooser.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
    typename Opr::AlgorithmDesc ret;
    auto set = opr->execution_policy().algo;
    if (set.valid()) {
        ret = set;
    } else {
        ret = opr->get_algorithm_info_heuristic(
                std::forward<Args>(args)..., std::numeric_limits<size_t>::max(),
                AlgoAttribute::DEFAULT, AlgoAttribute::DEFAULT).desc;
    }
    return static_cast<typename Opr::AlgoBase*>(
            opr->get_algorithm_from_desc(ret));
}

/*!
 * \brief get user-configured algorithm, or heuristic algorithm. used in opencl
 * whose algo need to be constructed each time.
 */
template <class Opr, typename... Args>
typename Opr::AlgoBase* get_algorithm_or_construct(Opr* opr, Args&&... args) {
    auto set = opr->execution_policy().algo;
    if (set.valid()) {
        return opr->algo_pack().construct_and_get_algo(set);
    } else {
        return static_cast<typename Opr::AlgoBase*>(
                opr->get_algorithm_heuristic(std::forward<Args>(args)...,
                                             std::numeric_limits<size_t>::max(),
                                             AlgoAttribute::DEFAULT,
                                             AlgoAttribute::DEFAULT));
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
 * \brief a helper function to get an algorithm match attribute. If require a
 * algorithm with specified attribute, and the given algorithm match that
 * attribute, return the given algorithm. Otherwise return nullptr
 */
template <typename Opr>
typename Opr::Algorithm* get_algo_match_attribute(
        typename Opr::AlgoBase* algo, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    if (algo->contain_attribute_all(positive_attr) &&
        !algo->contain_attribute_any(negative_attr)) {
        return algo;
    }
    return nullptr;
}

template <typename Opr>
typename Opr::Algorithm* get_algo_match_attribute(
        const std::vector<typename Opr::AlgoBase*>& algos,
        const typename Opr::AlgoBase::SizeArgs& args,
        size_t workspace_limit_in_bytes, const char* name,
        const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
        const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) {
    size_t min_workspace_limit_in_bytes = std::numeric_limits<size_t>::max();
    bool available_but_limited_by_workspace = false;
    bool available_but_attribute_mismatch = false;
    for (auto i : algos) {
        if (i->is_available_attribute(args, positive_attr, negative_attr,
                                         workspace_limit_in_bytes)) {
            return i;
        }
        if (i->is_available_attribute(args, positive_attr, negative_attr)) {
            if (i->get_workspace_in_bytes(args) > workspace_limit_in_bytes) {
                available_but_limited_by_workspace = true;
                min_workspace_limit_in_bytes =
                        std::min(min_workspace_limit_in_bytes,
                                 i->get_workspace_in_bytes(args));
            }
        }
        if (i->is_available(args)) {
            if (!(i->contain_attribute_all(positive_attr) &&
                  !i->contain_attribute_any(negative_attr)))
                available_but_attribute_mismatch = true;
        }
    }

    MEGDNN_MARK_USED_VAR(name);
    if (available_but_limited_by_workspace) {
        megdnn_throw(
                ssprintf("no %s algorithm without attribute(%s) with "
                         "attribute(%s) : %s workspace limit %zu is "
                         "less than mini workspace limit %zu",
                         name, Algorithm::attribute_str(negative_attr).c_str(),
                         Algorithm::attribute_str(positive_attr).c_str(),
                         args.to_string().c_str(), workspace_limit_in_bytes,
                         min_workspace_limit_in_bytes));
    } else if (available_but_attribute_mismatch) {
        megdnn_throw(ssprintf(
                "no %s algorithm without attribute(%s) with attribute(%s)", name,
                Algorithm::attribute_str(negative_attr).c_str(),
                Algorithm::attribute_str(positive_attr).c_str()));
    } else {
        megdnn_throw(ssprintf("no usable %s algorithm", name));
    }
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
