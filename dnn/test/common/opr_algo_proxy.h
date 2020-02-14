/**
 * \file dnn/test/common/opr_algo_proxy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/basic_types.h"
#include "test/common/opr_trait.h"
#include "test/common/utils.h"

namespace megdnn {
namespace test {

template <typename Opr, size_t Arity>
struct AlgoProxy;

template <typename Opr>
struct AlgoProxy<Opr, 3> {
    static std::vector<typename Opr::Algorithm*> get_all_algorithms(
            Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 3);
        return opr->get_all_algorithms(layouts[0], layouts[1], layouts[2]);
    }
    static typename Opr::Algorithm* get_algorithm_heuristic(
            Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 3);
        return opr->get_algorithm_heuristic(layouts[0], layouts[1], layouts[2]);
    }
};

template <typename Opr>
struct AlgoProxy<Opr, 5> {
    static std::vector<typename Opr::Algorithm*> get_all_algorithms(
            Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 5);
        return opr->get_all_algorithms(layouts[0], layouts[1], layouts[2],
                                       layouts[3], layouts[4]);
    }
    static typename Opr::Algorithm* get_algorithm_heuristic(
            Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 5);
        return opr->get_algorithm_heuristic(layouts[0], layouts[1], layouts[2],
                                            layouts[3], layouts[4]);
    }
};

template <typename Opr, size_t arity = OprTrait<Opr>::arity>
struct OprAlgoProxyDefaultImpl : public AlgoProxy<Opr, arity> {};

template <typename Opr>
struct OprAlgoProxy : public OprAlgoProxyDefaultImpl<Opr> {};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
