/**
 * \file dnn/test/common/deduce_layout_proxy.h
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

#include "test/common/utils.h"

namespace megdnn {
namespace test {

template <typename Opr, size_t Arity, bool can_deduce_layout>
struct DeduceLayoutProxy;

template <typename Opr, size_t Arity>
struct DeduceLayoutProxy<Opr, Arity, false> {
    static void deduce_layout(Opr*, TensorLayoutArray&) {}
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 2, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 2);
        opr->deduce_layout(layouts[0], layouts[1]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 3, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 3);
        opr->deduce_layout(layouts[0], layouts[1], layouts[2]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 4, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 4);
        opr->deduce_layout(layouts[0], layouts[1], layouts[2], layouts[3]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 5, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 5);
        opr->deduce_layout(layouts[0], layouts[1], layouts[2], layouts[3],
                           layouts[4]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 5, false> {
    static void deduce_layout(Opr*, TensorLayoutArray&) {}
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 6, false> {
    static void deduce_layout(Opr*, TensorLayoutArray&) {}
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 7, false> {
    static void deduce_layout(Opr*, TensorLayoutArray&) {}
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 8, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 8);
        opr->deduce_layout(layouts[0], layouts[1], layouts[2], layouts[3],
                           layouts[4], layouts[5], layouts[6], layouts[7]);
    }
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
