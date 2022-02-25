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
        opr->deduce_layout(layouts[0], layouts[1], layouts[2], layouts[3], layouts[4]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 6, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 6);
        opr->deduce_layout(
                layouts[0], layouts[1], layouts[2], layouts[3], layouts[4], layouts[5]);
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
struct DeduceLayoutProxy<Opr, 7, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 7);
        opr->deduce_layout(
                layouts[0], layouts[1], layouts[2], layouts[3], layouts[4], layouts[5],
                layouts[6]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 8, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 8);
        opr->deduce_layout(
                layouts[0], layouts[1], layouts[2], layouts[3], layouts[4], layouts[5],
                layouts[6], layouts[7]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 9, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 9);
        opr->deduce_layout(
                layouts[0], layouts[1], layouts[2], layouts[3], layouts[4], layouts[5],
                layouts[6], layouts[7], layouts[8]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 10, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 10);
        opr->deduce_layout(
                layouts[0], layouts[1], layouts[2], layouts[3], layouts[4], layouts[5],
                layouts[6], layouts[7], layouts[8], layouts[9]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 10, false> {
    static void deduce_layout(Opr*, TensorLayoutArray&) {}
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 13, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() == 13);
        opr->deduce_layout(
                layouts[0], layouts[1], layouts[2], layouts[3], layouts[4], layouts[5],
                layouts[6], layouts[7], layouts[8], layouts[9], layouts[10],
                layouts[11], layouts[12]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 13, false> {
    static void deduce_layout(Opr*, TensorLayoutArray&) {}
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
