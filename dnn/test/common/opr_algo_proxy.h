/**
 * \file dnn/test/common/opr_algo_proxy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "megdnn/basic_types.h"
#include "test/common/opr_trait.h"
#include "test/common/utils.h"

namespace megdnn {
namespace test {

template <typename Opr, size_t Arity>
struct AlgoProxy;

#define DEF_ALGO_PROXY(arity)                                                 \
    template <typename Opr>                                                   \
    struct AlgoProxy<Opr, arity> {                                            \
        static std::vector<typename Opr::AlgorithmInfo>                       \
        get_all_algorithms_info(Opr* opr, const TensorLayoutArray& layouts) { \
            megdnn_assert(layouts.size() == arity);                           \
            return opr->get_all_algorithms_info(LAYOUTS);                     \
        }                                                                     \
        static typename Opr::AlgorithmInfo get_algorithm_info_heuristic(      \
                Opr* opr, const TensorLayoutArray& layouts) {                 \
            megdnn_assert(layouts.size() == arity);                           \
            return opr->get_algorithm_info_heuristic(LAYOUTS);                \
        }                                                                     \
        static size_t get_workspace_in_bytes(                                 \
                Opr* opr, const TensorLayoutArray& layouts) {                 \
            megdnn_assert(layouts.size() == arity);                           \
            return opr->get_workspace_in_bytes(LAYOUTS);                      \
        }                                                                     \
        static void exec(Opr* opr, const TensorNDArray& tensors,              \
                         Workspace workspace) {                               \
            megdnn_assert(tensors.size() == arity);                           \
            return opr->exec(TENSORS, workspace);                             \
        }                                                                     \
    }

#define LAYOUTS layouts[0], layouts[1], layouts[2]
#define TENSORS tensors[0], tensors[1], tensors[2]
DEF_ALGO_PROXY(3);
#undef LAYOUTS
#undef TENSORS

#define LAYOUTS layouts[0], layouts[1], layouts[2], layouts[3], layouts[4]
#define TENSORS tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]
DEF_ALGO_PROXY(5);
#undef LAYOUTS
#undef TENSORS

#define LAYOUTS                                                             \
    layouts[0], layouts[1], layouts[2], layouts[3], layouts[4], layouts[5], \
            layouts[6], layouts[7]
#define TENSORS                                                             \
    tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], \
            tensors[6], tensors[7]
DEF_ALGO_PROXY(8);
#undef LAYOUTS
#undef TENSORS

#undef DEF_ALGO_PROXY

#define DEF_ALGO_PROXY(Opr, arity)                                             \
    template <>                                                                \
    struct AlgoProxy<Opr, arity> {                                             \
        static std::vector<typename Opr::AlgorithmInfo>                        \
        get_all_algorithms_info(Opr* opr, const TensorLayoutArray& layouts) {  \
            megdnn_assert(layouts.size() == arity);                            \
            return opr->get_all_algorithms_info(LAYOUTS);                      \
        }                                                                      \
        static typename Opr::AlgorithmInfo get_algorithm_info_heuristic(       \
                Opr* opr, const TensorLayoutArray& layouts) {                  \
            megdnn_assert(layouts.size() == arity);                            \
            return opr->get_algorithm_info_heuristic(LAYOUTS);                 \
        }                                                                      \
        static size_t get_workspace_in_bytes(                                  \
                Opr* opr, const TensorLayoutArray& layouts,                    \
                const typename Opr::PreprocessedFilter* preprocessed_filter =  \
                        nullptr) {                                             \
            megdnn_assert(layouts.size() == arity);                            \
            return opr->get_workspace_in_bytes(LAYOUTS, preprocessed_filter);  \
        }                                                                      \
        static void exec(                                                      \
                Opr* opr, const TensorNDArray& tensors,                        \
                const typename Opr::PreprocessedFilter* preprocessed_filter,   \
                Workspace workspace) {                                         \
            megdnn_assert(tensors.size() == arity);                            \
            return opr->exec(TENSORS, preprocessed_filter, workspace);         \
        }                                                                      \
        static void exec(Opr* opr, const TensorNDArray& tensors,               \
                         Workspace workspace) {                                \
            megdnn_assert(tensors.size() == arity);                            \
            return opr->exec(TENSORS, nullptr, workspace);                     \
        }                                                                      \
        static size_t get_preprocess_workspace_in_bytes(                       \
                Opr* opr, const TensorLayoutArray& layouts) {                  \
            megdnn_assert(layouts.size() == arity);                            \
            return opr->get_preprocess_workspace_in_bytes(LAYOUTS);            \
        }                                                                      \
        static SmallVector<TensorLayout> deduce_preprocessed_filter_layout(    \
                Opr* opr, const TensorLayoutArray& layouts) {                  \
            megdnn_assert(layouts.size() == arity);                            \
            return opr->deduce_preprocessed_filter_layout(LAYOUTS);            \
        }                                                                      \
        static void exec_preprocess(                                           \
                Opr* opr, const TensorNDArray& tensors,                        \
                const TensorLayoutArray& layouts,                              \
                Opr::PreprocessedFilter* preprocessed_filter,                  \
                _megdnn_workspace workspace) {                                 \
            megdnn_assert(layouts.size() == arity && tensors.size() == arity); \
            return opr->exec_preprocess(PREPROCESS_ARGS, preprocessed_filter,  \
                                        workspace);                            \
        }                                                                      \
    };

#define LAYOUTS layouts[0], layouts[1], layouts[2]
#define TENSORS tensors[0], tensors[1], tensors[2]
#define PREPROCESS_ARGS layouts[0], tensors[1], layouts[2]
DEF_ALGO_PROXY(ConvolutionForward, 3);
#undef PREPROCESS_ARGS
#undef LAYOUTS
#undef TENSORS

#define LAYOUTS layouts[0], layouts[1], layouts[2], layouts[3], layouts[4]
#define TENSORS tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]
#define PREPROCESS_ARGS \
    layouts[0], tensors[1], tensors[2], layouts[3], layouts[4]
DEF_ALGO_PROXY(ConvBias, 5);
#undef PREPROCESS_ARGS
#undef LAYOUTS
#undef TENSORS

#undef DEF_ALGO_PROXY

template <typename Opr, size_t arity = OprTrait<Opr>::arity>
struct OprAlgoProxyDefaultImpl : public AlgoProxy<Opr, arity> {};

template <typename Opr>
struct OprAlgoProxy : public OprAlgoProxyDefaultImpl<Opr> {};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
