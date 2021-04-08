/**
 * \file dnn/test/common/opr_proxy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "src/common/opr_trait.h"

#include "test/common/deduce_layout_proxy.h"
#include "test/common/exec_proxy.h"
#include "test/common/fast_run_cache.h"
#include "test/common/inspect_type.h"
#include "test/common/opr_algo_proxy.h"
#include "test/common/timer.h"
#include "test/common/workspace_wrapper.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <unordered_map>

namespace megdnn {
namespace test {

template <Algorithm::OprType>
struct OprFromOprTypeTrait;

template <typename Opr>
struct OprTypeFromOprTrait;

#define cb(_opr_type, _opr)                                     \
    template <>                                                 \
    struct OprFromOprTypeTrait<Algorithm::OprType::_opr_type> { \
        using Opr = megdnn::_opr;                               \
    };                                                          \
    template <>                                                 \
    struct OprTypeFromOprTrait<megdnn::_opr> {                  \
        constexpr static Algorithm::OprType opr_type =          \
                Algorithm::OprType::_opr_type;                  \
    }

cb(MATRIX_MUL_FORWARD, MatrixMulForward);
cb(BATCHED_MATRIX_MUL_FORWARD, BatchedMatrixMulForward);
cb(CONVOLUTION_FORWARD, ConvolutionForward);
cb(CONVOLUTION_BACKWARD_DATA, ConvolutionBackwardData);
cb(CONVOLUTION_BACKWARD_FILTER, ConvolutionBackwardFilter);
cb(CONVOLUTION3D_FORWARD, Convolution3DForward);
cb(CONVOLUTION3D_BACKWARD_DATA, Convolution3DBackwardData);
cb(CONVOLUTION3D_BACKWARD_FILTER, Convolution3DBackwardFilter);
cb(LOCAL_SHARE_FORWARD, LocalShareForward);
cb(LOCAL_SHARE_BACKWARD_DATA, LocalShareBackwardData);
cb(LOCAL_SHARE_BACKWARD_FILTER, LocalShareBackwardFilter);
cb(DEFORMABLE_CONV_FORWARD, DeformableConvForward);
cb(DEFORMABLE_CONV_BACKWARD_DATA, DeformableConvBackwardData);
cb(DEFORMABLE_CONV_BACKWARD_FILTER, DeformableConvBackwardFilter);
cb(BATCH_CONV_FORWARD, BatchConvBiasForward);
cb(CONVBIAS_FORWARD, ConvBiasForward);

#undef cb

// clang-format off
#define FOREACH_OPR_TYPE(cb) \
    cb(MATRIX_MUL_FORWARD) \
    cb(BATCHED_MATRIX_MUL_FORWARD) \
    cb(CONVOLUTION_FORWARD) \
    cb(CONVOLUTION_BACKWARD_DATA) \
    cb(CONVOLUTION_BACKWARD_FILTER) \
    cb(CONVOLUTION3D_FORWARD) \
    cb(CONVOLUTION3D_BACKWARD_DATA) \
    cb(CONVOLUTION3D_BACKWARD_FILTER) \
    cb(LOCAL_SHARE_FORWARD) \
    cb(LOCAL_SHARE_BACKWARD_DATA) \
    cb(LOCAL_SHARE_BACKWARD_FILTER) \
    cb(DEFORMABLE_CONV_FORWARD) \
    cb(DEFORMABLE_CONV_BACKWARD_DATA) \
    cb(DEFORMABLE_CONV_BACKWARD_FILTER) \
    cb(BATCH_CONV_FORWARD) \
    cb(CONVBIAS_FORWARD)

#define FOREACH_OPR_TYPE_WITH_STMT(cb, stmt) \
    cb(MATRIX_MUL_FORWARD, stmt) \
    cb(BATCHED_MATRIX_MUL_FORWARD, stmt) \
    cb(CONVOLUTION_FORWARD, stmt) \
    cb(CONVOLUTION_BACKWARD_DATA, stmt) \
    cb(CONVOLUTION_BACKWARD_FILTER, stmt) \
    cb(CONVOLUTION3D_FORWARD, stmt) \
    cb(CONVOLUTION3D_BACKWARD_DATA, stmt) \
    cb(CONVOLUTION3D_BACKWARD_FILTER, stmt) \
    cb(LOCAL_SHARE_FORWARD, stmt) \
    cb(LOCAL_SHARE_BACKWARD_DATA, stmt) \
    cb(LOCAL_SHARE_BACKWARD_FILTER, stmt) \
    cb(DEFORMABLE_CONV_FORWARD, stmt) \
    cb(DEFORMABLE_CONV_BACKWARD_DATA, stmt) \
    cb(DEFORMABLE_CONV_BACKWARD_FILTER, stmt) \
    cb(BATCH_CONV_FORWARD, stmt) \
    cb(CONVBIAS_FORWARD, stmt)

// clang-format on

#define _OPR_TYPE_CASE(_opr_type, _stmt)             \
    case Algorithm::OprType::_opr_type: {            \
        using _Opr = typename OprFromOprTypeTrait<   \
                Algorithm::OprType::_opr_type>::Opr; \
        _stmt;                                       \
        break;                                       \
    }

#define FOREACH_OPR_TYPE_DISPATCH(_search_items, _stmt)          \
    for (size_t _item_idx = 0; _item_idx < _search_items.size(); \
         _item_idx++) {                                          \
        auto&& _item = _search_items[_item_idx];                 \
        switch (_item.opr_type) {                                \
            FOREACH_OPR_TYPE_WITH_STMT(_OPR_TYPE_CASE, _stmt)    \
            default:                                             \
                megdnn_throw("unknown opr_type");                \
        }                                                        \
    }

template <typename Opr, size_t arity = OprTrait<Opr>::arity,
          bool has_workspace = OprTrait<Opr>::has_workspace,
          bool can_deduce_layout = OprTrait<Opr>::can_deduce_layout>
struct OprProxyDefaultImpl
        : public DeduceLayoutProxy<Opr, arity, can_deduce_layout>,
          public ExecProxy<Opr, arity, has_workspace> {};

template <typename Opr>
struct OprProxy : public OprProxyDefaultImpl<Opr> {};

template <typename Opr>
struct OprWeightPreprocessProxy : public OprProxyDefaultImpl<Opr> {};

template <typename Opr>
struct OprProxyVectorToSingle {};

template <>
struct OprProxy<ElemwiseForward> {
    static void deduce_layout(ElemwiseForward* opr,
                              TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() >= 2);
        auto inp = layouts;
        inp.pop_back();
        opr->deduce_layout(inp, layouts.back());
    }

    static void exec(ElemwiseForward* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() >= 2);
        auto inp = tensors;
        inp.pop_back();
        opr->exec(inp, tensors.back());
    }
};

template <>
struct OprProxy<ElemwiseMultiType> {
    static void deduce_layout(ElemwiseMultiType* opr,
                              TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() >= 2);
        auto inp = layouts;
        inp.pop_back();
        opr->deduce_layout(inp, layouts.back());
    }

    static void exec(ElemwiseMultiType* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() >= 2);
        auto inp = tensors;
        inp.pop_back();
        opr->exec(inp, tensors.back());
    }
};

template <>
struct OprProxy<ConcatForward> {
    static void deduce_layout(ConcatForward* opr, TensorLayoutArray& layouts) {
        megdnn_assert(layouts.size() >= 2);
        auto inp = layouts;
        inp.pop_back();
        opr->deduce_layout(inp, layouts.back());
    }

    static void exec(ConcatForward* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() >= 2);
        auto inp = tensors;
        inp.pop_back();

        TensorLayoutArray layouts(tensors.size());
        std::transform(tensors.begin(), tensors.end(), layouts.begin(),
                       [](const TensorND& tensor) { return tensor.layout; });
        auto inp_layouts = layouts;
        inp_layouts.pop_back();

        WorkspaceWrapper W(opr->handle(), opr->get_workspace_in_bytes(
                                                  inp_layouts, layouts.back()));

        auto inp_tensors = tensors;
        inp_tensors.pop_back();
        opr->exec(inp_tensors, tensors.back(), W.workspace());
    }
};

template <>
struct OprProxy<SplitForward> : DeduceLayoutProxy<SplitForward, 0, false> {
    static void exec(SplitForward* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() >= 2);
        auto out = tensors;
        out.erase(out.begin());

        TensorLayoutArray layouts(tensors.size());
        std::transform(tensors.begin(), tensors.end(), layouts.begin(),
                       [](const TensorND& tensor) { return tensor.layout; });
        auto out_layouts = layouts;
        out_layouts.erase(out_layouts.begin());

        WorkspaceWrapper W(
                opr->handle(),
                opr->get_workspace_in_bytes(layouts.front(), out_layouts));

        auto out_tensors = tensors;
        out_tensors.erase(out_tensors.begin());
        opr->exec(tensors.front(), out_tensors, W.workspace());
    }
};

//! OprProxy impl for tenary oprs with profiling support
template <class Opr>
struct OprProxyProfilingBase
        : public DeduceLayoutProxy<Opr, OprTrait<Opr>::arity,
                                   OprTrait<Opr>::can_deduce_layout> {
    static constexpr int arity = OprTrait<Opr>::arity;
    size_t warmup_times = 10, exec_times = 100;

    //! whether to enable profiling
    bool m_profiling;
    WorkspaceWrapper W;

    //! target algo setup by profiler; it can also be directly specified by the
    //! caller
    ExecutionPolicy target_execution_policy;

    OprProxyProfilingBase(bool profile = false) { m_profiling = profile; }

    //! used for alloc tensor for weight preprocess
    static std::shared_ptr<TensorNDArray> alloc_tensors(
            Handle* handle, const TensorLayoutArray& layouts) {
        auto deleter = [handle](TensorNDArray* ptr) {
            for (auto&& i : *ptr) {
                auto pdata = static_cast<dt_byte*>(i.raw_ptr) +
                             i.layout.span().low_byte;
                megdnn_free(handle, pdata);
            }
            delete ptr;
        };
        std::shared_ptr<TensorNDArray> ret{new TensorNDArray, deleter};
        for (size_t i = 0; i < layouts.size(); ++i) {
            auto span = layouts[i].span();
            ret->emplace_back(static_cast<dt_byte*>(
                                      megdnn_malloc(handle, span.dist_byte())) -
                                      span.low_byte,
                              layouts[i]);
        }
        return ret;
    }

    /**
     * flatten search space in postorder traversal
     * The subopr search construct a search tree
     *
     *           A
     *        /    \
     *       B1B2   C
     *      /     \
     *     D1D2D3   E
     * We use postorder traverse the search tree.
     * D1 -> D2 -> D3 -> E -> B1 -> B2 -> C -> A
     */
    static std::vector<Algorithm::SearchItem> flatten_search_space(
            const TensorLayoutArray layouts, const std::string& param,
            Handle* handle) {
        megdnn_assert(layouts.size() == arity);
        auto opr = handle->create_operator<Opr>();
        opr->param() =
                Algorithm::deserialize_read_pod<typename Opr::Param>(param);

        std::vector<Algorithm::SearchItem> ret;
        for (auto algo_info : AlgoProxy<Opr, arity>::get_all_algorithms_info(
                     opr.get(), layouts)) {
            Algorithm* algo = opr->get_algorithm_from_desc(algo_info.desc);
            std::vector<Algorithm::SearchItem>&& sub_items =
                    algo->get_subopr_list(layouts, opr.get());

            FOREACH_OPR_TYPE_DISPATCH(sub_items, {
                auto space = OprProxyProfilingBase<_Opr>::flatten_search_space(
                        _item.layouts, _item.param, handle);
                ret.insert(ret.end(), space.begin(), space.end());
            });
        }
        ret.push_back({OprTypeFromOprTrait<Opr>::opr_type, param, layouts});
        return ret;
    }

    static void construct_execution_policy(
            const TensorLayoutArray& layouts, const std::string& param,
            Handle* handle, FastRunCache& cache,
            ExecutionPolicy& policy) {
        megdnn_assert(layouts.size() == arity);
        auto opr = handle->create_operator<Opr>();
        opr->param() =
                Algorithm::deserialize_read_pod<typename Opr::Param>(param);
        if (!policy.algo.valid()) {
            policy.algo = cache.get(Algorithm::SearchItem{
                    OprTypeFromOprTrait<Opr>::opr_type, param, layouts});
            megdnn_assert(policy.algo.valid(),
                          "No cache found, maybe some error occured in "
                          "flatten_search_space or get_subopr_list");
        }
        policy.sub_policy.clear();
        Algorithm* algo = opr->get_algorithm_from_desc(policy.algo);
        std::vector<Algorithm::SearchItem>&& sub_items =
                algo->get_subopr_list(layouts, opr.get());
        FOREACH_OPR_TYPE_DISPATCH(sub_items, {
            policy.sub_policy.push_back({});
            OprProxyProfilingBase<_Opr>::construct_execution_policy(
                    _item.layouts, _item.param, handle, cache,
                    policy.sub_policy.back());
        });
        return;
    }

    /**
     * \brief search and get the best execution_policy
     */
    static void search(const TensorLayoutArray& layouts,
                       const std::string& param,
                       WorkspaceWrapper& workspace_wrapper, Handle* handle,
                       size_t warmup_times, size_t exec_times,
                       FastRunCache& cache) {
        megdnn_assert(layouts.size() == arity);
        auto opr = handle->create_operator<Opr>();

        opr->param() =
                Algorithm::deserialize_read_pod<typename Opr::Param>(param);
        SmallVector<size_t> sizes_in_bytes;
        for (const auto& layout : layouts) {
            sizes_in_bytes.push_back(layout.span().dist_byte());
        }

        float min_time = std::numeric_limits<float>::max();
        Algorithm::Info::Desc best_algo;

        std::string log_info = "Profiling start: ";
        for (auto&& layout : layouts) {
            log_info += layout.to_string() + " ";
        }
        megdnn_log("%s", log_info.c_str());
        best_algo = cache.get(Algorithm::SearchItem{
                OprTypeFromOprTrait<Opr>::opr_type, param, layouts});

        if (best_algo.valid()) {
            auto&& algo = opr->get_algorithm_from_desc(best_algo);
            MEGDNN_MARK_USED_VAR(algo);
            megdnn_log("Find best algo %s in cache", algo->name());
            return;
        }
        for (auto algo : AlgoProxy<Opr, arity>::get_all_algorithms_info(
                     opr.get(), layouts)) {
            //! construct execution_policy
            opr->execution_policy().algo = algo.desc;
            construct_execution_policy(layouts, param, handle, cache,
                                       opr->execution_policy());

            auto workspace_size = AlgoProxy<Opr, arity>::get_workspace_in_bytes(
                    opr.get(), layouts);
            sizes_in_bytes.push_back(workspace_size);

            WorkspaceBundle wb(nullptr, sizes_in_bytes);
            workspace_wrapper.update(wb.total_size_in_bytes());
            wb.set(workspace_wrapper.workspace().raw_ptr);
            TensorNDArray tensors;
            for (size_t i = 0; i < arity; i++) {
                tensors.push_back({wb.get(i), layouts[i]});
            }

            for (size_t times = 0; times < warmup_times; ++times) {
                AlgoProxy<Opr, arity>::exec(opr.get(), tensors,
                                            wb.get_workspace(arity));
            }
            megcoreSynchronize(opr->handle()->megcore_computing_handle());
            Timer timer;
            timer.start();
            for (size_t times = 0; times < exec_times; ++times) {
                AlgoProxy<Opr, arity>::exec(opr.get(), tensors,
                                            wb.get_workspace(arity));
            }
            megcoreSynchronize(opr->handle()->megcore_computing_handle());
            timer.stop();
            megdnn_log("%.3fms %s", timer.get_time_in_us() / 1e3,
                       algo.desc.name.c_str());
            if (min_time > timer.get_time_in_us()) {
                min_time = timer.get_time_in_us();
                best_algo = algo.desc;
            }

            sizes_in_bytes.pop_back();
        }
        auto&& algo = opr->get_algorithm_from_desc(best_algo);
        MEGDNN_MARK_USED_VAR(algo);
        megdnn_log("Profiling end, got best algo: %s", algo->name());
        cache.put(Algorithm::SearchItem{OprTypeFromOprTrait<Opr>::opr_type,
                                        param, layouts},
                  best_algo);
    }

    void exec(Opr* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() == arity);
        if (!W.valid()) {
            W = WorkspaceWrapper(opr->handle(), 0);
        }
        TensorLayoutArray layouts;
        for (auto&& tensor : tensors) {
            layouts.push_back(tensor.layout);
        }
        if (m_profiling && !target_execution_policy.algo.valid()) {
            FastRunCache cache;
            std::string param_str;
            Algorithm::serialize_write_pod(opr->param(), param_str);
            auto&& search_items =
                    flatten_search_space(layouts, param_str, opr->handle());
            FOREACH_OPR_TYPE_DISPATCH(search_items, {
                OprProxyProfilingBase<_Opr>::search(
                        _item.layouts, _item.param, W, opr->handle(),
                        warmup_times, exec_times, cache);
            });

            construct_execution_policy(layouts, param_str, opr->handle(), cache,
                                       opr->execution_policy());
            target_execution_policy = opr->execution_policy();
            auto workspace_size =
                    AlgoProxy<Opr, arity>::get_workspace_in_bytes(opr, layouts);
            W.update(workspace_size);
        }
        if (!target_execution_policy.algo.valid()) {
            auto workspace_size =
                    AlgoProxy<Opr, arity>::get_workspace_in_bytes(opr, layouts);
            W.update(workspace_size);
        }
        AlgoProxy<Opr, arity>::exec(opr, tensors, W.workspace());
    }
};

#define DEF_PROF(c)                                            \
    template <>                                                \
    struct OprProxy<c> : public OprProxyProfilingBase<c> {     \
        using OprProxyProfilingBase<c>::OprProxyProfilingBase; \
    }

DEF_PROF(MatrixMulForward);
DEF_PROF(ConvolutionForward);
DEF_PROF(ConvolutionBackwardData);
DEF_PROF(ConvolutionBackwardFilter);
DEF_PROF(LocalShareForward);
DEF_PROF(LocalShareBackwardData);
DEF_PROF(LocalShareBackwardFilter);

DEF_PROF(DeformableConvForward);
DEF_PROF(DeformableConvBackwardFilter);
DEF_PROF(BatchConvBiasForward);
DEF_PROF(ConvBiasForward);

DEF_PROF(DeformableConvBackwardData);
#undef DEF_PROF

template <class Opr>
struct OprWeightPreprocessProxyImpl : public OprProxyProfilingBase<Opr> {
    using Base = OprProxyProfilingBase<Opr>;
    static constexpr int arity = OprTrait<Opr>::arity;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        megdnn_assert(tensors.size() == arity);
        if (!Base::W.valid()) {
            Base::W = WorkspaceWrapper(opr->handle(), 0);
        }

        TensorLayoutArray layouts;
        for (auto&& tensor : tensors) {
            layouts.push_back(tensor.layout);
        }
        if (Base::m_profiling && !Base::target_execution_policy.algo.valid()) {
            size_t min_time = std::numeric_limits<size_t>::max();
            for (auto algo :
                 AlgoProxy<Opr, arity>::get_all_algorithms_info(opr, layouts)) {
                opr->execution_policy().algo = algo.desc;

                auto preprocess_tensors =
                        weight_prerocess(opr, tensors, algo.desc);
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                typename Opr::PreprocessedFilter preprocessed_filter{
                        nullptr, *preprocess_tensors};

                auto workspace_size =
                        AlgoProxy<Opr, arity>::get_workspace_in_bytes(
                                opr, layouts, &preprocessed_filter);
                Base::W.update(workspace_size);

                for (size_t times = 0; times < Base::warmup_times; ++times) {
                    AlgoProxy<Opr, arity>::exec(opr, tensors,
                                                &preprocessed_filter,
                                                Base::W.workspace());
                }
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                Timer timer;
                timer.start();
                for (size_t times = 0; times < Base::exec_times; ++times) {
                    AlgoProxy<Opr, arity>::exec(opr, tensors,
                                                &preprocessed_filter,
                                                Base::W.workspace());
                }
                megcoreSynchronize(opr->handle()->megcore_computing_handle());
                timer.stop();
                printf("%.3fms %s\n", timer.get_time_in_us() / 1e3,
                       algo.desc.name.c_str());
                if (min_time > timer.get_time_in_us()) {
                    min_time = timer.get_time_in_us();
                    Base::target_execution_policy.algo = algo.desc;
                }
            }
            opr->execution_policy() = Base::target_execution_policy;
            auto preprocess_tensors = weight_prerocess(
                    opr, tensors, Base::target_execution_policy.algo);
            megcoreSynchronize(opr->handle()->megcore_computing_handle());
            typename Opr::PreprocessedFilter preprocessed_filter{
                    nullptr, *preprocess_tensors};
            auto workspace_size = AlgoProxy<Opr, arity>::get_workspace_in_bytes(
                    opr, layouts, &preprocessed_filter);
            Base::W.update(workspace_size);
        }
        auto preprocess_tensors = weight_prerocess(
                opr, tensors, Base::target_execution_policy.algo);
        megcoreSynchronize(opr->handle()->megcore_computing_handle());
        typename Opr::PreprocessedFilter preprocessed_filter{
                nullptr, *preprocess_tensors};
        if (!Base::target_execution_policy.algo.valid()) {
            auto workspace_size = AlgoProxy<Opr, arity>::get_workspace_in_bytes(
                    opr, layouts, &preprocessed_filter);
            Base::W.update(workspace_size);
        }
        AlgoProxy<Opr, arity>::exec(opr, tensors, &preprocessed_filter,
                                    Base::W.workspace());
    }

    //! handle weight preprocess
    std::shared_ptr<TensorNDArray> weight_prerocess(
            Opr* opr, const TensorNDArray& tensors,
            const typename Opr::AlgorithmDesc&) {
        TensorLayoutArray layouts;
        for (auto&& tensor : tensors) {
            layouts.push_back(tensor.layout);
        }
        auto weight_perprocess_layouts =
                AlgoProxy<Opr, arity>::deduce_preprocessed_filter_layout(
                        opr, layouts);
        auto preprocessed_filter_tensors_ptr =
                Base::alloc_tensors(opr->handle(), weight_perprocess_layouts);
        typename Opr::PreprocessedFilter preprocessed_filter{
                nullptr, *preprocessed_filter_tensors_ptr};
        size_t preprocess_workspace_size =
                AlgoProxy<Opr, arity>::get_preprocess_workspace_in_bytes(
                        opr, layouts);
        WorkspaceWrapper preprocess_workspace(opr->handle(),
                                              preprocess_workspace_size);
        AlgoProxy<Opr, arity>::exec_preprocess(
                opr, tensors, layouts, &preprocessed_filter,
                preprocess_workspace.workspace());
        return preprocessed_filter_tensors_ptr;
    }
};

#define DEF_PROF(c)                                                          \
    template <>                                                              \
    struct OprWeightPreprocessProxy<c>                                       \
            : public OprWeightPreprocessProxyImpl<c> {                       \
        using OprWeightPreprocessProxyImpl<c>::OprWeightPreprocessProxyImpl; \
    }

DEF_PROF(ConvolutionForward);
DEF_PROF(ConvBias);
#undef DEF_PROF

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
