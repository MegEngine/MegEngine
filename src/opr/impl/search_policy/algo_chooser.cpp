/**
 * \file src/opr/impl/search_policy/algo_chooser.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/search_policy/algo_chooser.h"
#include <limits>
#include <unordered_set>
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/search_policy/algo_chooser_helper.h"
#include "megbrain/opr/search_policy/profiler.h"

#include "../internal/invoke.h"
#include "../internal/megdnn_opr_wrapper.inl"
#include "./workspace_need_limit_getter.inl"

//! TODO: here has to be know some megdnn::opr when there is produced midout.h
//! fix it if there is another graceful way.
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "megdnn/oprs/base.h"
#include "midout.h"
MIDOUT_DECL(megbrain_opr_algo_chooser)
#define MIDOUT_B(...) MIDOUT_BEGIN(megbrain_opr_algo_chooser, __VA_ARGS__) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using mgb::opr::intl::WorkspaceLimitGetter;
using namespace megdnn;
using namespace mgb;

#define APPLY(statement, ...)                                  \
    mgb::apply([&](const auto&... args) { return statement; }, \
               std::tuple_cat(__VA_ARGS__))

// timeout delta to be added with fastest known algorithm for new algos
constexpr double TIMEOUT_TOLERANCE = 2;

#define CACHE_KEY_VERSION "v5"

namespace {
template <typename Opr>
std::string profile_name(Opr* opr) {
    std::string ret =
            std::string(MegDNNOpr2MGBOpr<Opr>::MGBOpr::typeinfo()->name) +
            CACHE_KEY_VERSION;
    ret.append(opr->get_algorithm_set_name());
    return ret;
}

template <typename Opr>
std::string format_fixlayouts(
        const typename opr::AlgoChooser<Opr>::FixedTensorLayouts& layouts,
        size_t arity_in, size_t arity_out,
        const std::string& delimiter = " -> ") {
    std::string ret;
    if (arity_in) {
        ret.append("(");
        for (size_t i = 0; i < arity_in; ++i) {
            if (i) {
                ret.append(", ");
            }
            ret.append(layouts[i].to_string() + " ");
        }
        ret.append(")");
    }
    if (arity_in && arity_out) {
        ret.append(delimiter);
    }
    if (arity_out) {
        ret.append("(");
        for (size_t i = 0; i < arity_out; ++i) {
            if (i) {
                ret.append(", ");
            }
            ret.append(layouts[i + arity_in].to_string() + " ");
        }
        ret.append(")");
    }
    return ret;
}

/**
 * \brief Check if the sub opr list has circular dependence.
 */
class CircularDepsChecker {
    struct SearchItemStorage {
        std::string data_hold;
        size_t hash = 0;

        SearchItemStorage(const Algorithm::SearchItem& item) {
            Algorithm::serialize_write_pod(item.opr_type, data_hold);
            for (auto&& layout : item.layouts) {
                data_hold += layout.serialize();
            }
            data_hold += item.param;
        }

        SearchItemStorage& init_hash() {
            hash = XXHash64CT::hash(data_hold.data(), data_hold.size(),
                                    20201225);
            return *this;
        }

        bool operator==(const SearchItemStorage& rhs) const {
            return data_hold == rhs.data_hold;
        }

        struct Hash {
            size_t operator()(const SearchItemStorage& s) const {
                return s.hash;
            }
        };
    };
    std::unordered_set<SearchItemStorage, SearchItemStorage::Hash> m_set;

public:
    void put(const megdnn::Algorithm::SearchItem& key) {
        SearchItemStorage key_storage(key);
        key_storage.init_hash();
        mgb_assert(m_set.find(key_storage) == m_set.end(),
                   "Circular dependency during flatten search space");
        auto ret = m_set.insert(std::move(key_storage));
        mgb_assert(ret.second);
    }
    void remove(const megdnn::Algorithm::SearchItem& key) {
        SearchItemStorage key_storage(key);
        key_storage.init_hash();
        auto&& iter = m_set.find(key_storage);
        mgb_assert(iter != m_set.end());
        m_set.erase(iter);
    }
};

///////////////// OprTypeTrait /////////////////////////////
template <megdnn::Algorithm::OprType>
struct OprFromOprTypeTrait;

template <typename Opr>
struct OprTypeFromOprTrait;

#define cb(_opr_type, _opr)                                             \
    template <>                                                         \
    struct OprFromOprTypeTrait<megdnn::Algorithm::OprType::_opr_type> { \
        using Opr = megdnn::_opr;                                       \
    };                                                                  \
    template <>                                                         \
    struct OprTypeFromOprTrait<megdnn::_opr> {                          \
        constexpr static megdnn::Algorithm::OprType opr_type =          \
                megdnn::Algorithm::OprType::_opr_type;                  \
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
cb(POOLING_FORWARD, PoolingForward);
cb(POOLING_BACKWARD, PoolingBackward);

#undef cb

// clang-format off
#define FOREACH_OPR_TYPE_WITH_STMT(cb, stmt)  \
    cb(MATRIX_MUL_FORWARD, stmt)              \
    cb(BATCHED_MATRIX_MUL_FORWARD, stmt)      \
    cb(CONVOLUTION_FORWARD, stmt)             \
    cb(CONVOLUTION_BACKWARD_DATA, stmt)       \
    cb(CONVOLUTION_BACKWARD_FILTER, stmt)     \
    cb(CONVOLUTION3D_FORWARD, stmt)           \
    cb(CONVOLUTION3D_BACKWARD_DATA, stmt)     \
    cb(CONVOLUTION3D_BACKWARD_FILTER, stmt)   \
    cb(LOCAL_SHARE_FORWARD, stmt)             \
    cb(LOCAL_SHARE_BACKWARD_DATA, stmt)       \
    cb(LOCAL_SHARE_BACKWARD_FILTER, stmt)     \
    cb(DEFORMABLE_CONV_FORWARD, stmt)         \
    cb(DEFORMABLE_CONV_BACKWARD_DATA, stmt)   \
    cb(DEFORMABLE_CONV_BACKWARD_FILTER, stmt) \
    cb(BATCH_CONV_FORWARD, stmt)              \
    cb(CONVBIAS_FORWARD, stmt)                \
    cb(POOLING_FORWARD, stmt)                 \
    cb(POOLING_BACKWARD, stmt)
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
                mgb_throw(MegBrainError, "unknown opr_type");    \
        }                                                        \
    }

template <typename Opr>
TensorLayoutArray to_layout_array(
        const typename opr::AlgoChooser<Opr>::FixedTensorLayouts& layouts) {
    TensorLayoutArray ret;
    for (auto&& layout : layouts) {
        ret.push_back(layout);
    }
    return ret;
}

template <typename Opr>
typename opr::AlgoChooser<Opr>::FixedTensorLayouts to_fixed_layouts(
        const TensorLayoutArray& layouts) {
    typename opr::AlgoChooser<Opr>::FixedTensorLayouts ret;
    mgb_assert(ret.size() == layouts.size());
    size_t idx = 0;
    for (auto&& layout : layouts) {
        ret[idx++] = layout;
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
template <typename Opr>
std::vector<megdnn::Algorithm::SearchItem> flatten_search_space(
        const typename opr::AlgoChooser<Opr>::AlgoChooserHelper& helper,
        CircularDepsChecker& checker) {
    auto&& search_item = megdnn::Algorithm::SearchItem{
            OprTypeFromOprTrait<Opr>::opr_type, helper.param(),
            to_layout_array<Opr>(helper.fastrun_layouts())};
    checker.put(search_item);
    std::vector<megdnn::Algorithm::SearchItem> ret;
    for (auto algo_info : helper.get_all_candidates()) {
        megdnn::Algorithm* algo =
                helper.get_algorithm_from_desc(algo_info.desc);
        mgb_assert(algo, "Unknown algo description");
        std::vector<megdnn::Algorithm::SearchItem>&& sub_items =
                algo->get_subopr_list(
                        to_layout_array<Opr>(helper.fastrun_layouts()),
                        helper.megdnn_opr());

        FOREACH_OPR_TYPE_DISPATCH(sub_items, {
            auto&& megdnn_opr =
                    opr::intl::create_megdnn_opr<_Opr>(helper.comp_node());
            megdnn_opr->param() =
                    Algorithm::deserialize_read_pod<typename _Opr::Param>(
                            _item.param);
            typename opr::AlgoChooser<_Opr>::AlgoChooserHelper sub_helper(
                    to_fixed_layouts<_Opr>(_item.layouts), megdnn_opr.get(),
                    _item.param, helper.mgb_opr(), helper.comp_node(),
                    helper.execution_policy(),
                    helper.allow_weight_preprocess());
            auto space = flatten_search_space<_Opr>(sub_helper, checker);
            ret.insert(ret.end(), space.begin(), space.end());
        });
    }
    ret.push_back(search_item);
    checker.remove(search_item);
    return ret;
}

//! serialize a algo's desc to string. format is
//! handle_type|algo_type|size_of_param|size_of_name|string_of_param|string_of_name
static void serialize_write_pod(const Algorithm::Info::Desc& val,
                                std::string& result) {
    megdnn::Algorithm::serialize_write_pod(val.handle_type, result);
    megdnn::Algorithm::serialize_write_pod(val.type, result);
    uint32_t param_size = val.param.size();
    uint32_t name_size = val.name.size();
    megdnn::Algorithm::serialize_write_pod<uint32_t>(param_size, result);
    megdnn::Algorithm::serialize_write_pod<uint32_t>(name_size, result);
    result += val.param;
    result += val.name;
}

static Algorithm::Info::Desc deserialize_read_pod(const std::string& data,
                                                  size_t offset = 0) {
    Algorithm::Info::Desc ret;
#define cb(_val, _type)                                                \
    _val = megdnn::Algorithm::deserialize_read_pod<_type>(data.data(), \
                                                          offset);     \
    offset += sizeof(_val)

    cb(ret.handle_type, megdnn::Handle::HandleType);
    cb(ret.type, uint32_t);

    uint32_t param_size = 0;
    uint32_t name_size = 0;
    cb(param_size, uint32_t);
    cb(name_size, uint32_t);

    if (param_size > 0) {
        ret.param = std::string(data.data() + offset, param_size);
        offset += param_size;
    }
    if (name_size > 0) {
        ret.name = std::string(data.data() + offset, name_size);
        offset += name_size;
    }
    return ret;
}

}  // namespace

namespace mgb {
namespace opr {
template <class Opr>
class LayoutsModifier {
    using FixedTensorLayouts = typename AlgoChooser<Opr>::FixedTensorLayouts;

public:
    static void on(FixedTensorLayouts&, const typename Opr::Param&, size_t) {}

private:
    //! index of batch in tensor, 3 for CHWN4 e.g.
    static size_t index_of_batch(const typename Opr::Param&) { return 0; }

    //! indices contain batch in inputs and outputs, src(0) dst(2) for conv e.g.
    static std::vector<size_t> sm_indices_contain_batch;
};
template <class Opr>
std::vector<size_t> LayoutsModifier<Opr>::sm_indices_contain_batch = {};

#define DEFAULT_OPR_WITHOUT_INPUT_BROADCAST(opr, idxs)                       \
    template <>                                                              \
    class LayoutsModifier<opr> {                                             \
    public:                                                                  \
        using FixedTensorLayouts =                                           \
                typename AlgoChooser<opr>::FixedTensorLayouts;               \
        static void on(FixedTensorLayouts& layouts, const opr::Param& param, \
                       size_t new_batch_size) {                              \
            size_t batch_index = index_of_batch(param);                      \
            for (size_t index : sm_indices_contain_batch) {                  \
                layouts.at(index)[batch_index] = new_batch_size;             \
            }                                                                \
        }                                                                    \
                                                                             \
    private:                                                                 \
        static size_t index_of_batch(const opr::Param&) { return 0; }        \
        static std::vector<size_t> sm_indices_contain_batch;                 \
    };                                                                       \
    std::vector<size_t> LayoutsModifier<opr>::sm_indices_contain_batch = idxs;

DEFAULT_OPR_WITHOUT_INPUT_BROADCAST(megdnn::Convolution3DForward,
                                    (std::initializer_list<size_t>{0, 2}))
DEFAULT_OPR_WITHOUT_INPUT_BROADCAST(megdnn::Convolution3DBackwardData,
                                    (std::initializer_list<size_t>{1, 2}))
DEFAULT_OPR_WITHOUT_INPUT_BROADCAST(megdnn::Convolution3DBackwardFilter,
                                    (std::initializer_list<size_t>{0, 1}))
DEFAULT_OPR_WITHOUT_INPUT_BROADCAST(megdnn::BatchedMatrixMul,
                                    (std::initializer_list<size_t>{0, 1, 2}))
#undef DEFAULT_OPR_WITHOUT_INPUT_BROADCAST

#define CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST(opr, idxs)                     \
    template <>                                                              \
    class LayoutsModifier<opr> {                                             \
    public:                                                                  \
        using FixedTensorLayouts =                                           \
                typename AlgoChooser<opr>::FixedTensorLayouts;               \
        static void on(FixedTensorLayouts& layouts, const opr::Param& param, \
                       size_t new_batch_size) {                              \
            size_t batch_index = index_of_batch(param);                      \
            for (size_t index : sm_indices_contain_batch) {                  \
                layouts.at(index)[batch_index] = new_batch_size;             \
            }                                                                \
        }                                                                    \
                                                                             \
    private:                                                                 \
        static size_t index_of_batch(const opr::Param& param) {              \
            if (param.format == opr::Param::Format::CHWN4) {                 \
                return 3;                                                    \
            }                                                                \
            return 0;                                                        \
        }                                                                    \
        static std::vector<size_t> sm_indices_contain_batch;                 \
    };                                                                       \
    std::vector<size_t> LayoutsModifier<opr>::sm_indices_contain_batch = idxs;

CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST(megdnn::ConvolutionForward,
                                      (std::initializer_list<size_t>{0, 2}))
CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST(megdnn::ConvolutionBackwardData,
                                      (std::initializer_list<size_t>{1, 2}))
CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST(megdnn::ConvolutionBackwardFilter,
                                      (std::initializer_list<size_t>{0, 1}))
CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST(megdnn::LocalShareForward,
                                      (std::initializer_list<size_t>{0, 2}))
CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST(megdnn::LocalShareBackwardData,
                                      (std::initializer_list<size_t>{1, 2}))
CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST(megdnn::LocalShareBackwardFilter,
                                      (std::initializer_list<size_t>{0, 1}))
CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST(megdnn::DeformableConvForward,
                                      (std::initializer_list<size_t>{0, 2, 3,
                                                                     4}))
CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST(megdnn::DeformableConvBackwardData,
                                      (std::initializer_list<size_t>{0, 2, 3, 4,
                                                                     5, 6, 7}))
CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST(megdnn::DeformableConvBackwardFilter,
                                      (std::initializer_list<size_t>{0, 1, 2,
                                                                     3}))
CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST(megdnn::BatchConvBiasForward,
                                      (std::initializer_list<size_t>{0, 1, 2, 3,
                                                                     4}))
#undef CONV_LIKE_OPR_WITHOUT_INPUT_BROADCAST

template <>
class LayoutsModifier<megdnn::ConvBiasForward> {
public:
    using FixedTensorLayouts =
            typename AlgoChooser<megdnn::ConvBiasForward>::FixedTensorLayouts;
    static void on(FixedTensorLayouts& layouts,
                   const megdnn::ConvBiasForward::Param& param,
                   size_t new_batch_size) {
        size_t batch_index = index_of_batch(param);
        for (size_t index : sm_indices_contain_batch) {
            layouts.at(index)[batch_index] = new_batch_size;
        }
        for (size_t index : sm_indices_contain_batch_broadcast) {
            if (!check_bias_share_in_channel(layouts.at(index), param.format)) {
                layouts.at(index)[batch_index] = new_batch_size;
            }
        }
    }

private:
    static std::vector<size_t> sm_indices_contain_batch;
    static std::vector<size_t> sm_indices_contain_batch_broadcast;
    static size_t index_of_batch(const megdnn::ConvBiasForward::Param& param) {
        if (param.format == megdnn::ConvBiasForward::Param::Format::CHWN4) {
            return 3;
        }
        return 0;
    }
};
std::vector<size_t>
        LayoutsModifier<megdnn::ConvBiasForward>::sm_indices_contain_batch = {
                0, 3, 4};
std::vector<size_t> LayoutsModifier<
        megdnn::ConvBiasForward>::sm_indices_contain_batch_broadcast = {2};

template <>
class LayoutsModifier<megdnn::MatrixMul> {
public:
    using FixedTensorLayouts=
            typename AlgoChooser<megdnn::MatrixMul>::FixedTensorLayouts;
    static void on(FixedTensorLayouts& layouts,
                   const megdnn::MatrixMul::Param& param,
                   size_t new_batch_size) {
        //! Because we do not know whether the batch size is in the dimension m
        //! or the dimension n, we just ignore both m and n here.
        // FIXME Find a way to make mgb obtain batch size information from R or
        // automatically
        layouts.at(2)[0] = new_batch_size;
        layouts.at(2)[1] = new_batch_size;
        if (param.transposeA) {
            layouts.at(0)[1] = new_batch_size;
        } else {
            layouts.at(0)[0] = new_batch_size;
        }
        if (param.transposeB) {
            layouts.at(1)[0] = new_batch_size;
        } else {
            layouts.at(1)[1] = new_batch_size;
        }
    }
};

///////////////////////////// AlgoChooserHelper //////////////////////////
template <typename Opr>
AlgoChooser<Opr>::AlgoChooserHelper::AlgoChooserHelper(
        const FixedTensorLayouts& layouts, Opr* megdnn_opr,
        const std::string& param_str, const cg::OperatorNodeBase* mgb_opr,
        const CompNode& cn,
        const megdnn::param::ExecutionPolicy& execution_policy,
        bool allow_weight_preprocess)
        : m_fastrun_layouts{layouts},
          m_incache_layouts{layouts},
          m_dnn_opr{megdnn_opr},
          m_param{param_str},
          m_base_mgb_opr{mgb_opr},
          m_cn{cn},
          m_execution_policy{execution_policy},
          m_allow_weight_preprocess{allow_weight_preprocess} {
    auto fastrun_batch_size =
            owner_graph()->options().fast_run_config.shared_batch_size;

    if (fastrun_batch_size) {
        LayoutsModifier<Opr>::on(m_incache_layouts, m_dnn_opr->param(), 0);
        LayoutsModifier<Opr>::on(m_fastrun_layouts, m_dnn_opr->param(),
                                 fastrun_batch_size);
    }

    if (owner_graph()->options().no_profiling_on_shape_change) {
        for (size_t i = 0; i < m_incache_layouts.size(); i++) {
            for (size_t j = 0; j < m_incache_layouts.at(i).ndim; j++) {
                m_incache_layouts.at(i)[j] = 0;
                m_incache_layouts.at(i).stride[j] = 0;
            }
        }
    }

    mgb_assert(m_fastrun_layouts.size() == layouts.size());

    static_assert(
            std::tuple_size<FixedTensorLayouts>::value == 2 ||
                    std::tuple_size<FixedTensorLayouts>::value == 3 ||
                    std::tuple_size<FixedTensorLayouts>::value == 4 ||
                    std::tuple_size<FixedTensorLayouts>::value == 5 ||
                    std::tuple_size<FixedTensorLayouts>::value == 8,
            "Pooling assumes arity = 2 or 4,Convolution AlgoChooser assumes "
            "arity = 3 , 5 or 8 (for deformable conv)");
}

template <typename Opr>
typename AlgoChooser<Opr>::ImplExecutionPolicy
AlgoChooser<Opr>::AlgoChooserHelper::choose_by_heuristic(
        const ExecutionStrategy& selected_strategy) const {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("choose_by_heuristic")))
    ImplExecutionPolicy policy;
    auto workspace_limit = WorkspaceLimitGetter::get_workspace_limit(
            owner_graph(), m_cn, m_execution_policy.workspace_limit);
    auto attr = extract_algo_attribute(selected_strategy);
    policy.algo =
            APPLY(m_dnn_opr->get_algorithm_info_heuristic(
                          args..., workspace_limit, attr.first, attr.second),
                  m_fastrun_layouts)
                    .desc;

    Algorithm* algo = m_dnn_opr->get_algorithm_from_desc(policy.algo);
    mgb_assert(algo, "Unknown algo description");
    std::vector<Algorithm::SearchItem>&& sub_items = algo->get_subopr_list(
            to_layout_array<Opr>(m_fastrun_layouts), m_dnn_opr);

    FOREACH_OPR_TYPE_DISPATCH(sub_items, {
        auto&& megdnn_opr = intl::create_megdnn_opr<_Opr>(m_cn);
        megdnn_opr->param() =
                Algorithm::deserialize_read_pod<typename _Opr::Param>(
                        _item.param);
        typename AlgoChooser<_Opr>::AlgoChooserHelper sub_helper(
                to_fixed_layouts<_Opr>(_item.layouts), megdnn_opr.get(),
                _item.param, m_base_mgb_opr, m_cn, m_execution_policy,
                m_allow_weight_preprocess);
        policy.sub_policy.push_back(
                sub_helper.choose_by_heuristic(selected_strategy));
    });

    return policy;
    MIDOUT_E
}

template <typename Opr>
typename AlgoChooser<Opr>::ImplExecutionPolicy
AlgoChooser<Opr>::AlgoChooserHelper::choose_by_profile(
        const ExecutionStrategy& selected_strategy, bool enable_update) const {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("choose_by_profile")))
    if (owner_graph()->options().no_profiling_on_shape_change) {
        auto policy = m_dnn_opr->execution_policy();
        if (policy.algo.valid()) {
            return policy;
        }
    }

    typename AlgoChooser<Opr>::ImplExecutionPolicy tmp_policy;
    bool retrive_from_cache = true;
    bool allow_log = false;
    construct_execution_policy(selected_strategy, tmp_policy,
                               retrive_from_cache, allow_log);
    if (tmp_policy.algo.valid()) {
        // return policy when contruct successed
        return tmp_policy;
    }

    if (enable_update) {
        CircularDepsChecker circular_deps_checker;
        auto&& search_items =
                flatten_search_space<Opr>(*this, circular_deps_checker);
        FOREACH_OPR_TYPE_DISPATCH(search_items, {
            auto&& megdnn_opr = intl::create_megdnn_opr<_Opr>(m_cn);
            megdnn_opr->param() =
                    Algorithm::deserialize_read_pod<typename _Opr::Param>(
                            _item.param);
            typename AlgoChooser<_Opr>::AlgoChooserHelper sub_helper(
                    to_fixed_layouts<_Opr>(_item.layouts), megdnn_opr.get(),
                    _item.param, m_base_mgb_opr, m_cn, m_execution_policy,
                    m_allow_weight_preprocess);
            sub_helper.profile(selected_strategy);
        });
    }

    typename AlgoChooser<Opr>::ImplExecutionPolicy policy;
    construct_execution_policy(selected_strategy, policy);
    return policy;
    MIDOUT_E
}

template <typename Opr>
typename AlgoChooser<Opr>::ImplAlgoDesc
AlgoChooser<Opr>::AlgoChooserHelper::get_profile_result_from_cache(
        const ExecutionStrategy& selected_strategy) const {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("get_profile_result_from_cache")))
    AlgoChooserProfileCache cache(m_cn, profile_name(m_dnn_opr).c_str());

    typename Opr::Param origin_param = m_dnn_opr->param();
    AlgoChooserProfileCache::Key cache_key{m_incache_layouts.data(),
                                           m_incache_layouts.size(),
                                           &origin_param, sizeof(origin_param)};
    auto&& rst = cache.get(cache_key);
    if (!rst.valid())
        return {};

    auto&& prof = rst.val();
    if (prof.empty())
        return {};

    auto target_attr = extract_algo_attribute(selected_strategy);
    bool skip_by_negative = false;
    for (auto&& i : prof) {
        auto attr_of_algo =
                static_cast<megdnn::Algorithm::Attribute>(i.attribute);
        bool contain_attr_all_positive =
                (target_attr.first == (attr_of_algo & target_attr.first));
        bool contain_attr_any_negative =
                static_cast<bool>(attr_of_algo & target_attr.second);
        if (contain_attr_all_positive) {
            if (!contain_attr_any_negative) {
                Algorithm::Info::Desc algo_desc = deserialize_read_pod(i.algo);
                return algo_desc;
            } else {
                skip_by_negative = true;
            }
        }
    }

    std::string layouts_str =
            format_fixlayouts<Opr>(m_fastrun_layouts, arity_in, arity_out);
    if (skip_by_negative) {
        mgb_log_error(
                "opr: %s, layouts: %s, No usable algo. There are available "
                "algos match "
                "positive strategy(%s), but filtered by negative stategy(%s).",
                m_base_mgb_opr->dyn_typeinfo()->name, layouts_str.c_str(),
                Algorithm::attribute_str(target_attr.first).c_str(),
                Algorithm::attribute_str(target_attr.second).c_str());
    } else {
        mgb_log_error(
                "opr: %s, layouts: %s, No usable algo. algos read from cache "
                "could not "
                "satisfy positive strategy(%s)",
                m_base_mgb_opr->dyn_typeinfo()->name, layouts_str.c_str(),
                Algorithm::attribute_str(target_attr.first).c_str());
    }

    mgb_trap();
    MIDOUT_E
}

template <typename Opr>
void AlgoChooser<Opr>::AlgoChooserHelper::construct_execution_policy(
        const ExecutionStrategy& selected_strategy,
        typename AlgoChooser<Opr>::ImplExecutionPolicy& policy,
        bool retrive_from_cache, bool allow_log) const {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("construct_execution_policy")))
    if (!policy.algo.valid()) {
        if (retrive_from_cache) {
            policy.algo = get_profile_result_from_cache(selected_strategy);
            if (!policy.algo.valid()) {
                if (allow_log) {
                    auto target_attr =
                            extract_algo_attribute(selected_strategy);
                    std::string layouts_str = format_fixlayouts<Opr>(
                            m_fastrun_layouts, arity_in, arity_out);
                    std::string msg = ssprintf(
                            "(opr : %s, layouts %s, with attribute(%s) and "
                            "without attribute(%s)",
                            m_base_mgb_opr->dyn_typeinfo()->name,
                            layouts_str.c_str(),
                            Algorithm::attribute_str(target_attr.first).c_str(),
                            Algorithm::attribute_str(target_attr.second)
                                    .c_str());
                    mgb_log_warn(
                            "No algo get from cache for %s. This may caused by "
                            "mismatch with model and cache file or imcomplete "
                            "cache file. ex. profiling with version1, but "
                            "inferencing on version2 or profiling modelA but "
                            "inferencing modelB",
                            msg.c_str());
                }
                return;
            }
        } else {
            auto workspace_limit = WorkspaceLimitGetter::get_workspace_limit(
                    owner_graph(), m_cn, m_execution_policy.workspace_limit);

            auto attr = extract_algo_attribute(selected_strategy);
            policy.algo = APPLY(m_dnn_opr->get_algorithm_info_heuristic(
                                        args..., workspace_limit, attr.first,
                                        attr.second),
                                m_fastrun_layouts)
                                  .desc;
            mgb_assert(policy.algo.valid(),
                       "No algo found from heuristic with strategy %u and "
                       "workspace limit %zu",
                       static_cast<uint32_t>(selected_strategy),
                       workspace_limit);
        }
    }

    Algorithm* algo = m_dnn_opr->get_algorithm_from_desc(policy.algo);
    mgb_assert(algo, "Unknown algo description");
    std::vector<Algorithm::SearchItem>&& sub_items = algo->get_subopr_list(
            to_layout_array<Opr>(m_fastrun_layouts), m_dnn_opr);

    FOREACH_OPR_TYPE_DISPATCH(sub_items, {
        auto&& megdnn_opr = intl::create_megdnn_opr<_Opr>(m_cn);
        megdnn_opr->param() =
                Algorithm::deserialize_read_pod<typename _Opr::Param>(
                        _item.param);
        typename AlgoChooser<_Opr>::AlgoChooserHelper sub_helper(
                to_fixed_layouts<_Opr>(_item.layouts), megdnn_opr.get(),
                _item.param, m_base_mgb_opr, m_cn, m_execution_policy,
                m_allow_weight_preprocess);
        policy.sub_policy.push_back({});
        sub_helper.construct_execution_policy(selected_strategy,
                                              policy.sub_policy.back(),
                                              retrive_from_cache, allow_log);
        if (!policy.sub_policy.back().algo.valid()) {
            // means sub_helper.construct_execution_policy fails. clean up
            // policy.algo and return
            policy = {};
            return;
        }
    });
    MIDOUT_E
}

template <typename Opr>
size_t AlgoChooser<Opr>::AlgoChooserHelper::get_workspace_size_bytes(
        const ImplExecutionPolicy& policy,
        const FixedTensorLayouts& layouts) const {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("get_workspace_size_bytes")))
    m_dnn_opr->execution_policy() = policy;
    size_t result;
    const FixedTensorLayouts* layouts_ptr = &m_fastrun_layouts;
    if (layouts.at(0).ndim) {
        layouts_ptr = &layouts;
    }
    if_constexpr<opr_supports_preprocess<Opr>()>(
            [&](auto _) {
                auto&& opr = _(m_dnn_opr);
                auto prep =
                        this->construct_fake_preprocess_filter(*layouts_ptr);
                PreprocessFilter<Opr>* prep_ptr =
                        prep.valid() ? &prep.val() : nullptr;
                result = std::max(
                        APPLY(opr->get_preprocess_workspace_in_bytes(args...),
                              *layouts_ptr),
                        APPLY(opr->get_workspace_in_bytes(args..., prep_ptr),
                              *layouts_ptr));
            },
            /* else */
            [&](auto _) {
                result = APPLY(_(m_dnn_opr)->get_workspace_in_bytes(args...),
                               *layouts_ptr);
            });
    return result;
    MIDOUT_E
}

template <typename Opr>
std::vector<typename AlgoChooser<Opr>::ImplAlgo>
AlgoChooser<Opr>::AlgoChooserHelper::get_all_candidates() const {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("get_all_candidates")))
    auto heu = choose_by_heuristic(m_execution_policy.strategy);
    auto&& ret = APPLY(m_dnn_opr->get_all_algorithms_info(args...),
                       m_fastrun_layouts);
    bool found = false;
    for (size_t i = 0; i < ret.size(); ++i) {
        if (ret[i].desc == heu.algo) {
            found = true;
            std::swap(ret[i], ret[0]);
            break;
        }
    }

    Algorithm* palgo = m_dnn_opr->get_algorithm_from_desc(heu.algo);
    mgb_assert(palgo, "Unknown algo description");
    mgb_assert(found,
               "algo %s got by heuristic not found in "
               "candidate list",
               palgo->name());
    return std::move(ret);
    MIDOUT_E
}

template <typename Opr>
Maybe<AlgoChooserProfileCache::ResultEntry>
AlgoChooser<Opr>::AlgoChooserHelper::profile_single_algo(
        const ImplExecutionPolicy& policy, double& timeout) const {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("profile_single_algo")))
    typename TimedProfiler<Opr>::Param param;
    // force check copy size <= dest len-1 from gcc8 for safe
    param.execution_policy =
            TimedProfiler<Opr>::Param::ExecutionPolicyBlob::serialize(policy);
    param.workspace = get_workspace_size_bytes(policy);
    for (int i = 0; i < arity; ++i) {
        auto&& src = m_fastrun_layouts[i];
        bool cond_normal = src.format.is_default() &&
                           (src.dtype.category() == DTypeCategory::FLOAT ||
                            src.dtype.category() == DTypeCategory::INT ||
                            src.dtype.category() == DTypeCategory::QUANTIZED);

        bool cond_low_bit = src.dtype.is_low_bit() &&
                            src.format.is_lowbit_aligned() &&
                            (src.dtype.category() == DTypeCategory::QUANTIZED ||
                             src.dtype.category() == DTypeCategory::LOWBIT);
        MGB_MARK_USED_VAR(cond_normal);
        MGB_MARK_USED_VAR(cond_low_bit);
        mgb_assert(cond_normal || cond_low_bit,
                   "unsupported layout in profiling: %s",
                   src.to_string().c_str());
        param.dtypes[i] = src.dtype.enumv();
    }
    param.comp_node_loc = m_cn.locator();
    mgb_assert(param.shapes.size() == m_fastrun_layouts.size());
    for (size_t i = 0; i < param.shapes.size(); ++i)
        param.shapes[i] = m_fastrun_layouts[i];
    param.opr_param = m_dnn_opr->param();
    param.allow_weight_preprocess = m_allow_weight_preprocess;

    Algorithm* palgo = m_dnn_opr->get_algorithm_from_desc(policy.algo);
    mgb_assert(palgo, "can not find algo when profile single algo");

    auto rst = TimedProfiler<Opr>::profile(param, timeout);
    // MIOpen conv profiles all available algos when a specfic shape is
    // provided for the first time, which probably adds to the result time.
    // Therefore, a second profile execution is needed.
    if (strncmp(palgo->name(), "MIOpen", 6) == 0) {
        rst = TimedProfiler<Opr>::profile(param, timeout);
    }
    if (!rst.valid())
        return None;

    std::string algo_desc;
    serialize_write_pod(policy.algo, algo_desc);
    return AlgoChooserProfileCache::ResultEntry{
            algo_desc, static_cast<uint32_t>(palgo->attribute()),
            rst.val().time, param.workspace};
    MIDOUT_E
}

template <typename Opr>
void AlgoChooser<Opr>::AlgoChooserHelper::profile(
        const ExecutionStrategy& selected_strategy) const {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("profile")))
    if (get_profile_result_from_cache(selected_strategy).valid())
        return;
    AlgoChooserProfileCache::Result prof_rst;

    auto target_attr = extract_algo_attribute(selected_strategy);
    std::string layouts_str =
            format_fixlayouts<Opr>(m_fastrun_layouts, arity_in, arity_out);
    double cur_timeout = 0;

    auto workspace_limit = WorkspaceLimitGetter::get_workspace_limit(
            owner_graph(), m_cn, m_execution_policy.workspace_limit);
    RealTimer timer;
    for (auto algo : get_all_candidates()) {
        Maybe<AlgoChooserProfileCache::ResultEntry> cur_rst;

        ImplExecutionPolicy policy;
        policy.algo = algo.desc;

        //! check negative attribute : skip negative attribute
        auto palgo = m_dnn_opr->get_algorithm_from_desc(policy.algo);
        if (palgo->contain_attribute_any(target_attr.second)) {
            mgb_log_debug(
                    "skip algo %s, which matches the profile strategy required "
                    "'not contain attribute(%s).'",
                    algo.desc.name.c_str(),
                    Algorithm::attribute_str(target_attr.second).c_str());
            continue;
        }

        //! check workspace limit
        construct_execution_policy(selected_strategy, policy);
        mgb_assert(policy.algo.valid(),
                   "construct execution policy must success when profiling");
        if (get_workspace_size_bytes(policy) > workspace_limit) {
            continue;
        }

        std::string msg = ssprintf("profiling %s algorithm %s %s",
                                   m_base_mgb_opr->dyn_typeinfo()->name,
                                   algo.desc.name.c_str(), layouts_str.c_str());
        timer.reset();
        MGB_TRY { cur_rst = profile_single_algo(policy, cur_timeout); }
        MGB_CATCH(std::exception & exc, {
            mgb_log_warn("caught exception during %s: %s", msg.c_str(),
                         exc.what());
            continue;
        })
        MGB_CATCH(..., {
            mgb_log_warn("caught exception during %s", msg.c_str());
            continue;
        })
        if (!cur_rst.valid()) {
            mgb_log_warn("timeout when %s; timeout setting: %.3fsec",
                         msg.c_str(), cur_timeout);
            continue;
        }
        if (!cur_timeout) {
            cur_timeout = timer.get_secs() + TIMEOUT_TOLERANCE;
        } else {
            cur_timeout =
                    std::min(cur_timeout, timer.get_secs() + TIMEOUT_TOLERANCE);
        }
        auto&& rst = cur_rst.val();
        mgb_log_debug("%s: workspace: %zu; time: %.3gsec", msg.c_str(),
                      rst.workspace, rst.time);
        prof_rst.push_back(rst);
    }
    std::string msg = ssprintf(
            "no usable %s algorithm %s without attribute(%s) or could not meet "
            "workspace limite requirement(%zu)",
            m_base_mgb_opr->dyn_typeinfo()->name, layouts_str.c_str(),
            Algorithm::attribute_str(target_attr.second).c_str(),
            workspace_limit);
    mgb_assert(!prof_rst.empty(), "%s", msg.c_str());

    FixedTensorLayouts incache_layouts = m_incache_layouts;
    typename Opr::Param origin_param = m_dnn_opr->param();
    AlgoChooserProfileCache::Key cache_key{incache_layouts.data(),
                                           incache_layouts.size(), &origin_param,
                                           sizeof(origin_param)};

    AlgoChooserProfileCache cache(m_cn, profile_name(m_dnn_opr).c_str());
    cache.put(cache_key, prof_rst);
    MIDOUT_E
}

template <typename Opr>
Maybe<PreprocessFilter<Opr>>
AlgoChooser<Opr>::AlgoChooserHelper::construct_fake_preprocess_filter(
        const FixedTensorLayouts& layouts) const {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("construct_fake_preprocess_filter")))
    Maybe<PreprocessFilter<Opr>> result = None;
    const FixedTensorLayouts* layouts_ptr = &m_fastrun_layouts;
    if (layouts.at(0).ndim) {
        layouts_ptr = &layouts;
    }
    if_constexpr<opr_supports_preprocess<Opr>()>([&](auto _) {
        if (!m_allow_weight_preprocess)
            return;
        auto opr = _(m_dnn_opr);
        auto layouts = APPLY(opr->deduce_preprocessed_filter_layout(args...),
                             *layouts_ptr);
        //! No preprocess layout means no need weight preprocess
        if (layouts.empty()) {
            return;
        }
        //! all layouts arm empty means no need weight preprocess
        bool layout_valid = false;
        for (auto&& layout : layouts) {
            if (!layout.is_empty()) {
                layout_valid = true;
            }
        }
        if (!layout_valid) {
            return;
        }

        result = PreprocessFilter<Opr>{};
        auto& res = result.val();
        res.algorithm_id = nullptr;
        res.tensors.resize(layouts.size());
        for (size_t i = 0; i < layouts.size(); i++) {
            res.tensors[i] = megdnn::TensorND(nullptr, layouts[i]);
        }
    });
    return result;
    MIDOUT_E
}

template <typename Opr>
std::pair<AlgoAttribute, AlgoAttribute>
AlgoChooser<Opr>::AlgoChooserHelper::extract_algo_attribute(
        const ExecutionStrategy& strategy) const {
    std::pair<AlgoAttribute, AlgoAttribute> ret =
            std::make_pair(AlgoAttribute::DEFAULT, AlgoAttribute::DEFAULT);

    //! from strategy
    if (strategy & ExecutionStrategy::REPRODUCIBLE) {
        ret.first |= AlgoAttribute::REPRODUCIBLE;
    }
    if (strategy & ExecutionStrategy::OPTMIZED) {
        ret.second |= AlgoAttribute::NAIVE;
    }

    //! from graph option
    // FIXME: no_profiling_on_shape_change extract USABLE_DEPEND_ON_SHAPE attribute when fixed usable
    if (owner_graph()->options().fast_run_config.shared_batch_size) {
        ret.second |= AlgoAttribute::USABLE_DEPEND_ON_SHAPE;
    }

    if (owner_graph()->options().fast_run_config.binary_equal_between_batch) {
        ret.first |= AlgoAttribute::REPRODUCIBLE;
        ret.second |= AlgoAttribute::ACCURACY_DEPEND_ON_BATCH;
    }

    return ret;
}

#define INST(Opr)                                                              \
    template AlgoChooser<megdnn::Opr>::AlgoChooserHelper::AlgoChooserHelper(   \
            const FixedTensorLayouts& layouts, megdnn::Opr* megdnn_opr,        \
            const std::string& param_str, const cg::OperatorNodeBase* mgb_opr, \
            const CompNode& cn,                                                \
            const megdnn::param::ExecutionPolicy& execution_policy,            \
            bool allow_weight_preprocess);                                     \
    template typename AlgoChooser<megdnn::Opr>::ImplExecutionPolicy            \
    AlgoChooser<megdnn::Opr>::AlgoChooserHelper::choose_by_heuristic(          \
            const ExecutionStrategy& select_strategy) const;                   \
    template typename AlgoChooser<megdnn::Opr>::ImplExecutionPolicy            \
    AlgoChooser<megdnn::Opr>::AlgoChooserHelper::choose_by_profile(            \
            const ExecutionStrategy& select_strategy, bool enable_update)      \
            const;                                                             \
    template typename AlgoChooser<megdnn::Opr>::ImplAlgoDesc                   \
    AlgoChooser<megdnn::Opr>::AlgoChooserHelper::                              \
            get_profile_result_from_cache(                                     \
                    const ExecutionStrategy& select_strategy) const;           \
    template void                                                              \
    AlgoChooser<megdnn::Opr>::AlgoChooserHelper::construct_execution_policy(   \
            const ExecutionStrategy& select_strategy,                          \
            typename AlgoChooser<megdnn::Opr>::ImplExecutionPolicy& policy,    \
            bool retrive_from_cache, bool allow_log) const;                    \
    template size_t                                                            \
    AlgoChooser<megdnn::Opr>::AlgoChooserHelper::get_workspace_size_bytes(     \
            const typename AlgoChooser<megdnn::Opr>::ImplExecutionPolicy&      \
                    policy,                                                    \
            const FixedTensorLayouts& layouts) const;                          \
    template std::vector<typename AlgoChooser<megdnn::Opr>::ImplAlgo>          \
    AlgoChooser<megdnn::Opr>::AlgoChooserHelper::get_all_candidates() const;   \
    template Maybe<AlgoChooserProfileCache::ResultEntry>                       \
    AlgoChooser<megdnn::Opr>::AlgoChooserHelper::profile_single_algo(          \
            const typename AlgoChooser<megdnn::Opr>::ImplExecutionPolicy&      \
                    policy,                                                    \
            double& timeout) const;                                            \
    template std::pair<AlgoAttribute, AlgoAttribute>                           \
    AlgoChooser<megdnn::Opr>::AlgoChooserHelper::extract_algo_attribute(       \
            const ExecutionStrategy& strategy) const;                          \
    template void AlgoChooser<megdnn::Opr>::AlgoChooserHelper::profile(        \
            const ExecutionStrategy& selected_strategy) const;

MGB_FOREACH_FASTRUN_OPR(INST)
#undef INST

//////////////////////////////// AlgoChoose /////////////////////////////
template <typename Opr>
typename AlgoChooser<Opr>::ImplExecutionPolicy AlgoChooser<Opr>::get_policy(
        const AlgoChooserHelper& helper) {
    auto opr_strategy = helper.execution_policy().strategy;
    auto strategy2str = [](auto strategy) {
        std::string ret;
        if (strategy & ExecutionStrategy::HEURISTIC) {
            ret += "HEURISTIC ";
        }
        if (strategy & ExecutionStrategy::PROFILE) {
            ret += "PROFILE ";
        }
        if (strategy & ExecutionStrategy::REPRODUCIBLE) {
            ret += "REPRODUCIBLE ";
        }
        if (strategy & ExecutionStrategy::OPTIMIZED) {
            ret += "OPTIMIZED ";
        }
        return ret;
    };
    mgb_log_debug("Use Stragegy :%s", strategy2str(opr_strategy).c_str());
    if (opr_strategy & ExecutionStrategy::HEURISTIC) {
        if (opr_strategy & ExecutionStrategy::PROFILE) {
            //! this strategy will choose from cache first, then choost by
            //! heuristic if fail.
            ImplExecutionPolicy policy =
                    helper.choose_by_profile(opr_strategy, false);
            if (!policy.algo.valid()) {
                policy = helper.choose_by_heuristic(opr_strategy);
            }
            return policy;
        } else {
            return helper.choose_by_heuristic(opr_strategy);
        }
    }
#if MGB_ENABLE_FASTRUN
    else if (opr_strategy & ExecutionStrategy::PROFILE) {
        return helper.choose_by_profile(opr_strategy, true);
    }
#endif
    else {
        mgb_throw(GraphError, "bad ExecutionPolicy strategy");
    }
}

template <typename Opr>
size_t AlgoChooser<Opr>::setup_algo(const FixedTensorLayouts& layouts,
                                    Opr* megdnn_opr, const MGBOpr* mgb_opr,
                                    bool allow_weight_preprocess) {
    if (WorkspaceLimitGetter::is_prealloc_run(mgb_opr->owner_graph())) {
        return 0;
    }

    std::string param_str;
    Algorithm::serialize_write_pod(megdnn_opr->param(), param_str);
    AlgoChooserHelper helper(layouts, megdnn_opr, param_str, mgb_opr,
                             mgb_opr->comp_node(), mgb_opr->execution_policy(),
                             allow_weight_preprocess);

    ImplExecutionPolicy policy;
    if (auto algo_choose_hook = mgb_opr->algo_chooser()) {
        policy = algo_choose_hook(mgb_opr);
        auto strategy =
                ExecutionStrategy::HEURISTIC | ExecutionStrategy::REPRODUCIBLE;
        bool retrive_from_cache = false;
        helper.construct_execution_policy(strategy, policy, retrive_from_cache);
    }
    if (!policy.algo.valid()) {
        policy = get_policy(helper);
    }
    size_t workspace = helper.get_workspace_size_bytes(policy, layouts);

    std::string ret;
    ret.append(mgb_opr->dyn_typeinfo()->name);
    ret.append(": tensor layouts");
    ret += format_fixlayouts<Opr>(layouts, arity_in, arity_out);
    Algorithm* palgo = megdnn_opr->get_algorithm_from_desc(policy.algo);
    mgb_assert(palgo, "Unknown algo description");
    ret.append("): algo=" + std::string(palgo->name()));
    ret.append(ssprintf(" workspace=%.2fMiB attribute=%d",
                        workspace / (1024 * 1024.0),
                        static_cast<uint32_t>(palgo->attribute())));
    mgb_log_debug("%s", ret.c_str());

    megdnn_opr->execution_policy() = policy;
    return workspace;
}

#define INST(Opr)                                                         \
    template AlgoChooser<megdnn::Opr>::ImplExecutionPolicy                \
    AlgoChooser<megdnn::Opr>::get_policy(const AlgoChooserHelper& proxy); \
    template size_t AlgoChooser<megdnn::Opr>::setup_algo(                 \
            const FixedTensorLayouts& layouts, megdnn::Opr* megdnn_opr,   \
            const MGBOpr* mgb_opr, bool allow_weight_preprocess);

MGB_FOREACH_FASTRUN_OPR(INST)
#undef INST

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
