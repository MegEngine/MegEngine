/**
 * \file src/opr/impl/dnn/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/convolution.h"

#include "megbrain/graph/grad_impl.h"
#include "megbrain/system.h"
#include "megbrain/utils/hash_ct.h"
#include "megbrain/utils/timer.h"

#include "megdnn/oprs/utils.h"

#include "../internal/megdnn_opr_wrapper.inl"

#include <array>
#include <chrono>
#include <cstring>
#include <thread>


using namespace mgb;
using namespace opr;
using namespace cg::static_infer;
using intl::WorkspaceLimitGetter;

#define CACHE_KEY_VERSION "v2"

#define MGB_FOREACH_FASTRUN_OPR(cb)   \
    cb(ConvolutionForward);           \
    cb(ConvBiasForward);              \
    cb(ConvolutionBackwardData);      \
    cb(ConvolutionBackwardFilter);    \
    cb(Convolution3DForward);         \
    cb(Convolution3DBackwardData);    \
    cb(Convolution3DBackwardFilter);  \
    cb(LocalShareForward);            \
    cb(LocalShareBackwardData);       \
    cb(LocalShareBackwardFilter);     \
    cb(DeformableConvForward);        \
    cb(DeformableConvBackwardFilter); \
    cb(DeformableConvBackwardData);   \
    cb(BatchConvBiasForward);

namespace mgb {
namespace opr {
namespace intl {

#define cb(_Opr)                                           \
    template <>                                            \
    struct AutoAddWorkspaceNeedLimitGetter<megdnn::_Opr> { \
        static constexpr bool val = true;                  \
    };
MGB_FOREACH_FASTRUN_OPR(cb)

#undef cb

}  // namespace intl
}  // namespace opr
}  // namespace mgb

namespace {

template <class MegDNNOpr>
struct MegDNNOpr2MGBOpr;

#define cb(_Opr)                            \
    template <>                             \
    struct MegDNNOpr2MGBOpr<megdnn::_Opr> { \
        using MGBOpr = opr::_Opr;           \
    };

MGB_FOREACH_FASTRUN_OPR(cb)

#undef cb

template <class MGBOpr>
struct OprAttributeTrait {
    static bool is_weights_persistent(const MGBOpr*) { return false; }
};

template <>
struct OprAttributeTrait<opr::ConvBias> {
    //! return true if the flag of weights is PERSISTENT_DEVICE_VALUE, false
    //! otherwise. True means weights can be tranformed in the first run.
    static bool is_weights_persistent(const opr::ConvBias* opr) {
        return opr->input()[1]->contain_flag(
                VarNode::Flag::PERSISTENT_DEVICE_VALUE);
    }
};

template <typename Opr>
struct OprArityTrait;

#define cb(x) (x)
#define cb_ref(x) (&(x))
#define cb_dnn(x) ((x).as_megdnn())

#define INST_ARITY(_Opr, _in, _out)                                           \
    template <>                                                               \
    struct OprArityTrait<_Opr> {                                              \
        static constexpr int arity_in = _in;                                  \
        static constexpr int arity_out = _out;                                \
        static constexpr int arity = _in + _out;                              \
        using TensorLayoutArray = std::array<TensorLayout, arity>;            \
        static size_t get_workspace_in_bytes(                                 \
                _Opr* opr, typename _Opr::Algorithm* algo,                    \
                const TensorLayoutArray& layouts) {                           \
            opr->execution_policy() = {algo};                                 \
            return opr->get_workspace_in_bytes(LAYOUTS(cb));                  \
        }                                                                     \
                                                                              \
        static std::vector<typename _Opr::Algorithm*> get_all_algorithms(     \
                _Opr* opr, const TensorLayoutArray& layouts) {                \
            return opr->get_all_algorithms(LAYOUTS(cb));                      \
        }                                                                     \
                                                                              \
        static typename _Opr::Algorithm* get_algorithm_heuristic(             \
                _Opr* opr, const TensorLayoutArray& layouts,                  \
                size_t workspace_limit, bool reproducible) {                  \
            return opr->get_algorithm_heuristic(LAYOUTS(cb), workspace_limit, \
                                                reproducible);                \
        }                                                                     \
                                                                              \
        static void exec(_Opr* opr, const DeviceTensorND* inp_val,            \
                         const DeviceTensorND* out_val,                       \
                         megdnn::Workspace& workspace) {                      \
            opr->exec(TENSORS(cb_dnn), workspace);                            \
        }                                                                     \
                                                                              \
        static void modify_input_layouts(_Opr* opr,                           \
                                         const TensorLayoutArray& layouts) {  \
            intl::MegDNNOprInputsLayoutModifier<_Opr>::apply(                 \
                    opr->param(), {LAYOUTS(cb_ref)});                         \
        }                                                                     \
    }

#define TENSORS(cb) cb(inp_val[0]), cb(inp_val[1]), cb(out_val[0])
#define LAYOUTS(cb) cb(layouts[0]), cb(layouts[1]), cb(layouts[2])
#define INST_ARITY_2_1(Opr) INST_ARITY(Opr, 2, 1)
INST_ARITY_2_1(megdnn::Convolution);
INST_ARITY_2_1(megdnn::ConvolutionBackwardData);
INST_ARITY_2_1(megdnn::ConvolutionBackwardFilter);
INST_ARITY_2_1(megdnn::Convolution3DForward);
INST_ARITY_2_1(megdnn::Convolution3DBackwardData);
INST_ARITY_2_1(megdnn::Convolution3DBackwardFilter);
INST_ARITY_2_1(megdnn::LocalShareForward);
INST_ARITY_2_1(megdnn::LocalShareBackwardData);
INST_ARITY_2_1(megdnn::LocalShareBackwardFilter);
#undef TENSORS
#undef LAYOUTS
#undef INST_ARITY_2_1

#define TENSORS(cb)                                                 \
    cb(inp_val[0]), cb(inp_val[1]), cb(inp_val[2]), cb(inp_val[3]), \
            cb(out_val[0])
#define LAYOUTS(cb)                                                 \
    cb(layouts[0]), cb(layouts[1]), cb(layouts[2]), cb(layouts[3]), \
            cb(layouts[4])
#define INST_ARITY_4_1(Opr) INST_ARITY(Opr, 4, 1)
INST_ARITY_4_1(megdnn::ConvBias);
INST_ARITY_4_1(megdnn::DeformableConvForward);
INST_ARITY_4_1(megdnn::DeformableConvBackwardFilter);
INST_ARITY_4_1(megdnn::BatchConvBiasForward);
#undef TENSORS
#undef LAYOUTS
#undef INST_ARITY_4_1

#define TENSORS(cb) cb(inp_val[0]), cb(inp_val[1]), cb(inp_val[2]), \
        cb(inp_val[3]), cb(inp_val[4]), cb(out_val[0]),             \
        cb(out_val[1]), cb(out_val[2])
#define LAYOUTS(cb) cb(layouts[0]), cb(layouts[1]), cb(layouts[2]), \
        cb(layouts[3]), cb(layouts[4]), cb(layouts[5]),             \
        cb(layouts[6]), cb(layouts[7])

#define INST_ARITY_5_3(Opr) INST_ARITY(Opr, 5, 3)
INST_ARITY_5_3(megdnn::DeformableConvBackwardData);
#undef TENSORS
#undef LAYOUTS
#undef INST_ARITY_5_3
#undef cb
#undef cb_ref
#undef cb_dnn
#undef INST_ARITY

// timeout delta to be added with fastest known algorithm for new algos
constexpr double TIMEOUT_TOLERANCE = 2;

template <typename Opr>
struct AlgoChooserFuncId {};

#define DEF_FUNC_ID(func)                                                     \
    template <>                                                               \
    struct AlgoChooserFuncId<megdnn::func> {                                  \
        __attribute__(                                                        \
                (unused)) static constexpr sys::TimedFuncInvoker::FuncId ID = \
                static_cast<sys::TimedFuncInvoker::FuncId>(                   \
                        MGB_HASH_STR("megdnn::" #func));                      \
    };

MGB_FOREACH_FASTRUN_OPR(DEF_FUNC_ID)

#undef DEF_FUNC_ID

/* =================== TimedProfiler =================== */

/*!
 * \brief profile a megdnn opr conv with given param
 *
 * This class only provides static methods, and the entry point is
 * TimedProfiler::profile; it would run profiler in a timed environment by
 * sys::TimedFuncInvoker
 *
 * \tparam Opr megdnn opr impl
 */
template <typename Opr>
class TimedProfiler {
    static constexpr int arity_in = OprArityTrait<Opr>::arity_in;
    static constexpr int arity_out = OprArityTrait<Opr>::arity_out;
    static constexpr int arity = OprArityTrait<Opr>::arity;
    using ConvTensorShapes = std::array<TensorShape, arity>;

public:
    struct Param {
        char algo_name[128];
        size_t workspace;
        DTypeEnum dtypes[arity];
        CompNode::Locator comp_node_loc;
        ConvTensorShapes shapes;
        typename Opr::Param opr_param;

        //! filled by profile()
        mutable double actual_timeout;
    };

    struct Result {
        double time;
    };

    static Maybe<Result> profile(const Param& param, double& timeout) {
        mgb_assert(timeout >= 0);
        if (!timeout) {
            timeout = timeout_setting;
        } else if (timeout_setting) {
            timeout = std::min(timeout, timeout_setting);
        }
        param.actual_timeout =
                timeout ? timeout : std::numeric_limits<double>::infinity();
        auto res = sys::TimedFuncInvoker::ins().invoke(
                AlgoChooserFuncId<Opr>::ID,
                TParam::from_pod(const_cast<Param&>(param)), timeout);
        if (res.valid())
            return res.val().template as_single_pod<Result>();
        return None;
    }
private:
    using TParam = sys::TimedFuncInvoker::Param;
    using TResult = sys::TimedFuncInvoker::Result;

    static const double timeout_setting;

    static double init_timeout_setting();
    static TResult prof_impl(const TParam& raw_param);
    static void prof_init_device(const TParam& raw_param);
};
template <typename Opr>
const double TimedProfiler<Opr>::timeout_setting =
        TimedProfiler<Opr>::init_timeout_setting();

template <typename Opr>
double TimedProfiler<Opr>::init_timeout_setting() {
#if MGB_ENABLE_FASTRUN
    sys::TimedFuncInvoker::ins().register_func(
            AlgoChooserFuncId<Opr>::ID, &TimedProfiler<Opr>::prof_impl,
            &TimedProfiler<Opr>::prof_init_device);
    auto to_set = MGB_GETENV("MGB_CONV_PROFILING_TIMEOUT");
    if (to_set)
        return std::stod(to_set);
#endif
    return 0;
}

template <typename Opr>
typename TimedProfiler<Opr>::TResult TimedProfiler<Opr>::prof_impl(
        const TParam& raw_param) {
    auto&& param = raw_param.as_single_pod<Param>();
    CompNode cn = CompNode::load(param.comp_node_loc, param.comp_node_loc);
    auto megdnn_opr = intl::create_megdnn_opr<Opr>(cn);
    std::array<TensorLayout, arity> layouts;

    auto from_enum = [&](DTypeEnum enumv) -> DType {
        switch (enumv) {

#define cb(_dt)                  \
    case DTypeTrait<_dt>::enumv: \
        return _dt(1.0f, static_cast<uint8_t>(0))
            cb(dtype::Quantized8Asymm);
#undef cb

#define cb(_dt)                  \
    case DTypeTrait<_dt>::enumv: \
        return _dt(1.0f)

            cb(dtype::QuantizedS8);
            cb(dtype::QuantizedS32);
            default:
                return DType::from_enum(enumv);
#undef cb
        }
    };
    for (int i = 0; i < arity; ++i) {
        layouts[i] = {param.shapes[i], from_enum(param.dtypes[i])};
    }

    megdnn_opr->param() = param.opr_param;
    {
        typename Opr::Algorithm* algo = nullptr;
        for (auto i : OprArityTrait<Opr>::get_all_algorithms(megdnn_opr.get(),
                                                             layouts)) {
            if (!strcmp(i->name(), param.algo_name)) {
                algo = i;
                break;
            }
        }
        mgb_assert(algo, "algorithm %s not found", param.algo_name);
        megdnn_opr->execution_policy() = {algo};
    }

    {
        // first allocate a whole chunk to avoid memory fragmentation (here we
        // rely on memory allocator to reuse memory)
        auto align = cn.get_mem_addr_alignment();
        size_t tot_size = align;
        for (int i = 0; i < arity; ++i) {
            tot_size += layouts[i].span().high_byte + align;
        }
        tot_size += param.workspace;
        DeviceTensorStorage storage{cn};
        storage.ensure_size(tot_size);
    }

    // allocate input and output memory
    DeviceTensorND inp_val[arity_in], out_val[arity_out], workspace;
    for (int i = 0; i < arity_in; ++i) {
        inp_val[i]
                .comp_node(cn)
                .dtype(layouts[i].dtype)
                .resize(layouts[i]);
    }
    for (int i = 0; i < arity_out; ++i) {
        out_val[i]
                .comp_node(cn)
                .dtype(layouts[arity_in + i].dtype)
                .resize(layouts[arity_in + i]);
    }
    megdnn::Workspace mdn_workspace;

    // allocate workspace
    if (param.workspace) {
        workspace.comp_node(cn).dtype(dtype::Byte()).resize({param.workspace});
        mdn_workspace.size = param.workspace;
        mdn_workspace.raw_ptr = workspace.raw_ptr();
    }

    for (int i = 0; i < arity_in; ++i) {
        fill_zero_dev_tensor(inp_val[i]);
    }

    RealTimer timer;
    auto ev_start = cn.create_event(CompNode::Event::NEED_TIMER),
         ev_end = cn.create_event(CompNode::Event::NEED_TIMER);
    ev_start->record();
    OprArityTrait<Opr>::exec(megdnn_opr.get(), inp_val, out_val, mdn_workspace);
    ev_end->record();

    double next_report_time = 0.5;
    while (!ev_end->finished()) {
        if (timer.get_secs() >= next_report_time) {
            mgb_log_warn(
                    "profiling conv algo %s already took %.3f/%.3f secs"
                    " (limit can be set by MGB_CONV_PROFILING_TIMEOUT) ",
                    param.algo_name, timer.get_secs(), param.actual_timeout);
            next_report_time = timer.get_secs() + 1;
        }
        using namespace std::literals;
        std::this_thread::sleep_for(1000us);
    }

    mgb_assert(ev_start->finished());
    return TResult::from_pod(Result{ev_start->elapsed_time_until(*ev_end)});
};

template <typename Opr>
void TimedProfiler<Opr>::prof_init_device(const TParam& raw_param) {
    auto&& param = raw_param.as_single_pod<Param>();
    CompNode cn = CompNode::load(param.comp_node_loc, param.comp_node_loc);
    // wait for cuda init, so its time does not get accounted in timeout
    cn.sync();
}

/* =================== AlgoChooser =================== */
/*!
 * \brief choose algorithm according to ExecutionPolicy
 *
 * This class only provides static methods, and the entry point is
 * AlgoChooser::setup_algo. When profiling is needed, it would first try to
 * retrive profiling stats from cache, and run TimedProfiler when necessary
 *
 * \tparam Opr megdnn operator impl
 */
template <typename Opr>
class AlgoChooser {
    static constexpr int arity_in = OprArityTrait<Opr>::arity_in;
    static constexpr int arity_out = OprArityTrait<Opr>::arity_out;
    static constexpr int arity = OprArityTrait<Opr>::arity;
    using ImplAlgo = typename Opr::Algorithm*;
    using MGBOpr = typename MegDNNOpr2MGBOpr<Opr>::MGBOpr;
    using ConvTensorLayouts = std::array<TensorLayout, arity>;

    class ExeContext {
        const ConvTensorLayouts& m_layouts;
        Opr* m_megdnn_opr;
        const MGBOpr* m_mgb_opr;

    public:
        ExeContext(const ConvTensorLayouts& layouts, Opr* megdnn_opr,
                   const MGBOpr* mgb_opr)
                : m_layouts{layouts},
                  m_megdnn_opr{megdnn_opr},
                  m_mgb_opr{mgb_opr} {
            mgb_assert(m_layouts.size() == layouts.size());
            static_assert(
                    std::tuple_size<ConvTensorLayouts>::value == 3 ||
                            std::tuple_size<ConvTensorLayouts>::value == 5 ||
                            std::tuple_size<ConvTensorLayouts>::value == 8,
                    "Convolution AlgoChooser assumes arity = 3 , 5 or 8 (for "
                    "deformable conv)");
            OprArityTrait<Opr>::modify_input_layouts(megdnn_opr, m_layouts);
        }

        Opr* megdnn_opr() const { return m_megdnn_opr; }

        const MGBOpr* mgb_opr() const { return m_mgb_opr; }

        const TensorLayout& inp_layout(size_t idx) const {
            return m_layouts[idx];
        }

        const ConvTensorLayouts& layouts() const { return m_layouts; }

        ImplAlgo choose_by_heuristic(bool reproducible = false) const {
            auto opr = m_mgb_opr;
            auto workspace_limit = WorkspaceLimitGetter::get_workspace_limit(
                    opr->owner_graph(), opr->comp_node(),
                    opr->execution_policy().workspace_limit);
            return OprArityTrait<Opr>::get_algorithm_heuristic(
                    m_megdnn_opr, m_layouts, workspace_limit, reproducible);
        }

        //! get all candidate algos, and the one choose_by_heuristic() is
        //! put first
        std::vector<ImplAlgo> get_all_candidates() const {
            auto heu = choose_by_heuristic();
            auto&& ret = OprArityTrait<Opr>::get_all_algorithms(
                    m_megdnn_opr, m_layouts);
            bool found = false;
            for (size_t i = 0; i < ret.size(); ++i) {
                if (ret[i] == heu) {
                    found = true;
                    std::swap(ret[i], ret[0]);
                    break;
                }
            }
            mgb_assert(found,
                       "algo got by heuristic not found in "
                       "candidate list");
            return std::move(ret);
        }

        //! get candidate algos with workspace limit.
        std::vector<ImplAlgo> get_all_candidates_with_workspace_limit() const {
            auto && all_algos = get_all_candidates();
            auto opr = m_mgb_opr;
            auto workspace_limit = WorkspaceLimitGetter::get_workspace_limit(
                    opr->owner_graph(), opr->comp_node(),
                    opr->execution_policy().workspace_limit);
            std::vector<ImplAlgo> ret;
            for (auto&& algo : all_algos) {
                if (get_workspace_size_bytes(algo) <= workspace_limit) {
                    ret.push_back(algo);
                }
            }
            return ret;
        }

        //! get workspace size required for specific algo
        size_t get_workspace_size_bytes(ImplAlgo algo) const {
            return OprArityTrait<Opr>::get_workspace_in_bytes(m_megdnn_opr,
                                                              algo, m_layouts);
        }

        /*!
         * \brief profile a single algorithm
         *
         * This is actually a wrapper that constructs param and call
         * TimedProfiler<Opr>::profile for the actual profiling
         *
         * \param[in,out] timeout set the timeout, and return the actual
         *      timeout used during profiling
         */
        Maybe<AlgoChooserProfileCache::ResultEntry> profile_single_algo(
                ImplAlgo algo, double& timeout) const;

    private:
        /*!
         * \brief modify param passed to prof_impl by weights preprcess.
         *
         * \param param: param passed.
         *
         * \warning invoke when is_weights_persistent is true.
         */
        void modify_param_with_weights_preprocessed(
                typename TimedProfiler<Opr>::Param& param) const {}
    };

    //! entrance for getting algorithm according to execution strategy
    static ImplAlgo get_algo(ExeContext& ctx) {
        using S = mixin::Convolution::ExecutionPolicy::Strategy;
        MGB_MARK_USED_VAR(TIMEOUT_TOLERANCE);
        switch (ctx.mgb_opr()->execution_policy().strategy) {
            case S::HEURISTIC:
                return ctx.choose_by_heuristic();
            case S::HEURISTIC_REPRODUCIBLE:
                return ctx.choose_by_heuristic(true);
            case S::PROFILE_HEURISTIC: {
                ImplAlgo algo = choose_by_profile(ctx, false, false);
                if (algo == nullptr)
                    algo = ctx.choose_by_heuristic();
                return algo;
            }
#if MGB_ENABLE_FASTRUN
            case S::PROFILE:
                return choose_by_profile(ctx, false);
            case S::PROFILE_REPRODUCIBLE:
                return choose_by_profile(ctx, true);
#endif
            default:
                mgb_throw(GraphError,
                          "bad convolution ExecutionPolicy strategy");
        }
    }

    //! get all profile result, either by retrieving cache or profiling
    static AlgoChooserProfileCache::Result get_profile_result(
            ExeContext& ctx, bool enable_update);

    static ImplAlgo choose_by_profile(ExeContext& ctx,
                                      bool require_reproducible,
                                      bool enable_update = true);

public:
    /*!
     * \brief setup algorithm and return workspace size
     */
    static size_t setup_algo(const ConvTensorLayouts& layouts, Opr* megdnn_opr,
                             const MGBOpr* mgb_opr) {
        if (WorkspaceLimitGetter::is_prealloc_run(mgb_opr->owner_graph())) {
            return 0;
        }

        ExeContext ctx(layouts, megdnn_opr, mgb_opr);

        auto algo = get_algo(ctx);
        size_t workspace = ctx.get_workspace_size_bytes(algo);
        mgb_log_debug(
                "%s: input shapes (%s, %s): algo=%s "
                "workspace=%.2fMiB reproducible=%d",
                mgb_opr->dyn_typeinfo()->name,
                layouts[0].TensorShape::to_string().c_str(),
                layouts[1].TensorShape::to_string().c_str(), algo->name(),
                workspace / (1024 * 1024.0), algo->is_reproducible());
        megdnn_opr->execution_policy() = {algo};
        return workspace;
    }
};

template <typename Opr>
AlgoChooserProfileCache::Result AlgoChooser<Opr>::get_profile_result(
        ExeContext& ctx, bool enable_update) {
    AlgoChooserProfileCache& cache = ctx.mgb_opr()->profile_cache();
    auto param_blob = ctx.mgb_opr()->param_blob();
    AlgoChooserProfileCache::Key cache_key{ctx.layouts().data(),
                                           ctx.layouts().size(),
                                           param_blob.first, param_blob.second};
    {
        auto&& rst = cache.get(cache_key);
        if (rst.valid())
            return rst.val();
    }

    AlgoChooserProfileCache::Result prof_rst;
    if (!enable_update)
        return prof_rst;

    std::string str_on_inp_shape = ssprintf(
            "on input layouts (%s, %s)", ctx.layouts()[0].to_string().c_str(),
            ctx.layouts()[1].to_string().c_str());
    double cur_timeout = 0;
    RealTimer timer;
    for (auto algo : ctx.get_all_candidates_with_workspace_limit()) {
        Maybe<AlgoChooserProfileCache::ResultEntry> cur_rst;
        std::string msg = ssprintf("profiling %s algorithm %s %s",
                                   ctx.mgb_opr()->dyn_typeinfo()->name,
                                   algo->name(), str_on_inp_shape.c_str());
        timer.reset();
        MGB_TRY { cur_rst = ctx.profile_single_algo(algo, cur_timeout); }
        MGB_CATCH(std::exception & exc,
                  {
                      mgb_log_warn("caught exception during %s: %s",
                                   msg.c_str(), exc.what());
                      continue;
                  })
        MGB_CATCH(..., {
            mgb_log_warn("caught exception during %s", msg.c_str());
            continue;
        }) if (!cur_rst.valid()) {
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
    mgb_assert(!prof_rst.empty(), "no usable convolution algorithm %s",
               str_on_inp_shape.c_str());

    cache.put(cache_key, prof_rst);
    return prof_rst;
}

template <typename Opr>
typename AlgoChooser<Opr>::ImplAlgo AlgoChooser<Opr>::choose_by_profile(
        ExeContext& ctx, bool require_reproducible, bool enable_update) {
    auto opr = ctx.mgb_opr();
    if (opr->owner_graph()->options().no_profiling_on_shape_change) {
        auto algo = ctx.megdnn_opr()->execution_policy().algorithm;
        if (algo)
            return algo;
    }

    std::unordered_map<std::string, ImplAlgo> algo_map;
    for (auto i : ctx.get_all_candidates()) {
        auto ins = algo_map.emplace(i->name(), i);
        mgb_assert(ins.second, "duplicated algo name: %s", i->name());
    }

    auto&& prof = get_profile_result(ctx, enable_update);
    if (prof.empty())
        return nullptr;
    for (auto&& i : prof) {
        if ((!require_reproducible || i.reproducible)) {
            auto iter = algo_map.find(i.algo);
            mgb_assert(
                    iter != algo_map.end(),
                    "algorithm %s exists in "
                    "profiling result but not in algo_map; please report this "
                    "bug; opr: %s{%s}, shapes: %s %s %s",
                    ctx.mgb_opr()->cname(), ctx.mgb_opr()->dyn_typeinfo()->name,
                    ctx.layouts()[0].TensorShape::to_string().c_str(),
                    ctx.layouts()[1].TensorShape::to_string().c_str(),
                    ctx.layouts()[2].TensorShape::to_string().c_str(),
                    i.algo.c_str());
            return iter->second;
        }
    }

    mgb_log_error(
            "Workspace requirement (%zu) could not be satisfied. Abort now to "
            "avoid further problems",
            WorkspaceLimitGetter::get_workspace_limit(
                    opr->owner_graph(), opr->comp_node(),
                    opr->execution_policy().workspace_limit));
    mgb_trap();
}

template <>
void AlgoChooser<megdnn::ConvBias>::ExeContext::
        modify_param_with_weights_preprocessed(
                typename TimedProfiler<megdnn::ConvBias>::Param& param) const {
    if (param.opr_param.format == megdnn::ConvBias::Param::Format::NCHW) {
        auto winograd_param =
                megdnn::ConvBias::parse_winograd_name(param.algo_name);
        if (winograd_param == megdnn::ConvBias::INVALID_WINOGRAD_PARAM) {
            return;
        }
        ConvBiasForward::check_winograd_param_valid(winograd_param,
                                                    m_layouts[1].dtype);
        auto winograd_preprocess_opr =
                intl::create_megdnn_opr<megdnn::WinogradFilterPreprocess>(
                        m_mgb_opr->output(0)->comp_node());
        winograd_preprocess_opr->param().format =
                ConvBiasForward::get_matmul_format(winograd_param);
        winograd_preprocess_opr->param().output_block_size =
                winograd_param.output_block_size;
        TensorLayout filter_transform_layout;
        winograd_preprocess_opr->deduce_layout(m_layouts[1],
                                               filter_transform_layout);
        param.shapes[1] = filter_transform_layout;
        param.dtypes[1] = filter_transform_layout.dtype.enumv();

        param.opr_param.format = megdnn::ConvBias::Param::Format::NCHW_WINOGRAD;
        param.opr_param.output_block_size = winograd_param.output_block_size;
    }
}

template <typename Opr>
Maybe<AlgoChooserProfileCache::ResultEntry>
AlgoChooser<Opr>::ExeContext::profile_single_algo(ImplAlgo algo,
                                                  double& timeout) const {
    typename TimedProfiler<Opr>::Param param;
    bool is_weights_persistent =
            OprAttributeTrait<typename MegDNNOpr2MGBOpr<Opr>::MGBOpr>::
                    is_weights_persistent(m_mgb_opr);
    auto name = algo->name();
    // force check copy size <= dest len-1 from gcc8 for safe
    auto len = sizeof(param.algo_name);
    strncpy(param.algo_name, name, len - 1);
    param.algo_name[len - 1] = '\0';
    mgb_assert(!param.algo_name[sizeof(param.algo_name) - 2],
               "algo name too long: %s; len=%zu", name, strlen(name));
    param.workspace = get_workspace_size_bytes(algo);
    for (int i = 0; i < arity; ++i) {
        auto&& src = m_layouts[i];
        mgb_assert(src.format.is_default() &&
                           (src.dtype.category() == DTypeCategory::FLOAT ||
                            src.dtype.category() == DTypeCategory::INT ||
                            src.dtype.category() == DTypeCategory::QUANTIZED),
                   "unsupported layout in profiling: %s",
                   src.to_string().c_str());
        param.dtypes[i] = src.dtype.enumv();
    }
    param.comp_node_loc = m_mgb_opr->output(0)->comp_node().locator();
    mgb_assert(param.shapes.size() == m_layouts.size());
    for (size_t i = 0; i < param.shapes.size(); ++i)
        param.shapes[i] = m_layouts[i];
    param.opr_param = m_megdnn_opr->param();

    if (is_weights_persistent) {
        modify_param_with_weights_preprocessed(param);
    }

    auto rst = TimedProfiler<Opr>::profile(param, timeout);
    // MIOpen conv profiles all available algos when a specfic shape is
    // provided for the first time, which probably adds to the result time.
    // Therefore, a second profile execution is needed.
    if (strncmp(name, "MIOpen", 6) == 0)
        rst = TimedProfiler<Opr>::profile(param, timeout);
    if (!rst.valid())
        return None;
    return AlgoChooserProfileCache::ResultEntry{
            algo->name(), algo->is_reproducible(), rst.val().time,
            param.workspace};
}

}  // anonymous namespace

/* ==================== misc impl  ==================== */

mixin::Convolution::~Convolution() = default;

void mixin::Convolution::set_execution_policy(const ExecutionPolicy& policy) {
    mgb_throw_if(
            m_policy_accessed, InternalError,
            "attempt to modify ExecutionPolicy after it has been accessed");
    m_policy = policy;
}

template <class MgbOpr, class MegDNNOpr>
void mixin::Convolution::init_output_static_infer_desc_for_bwd_data(
        cg::OperatorNodeBase* self) {
    using namespace cg::static_infer;
    auto&& mgr = self->owner_graph()->static_infer_manager();

    DepVal inp_deps;
    inp_deps.reserve(4);
    for (int i = 0; i < 2; ++i) {
        inp_deps.push_back({self->input(i), DepType::SHAPE});
    }

    // output shape
    if (self->input().size() == 3) {
        mgr.register_shape_infer(self->output(0),
                                 ShapeInferDesc::make_identity(self->input(2)));
    } else {
        auto infer_shp = [self](TensorShape& dest, const InpVal& inp) {
            TensorLayout ol{self->output(0)->dtype()};
            static_cast<MgbOpr*>(self)->megdnn_opr()->deduce_layout(
                    {inp.val.at(0).shape(), self->input(0)->dtype()},
                    {inp.val.at(1).shape(), self->input(1)->dtype()}, ol);
            dest = ol;
            return true;
        };
        mgr.register_shape_infer(self->output(0),
                                 {SourceType::DEP, inp_deps, infer_shp});
    }

    // workspace size
    auto infer_wk = [self](TensorShape& dest, const InpVal& inp) {
        auto&& iv = inp.val;
        dest.ndim = 1;
        dest.shape[0] = AlgoChooser<MegDNNOpr>::setup_algo(
                {TensorLayout{iv[0].shape(), self->input(0)->dtype(),
                              self->input(0)->format()},
                 {iv[1].shape(), self->input(1)->dtype(),
                  self->input(1)->format()},
                 {iv.at(2).shape(), self->output(0)->dtype(),
                  self->output(0)->format()}},
                static_cast<MgbOpr*>(self)->megdnn_opr(),
                static_cast<MgbOpr*>(self));
        return true;
    };
    inp_deps.push_back({self->output(0), DepType::SHAPE});
    auto workspace_dep_var =
            WorkspaceLimitGetter::register_to_graph(self->owner_graph());
    if (workspace_dep_var) {
        inp_deps.push_back({workspace_dep_var, DepType::VALUE});
    }
    mgr.register_shape_infer(self->output(1),
                             {SourceType::DEP, inp_deps, infer_wk});
}

#define IMPL_CONV(_cls, _prof_name)                                  \
    void _cls::init_profile_cache() {                                \
        std::string name(_prof_name CACHE_KEY_VERSION);              \
        name.append(megdnn_opr()->get_algorithm_set_name());         \
        m_profile_cache = std::make_unique<AlgoChooserProfileCache>( \
                comp_node(), name.c_str());                          \
    }                                                                \
    std::pair<const void*, size_t> _cls::param_blob() const {        \
        return {&param(), sizeof(Param)};                            \
    }                                                                \
    MGB_DYN_TYPE_OBJ_FINAL_IMPL(_cls)

AlgoChooserProfileCache& mixin::Convolution::profile_cache() const {
    if (!m_profile_cache) {
        const_cast<Convolution*>(this)->init_profile_cache();
        mgb_assert(m_profile_cache);
    }
    return *m_profile_cache;
}

/* ==================== ConvolutionForward  ==================== */

IMPL_CONV(ConvolutionForward, "conv_fwd");

ConvolutionForward::ConvolutionForward(VarNode* src, VarNode* filter,
                                       const Param& param,
                                       const ExecutionPolicy& policy,
                                       const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "conv", {src, filter}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter});
}

SymbolVar ConvolutionForward::make(SymbolVar src, SymbolVar filter,
                                   const Param& param,
                                   const ExecutionPolicy& policy,
                                   const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<ConvolutionForward>(
            src.node(), filter.node(), param, policy, config);
}

void ConvolutionForward::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    megdnn_opr()->deduce_dtype(input(0)->dtype(), input(1)->dtype(),
                               output_dtype);
    output(0)->dtype(output_dtype);
}

MGB_IMPL_OPR_GRAD(ConvolutionForward) {
    mgb_assert(opr.input(0)->dtype().category() == DTypeCategory::FLOAT,
               "only float data type supported for grad");
    mgb_assert(wrt_idx == 0 || wrt_idx == 1);
    mgb_assert(out_grad.size() == 2);
    if (wrt_idx == 0) {
        // data
        SymbolVar grad = ConvolutionBackwardData::make(
                opr.input(1), out_grad[0], opr.input(0), opr.param(),
                opr.execution_policy());
        return grad.node();
    } else {
        // filter
        SymbolVar grad = ConvolutionBackwardFilter::make(
                opr.input(0), out_grad[0], opr.input(1), opr.param(),
                opr.execution_policy());
        return grad.node();
    }
}

size_t ConvolutionForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 2 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::ConvolutionForward>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

void ConvolutionForward::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(0)->format());
}

/* ==================== ConvolutionBackwardData  ==================== */
IMPL_CONV(ConvolutionBackwardData, "conv_bwd_data");

ConvolutionBackwardData::ConvolutionBackwardData(
        VarNode* filter, VarNode* diff, VarNode* src_for_shp,
        const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config)
        : Super{filter->owner_graph(),
                config,
                "conv_bwd_data",
                {filter, diff}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({filter, diff});
    if (src_for_shp) {
        add_input({src_for_shp});
    }
}

SymbolVar ConvolutionBackwardData::make(SymbolVar filter, SymbolVar diff,
                                        SymbolVar src, const Param& param,
                                        const ExecutionPolicy& policy,
                                        const OperatorNodeConfig& config) {
    return filter.insert_single_output_opr<ConvolutionBackwardData>(
            filter.node(), diff.node(), src.node(), param, policy, config);
}

SymbolVar ConvolutionBackwardData::make(SymbolVar filter, SymbolVar data,
                                        const Param& param,
                                        const ExecutionPolicy& policy,
                                        const OperatorNodeConfig& config) {
    return make(filter, data, {}, param, policy, config);
}

void ConvolutionBackwardData::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void ConvolutionBackwardData::init_output_static_infer_desc() {
    init_output_static_infer_desc_for_bwd_data<ConvolutionBackwardData,
                                               megdnn::ConvolutionBackwardData>(
            this);
}

void ConvolutionBackwardData::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    megdnn_opr()->deduce_dtype(input(0)->dtype(), input(1)->dtype(),
                               output_dtype);
    output(0)->dtype(output_dtype);
}

void ConvolutionBackwardData::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(1)->format());
}

cg::OperatorNodeBase::NodeProp* ConvolutionBackwardData::do_make_node_prop()
        const {
    auto prop = Super::Super::do_make_node_prop();
    if (input().size() == 3) {
        using D = NodeProp::DepType;
        prop->reset_dep_type(input(), {D::DEV_VALUE, D::DEV_VALUE, D::SHAPE});
    }
    return prop;
}

void ConvolutionBackwardData::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                       input(1)->dev_tensor().as_megdnn(),
                       output(0)->dev_tensor().as_megdnn(),
                       intl::get_megdnn_workspace_from_var(output(1)));
}

MGB_IMPL_OPR_GRAD(ConvolutionBackwardData) {
    mgb_assert(!out_grad[1]);
    if (wrt_idx == 0) {
        return ConvolutionBackwardFilter::make(out_grad[0], opr.input(1),
                                               opr.input(0), opr.param(),
                                               opr.execution_policy())
                .node();
    }
    if (wrt_idx == 1) {
        return Convolution::make(out_grad[0], opr.input(0), opr.param(),
                                 opr.execution_policy())
                .node();
    }
    return nullptr;
}

/* ==================== ConvolutionBackwardFilter  ==================== */
IMPL_CONV(ConvolutionBackwardFilter, "conv_bwd_filter");

ConvolutionBackwardFilter::ConvolutionBackwardFilter(
        VarNode* src, VarNode* diff, VarNode* filter, const Param& param,
        const ExecutionPolicy& policy, const OperatorNodeConfig& config)
        : Super({src->owner_graph(),
                 config,
                 "conv_bwd_filter",
                 {src, diff, filter}},
                2, false) {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, diff, filter});
}

SymbolVar ConvolutionBackwardFilter::make(SymbolVar src, SymbolVar diff,
                                          SymbolVar filter, const Param& param,
                                          const ExecutionPolicy& policy,
                                          const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<ConvolutionBackwardFilter>(
            src.node(), diff.node(), filter.node(), param, policy, config);
}

size_t ConvolutionBackwardFilter::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 3 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::ConvolutionBackwardFilter>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

MGB_IMPL_OPR_GRAD(ConvolutionBackwardFilter) {
    mgb_assert(!out_grad[1]);
    if (wrt_idx == 0) {
        return ConvolutionBackwardData::make(out_grad[0], opr.input(1),
                                             opr.input(0), opr.param(),
                                             opr.execution_policy())
                .node();
    }
    if (wrt_idx == 1) {
        return Convolution::make(opr.input(0), out_grad[0], opr.param(),
                                 opr.execution_policy())
                .node();
    }
    return nullptr;
}
/* ==================== Convolution3DForward ==================== */

IMPL_CONV(Convolution3DForward, "conv3d_fwd");

Convolution3DForward::Convolution3DForward(VarNode* src, VarNode* filter,
                                           const Param& param,
                                           const ExecutionPolicy& policy,
                                           const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "conv3d", {src, filter}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter});
}

SymbolVar Convolution3DForward::make(SymbolVar src, SymbolVar filter,
                                     const Param& param,
                                     const ExecutionPolicy& policy,
                                     const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<Convolution3DForward>(
            src.node(), filter.node(), param, policy, config);
}

void Convolution3DForward::init_output_dtype() {
    switch (param().data_type) {
        case Param::DataType::FLOAT:
            output(0)->dtype(input(0)->dtype());
            break;
#if !MEGDNN_DISABLE_FLOAT16
        case Param::DataType::FLOAT_IO16xC32:
            mgb_assert(input(0)->dtype() == dtype::Float16(),
                       "invalid input dtype %s", input(0)->name().c_str());
            output(0)->dtype(input(0)->dtype());
            break;
#endif
        default:
            mgb_throw(MegBrainError, "bad data_type enum");
    }
}

MGB_IMPL_OPR_GRAD(Convolution3DForward) {
    mgb_assert(opr.param().data_type ==
                       Convolution3DForward::Param::DataType::FLOAT,
               "only float data type supported for grad");
    mgb_assert(wrt_idx == 0 || wrt_idx == 1);
    mgb_assert(out_grad.size() == 2);
    if (wrt_idx == 0) {
        // data
        SymbolVar grad = Convolution3DBackwardData::make(
                opr.input(1), out_grad[0], opr.input(0), opr.param(),
                opr.execution_policy());
        return grad.node();
    } else {
        // filter
        SymbolVar grad = Convolution3DBackwardFilter::make(
                opr.input(0), out_grad[0], opr.input(1), opr.param(),
                opr.execution_policy());
        return grad.node();
    }
}

size_t Convolution3DForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 2 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::Convolution3DForward>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

/* ==================== Convolution3DBackwardData  ==================== */
IMPL_CONV(Convolution3DBackwardData, "conv3d_bwd_data");

Convolution3DBackwardData::Convolution3DBackwardData(
        VarNode* filter, VarNode* diff, VarNode* src_for_shp,
        const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config)
        : Super{filter->owner_graph(),
                config,
                "conv3d_bwd_data",
                {filter, diff}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({filter, diff});
    if (src_for_shp) {
        add_input({src_for_shp});
    }
}

SymbolVar Convolution3DBackwardData::make(SymbolVar filter, SymbolVar diff,
                                          SymbolVar src, const Param& param,
                                          const ExecutionPolicy& policy,
                                          const OperatorNodeConfig& config) {
    return filter.insert_single_output_opr<Convolution3DBackwardData>(
            filter.node(), diff.node(), src.node(), param, policy, config);
}

SymbolVar Convolution3DBackwardData::make(SymbolVar filter, SymbolVar data,
                                          const Param& param,
                                          const ExecutionPolicy& policy,
                                          const OperatorNodeConfig& config) {
    return make(filter, data, {}, param, policy, config);
}

void Convolution3DBackwardData::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void Convolution3DBackwardData::init_output_static_infer_desc() {
    init_output_static_infer_desc_for_bwd_data<
            Convolution3DBackwardData, megdnn::Convolution3DBackwardData>(this);
}

cg::OperatorNodeBase::NodeProp* Convolution3DBackwardData::do_make_node_prop()
        const {
    auto prop = Super::Super::do_make_node_prop();
    if (input().size() == 3) {
        using D = NodeProp::DepType;
        prop->reset_dep_type(input(), {D::DEV_VALUE, D::DEV_VALUE, D::SHAPE});
    }
    return prop;
}

void Convolution3DBackwardData::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                       input(1)->dev_tensor().as_megdnn(),
                       output(0)->dev_tensor().as_megdnn(),
                       intl::get_megdnn_workspace_from_var(output(1)));
}

MGB_IMPL_OPR_GRAD(Convolution3DBackwardData) {
    mgb_assert(!out_grad[1]);
    if (wrt_idx == 0) {
        return Convolution3DBackwardFilter::make(out_grad[0], opr.input(1),
                                                 opr.input(0), opr.param(),
                                                 opr.execution_policy())
                .node();
    }
    if (wrt_idx == 1) {
        return Convolution3D::make(out_grad[0], opr.input(0), opr.param(),
                                   opr.execution_policy())
                .node();
    }
    return nullptr;
}

/* ==================== Convolution3DBackwardFilter  ==================== */
IMPL_CONV(Convolution3DBackwardFilter, "conv3d_bwd_filter");

Convolution3DBackwardFilter::Convolution3DBackwardFilter(
        VarNode* src, VarNode* diff, VarNode* filter, const Param& param,
        const ExecutionPolicy& policy, const OperatorNodeConfig& config)
        : Super({src->owner_graph(),
                 config,
                 "conv3d_bwd_filter",
                 {src, diff, filter}},
                2, false) {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, diff, filter});
}

SymbolVar Convolution3DBackwardFilter::make(SymbolVar src, SymbolVar diff,
                                            SymbolVar filter,
                                            const Param& param,
                                            const ExecutionPolicy& policy,
                                            const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<Convolution3DBackwardFilter>(
            src.node(), diff.node(), filter.node(), param, policy, config);
}

size_t Convolution3DBackwardFilter::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 3 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::Convolution3DBackwardFilter>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

/* ========================== MaskConvolution  ========================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MaskConvolution);

MaskConvolution::MaskConvolution(VarNode* src, VarNode* filter, VarNode* mask,
                                 const Param& param,
                                 const OperatorNodeConfig& config)
        : Super(src->owner_graph(), config, "mask_conv_fwd",
                {src, filter, mask}) {
    init_megdnn_opr(*this, param);
    add_input({src, filter, mask});
}

SymbolVar MaskConvolution::make(SymbolVar src, SymbolVar filter, SymbolVar mask,
                                const Param& param,
                                const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<MaskConvolution>(
            src.node(), filter.node(), mask.node(), param, config);
}

void MaskConvolution::init_output_dtype() {
    auto dtype = input(2)->dtype();
    mgb_assert(dtype == dtype::Int32() || dtype == dtype::Int16() ||
                       dtype == dtype::Int8(),
               "dtype must be int8, int16 or int32, while get %s",
               dtype.name());
    output(0)->dtype(input(0)->dtype());
}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MaskPropagate);

MaskPropagate::MaskPropagate(VarNode* src, const Param& param,
                             const OperatorNodeConfig& config)
        : Super(src->owner_graph(), config, "mask_propagate", {src}) {
    init_megdnn_opr(*this, param);
    add_input({src});
}

void MaskPropagate::init_output_dtype() {
    auto dtype = input(0)->dtype();
    mgb_assert(dtype == dtype::Int32() || dtype == dtype::Int16() ||
               dtype == dtype::Int8());
    output(0)->dtype(dtype);
}

SymbolVar MaskPropagate::make(SymbolVar src, const Param& param,
                              const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<MaskPropagate>(src.node(), param,
                                                       config);
}

/* ==================== ConvBiasForward  ==================== */
IMPL_CONV(ConvBiasForward, "conv_bias_fwd");

ConvBiasForward::ConvBiasForward(VarNode* src, VarNode* filter,
                                 const Param& param,
                                 const ExecutionPolicy& policy,
                                 const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "conv_bias", {src, filter}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter});
}

ConvBiasForward::ConvBiasForward(VarNode* src, VarNode* filter, VarNode* bias,
                                 const Param& param,
                                 const ExecutionPolicy& policy,
                                 const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "conv_bias", {src, filter, bias}} {
    m_policy = policy;
    init_megdnn_opr(*this, param);
    add_input({src, filter, bias});
}

ConvBiasForward::ConvBiasForward(VarNode* src, VarNode* filter, VarNode* bias,
                                 VarNode* z, const Param& param,
                                 const ExecutionPolicy& policy,
                                 const OperatorNodeConfig& config)
        : Super{src->owner_graph(),
                config,
                "conv_bias",
                {src, filter, bias, z}} {
    m_policy = policy;
    init_megdnn_opr(*this, param);
    add_input({src, filter, bias, z});
}

void ConvBiasForward::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

SymbolVar ConvBiasForward::make(SymbolVar src, SymbolVar filter,
                                const Param& param,
                                const ExecutionPolicy& policy,
                                const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<ConvBiasForward>(
            src.node(), filter.node(), param, policy, config);
}

SymbolVar ConvBiasForward::make(SymbolVar src, SymbolVar filter, SymbolVar bias,
                                const Param& param,
                                const ExecutionPolicy& policy,
                                const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<ConvBiasForward>(
            src.node(), filter.node(), bias.node(), param, policy, config);
}

SymbolVar ConvBiasForward::make(SymbolVar src, SymbolVar filter, SymbolVar bias,
                                SymbolVar z, const Param& param,
                                const ExecutionPolicy& policy,
                                const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<ConvBiasForward>(
            src.node(), filter.node(), bias.node(), z.node(), param, policy,
            config);
}

void ConvBiasForward::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    DType i0, i1, i2, i3;
    mgb_assert(input().size() >= 2 && input().size() <= 4);
    i0 = input(0)->dtype();
    i1 = input(1)->dtype();
    if (input().size() >= 3)
        i2 = input(2)->dtype();
    if (input().size() == 4)
        i3 = input(3)->dtype();
    megdnn_opr()->deduce_dtype(i0, i1, i2, i3, output_dtype);
    output(0)->dtype(output_dtype);
}

size_t ConvBiasForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    auto mo = megdnn_opr();
    TensorLayout i0, i1, i2, i3;
    mgb_assert(input_shapes.size() >= 2 && input_shapes.size() <= 4);
    i0 = {input_shapes[0], input(0)->dtype(), input(0)->format()};
    i1 = {input_shapes[1], input(1)->dtype(), input(1)->format()};
    if (input_shapes.size() >= 3)
        i2 = {input_shapes[2], input(2)->dtype(), input(2)->format()};
    else {
        DType dtype;
        mo->deduce_dtype(input(0)->dtype(), input(1)->dtype(), DType{}, DType{},
                         dtype);
        i2 = {{}, dtype};
    }
    if (input_shapes.size() == 4)
        i3 = {input_shapes[3], input(3)->dtype(), input(3)->format()};
    else
        i3 = {{}, output(0)->dtype(), output(0)->format()};

    return AlgoChooser<megdnn::ConvBias>::setup_algo(
            {i0,
             i1,
             i2,
             i3,
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            mo, this);
}

void ConvBiasForward::scn_do_execute() {
    auto&& inp = input();
    auto mo = megdnn_opr();
    if (inp.size() == 2) {
        TensorLayout bias_layout;
        bias_layout.ndim = 0;
        if (output(0)->dtype().enumv() == DTypeEnum::QuantizedS8) {
            bias_layout.dtype = dtype::QuantizedS32(
                    output(0)->dtype().param<dtype::QuantizedS8>().scale);
        } else {
            bias_layout.dtype = output(0)->dtype();
        }
        TensorLayout z_layout;
        z_layout.ndim = 0;
        z_layout.dtype = output(0)->dtype();
        megdnn::TensorND bias_tensor{nullptr, bias_layout};
        megdnn::TensorND z_tensor{nullptr, z_layout};
        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(), bias_tensor, z_tensor,
                 output(0)->dev_tensor().as_megdnn(),
                 intl::get_megdnn_workspace_from_var(output().back()));

    } else if (inp.size() == 3) {
        TensorLayout z_layout;
        z_layout.ndim = 0;
        z_layout.dtype = output(0)->dtype();
        megdnn::TensorND z_tensor{nullptr, z_layout};

        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(),
                 inp[2]->dev_tensor().as_megdnn(), z_tensor,
                 output(0)->dev_tensor().as_megdnn(),
                 intl::get_megdnn_workspace_from_var(output().back()));
    } else {
        mgb_assert(inp.size() == 4);
        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(),
                 inp[2]->dev_tensor().as_megdnn(),
                 inp[3]->dev_tensor().as_megdnn(),
                 output(0)->dev_tensor().as_megdnn(),
                 intl::get_megdnn_workspace_from_var(output().back()));
    }
}

void ConvBiasForward::get_output_var_shape(const TensorShapeArray& inp_shape,
                                           TensorShapeArray& out_shape) const {
    auto mo = megdnn_opr();
    TensorLayout dst;
    mo->deduce_layout({inp_shape[0], input(0)->dtype(), input(0)->format()},
                      {inp_shape[1], input(1)->dtype(), input(0)->format()}, {},
                      {}, dst);
    out_shape[0] = dst;
}

void ConvBiasForward::init_output_static_infer_desc() {
    Super::set_nr_managed_outputs(this->output().size() - 1);
    Super::init_output_static_infer_desc();
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<
                    megdnn::ConvBiasForward>::val);
}

void ConvBiasForward::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(0)->format());
}

void ConvBiasForward::check_winograd_param_valid(
        const megdnn::ConvBias::WinogradParam& param,
        const DType& dtype) {
    if (dtype.enumv() == DTypeEnum::Float32) {
        mgb_assert(param.channel_block_size == 1 ||
                           param.channel_block_size == 4 ||
                           param.channel_block_size == 8,
                   "only support 1/4/8 for the channel_block_size of "
                   "winograd param, got %u",
                   param.channel_block_size);
    } else {
        mgb_assert((MEGDNN_FLOAT16_SELECT(dtype.enumv() == DTypeEnum::Float16,
                                          false) ||
                    dtype.enumv() == DTypeEnum::QuantizedS8 ||
                    dtype.enumv() == DTypeEnum::Quantized8Asymm) &&
                           (param.channel_block_size == 1 ||
                            param.channel_block_size == 8),
                   "only support 1/8 for the channel_block_size of "
                   "winograd param, got %u",
                   param.channel_block_size);
    }
}

megdnn::param::MatrixMul::Format ConvBiasForward::get_matmul_format(
        const megdnn::ConvBias::WinogradParam& param) {
    switch (param.channel_block_size) {
        case 1:
            return megdnn::param::MatrixMul::Format::DEFAULT;
            break;
        case 4:
            return megdnn::param::MatrixMul::Format::MK4;
            break;
        case 8:
            return megdnn::param::MatrixMul::Format::MK8;
            break;
        default:
            mgb_throw(InternalError,
                      "Only Support 1/4/8 for "
                      "channel_block_size, got: %u",
                      param.channel_block_size);
    }
}

/* ===================== LocalShareForward ==================== */

IMPL_CONV(LocalShareForward, "local_share");

LocalShareForward::LocalShareForward(VarNode* src, VarNode* filter,
                                     const Param& param,
                                     const ExecutionPolicy& policy,
                                     const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "local_share", {src, filter}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter});
}

SymbolVar LocalShareForward::make(SymbolVar src, SymbolVar filter,
                                  const Param& param,
                                  const ExecutionPolicy& policy,
                                  const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<LocalShareForward>(
            src.node(), filter.node(), param, policy, config);
}

void LocalShareForward::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    mgb_assert(!output_dtype.valid() || output_dtype == dtype::Float32());
    output_dtype = dtype::Float32();
    output(0)->dtype(output_dtype);
}

void LocalShareForward::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(0)->format());
}

size_t LocalShareForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 2 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::LocalShareForward>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

MGB_IMPL_OPR_GRAD(LocalShareForward) {
    mgb_assert(opr.input(0)->dtype().category() == DTypeCategory::FLOAT,
            "only float data type supported for grad");
    mgb_assert(wrt_idx == 0 || wrt_idx == 1);
    mgb_assert(out_grad.size() == 2);
    if (wrt_idx == 0) {
        // data
        SymbolVar grad = LocalShareBackwardData::make(
                opr.input(1), out_grad[0], opr.input(0),
                opr.param(), opr.execution_policy());
        return grad.node();
    } else {
        // filter
        SymbolVar grad = LocalShareBackwardFilter::make(
                opr.input(0), out_grad[0], opr.input(1),
                opr.param(), opr.execution_policy());
        return grad.node();
    }
}

/* ===================== LocalShareBackwardData ==================== */

IMPL_CONV(LocalShareBackwardData, "local_share_bwd_data");

LocalShareBackwardData::LocalShareBackwardData(VarNode* filter, VarNode* diff,
                                               VarNode* src_for_shp,
                                               const Param& param,
                                               const ExecutionPolicy& policy,
                                               const OperatorNodeConfig& config)
        : Super{filter->owner_graph(), config, "local_share_bwd_data", {filter, diff}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({filter, diff});
    if (src_for_shp) {
        add_input({src_for_shp});
    }
}

SymbolVar LocalShareBackwardData::make(SymbolVar filter, SymbolVar diff,
                                       SymbolVar src, const Param& param,
                                       const ExecutionPolicy& policy,
                                       const OperatorNodeConfig& config) {
    return filter.insert_single_output_opr<LocalShareBackwardData>(
            filter.node(), diff.node(), src.node(), param, policy, config);
}

void LocalShareBackwardData::init_output_static_infer_desc() {
    init_output_static_infer_desc_for_bwd_data<LocalShareBackwardData,
                                               megdnn::LocalShareBackwardData>(
            this);
}

void LocalShareBackwardData::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    mgb_assert(!output_dtype.valid() || output_dtype == dtype::Float32());
    output_dtype = dtype::Float32();
    output(0)->dtype(output_dtype);
}

void LocalShareBackwardData::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

cg::OperatorNodeBase::NodeProp* LocalShareBackwardData::do_make_node_prop()
        const {
    auto prop = Super::Super::do_make_node_prop();
    mgb_assert(input().size() == 3);
    using D = NodeProp::DepType;
    prop->reset_dep_type(input(), {D::DEV_VALUE, D::DEV_VALUE, D::SHAPE});
    return prop;
}

void LocalShareBackwardData::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                       input(1)->dev_tensor().as_megdnn(),
                       output(0)->dev_tensor().as_megdnn(),
                       intl::get_megdnn_workspace_from_var(output(1)));
}

MGB_IMPL_OPR_GRAD(LocalShareBackwardData) {
    mgb_assert(!out_grad[1]);
    if (wrt_idx == 0) {
        return LocalShareBackwardFilter::make(out_grad[0], opr.input(1),
                                              opr.input(0), opr.param(),
                                              opr.execution_policy())
                .node();
    }
    if (wrt_idx == 1) {
        return LocalShare::make(out_grad[0], opr.input(0), opr.param(),
                                opr.execution_policy())
                .node();
    }
    return nullptr;
}

/* ==================== LocalShareBackwardFilter  ==================== */

IMPL_CONV(LocalShareBackwardFilter, "local_share_bwd_filter");

LocalShareBackwardFilter::LocalShareBackwardFilter(
        VarNode* src, VarNode* diff, VarNode* filter, const Param& param,
        const ExecutionPolicy& policy, const OperatorNodeConfig& config)
        : Super({src->owner_graph(),
                 config,
                 "local_share_bwd_filter",
                 {src, diff, filter}},
                2, false) {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, diff, filter});
}

SymbolVar LocalShareBackwardFilter::make(
        SymbolVar src, SymbolVar diff, SymbolVar filter,
        const Param &param,
        const ExecutionPolicy &policy,
        const OperatorNodeConfig &config) {

    return src.insert_single_output_opr<LocalShareBackwardFilter>(
            src.node(), diff.node(), filter.node(), param, policy, config);
}

size_t LocalShareBackwardFilter::get_workspace_size_bytes(
        const TensorShapeArray &input_shapes,
        const TensorShapeArray &output_shapes) const {
    mgb_assert(input_shapes.size() == 3 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::LocalShareBackwardFilter>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
            {input_shapes[1], input(1)->dtype(), input(1)->format()},
            {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

MGB_IMPL_OPR_GRAD(LocalShareBackwardFilter) {
    mgb_assert(!out_grad[1]);
    if (wrt_idx == 0) {
        return LocalShareBackwardData::make(out_grad[0], opr.input(1),
                opr.input(0), opr.param(), opr.execution_policy()).node();
    }
    if (wrt_idx == 1) {
        return LocalShare::make(
                opr.input(0), out_grad[0], opr.param(), opr.execution_policy()).
            node();
    }
    return nullptr;
}

/* ===================== DeformableConvForward ==================== */

IMPL_CONV(DeformableConvForward, "deformable_conv");

DeformableConvForward::DeformableConvForward(VarNode* src, VarNode* filter,
                                             VarNode* offset, VarNode* mask,
                                             const Param& param,
                                             const ExecutionPolicy& policy,
                                             const OperatorNodeConfig& config)
        : Super{src->owner_graph(),
                config,
                "deformable_conv",
                {src, filter, offset, mask}} {
    mgb_assert(src->dtype() == dtype::Float32() &&
                       filter->dtype() == dtype::Float32() &&
                       offset->dtype() == dtype::Float32() &&
                       mask->dtype() == dtype::Float32(),
               "input should be float32, got %s, %s, %s, %s",
               src->dtype().name(), filter->dtype().name(),
               offset->dtype().name(), mask->dtype().name());

    init_megdnn_opr(*this, param);
    m_policy = policy;

    add_input({src, filter, offset, mask});
}

SymbolVar DeformableConvForward::make(SymbolVar src, SymbolVar filter,
                                      SymbolVar offset, SymbolVar mask,
                                      const Param& param,
                                      const ExecutionPolicy& policy,
                                      const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<DeformableConvForward>(
            src.node(), filter.node(), offset.node(), mask.node(), param,
            policy, config);
}

void DeformableConvForward::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    mgb_assert(!output_dtype.valid() || output_dtype == dtype::Float32());
    output_dtype = dtype::Float32();
    output(0)->dtype(output_dtype);
}

void DeformableConvForward::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(0)->format());
}

size_t DeformableConvForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 4 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::DeformableConvForward>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {input_shapes[2], input(2)->dtype(), input(2)->format()},
             {input_shapes[3], input(3)->dtype(), input(3)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

MGB_IMPL_OPR_GRAD(DeformableConvForward) {
    mgb_assert(opr.input(0)->dtype() == dtype::Float32(),
               "only float data type supported for grad");
    mgb_assert(wrt_idx < 4);
    mgb_assert(!out_grad[1]);
    mgb_assert(out_grad.size() == 2);

    // data, offset and mask
    auto grad_arr = DeformableConvBackwardData::make_all(
            opr.input(0), opr.input(1), opr.input(2), opr.input(3), out_grad[0],
            opr.param(), opr.execution_policy(), opr.config());
    // filter
    auto filter_grad = DeformableConvBackwardFilter::make(
            opr.input(0), opr.input(1), opr.input(2), opr.input(3), out_grad[0],
            opr.param(), opr.execution_policy(), opr.config());

    SymbolVarArray grads = {grad_arr[0], filter_grad, grad_arr[1], grad_arr[2]};
    return grads[wrt_idx].node();
}

/* ==================== DeformableConvBackwardData  ==================== */

IMPL_CONV(DeformableConvBackwardData, "deformalbe_conv_backward_data");

DeformableConvBackwardData::DeformableConvBackwardData(
        VarNode* src, VarNode* filter, VarNode* offset, VarNode* mask,
        VarNode* diff, const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config)
        : Super{filter->owner_graph(),
                config,
                "deformable_conv_backward_data",
                {src, filter, offset, mask, diff}} {
    mgb_assert(src->dtype() == dtype::Float32() and
                       filter->dtype() == dtype::Float32() and
                       offset->dtype() == dtype::Float32() and
                       mask->dtype() == dtype::Float32() and
                       diff->dtype() == dtype::Float32(),
               "input should be float32, got %s, %s, %s, %s %s",
               src->dtype().name(), filter->dtype().name(),
               offset->dtype().name(), mask->dtype().name(),
               diff->dtype().name());

    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter, offset, mask, diff});
}

SymbolVarArray DeformableConvBackwardData::make_all(
        SymbolVar src, SymbolVar filter, SymbolVar offset, SymbolVar mask,
        SymbolVar diff, const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config) {
    auto graph = src.node()->owner_graph();

    auto back_node =
            graph->insert_opr(std::make_unique<DeformableConvBackwardData>(
                    src.node(), filter.node(), offset.node(), mask.node(),
                    diff.node(), param, policy, config));

    return {back_node->output(0), back_node->output(1), back_node->output(2)};
}

SymbolVar DeformableConvBackwardData::make(SymbolVar src, SymbolVar filter,
                                           SymbolVar offset, SymbolVar mask,
                                           SymbolVar diff, const Param& param,
                                           const ExecutionPolicy& policy,
                                           const OperatorNodeConfig& config) {
    auto&& all =
            make_all(src, filter, offset, mask, diff, param, policy, config);
    return all[0];
}

void DeformableConvBackwardData::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),   // src
                       input(1)->dev_tensor().as_megdnn(),   // filter
                       input(2)->dev_tensor().as_megdnn(),   // offset
                       input(3)->dev_tensor().as_megdnn(),   // mask
                       input(4)->dev_tensor().as_megdnn(),   // diff
                       output(0)->dev_tensor().as_megdnn(),  // src_grad
                       output(1)->dev_tensor().as_megdnn(),  // offset_grad
                       output(2)->dev_tensor().as_megdnn(),  // mask_grad
                       intl::get_megdnn_workspace_from_var(output(3)));
}

void DeformableConvBackwardData::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    TensorShape im_shp = inp_shape[0];
    TensorShape offset_shp = inp_shape[2];
    TensorShape mask_shp = inp_shape[3];

    mgb_assert(im_shp.ndim == 4, "invalid src shape: %s",
               im_shp.to_string().c_str());
    mgb_assert(offset_shp.ndim == 4, "invalid offset shape: %s",
               offset_shp.to_string().c_str());
    mgb_assert(mask_shp.ndim == 4, "invalid mask shape: %s",
               mask_shp.to_string().c_str());
    mgb_assert(out_shape.size() == 3);

    out_shape[0] = im_shp;
    out_shape[1] = offset_shp;
    out_shape[2] = mask_shp;
}

size_t DeformableConvBackwardData::get_workspace_size_bytes(
        const TensorShapeArray& inp_shape,
        const TensorShapeArray& out_shape) const {
    size_t ws = AlgoChooser<megdnn::DeformableConvBackwardData>::setup_algo(
            {TensorLayout{inp_shape[0], input(0)->dtype(), input(0)->format()},
             {inp_shape[1], input(1)->dtype(), input(1)->format()},
             {inp_shape[2], input(2)->dtype(), input(2)->format()},
             {inp_shape[3], input(3)->dtype(), input(3)->format()},
             {inp_shape[4], input(4)->dtype(), input(4)->format()},
             {out_shape[0], output(0)->dtype(), output(0)->format()},
             {out_shape[1], output(1)->dtype(), output(1)->format()},
             {out_shape[2], output(2)->dtype(), output(2)->format()}},
            megdnn_opr(), this);
    return ws;
}

void DeformableConvBackwardData::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    mgb_assert(!output_dtype.valid() || output_dtype == dtype::Float32());
    output_dtype = dtype::Float32();
    output(0)->dtype(output_dtype);
    output(1)->dtype(output_dtype);
    output(2)->dtype(output_dtype);
}

void DeformableConvBackwardData::init_output_format() {
    mgb_assert(output().size() == 4);
    output(0)->format(input(0)->format());
    output(1)->format(input(2)->format());
    output(2)->format(input(3)->format());
}

cg::OperatorNodeBase::NodeProp* DeformableConvBackwardData::do_make_node_prop()
        const {
    auto prop = Super::Super::do_make_node_prop();
    using D = NodeProp::DepType;
    mgb_assert(input().size() == 5);
    prop->reset_dep_type(input(), {D::DEV_VALUE, D::DEV_VALUE, D::DEV_VALUE,
                                   D::DEV_VALUE, D::DEV_VALUE});
    return prop;
}

void DeformableConvBackwardData::init_output_static_infer_desc() {
    Super::set_nr_managed_outputs(this->output().size() - 1);
    Super::init_output_static_infer_desc();
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<
                    megdnn::DeformableConvBackwardData>::val);
}

/* ==================== DeformableConvBackwardFilter  ==================== */

IMPL_CONV(DeformableConvBackwardFilter, "deformalbe_conv_backward_filter");

DeformableConvBackwardFilter::DeformableConvBackwardFilter(
        VarNode* src, VarNode* filter, VarNode* offset, VarNode* mask,
        VarNode* diff, const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config)
        : Super({src->owner_graph(),
                 config,
                 "deformable_conv_backward_filter",
                 {src, filter, offset, mask, diff}},
                1, false) {
    mgb_assert(src->dtype() == dtype::Float32() and
                       filter->dtype() == dtype::Float32() and
                       offset->dtype() == dtype::Float32() and
                       mask->dtype() == dtype::Float32() and
                       diff->dtype() == dtype::Float32(),
               "input should be float32, got %s, %s, %s, %s %s",
               src->dtype().name(), filter->dtype().name(),
               offset->dtype().name(), mask->dtype().name(),
               diff->dtype().name());
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter, offset, mask, diff});
}

SymbolVar DeformableConvBackwardFilter::make(SymbolVar src, SymbolVar filter,
                                             SymbolVar offset, SymbolVar mask,
                                             SymbolVar diff, const Param& param,
                                             const ExecutionPolicy& policy,
                                             const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<DeformableConvBackwardFilter>(
            src.node(), filter.node(), offset.node(), mask.node(), diff.node(),
            param, policy, config);
}

void DeformableConvBackwardFilter::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),   // src
                       input(2)->dev_tensor().as_megdnn(),   // offset
                       input(3)->dev_tensor().as_megdnn(),   // mask
                       input(4)->dev_tensor().as_megdnn(),   // diff
                       output(0)->dev_tensor().as_megdnn(),  // filter_diff
                       intl::get_megdnn_workspace_from_var(output(1)));
}

size_t DeformableConvBackwardFilter::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 5 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::DeformableConvBackwardFilter>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[2], input(2)->dtype(), input(2)->format()},
             {input_shapes[3], input(3)->dtype(), input(3)->format()},
             {input_shapes[4], input(4)->dtype(), input(4)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

/* ==================== BatchConvBiasForward  ==================== */
IMPL_CONV(BatchConvBiasForward, "batch_conv_bias_fwd");

BatchConvBiasForward::BatchConvBiasForward(VarNode* src, VarNode* filter,
                                           const Param& param,
                                           const ExecutionPolicy& policy,
                                           const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "batch_conv_bias", {src, filter}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter});
}

BatchConvBiasForward::BatchConvBiasForward(VarNode* src, VarNode* filter,
                                           VarNode* bias, const Param& param,
                                           const ExecutionPolicy& policy,
                                           const OperatorNodeConfig& config)
        : Super{src->owner_graph(),
                config,
                "batch_conv_bias",
                {src, filter, bias}} {
    m_policy = policy;
    init_megdnn_opr(*this, param);
    add_input({src, filter, bias});
}

BatchConvBiasForward::BatchConvBiasForward(VarNode* src, VarNode* filter,
                                           VarNode* bias, VarNode* z,
                                           const Param& param,
                                           const ExecutionPolicy& policy,
                                           const OperatorNodeConfig& config)
        : Super{src->owner_graph(),
                config,
                "batch_conv_bias",
                {src, filter, bias, z}} {
    m_policy = policy;
    init_megdnn_opr(*this, param);
    add_input({src, filter, bias, z});
}

void BatchConvBiasForward::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

SymbolVar BatchConvBiasForward::make(SymbolVar src, SymbolVar filter,
                                     const Param& param,
                                     const ExecutionPolicy& policy,
                                     const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<BatchConvBiasForward>(
            src.node(), filter.node(), param, policy, config);
}

SymbolVar BatchConvBiasForward::make(SymbolVar src, SymbolVar filter,
                                     SymbolVar bias, const Param& param,
                                     const ExecutionPolicy& policy,
                                     const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<BatchConvBiasForward>(
            src.node(), filter.node(), bias.node(), param, policy, config);
}

SymbolVar BatchConvBiasForward::make(SymbolVar src, SymbolVar filter,
                                     SymbolVar bias, SymbolVar z,
                                     const Param& param,
                                     const ExecutionPolicy& policy,
                                     const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<BatchConvBiasForward>(
            src.node(), filter.node(), bias.node(), z.node(), param, policy,
            config);
}

void BatchConvBiasForward::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    DType i0, i1, i2, i3;
    mgb_assert(input().size() >= 2 && input().size() <= 4);
    i0 = input(0)->dtype();
    i1 = input(1)->dtype();
    if (input().size() >= 3)
        i2 = input(2)->dtype();
    if (input().size() == 4)
        i3 = input(3)->dtype();
    megdnn_opr()->deduce_dtype(i0, i1, i2, i3, output_dtype);
    output(0)->dtype(output_dtype);
}

size_t BatchConvBiasForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    auto mo = megdnn_opr();
    TensorLayout i0, i1, i2, i3;
    mgb_assert(input_shapes.size() >= 2 && input_shapes.size() <= 4);
    i0 = {input_shapes[0], input(0)->dtype(), input(0)->format()};
    i1 = {input_shapes[1], input(1)->dtype(), input(1)->format()};
    if (input_shapes.size() >= 3)
        i2 = {input_shapes[2], input(2)->dtype(), input(2)->format()};
    else {
        DType dtype;
        mo->deduce_dtype(input(0)->dtype(), input(1)->dtype(), DType{}, DType{},
                         dtype);
        i2 = {{}, dtype};
    }
    if (input_shapes.size() == 4)
        i3 = {input_shapes[3], input(3)->dtype(), input(3)->format()};
    else
        i3 = {{}, output(0)->dtype(), output(0)->format()};

    return AlgoChooser<megdnn::BatchConvBias>::setup_algo(
            {i0,
             i1,
             i2,
             i3,
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            mo, this);
}

void BatchConvBiasForward::scn_do_execute() {
    auto&& inp = input();
    auto mo = megdnn_opr();
    if (inp.size() == 2) {
        TensorLayout bias_layout;
        bias_layout.ndim = 0;
        if (output(0)->dtype().enumv() == DTypeEnum::QuantizedS8) {
            bias_layout.dtype = dtype::QuantizedS32(
                    output(0)->dtype().param<dtype::QuantizedS8>().scale);
        } else {
            bias_layout.dtype = output(0)->dtype();
        }
        TensorLayout z_layout;
        z_layout.ndim = 0;
        z_layout.dtype = output(0)->dtype();
        megdnn::TensorND bias_tensor{nullptr, bias_layout};
        megdnn::TensorND z_tensor{nullptr, z_layout};
        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(), bias_tensor, z_tensor,
                 output(0)->dev_tensor().as_megdnn(),
                 intl::get_megdnn_workspace_from_var(output().back()));

    } else if (inp.size() == 3) {
        TensorLayout z_layout;
        z_layout.ndim = 0;
        z_layout.dtype = output(0)->dtype();
        megdnn::TensorND z_tensor{nullptr, z_layout};

        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(),
                 inp[2]->dev_tensor().as_megdnn(), z_tensor,
                 output(0)->dev_tensor().as_megdnn(),
                 intl::get_megdnn_workspace_from_var(output().back()));
    } else {
        mgb_assert(inp.size() == 4);
        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(),
                 inp[2]->dev_tensor().as_megdnn(),
                 inp[3]->dev_tensor().as_megdnn(),
                 output(0)->dev_tensor().as_megdnn(),
                 intl::get_megdnn_workspace_from_var(output().back()));
    }
}

void BatchConvBiasForward::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    auto mo = megdnn_opr();
    TensorLayout dst;
    mo->deduce_layout({inp_shape[0], input(0)->dtype(), input(0)->format()},
                      {inp_shape[1], input(1)->dtype(), input(0)->format()}, {},
                      {}, dst);
    out_shape[0] = dst;
}

void BatchConvBiasForward::init_output_static_infer_desc() {
    Super::set_nr_managed_outputs(this->output().size() - 1);
    Super::init_output_static_infer_desc();
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<
                    megdnn::BatchConvBiasForward>::val);
}

void BatchConvBiasForward::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(0)->format());
}

#undef IMPL_CONV
#undef MGB_FOREACH_FASTRUN_OPR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
