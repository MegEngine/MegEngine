/**
 * \file src/opr/impl/search_policy/profile.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/search_policy/profiler.h"

#include "../internal/invoke.h"
#include "../internal/megdnn_opr_wrapper.inl"
#include "megdnn/handle.h"
#include "megdnn/oprs/base.h"

#if MGB_ROCM
#include "hcc_detail/hcc_defs_prologue.h"
#include "megcore_rocm.h"
#endif

//! TODO: here has to be know some megdnn::opr when there is produced midout.h
//! fix it if there is another graceful way.
#include "megdnn/oprs.h"

#include "midout.h"

MIDOUT_DECL(megbrain_opr_profile)
#define MIDOUT_B(...) MIDOUT_BEGIN(megbrain_opr_profile, __VA_ARGS__) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

namespace {
std::string serialize_policy(const megdnn::ExecutionPolicy& policy) {
    std::string ret;
    //! serialize AlgorithmDesc
    megdnn::Algorithm::serialize_write_pod(policy.algo.handle_type, ret);
    megdnn::Algorithm::serialize_write_pod(policy.algo.type, ret);
    uint32_t param_size = policy.algo.param.size();
    uint32_t name_size = policy.algo.name.size();
    megdnn::Algorithm::serialize_write_pod<uint32_t>(param_size, ret);
    megdnn::Algorithm::serialize_write_pod<uint32_t>(name_size, ret);
    ret += policy.algo.param;
    ret += policy.algo.name;

    //! serialize sub_policy
    uint32_t size = policy.sub_policy.size();
    megdnn::Algorithm::serialize_write_pod(size, ret);
    for (auto&& sub : policy.sub_policy) {
        ret += serialize_policy(sub);
    }
    return ret;
}

megdnn::ExecutionPolicy deserialize_policy(const char* buf, uint32_t size,
                                           uint32_t& offset) {
    megdnn::ExecutionPolicy ret;
#define cb(_val, _type)                                                 \
    _val = megdnn::Algorithm::deserialize_read_pod<_type>(buf, offset); \
    offset += sizeof(_val)

    cb(ret.algo.handle_type, megdnn::Handle::HandleType);
    cb(ret.algo.type, uint32_t);

    uint32_t param_size = 0;
    uint32_t name_size = 0;
    cb(param_size, uint32_t);
    cb(name_size, uint32_t);
    if (param_size > 0) {
        ret.algo.param = std::string(buf + offset, param_size);
        offset += param_size;
    }
    if (name_size > 0) {
        ret.algo.name = std::string(buf + offset, name_size);
        offset += name_size;
    }

    uint32_t nr_policy = 0;
    cb(nr_policy, uint32_t);
#undef cb

    for (uint32_t i = 0; i < nr_policy; i++) {
        ret.sub_policy.push_back(deserialize_policy(buf, size, offset));
    }
    return ret;
}
}

namespace mgb {
namespace opr {
#define APPLY(statement, ...)                                  \
    mgb::apply([&](const auto&... args) { return statement; }, \
               std::tuple_cat(__VA_ARGS__))

////////////// TimedProfiler::Param::ExecutionPolicyBlob //////////////////////

template <typename Opr>
typename TimedProfiler<Opr>::Param::ExecutionPolicyBlob
TimedProfiler<Opr>::Param::ExecutionPolicyBlob::serialize(
        const megdnn::ExecutionPolicy& policy) {
    ExecutionPolicyBlob ret;
    std::string serialize_bin = serialize_policy(policy);
    mgb_assert(serialize_bin.size() < MAX_SIZE_IN_BYTES);
    memcpy(ret.data, serialize_bin.data(), serialize_bin.size());
    ret.size = serialize_bin.size();
    return ret;
}

template <typename Opr>
megdnn::ExecutionPolicy
TimedProfiler<Opr>::Param::ExecutionPolicyBlob::deserialize() const {
    uint32_t offset = 0;
    auto&& ret = deserialize_policy(data, size, offset);
    mgb_assert(offset == size);
    return std::move(ret);
}

#define INST(Opr)                                                            \
    template typename TimedProfiler<megdnn::Opr>::Param::ExecutionPolicyBlob \
    TimedProfiler<megdnn::Opr>::Param::ExecutionPolicyBlob::serialize(       \
            const megdnn::ExecutionPolicy& policy);                          \
    template megdnn::ExecutionPolicy                                         \
    TimedProfiler<megdnn::Opr>::Param::ExecutionPolicyBlob::deserialize()    \
            const;

MGB_FOREACH_FASTRUN_OPR(INST)
#undef INST


////////////////// TimedProfiler //////////////////////////////

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

#define APPLY(statement, ...)                                  \
    mgb::apply([&](const auto&... args) { return statement; }, \
               std::tuple_cat(__VA_ARGS__))

template <typename Opr>
typename TimedProfiler<Opr>::TResult TimedProfiler<Opr>::prof_impl(
        const TParam& raw_param) {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("TimedProfiler::prof_impl")))
#if MGB_ROCM
    bool miopen_algo_search_enabled;
    megcore::getMIOpenAlgoSearchStatus(&miopen_algo_search_enabled);
    mgb_assert(miopen_algo_search_enabled, "MIOpen algo search not enabled");
#endif
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
            cb(dtype::QuantizedS16);
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
    megdnn_opr->execution_policy() = param.execution_policy.deserialize();

    // Allocate preprocessed weight buffers.
    TensorLayoutArray preprocessed_layout;
    if_constexpr<opr_supports_preprocess<Opr>()>([&](auto _) {
        if (param.allow_weight_preprocess) {
            preprocessed_layout = APPLY(
                    _(megdnn_opr)->deduce_preprocessed_filter_layout(args...),
                    layouts);
        }
    });

    {
        // first allocate a whole chunk to avoid memory fragmentation (here we
        // rely on memory allocator to reuse memory)
        auto align = cn.get_mem_addr_alignment();
        size_t tot_size = align;
        for (int i = 0; i < arity; ++i) {
            tot_size += layouts[i].span().high_byte + align;
        }
        for (const auto& layout : preprocessed_layout) {
            tot_size += layout.span().high_byte + align;
        }
        tot_size += param.workspace;
        DeviceTensorStorage storage{cn};
        storage.ensure_size(tot_size);
    }

    // allocate input and output memory
    std::array<DeviceTensorND, arity_in> inp_val;
    std::array<DeviceTensorND, arity_out> out_val;
    DeviceTensorND workspace;
    for (int i = 0; i < arity_in; ++i) {
        inp_val[i].comp_node(cn).dtype(layouts[i].dtype).resize(layouts[i]);
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

    // allocate storage for preprocessed filter
    SmallVector<DeviceTensorND> flt_val(preprocessed_layout.size());
    for (size_t i = 0; i < preprocessed_layout.size(); i++) {
        flt_val[i] = {cn, preprocessed_layout[i], preprocessed_layout[i].dtype,
                      preprocessed_layout[i].format};
    }

    for (int i = 0; i < arity_in; ++i) {
        fill_zero_dev_tensor(inp_val[i]);
    }

    PreprocessFilter<Opr> prep_flt;
    if_constexpr<opr_supports_preprocess<Opr>()>([&](auto _) {
        if (!preprocessed_layout.empty()) {
            auto&& pf = _(prep_flt);
            pf.algorithm_id = nullptr;
            pf.tensors.resize(flt_val.size());
            for (size_t i = 0; i < flt_val.size(); i++) {
                pf.tensors[i] = flt_val[i].as_megdnn();
            }
            if_constexpr<opr_contain_bias<Opr>()>(
                    //! convbias
                    [&](auto __) {
                        APPLY(__(megdnn_opr)
                                      ->exec_preprocess(args..., &pf,
                                                        mdn_workspace),
                              std::forward_as_tuple(layouts[0],
                                                    inp_val[1].as_megdnn(),
                                                    inp_val[2].as_megdnn()),
                              array_skip<arity_in - 1>(layouts));
                    },
                    //! Convolution
                    [&](auto __) {
                        APPLY(__(megdnn_opr)
                                      ->exec_preprocess(args..., &pf,
                                                        mdn_workspace),
                              std::forward_as_tuple(layouts[0],
                                                    inp_val[1].as_megdnn()),
                              array_skip<2>(layouts));
                    });
        }
    });

    RealTimer timer;
    auto ev_start = cn.create_event(CompNode::Event::NEED_TIMER),
         ev_end = cn.create_event(CompNode::Event::NEED_TIMER);
    ev_start->record();
    if_constexpr<opr_supports_preprocess<Opr>()>(
            [&](auto _) {
                auto&& opr = _(megdnn_opr);
                PreprocessFilter<Opr>* pf =
                        preprocessed_layout.empty() ? nullptr : &prep_flt;
                APPLY(opr->exec(args.as_megdnn()..., pf, mdn_workspace),
                      inp_val, out_val);
            },
            /* else */
            [&](auto _) {
                APPLY(_(megdnn_opr)->exec(args.as_megdnn()..., mdn_workspace),
                      inp_val, out_val);
            });
    ev_end->record();

    megdnn::Algorithm* algo = megdnn_opr->get_algorithm_from_desc(
            megdnn_opr->execution_policy().algo);
    mgb_assert(algo);
    double next_report_time = 0.5;
    while (!ev_end->finished()) {
        if (timer.get_secs() >= next_report_time) {
            mgb_log_warn(
                    "profiling conv algo %s already took %.3f/%.3f secs"
                    " (limit can be set by MGB_CONV_PROFILING_TIMEOUT) ",
                    algo->name(), timer.get_secs(), param.actual_timeout);
            next_report_time = timer.get_secs() + 1;
        }
        using namespace std::literals;
        std::this_thread::sleep_for(1000us);
    }
    // release all free blocks owned by child process,
    // in order to avoid main process running out of memory
    cn.try_coalesce_all_free_memory();

    mgb_assert(ev_start->finished());
    return TResult::from_pod(Result{ev_start->elapsed_time_until(*ev_end)});
    MIDOUT_E
};

template <typename Opr>
Maybe<typename TimedProfiler<Opr>::Result> TimedProfiler<Opr>::profile(
        const Param& param, double& timeout) {
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

template <typename Opr>
void TimedProfiler<Opr>::prof_init_device(const TParam& raw_param) {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("TimedProfiler::prof_init_device")))
#if MGB_ROCM
    megcore::enableMIOpenAlgoSearch(true);
#endif
    auto&& param = raw_param.as_single_pod<Param>();
    CompNode cn = CompNode::load(param.comp_node_loc, param.comp_node_loc);
    // wait for cuda init, so its time does not get accounted in timeout
    cn.sync();
    MIDOUT_E
}

#define INST(Opr)                                                             \
    template const double TimedProfiler<megdnn::Opr>::timeout_setting;        \
    template double TimedProfiler<megdnn::Opr>::init_timeout_setting();       \
    template typename TimedProfiler<megdnn::Opr>::TResult                     \
    TimedProfiler<megdnn::Opr>::prof_impl(const TParam& raw_param);           \
    template Maybe<typename TimedProfiler<megdnn::Opr>::Result>               \
    TimedProfiler<megdnn::Opr>::profile(const Param& param, double& timeout); \
    template void TimedProfiler<megdnn::Opr>::prof_init_device(               \
            const TParam& raw_param);

MGB_FOREACH_FASTRUN_OPR(INST)
#undef INST
}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
