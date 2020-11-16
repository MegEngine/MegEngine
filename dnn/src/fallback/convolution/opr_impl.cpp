/**
 * \file dnn/src/fallback/convolution/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/algo_chooser.h"
#include "src/common/metahelper.h"
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"
#include "src/fallback/convolution/algos.h"
#include "src/fallback/convolution/opr_impl.h"
#include "src/fallback/convolution/run_conv.h"
#include "src/naive/convolution/helper.h"
#include "src/naive/handle.h"

#include "midout.h"

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
#include "src/arm_common/convolution/opr_impl.h"
#endif

#include <cstring>
#include <unordered_map>

MIDOUT_DECL(megdnn_fb_convbwd_float)

using namespace megdnn;
using namespace fallback;

namespace {
template <typename T>
void incr_ptr(T*& dst, ptrdiff_t delta) {
    dst = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(dst) + delta);
}

}  // namespace

class ConvolutionImpl::AlgoPack : NonCopyableObj {
    AlgoFallback algo_fallback;
    AlgoNaive algo_naive;
    SmallVector<std::unique_ptr<AlgoBase>> refhold;
    SmallVector<AlgoBase*> m_all_algos;
    AlgoBase::Mapper m_all_algos_map;
public:
    AlgoPack() {
        static CpuOprDelegationStorage<1> storage;
        auto conv_bias_opr = storage.get<ConvBias, 0>();
        auto&& conv_bias_algo =
                static_cast<ConvBiasImpl*>(conv_bias_opr)->get_all_packed_algo();
        for (auto&& algorithm : conv_bias_algo) {
            // fallback algo
            refhold.emplace_back(new AlgoDefault(algorithm));
            m_all_algos.emplace_back(refhold.back().get());
        }

        m_all_algos.emplace_back(&algo_fallback);
        m_all_algos.emplace_back(&algo_naive);

        for (auto&& algo : m_all_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
    }

    const SmallVector<AlgoBase*>& all_algos() const { return m_all_algos; }
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

const ConvolutionImpl::AlgoPack& ConvolutionImpl::algo_pack() {
    static AlgoPack algo_pack;
    return algo_pack;
}

SmallVector<ConvolutionImpl::AlgoBase*> ConvolutionImpl::get_all_packed_algo() {
    return algo_pack().all_algos();
}

SmallVector<ConvolutionImpl::AlgoBase*> ConvolutionImpl::select_algo_type(
        ConvAlgoTypePack target_type) {
    megdnn_assert(nr_type_contain(target_type.data_type),
                  "ConvBias algo selection only support one type");
    SmallVector<ConvolutionImpl::AlgoBase*> algos;
    for (auto&& algo : get_all_packed_algo()) {
        auto algo_type = algo->get_algo_type();
        if (contain_data_type(algo_type.data_type, target_type.data_type) &&
            algo_type.algo_category == target_type.algo_category) {
            algos.push_back(algo);
        }
    }
    return algos;
}

bool ConvolutionImpl::is_naive_algo(ConvolutionImpl::Algorithm* algo) {
    return algo == nullptr || strcmp(algo->name(), "DEFAULT") == 0;
}

#define NCB_ALGO_FUNC(name, algo, param) \
    static_cast<AlgoBase*>(algo)->name(param)

void ConvolutionImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                           _megdnn_tensor_out dst,
                           const PreprocessedFilter* preprocessed_filter,
                           _megdnn_workspace workspace) {
    auto fparam = make_ncb_kern_param(src, filter, dst, preprocessed_filter,
                                      workspace);
    auto&& algo = get_algorithm(fparam, workspace.size);
    if (!is_naive_algo(algo) &&
        NCB_ALGO_FUNC(get_workspace, algo, fparam) <= workspace.size) {
        exec_with_ncb_kern(fparam, algo);
    } else {
        naive::ConvolutionForwardImpl::exec(src, filter, dst,
                                            preprocessed_filter, workspace);
    }
}

void ConvolutionImpl::exec_preprocess(const TensorLayout& src_layout,
                                      _megdnn_tensor_in filter,
                                      const TensorLayout& dst_layout,
                                      PreprocessedFilter* preprocessed_filter,
                                      _megdnn_workspace workspace) {
    //! exec_preprocess currently only support preprocess weights before exec,
    //! src/dst will be ignored, just set to nullptr
    TensorND src{nullptr, src_layout}, dst{nullptr, dst_layout};
    auto fparam = make_ncb_kern_param(src, filter, dst, preprocessed_filter,
                                      workspace);

    //! should not pass workspace_size limit otherwise can not find match algo
    auto&& algo = get_algorithm(fparam);
    if (!is_naive_algo(algo) &&
        NCB_ALGO_FUNC(get_preprocess_workspace, algo, fparam) <=
                workspace.size) {
        exec_preprocess_with_ncb_kern(fparam, algo);
    } else {
        naive::ConvolutionForwardImpl::exec_preprocess(
                src_layout, filter, dst_layout, preprocessed_filter, workspace);
    }
}

size_t ConvolutionImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst,
        const PreprocessedFilter* preprocessed_filter) {
    auto fparam =
            make_ncb_kern_size_param(src, filter, dst, preprocessed_filter);
    auto&& algo = get_algorithm(fparam);
    if (is_naive_algo(algo)) {
        return naive::ConvolutionForwardImpl::get_workspace_in_bytes(
                src, filter, dst, preprocessed_filter);
    } else {
        return NCB_ALGO_FUNC(get_workspace, algo, fparam);
    }
}

size_t ConvolutionImpl::get_preprocess_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst) {
    auto fparam = make_ncb_kern_size_param(src, filter, dst, nullptr);
    auto&& algo = get_algorithm(fparam);
    if (is_naive_algo(algo)) {
        return naive::ConvolutionForwardImpl::get_preprocess_workspace_in_bytes(
                src, filter, dst);
    } else {
        return NCB_ALGO_FUNC(get_preprocess_workspace, algo, fparam);
    }
}

SmallVector<TensorLayout> ConvolutionImpl::deduce_preprocessed_filter_layout(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst) {
    auto fparam = make_ncb_kern_size_param(src, filter, dst, nullptr);
    auto&& algo = get_algorithm(fparam);
    if (is_naive_algo(algo)) {
        return naive::ConvolutionForwardImpl::deduce_preprocessed_filter_layout(
                src, filter, dst);
    } else {
        return NCB_ALGO_FUNC(deduce_preprocessed_filter_layout, algo, fparam);
    }
}

std::vector<ConvolutionImpl::Algorithm*> ConvolutionImpl::get_all_algorithms(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst) {
    auto fparam = make_ncb_kern_size_param(src, filter, dst, nullptr);
    auto ret = get_all_algorithms_with_ncb(fparam);
    if (ret.empty()) {
        return naive::ConvolutionForwardImpl::get_all_algorithms(src, filter,
                                                                 dst);
    }
    return ret;
}

ConvolutionImpl::Algorithm* ConvolutionImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst, size_t workspace_limit_in_bytes,
        bool reproducible) {
    auto fparam = make_ncb_kern_size_param(src, filter, dst, nullptr);
    auto result = get_algorithm_heuristic_with_ncb(
            fparam, workspace_limit_in_bytes, reproducible);
    if (result == nullptr) {
        result = naive::ConvolutionForwardImpl::get_algorithm_heuristic(
                src, filter, dst, workspace_limit_in_bytes, reproducible);
    }
    return result;
}

ConvolutionImpl::NCBKernSizeParam ConvolutionImpl::make_ncb_kern_size_param(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst,
        const PreprocessedFilter* preprocessed_filter) {
    auto safe_u32 = [](size_t v) -> uint32_t {
        megdnn_assert(v <= std::numeric_limits<uint32_t>::max(),
                      "value too large: %zu", v);
        return v;
    };
    size_t spatial_pos;
    if (param().format == Param::Format::NCHW88 ||
        param().format == Param::Format::NCHW8 ||
        param().format == Param::Format::NCHW4 ||
        param().format == Param::Format::NCHW44_DOT ||
        param().format == Param::Format::NCHW44) {
        spatial_pos = 2;
    } else if (param().format == Param::Format::NCHW ||
               param().format == Param::Format::NCHW_WINOGRAD) {
        spatial_pos = 2;
    } else if (param().format == Param::Format::NHWC) {
        spatial_pos = 1;
    } else {
        megdnn_assert(0, "invalid conv format %d",
                      static_cast<int>(param().format));
    }
    size_t nr_threads = static_cast<naive::HandleImpl*>(handle())
                                ->megcore_dispatcher()
                                ->nr_threads();

    return {safe_u32(src[0]),
            {{safe_u32(src[spatial_pos]), safe_u32(src[spatial_pos + 1])}},
            {{safe_u32(dst[spatial_pos]), safe_u32(dst[spatial_pos + 1])}},
            check_layout_fwd(src, filter, dst),
            src.dtype,
            filter.dtype,
            dst.dtype,
            src.stride[0],
            dst.stride[0],
            {src.stride[0], src.stride[1], src.stride[2], src.stride[3]},
            {dst.stride[0], dst.stride[1], dst.stride[2], dst.stride[3]},
            param().compute_mode,
            nr_threads,
            preprocessed_filter};
}

ConvolutionImpl::NCBKernParam ConvolutionImpl::make_ncb_kern_param(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
        const PreprocessedFilter* preprocessed_filter,
        _megdnn_workspace workspace) {
    NCBKernParam ret;
    static_cast<NCBKernSizeParam&>(ret) = make_ncb_kern_size_param(
            src.layout, filter.layout, dst.layout, preprocessed_filter);
    ret.src_ptr = src.raw_ptr;
    ret.filter_ptr = filter.raw_ptr;
    ret.dst_ptr = dst.raw_ptr;
    ret.workspace_ptr = workspace.raw_ptr;
    ret.workspace_size = workspace.size;
    return ret;
}

void ConvolutionImpl::exec_preprocess_with_ncb_kern(const NCBKernParam& param,
                                                    Algorithm* algo) {
    auto&& kerns = NCB_ALGO_FUNC(dispatch_preprocess_kern, algo, param);
    auto&& fallback_handle = handle();
    for (auto&& kernel : kerns) {
        megdnn_assert(
                param.filter_meta.format == Param::Format::NCHW ||
                        param.filter_meta.format == Param::Format::NHWC ||
                        param.filter_meta.format == Param::Format::NCHW88 ||
                        param.filter_meta.format == Param::Format::NCHW44 ||
                        param.filter_meta.format == Param::Format::NCHW44_DOT,
                "invalid conv format");
        auto run = [param, kernel](size_t index, size_t thread_id) {
            CpuNDRange ndrange_id(kernel.global_size, index);
            kernel.kern(param, {thread_id, ndrange_id});
        };
        static_cast<naive::HandleImpl*>(fallback_handle)
                ->dispatch_kern(run, kernel.global_size.total_size());
    }
}

void ConvolutionImpl::exec_with_ncb_kern(const NCBKernParam& param,
                                         Algorithm* algo) {
    auto&& kerns = NCB_ALGO_FUNC(dispatch_kern, algo, param);
    auto&& fallback_handle = handle();
    for (auto&& kernel : kerns) {
        megdnn_assert(
                param.filter_meta.format == Param::Format::NCHW ||
                        param.filter_meta.format == Param::Format::NHWC ||
                        param.filter_meta.format == Param::Format::NCHW88 ||
                        param.filter_meta.format == Param::Format::NCHW44 ||
                        param.filter_meta.format == Param::Format::NCHW44_DOT,
                "invalid conv format");
        auto run = [param, kernel](size_t index, size_t thread_id) {
            CpuNDRange ndrange_id(kernel.global_size, index);
            kernel.kern(param, {thread_id, ndrange_id});
        };
        static_cast<naive::HandleImpl*>(fallback_handle)
                ->dispatch_kern(run, kernel.global_size.total_size());
    }
}

ConvolutionImpl::Algorithm* ConvolutionImpl::get_algorithm_heuristic_with_ncb(
        const NCBKernSizeParam& param, size_t workspace_limit_in_bytes,
        bool reproducible) {
    auto algo_data_type = param.deduce_algo_data_type();
    auto suggest_category_order = suggest_algo_category_order(param);
    for (auto category : suggest_category_order) {
        auto&& origin_algos = select_algo_type({algo_data_type, category});
        ConvolutionImpl::Algorithm* heuristic_algo = nullptr;
        for (auto i : origin_algos) {
            bool usable_reproducible =
                    static_cast<AlgoBase*>(i)->usable_reproducible(
                            param, AlgoSelectionStrategy::HEURISTIC,
                            reproducible);
            if (usable_reproducible &&
                static_cast<AlgoBase*>(i)->get_workspace(param) <=
                        workspace_limit_in_bytes) {
                //! store the first usable algo if no prefer algo, choose it as
                //! the target algo
                if (!heuristic_algo) {
                    heuristic_algo = i;
                }
                //! choose the first prefer algo
                if (i->is_preferred(param)) {
                    return i;
                }
            }
        }
        if (heuristic_algo) {
            return heuristic_algo;
        }
    }
    return nullptr;
}

std::vector<ConvolutionImpl::Algorithm*>
ConvolutionImpl::get_all_algorithms_with_ncb(const NCBKernSizeParam& param) {
    std::vector<Algorithm*> ret;
    std::vector<Algorithm*> prefer_algos;
    for (auto&& i : get_all_packed_algo()) {
        if (i->usable(param, AlgoSelectionStrategy::FULL_RUN)) {
            if (i->is_preferred(param)) {
                prefer_algos.push_back(i);
            } else {
                ret.push_back(i);
            }
        }
    }
    ret.insert(ret.begin(), prefer_algos.begin(), prefer_algos.end());
    return ret;
}

ConvolutionImpl::Algorithm* ConvolutionImpl::get_algo_from_desc(
        const AlgorithmDesc& desc) const {
    if (!desc.valid()) {
        return nullptr;
    } else {
        switch (desc.handle_type) {
            case Handle::HandleType::FALLBACK: {
                const auto& map = algo_pack().all_algos_map();
                megdnn_assert(map.find(desc) != map.end());
                return map.at(desc);
            }
            case Handle::HandleType::NAIVE: {
                auto algo = static_cast<naive::HandleImpl*>(handle())
                                    ->default_conv_fwd_algo();
                megdnn_assert(algo->info().desc == desc);
                return algo;
            }
            default:
                megdnn_throw("Unknown handle type");
                return nullptr;
        }
    }
}

ConvolutionImpl::Algorithm* ConvolutionImpl::get_algorithm(
        const NCBKernSizeParam& param, size_t workspace_size) {
    if (auto algo = get_algo_from_desc(execution_policy().algo.desc)) {
        return algo;
    }
    if (!m_prev_selected_algo ||
        memcmp(&m_prev_selected_algo_sizep, &param, sizeof(NCBKernSizeParam))) {
        m_prev_selected_algo =
                get_algorithm_heuristic_with_ncb(param, workspace_size);
        m_prev_selected_algo_sizep = param;
    }
    return m_prev_selected_algo;
}

SmallVector<AlgoCategory> ConvolutionImpl::suggest_algo_category_order(
        const NCBKernSizeParam& param) const {
    static CpuOprDelegationStorage<1> storage;
    auto conv_bias_opr = storage.get<ConvBias, 0>();
    auto conv_bias_param =
            ConvolutionImpl::AlgoDefault::init_conv_bias_param(param);
    return static_cast<ConvBiasImpl*>(conv_bias_opr)
            ->suggest_algo_category_order(conv_bias_param);
}

const char* ConvolutionImpl::get_algorithm_set_name() const {
    // fallback version 0
    return "F0";
}

ConvolutionImpl::AlgoDataType
ConvolutionImpl::NCBKernSizeParam::deduce_algo_data_type() const {
    if (src_type.enumv() == DTypeEnum::Float32) {
        return ConvolutionImpl::AlgoDataType::FLOAT32;
#if !MEGDNN_DISABLE_FLOAT16
    } else if (src_type.enumv() == DTypeEnum::Float16) {
        return ConvolutionImpl::AlgoDataType::FLOAT16;
#endif
    } else if (src_type.enumv() == DTypeEnum::Int8 ||
               src_type.enumv() == DTypeEnum::QuantizedS8) {
        if (dst_type.enumv() == DTypeEnum::Int16) {
            return ConvolutionImpl::AlgoDataType::INT8X8X16;
        } else {
            return ConvolutionImpl::AlgoDataType::QINT8X8X32;
        }
    } else if (src_type.enumv() == DTypeEnum::Quantized8Asymm) {
        return ConvolutionImpl::AlgoDataType::QUINT8X8X32;
    } else {
        megdnn_throw(ssprintf("megdnn not support data type of %s * %s -> %s\n",
                              src_type.name(), filter_type.name(),
                              dst_type.name()));
    }
}

/* ===================== ConvolutionBackwardData ===================== */

class ConvolutionBackwardDataImpl::AlgoPack : NonCopyableObj {
    AlgoNaive algo_naive;
    AlgoDirect algo_direct;
    AlgoMatrixMul algo_matmul;
    SmallVector<AlgoBase*> m_all_algos;
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack() {
        m_all_algos.emplace_back(&algo_matmul);
        m_all_algos.emplace_back(&algo_direct);
        m_all_algos.emplace_back(&algo_naive);

        for (auto&& algo : m_all_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
    }
    const SmallVector<AlgoBase*>& all_algos() const { return m_all_algos; }
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};
const ConvolutionBackwardDataImpl::AlgoPack&
ConvolutionBackwardDataImpl::algo_pack() {
    static AlgoPack algo_pack;
    return algo_pack;
}

SmallVector<ConvolutionBackwardDataImpl::AlgoBase*>
ConvolutionBackwardDataImpl::get_all_packed_algo() {
    return algo_pack().all_algos();
}

void ConvolutionBackwardDataImpl::exec(_megdnn_tensor_in filter,
                                       _megdnn_tensor_in diff,
                                       _megdnn_tensor_out grad,
                                       _megdnn_workspace workspace) {
    if (param().format == param::Convolution::Format::NHWCD4 ||
        param().format == param::Convolution::Format::NCHW4) {
        return naive::ConvolutionBackwardDataImpl::exec(filter, diff, grad,
                                                        workspace);
    }
    auto fparam = make_ncb_kern_param(filter, diff, grad, workspace);
    return exec_with_ncb_kern(fparam);
}

size_t ConvolutionBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad) {
    if (param().format == param::Convolution::Format::NHWCD4 ||
        param().format == param::Convolution::Format::NCHW4) {
        return naive::ConvolutionBackwardDataImpl::get_workspace_in_bytes(
                filter, diff, grad);
    }
    auto fparam = make_ncb_kern_size_param(filter, diff, grad);
    return get_workspace_with_ncb(fparam);
}

std::vector<ConvolutionBackwardDataImpl::Algorithm*>
ConvolutionBackwardDataImpl::get_all_algorithms(const TensorLayout& filter,
                                                const TensorLayout& diff,
                                                const TensorLayout& grad) {
    if (param().format == param::Convolution::Format::NHWCD4 ||
        param().format == param::Convolution::Format::NCHW4) {
        return naive::ConvolutionBackwardDataImpl::get_all_algorithms(
                filter, diff, grad);
    }
    auto fparam = make_ncb_kern_size_param(filter, diff, grad);
    auto ret = get_all_algorithms_with_ncb(fparam);
    megdnn_assert(!ret.empty(), "no usable conv fwd algorithm");
    return ret;
}

ConvolutionBackwardDataImpl::Algorithm*
ConvolutionBackwardDataImpl::get_algorithm_heuristic(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_limit_in_bytes,
        bool reproducible) {
    if (param().format == param::Convolution::Format::NHWCD4 ||
        param().format == param::Convolution::Format::NCHW4) {
        return naive::ConvolutionBackwardDataImpl::get_algorithm_heuristic(
                filter, diff, grad, workspace_limit_in_bytes, reproducible);
    }
    auto fparam = make_ncb_kern_size_param(filter, diff, grad);
    return get_algorithm_heuristic_with_ncb(fparam, workspace_limit_in_bytes,
                                            reproducible);
}

ConvolutionBackwardDataImpl::NCBKernSizeParam
ConvolutionBackwardDataImpl::make_ncb_kern_size_param(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad) {
    auto safe_u32 = [](size_t v) -> uint32_t {
        megdnn_assert(v <= std::numeric_limits<uint32_t>::max(),
                      "value too large: %zu", v);
        return v;
    };
    size_t spatial_pos;
    if (param().format == Param::Format::NCHW) {
        spatial_pos = 2;
    } else {
        megdnn_assert(param().format == Param::Format::NHWC,
                      "invalid conv format");
        spatial_pos = 1;
    }
    auto grad_fwd = grad;
    auto filter_fwd = filter;
    auto diff_fwd = diff;

    std::swap(grad_fwd.dtype, diff_fwd.dtype);

    return {
            safe_u32(diff[0]),
            {{safe_u32(diff[spatial_pos]), safe_u32(diff[spatial_pos + 1])}},
            {{safe_u32(grad[spatial_pos]), safe_u32(grad[spatial_pos + 1])}},
            check_layout_fwd(grad_fwd, filter_fwd, diff_fwd),
            diff.dtype,
            filter.dtype,
            grad.dtype,
            diff,
            filter,
            grad,
            diff.stride[0],
            grad.stride[0],
            0,
            0,
            0,
            param().compute_mode,
    };
}

ConvolutionBackwardDataImpl::NCBKernParam
ConvolutionBackwardDataImpl::make_ncb_kern_param(_megdnn_tensor_in filter,
                                                 _megdnn_tensor_in diff,
                                                 _megdnn_tensor_out grad,
                                                 _megdnn_workspace workspace) {
    NCBKernParam ret;
    static_cast<NCBKernSizeParam&>(ret) =
            make_ncb_kern_size_param(filter.layout, diff.layout, grad.layout);

    auto required_workspace_in_bytes = get_workspace_with_ncb(ret);
    megdnn_assert(workspace.size >= required_workspace_in_bytes,
                  "required workspace: %zu; provided workspace: %zu",
                  required_workspace_in_bytes, workspace.size);
    ret.filter_ptr = filter.raw_ptr;
    ret.diff_ptr = diff.raw_ptr;
    ret.grad_ptr = grad.raw_ptr;
    ret.workspace_ptr = workspace.raw_ptr;
    ret.workspace_size = workspace.size;
    return ret;
}

void ConvolutionBackwardDataImpl::exec_with_ncb_kern(
        const NCBKernParam& param) {
    auto p1g = param;
    auto group = p1g.filter_meta.group;
    p1g.filter_meta.group = 1;
    auto&& algo = get_algorithm(p1g);
    auto kptr = ncb_1g_dispatch_kern(algo, p1g);
    if (group == 1 || static_cast<AlgoBase*>(algo)->is_naive()) {
        auto run = [kptr, param]() { kptr(param); };
        static_cast<naive::HandleImpl*>(handle())->dispatch_kern(run);
    } else {
        megdnn_assert(p1g.filter_meta.format == Param::Format::NCHW ||
                              p1g.filter_meta.format == Param::Format::NHWC,
                      "invalid conv format");
        auto run = [kptr, p1g_orig = p1g, group]() {
            auto p1g = p1g_orig;
            ptrdiff_t istrd, fstrd, ostrd;
            fstrd = p1g.filter_meta.icpg * p1g.filter_meta.ocpg *
                    p1g.filter_meta.spatial[0] * p1g.filter_meta.spatial[1] *
                    p1g.filter_type.size();
            istrd = p1g.filter_meta.ocpg * p1g.diff_type.size();
            ostrd = p1g.filter_meta.icpg * p1g.grad_type.size();
            p1g.diff_extra_mem_size =
                    (group - 1) * p1g.filter_meta.ocpg * p1g.diff_type.size();
            p1g.filter_extra_mem_size =
                    (group - 1) * p1g.filter_meta.icpg * p1g.filter_meta.ocpg *
                    p1g.filter_meta.spatial[0] * p1g.filter_meta.spatial[1] *
                    p1g.filter_type.size();
            p1g.grad_extra_mem_size =
                    (group - 1) * p1g.filter_meta.icpg * p1g.grad_type.size();
            if (p1g.filter_meta.format == Param::Format::NCHW) {
                istrd *= p1g.isz[0] * p1g.isz[1];
                ostrd *= p1g.osz[0] * p1g.osz[1];
                p1g.diff_extra_mem_size *= p1g.isz[0] * p1g.isz[1];
                p1g.grad_extra_mem_size *= p1g.osz[0] * p1g.osz[1];
            } else {
                // must be NHWC. No action performed.
            }
            for (size_t i = 0; i < group; ++i) {
                kptr(p1g);
                incr_ptr(p1g.diff_ptr, istrd);
                incr_ptr(p1g.filter_ptr, fstrd);
                incr_ptr(p1g.grad_ptr, ostrd);
                p1g.diff_extra_mem_size -= istrd;
                p1g.filter_extra_mem_size -= fstrd;
                p1g.grad_extra_mem_size -= ostrd;
            }
        };
        static_cast<naive::HandleImpl*>(handle())->dispatch_kern(run);
    }
}

size_t ConvolutionBackwardDataImpl::get_workspace_with_ncb(
        const NCBKernSizeParam& param) {
    if (param.filter_meta.group != 1) {
        auto p1g = param;
        p1g.filter_meta.group = 1;
        auto algo = get_algorithm(p1g);
        return ncb_1g_get_workspace(algo, p1g);
    }
    auto algo = get_algorithm(param);
    return ncb_1g_get_workspace(algo, param);
}

std::vector<ConvolutionBackwardDataImpl::Algorithm*>
ConvolutionBackwardDataImpl::get_all_algorithms_with_ncb(
        const NCBKernSizeParam& param) {
    if (param.filter_meta.group != 1) {
        auto p1g = param;
        p1g.filter_meta.group = 1;
        return ncb_1g_get_all_algorithms(p1g);
    }
    return ncb_1g_get_all_algorithms(param);
}

ConvolutionBackwardDataImpl::Algorithm*
ConvolutionBackwardDataImpl::get_algorithm_heuristic_with_ncb(
        const NCBKernSizeParam& param, size_t workspace_limit_in_bytes,
        bool reproducible) {
    if (param.filter_meta.group != 1) {
        auto p1g = param;
        p1g.filter_meta.group = 1;
        return ncb_1g_get_algorithm_heuristic(p1g, workspace_limit_in_bytes,
                                              reproducible);
    }
    return ncb_1g_get_algorithm_heuristic(param, workspace_limit_in_bytes,
                                          reproducible);
}

size_t ConvolutionBackwardDataImpl::ncb_1g_get_workspace(
        Algorithm* algo, const NCBKernSizeParam& param) {
    megdnn_assert(param.filter_meta.group == 1);
    if (algo->handle_type() == Handle::HandleType::FALLBACK) {
        return static_cast<AlgoBase*>(algo)->get_workspace(this, param);
    }
    return 0;
}

ConvolutionBackwardDataImpl::ncb_kern_t
ConvolutionBackwardDataImpl::ncb_1g_dispatch_kern(
        Algorithm* algo, const NCBKernSizeParam& param) {
    megdnn_assert(param.filter_meta.group == 1);

    if (algo->handle_type() == Handle::HandleType::FALLBACK) {
        return static_cast<AlgoBase*>(algo)->dispatch_kern(this, param);
    }

    megdnn_throw(
            megdnn_mangle("no suitable ConvolutionBackwardData algorithm"));
}

bool ConvolutionBackwardDataImpl::is_matrix_mul_preferred(
        const NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto OC = fm.ocpg, IC = fm.icpg;

    return (OC * IC >= 32) ||
           (fm.spatial[0] == 1 && fm.spatial[1] == 1 && fm.padding[0] == 0 &&
            fm.padding[1] == 0 && fm.stride[0] == 1 && fm.stride[1] == 1);
}

std::vector<ConvolutionBackwardDataImpl::Algorithm*>
ConvolutionBackwardDataImpl::ncb_1g_get_all_algorithms(
        const NCBKernSizeParam& param) {
    std::vector<Algorithm*> ret;
    std::vector<Algorithm*> prefer_algos;
    for (auto&& i : get_all_packed_algo()) {
        if (i->usable(this, param)) {
            if (i->is_preferred(param)) {
                prefer_algos.push_back(i);
            } else {
                ret.push_back(i);
            }
        }
    }
    ret.insert(ret.begin(), prefer_algos.begin(), prefer_algos.end());
    return ret;
}

ConvolutionBackwardDataImpl::Algorithm*
ConvolutionBackwardDataImpl::ncb_1g_get_algorithm_heuristic(
        const NCBKernSizeParam& param, size_t workspace_limit_in_bytes,
        bool reproducible) {
    for (auto i : ncb_1g_get_all_algorithms(param)) {
        if (ncb_1g_get_workspace(i, param) <= workspace_limit_in_bytes) {
            if (reproducible) {
                if (i->is_reproducible()) {
                    return i;
                }
            } else {
                return i;
            }
        }
    }
    megdnn_assert(0,
                  "no suitable algorithm found within given workspace limit");
}

ConvolutionBackwardDataImpl::Algorithm*
ConvolutionBackwardDataImpl::get_algo_from_desc(
        const AlgorithmDesc& desc) const {
    if (!desc.valid()) {
        return nullptr;
    } else {
        switch (desc.handle_type) {
            case Handle::HandleType::FALLBACK: {
                const auto& map = algo_pack().all_algos_map();
                megdnn_assert(map.find(desc) != map.end());
                return map.at(desc);
            }
#if MEGDNN_AARCH64 || MEGDNN_ARMV7
            case Handle::HandleType::ARM_COMMON:
            case Handle::HandleType::AARCH64:
            case Handle::HandleType::ARMV7:
                return arm_common::ConvolutionBackwardDataImpl::
                        get_algo_from_desc(desc);
#endif
            case Handle::HandleType::NAIVE: {
                auto algo = static_cast<naive::HandleImpl*>(handle())
                                    ->default_conv_bwd_data_algo();
                megdnn_assert(algo->info().desc == desc);
                return algo;
            }
            default:
                megdnn_throw("Unknown handle type");
                return nullptr;
        }
    }
}


ConvolutionBackwardDataImpl::Algorithm*
ConvolutionBackwardDataImpl::get_algorithm(const NCBKernSizeParam& param) {
    if (auto algo = get_algo_from_desc(execution_policy().algo.desc)) {
        return algo;
    }
    if (!m_prev_selected_algo ||
        memcmp(&m_prev_selected_algo_sizep, &param, sizeof(NCBKernSizeParam))) {
        m_prev_selected_algo = ncb_1g_get_algorithm_heuristic(
                param, std::numeric_limits<size_t>::max());
        m_prev_selected_algo_sizep = param;
    }
    return m_prev_selected_algo;
}

const char* ConvolutionBackwardDataImpl::get_algorithm_set_name() const {
    // fallback version 0
    return "FALLBACK_CONVOLUTION_BACKWARD_DATA_IMPL0";
}

// vim: syntax=cpp.doxygen
