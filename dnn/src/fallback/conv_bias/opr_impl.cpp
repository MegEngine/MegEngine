/**
 * \file dnn/src/fallback/conv_bias/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/convolution/opr_impl.h"
#include "src/common/algo_chooser.h"
#include "src/common/metahelper.h"
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/algos.h"
#include "src/fallback/conv_bias/im2col/algos.h"
#include "src/fallback/conv_bias/opr_impl.h"
#include "src/naive/convolution/algorithms.h"
#include "src/naive/handle.h"

#include <cstring>

using namespace megdnn;
using namespace fallback;

namespace {
template <typename T>
void incr_ptr(T*& dst, ptrdiff_t delta) {
    dst = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(dst) + delta);
}

}  // namespace

class ConvBiasImpl::AlgoPack : NonCopyableObj {
    AlgoNaive algo_naive;
    SmallVector<std::unique_ptr<AlgoBase>> refhold;

public:
    AlgoPack() {
        static CpuOprDelegationStorage<> storage;
        auto matmul_opr = storage.get<MatrixMul>();
        auto&& matmul_algos =
                static_cast<fallback::MatrixMulImpl*>(matmul_opr)->algo_pack();
        for (auto&& algo : matmul_algos) {
            if (algo->algoset() ==
                //! TODO: threre should filter MK matmul
                MatrixMulImpl::AlgoBase::AlgoSet::ALGO_TYPE_GEMV) {
                continue;
            }
            for (size_t ohw_tile_size : {192, 384, 96, 48, 24}) {
                refhold.emplace_back(new AlgoIm2col(
                        static_cast<MatrixMulImpl::AlgoBase*>(algo),
                        ohw_tile_size));
                all_algos.emplace_back(refhold.back().get());
            }
#if 0
        //! As these algos maybe very slow, it will make fastrun search slow, so
        //! we disable it, but for the test of strategyhelper, we just keep it.
        //! FIXME: I do not know a better way to do it.
            refhold.emplace_back(new AlgoWinogradF32(
                    static_cast<MatrixMulImpl::AlgoBase*>(algo)));
            all_algos.emplace_back(refhold.back().get());
            refhold.emplace_back(new AlgoWinogradF32_4x4(
                    static_cast<MatrixMulImpl::AlgoBase*>(algo)));
            all_algos.emplace_back(refhold.back().get());
            refhold.emplace_back(new AlgoWinogradQS8(
                    static_cast<MatrixMulImpl::AlgoBase*>(algo)));
            all_algos.emplace_back(refhold.back().get());
            refhold.emplace_back(new AlgoWinogradQS8_8x8(
                    static_cast<MatrixMulImpl::AlgoBase*>(algo)));
            all_algos.emplace_back(refhold.back().get());
#endif
        }
        //! reverse matmul algo, when the algo is_prefer can be selected first
        std::reverse(all_algos.begin(), all_algos.end());
        all_algos.emplace_back(&algo_naive);
    }
    SmallVector<AlgoBase*> all_algos;
};

SmallVector<ConvBiasImpl::AlgoBase*> ConvBiasImpl::algo_pack() {
    static AlgoPack sl_algo_pack;
    return sl_algo_pack.all_algos;
}
bool ConvBiasImpl::is_naive_algo(ConvBiasImpl::Algorithm* algo) {
    return algo == nullptr || strcmp(algo->name(), "DEFAULT") == 0;
}
void ConvBiasImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                        _megdnn_tensor_in bias, _megdnn_tensor_in z,
                        _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, filter.layout, bias.layout, z.layout, dst.layout,
               workspace.size);
    auto fparam = make_ncb_kern_param(src, filter, bias, dst, workspace);
    ConvBiasImpl::Algorithm* algo = get_algorithm(fparam, workspace.size);
    if (!is_naive_algo(algo) &&
        ncb_algo_get_workspace(algo, fparam) <= workspace.size) {
        exec_with_ncb_kern(fparam, algo);
    } else {
        naive::ConvBiasForwardImpl::exec(src, filter, bias, z, dst, workspace);
    }
}

size_t ConvBiasImpl::get_workspace_in_bytes(const TensorLayout& src,
                                            const TensorLayout& filter,
                                            const TensorLayout& bias,
                                            const TensorLayout& z,
                                            const TensorLayout& dst) {
    auto fparam = make_ncb_kern_size_param(src, filter, bias, dst);
    ConvBiasImpl::Algorithm* algo = get_algorithm(fparam);
    if (is_naive_algo(algo)) {
        return naive::ConvBiasForwardImpl::get_workspace_in_bytes(src, filter,
                                                                  bias, z, dst);
    } else {
        return ncb_algo_get_workspace(algo, fparam);
    }
}

std::vector<ConvBiasImpl::Algorithm*> ConvBiasImpl::get_all_algorithms(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst) {
    auto fparam = make_ncb_kern_size_param(src, filter, bias, dst);
    auto ret = get_all_algorithms_with_ncb(fparam);
    if (ret.empty()) {
        return naive::ConvBiasForwardImpl::get_all_algorithms(src, filter, bias,
                                                              z, dst);
    }
    return ret;
}

ConvBiasImpl::Algorithm* ConvBiasImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst, size_t workspace_limit_in_bytes,
        bool reproducible) {
    auto fparam = make_ncb_kern_size_param(src, filter, bias, dst);
    auto result = get_algorithm_heuristic_with_ncb(
            fparam, workspace_limit_in_bytes, reproducible);
    if (result == nullptr) {
        result = naive::ConvBiasForwardImpl::get_algorithm_heuristic(
                src, filter, bias, z, dst, workspace_limit_in_bytes,
                reproducible);
    }
    return result;
}

ConvBiasImpl::NCBKernSizeParam ConvBiasImpl::make_ncb_kern_size_param(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& dst) {
    auto safe_u32 = [](size_t v) -> uint32_t {
        megdnn_assert(v <= std::numeric_limits<uint32_t>::max(),
                      "value too large: %zu", v);
        return v;
    };
    size_t spatial_pos;
    if (param().format == Param::Format::NCHW88 ||
        param().format == Param::Format::NCHW8 ||
        param().format == Param::Format::NCHW4 ||
        param().format == Param::Format::NCHW ||
        param().format == Param::Format::NCHW_WINOGRAD ||
        param().format == Param::Format::NCHW88_WINOGRAD) {
        spatial_pos = 2;
    } else if (param().format == Param::Format::NHWC) {
        spatial_pos = 1;
    } else {
        megdnn_assert(0, "invalid conv format %d",
                      static_cast<int>(param().format));
    }
    BiasMode bias_mode;
    if (bias.ndim == 0) {
        bias_mode = BiasMode::NO_BIAS;
    } else if (bias.eq_shape(dst)) {
        bias_mode = BiasMode::BIAS;
    } else {
        //! just check the ndim, the detail shape check is in check_exec
        megdnn_assert(bias.ndim == dst.ndim);
        bias_mode = BiasMode::BROADCAST_CHANNEL_BIAS;
    }

    static_assert(sizeof(CanonizedFilterMeta) ==
                          sizeof(ConvolutionImpl::CanonizedFilterMeta),
                  "sizeof CanonizedFilterMeta in convolution and conv_bias "
                  "should be equal");
    CanonizedFilterMeta fm = check_layout_fwd(src, filter, dst);
    ConvolutionImpl::CanonizedFilterMeta conv_fm;
    conv_fm.copy_from(fm);

    param::MatrixMul::Format format = param::MatrixMul::Format::DEFAULT;
    if (param().format == Param::Format::NCHW_WINOGRAD ||
        param().format == Param::Format::NCHW88_WINOGRAD) {
        size_t flt_start = 0;
        if (param().sparse == Param::Sparse::GROUP) {
            flt_start = 1;
        }

        if (filter.ndim == 6 + flt_start) {
            if (filter[5] == 4) {
                format = param::MatrixMul::Format::MK4;
            } else {
                megdnn_assert(filter[5] == 8);
                format = param::MatrixMul::Format::MK8;
            }
        }
    }
    size_t nr_threads = static_cast<naive::HandleImpl*>(handle())
                                ->megcore_dispatcher()
                                ->nr_threads();
    return {{safe_u32(src[0]),
             {{safe_u32(src[spatial_pos]), safe_u32(src[spatial_pos + 1])}},
             {{safe_u32(dst[spatial_pos]), safe_u32(dst[spatial_pos + 1])}},
             conv_fm,
             src.dtype,
             filter.dtype,
             dst.dtype,
             src.stride[0],
             dst.stride[0],
             {src.stride[0], src.stride[1], src.stride[2], src.stride[3]},
             {dst.stride[0], dst.stride[1], dst.stride[2], dst.stride[3]},
             param().compute_mode,
             nr_threads},
            param().output_block_size,
            format,
            bias.dtype,
            bias.stride[0],
            bias_mode,
            param().nonlineMode};
}

ConvBiasImpl::NCBKernParam ConvBiasImpl::make_ncb_kern_param(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    NCBKernParam ret;
    static_cast<NCBKernSizeParam&>(ret) = make_ncb_kern_size_param(
            src.layout, filter.layout, bias.layout, dst.layout);
    ret.src_ptr = src.raw_ptr;
    ret.filter_ptr = filter.raw_ptr;
    ret.bias_ptr = bias.raw_ptr;
    ret.dst_ptr = dst.raw_ptr;
    ret.workspace_ptr = workspace.raw_ptr;
    ret.workspace_size = workspace.size;
    return ret;
}

void ConvBiasImpl::exec_with_ncb_kern(const NCBKernParam& param,
                                      ConvBiasImpl::Algorithm* algo) {
    auto ncb_kerns = ncb_algo_dispatch_kerns(algo, param);
    size_t src_batch_stride = param.inp_bs * param.src_type.size();
    size_t dst_batch_stride = param.out_bs * param.dst_type.size();
    size_t bias_batch_stride = 0;
    if (param.bias_mode == BiasMode::BIAS) {
        bias_batch_stride = param.bias_bs * param.bias_type.size();
    }
    for (auto&& kernel : ncb_kerns) {
        megdnn_assert(
                param.filter_meta.format == Param::Format::NCHW ||
                        param.filter_meta.format == Param::Format::NHWC ||
                        param.filter_meta.format ==
                                Param::Format::NCHW_WINOGRAD ||
                        param.filter_meta.format == Param::Format::NCHW88 ||
                        param.filter_meta.format ==
                                Param::Format::NCHW88_WINOGRAD,
                "invalid conv format");
        ptrdiff_t istrd = 0, fstrd = 0, bstrd = 0, ostrd = 0;
        if (param.filter_meta.format == Param::Format::NCHW_WINOGRAD ||
            param.filter_meta.format == Param::Format::NCHW88_WINOGRAD) {
            fstrd = param.filter_meta.icpg * param.filter_meta.ocpg *
                    (param.filter_meta.spatial[0] + param.output_block_size -
                     1) *
                    (param.filter_meta.spatial[1] + param.output_block_size -
                     1) *
                    param.filter_type.size();
        } else {
            fstrd = param.filter_meta.icpg * param.filter_meta.ocpg *
                    param.filter_meta.spatial[0] *
                    param.filter_meta.spatial[1] * param.filter_type.size();
        }
        istrd = param.filter_meta.icpg * param.src_type.size();
        ostrd = param.filter_meta.ocpg * param.dst_type.size();
        if (param.bias_mode != BiasMode::NO_BIAS) {
            bstrd = param.filter_meta.ocpg * param.bias_type.size();
        }
        if (param.filter_meta.format == Param::Format::NCHW ||
            param.filter_meta.format == Param::Format::NCHW_WINOGRAD ||
            param.filter_meta.format == Param::Format::NCHW88_WINOGRAD) {
            istrd *= param.isz[0] * param.isz[1];
            ostrd *= param.osz[0] * param.osz[1];
            if (param.bias_mode == BiasMode::BIAS) {
                bstrd *= param.osz[0] * param.osz[1];
            }
        } else {
            // must be NHWC. No action performed.
        }
        auto run = [=](size_t index, size_t thread_id) {
            auto copy_param = param;
            CpuNDRange ndrange_id(kernel.global_size, index);
            size_t group_id = ndrange_id[0];
            size_t batch_id = ndrange_id[1];
            //! The kernel ptr point to batch index
            incr_ptr(copy_param.src_ptr,
                     group_id * istrd + batch_id * src_batch_stride);
            incr_ptr(copy_param.filter_ptr, group_id * fstrd);
            incr_ptr(copy_param.bias_ptr,
                     group_id * bstrd + batch_id * bias_batch_stride);
            incr_ptr(copy_param.dst_ptr,
                     group_id * ostrd + batch_id * dst_batch_stride);
            kernel.kern(copy_param, {thread_id, ndrange_id});
        };
        static_cast<naive::HandleImpl*>(handle())->dispatch_kern(
                run, kernel.global_size.total_size());
    }
}

ConvBiasImpl::Algorithm* ConvBiasImpl::get_algorithm_heuristic_with_ncb(
        const NCBKernSizeParam& param, size_t workspace_limit_in_bytes,
        bool reproducible) {
    return ncb_algo_get_algorithm_heuristic(param, workspace_limit_in_bytes,
                                            reproducible);
}

size_t ConvBiasImpl::ncb_algo_get_workspace(Algorithm* algo,
                                            const NCBKernSizeParam& param) {
    return static_cast<AlgoBase*>(algo)->get_workspace(this, param);
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::ncb_algo_dispatch_kerns(
        Algorithm* algo, const NCBKernSizeParam& param) {
    return static_cast<AlgoBase*>(algo)->dispatch_kerns(this, param);
}

std::vector<ConvBiasImpl::Algorithm*> ConvBiasImpl::get_all_algorithms_with_ncb(
        const NCBKernSizeParam& param) {
    MEGDNN_MARK_USED_VAR(param);
    std::vector<Algorithm*> algos;
    std::vector<Algorithm*> prefer_algos;
    for (auto&& algo : algo_pack()) {
        if (algo->usable(this, param, AlgoSelectionStrategy::FULL_RUN)) {
            if (algo->is_preferred(this, param)) {
                prefer_algos.push_back(algo);
            } else {
                algos.push_back(algo);
            }
        }
    }
    std::reverse(prefer_algos.begin(), prefer_algos.end());
    //! Prefer algo inserted from begin
    algos.insert(algos.begin(), prefer_algos.begin(), prefer_algos.end());
    return algos;
}

ConvBiasImpl::Algorithm* ConvBiasImpl::ncb_algo_get_algorithm_heuristic(
        const NCBKernSizeParam& param, size_t workspace_limit_in_bytes,
        bool reproducible) {
    for (auto i : get_all_algorithms_with_ncb(param)) {
        if (static_cast<AlgoBase*>(i)->usable_reproducible(
                    this, param, AlgoSelectionStrategy::HEURISTIC,
                    reproducible) &&
            ncb_algo_get_workspace(i, param) <= workspace_limit_in_bytes) {
            return i;
        }
    }
    return nullptr;
}

ConvBiasImpl::Algorithm* ConvBiasImpl::get_algorithm(
        const NCBKernSizeParam& param, size_t workspace_size) {
    if (auto set = execution_policy().algorithm) {
        return set;
    }
    if (!m_prev_selected_algo ||
        memcmp(&m_prev_selected_algo_sizep, &param, sizeof(NCBKernSizeParam))) {
        m_prev_selected_algo =
                get_algorithm_heuristic_with_ncb(param, workspace_size);
        m_prev_selected_algo_sizep = param;
    }
    return m_prev_selected_algo;
}

const char* ConvBiasImpl::get_algorithm_set_name() const {
    // fallback version 0
    return "F0";
}

// vim: syntax=cpp.doxygen
