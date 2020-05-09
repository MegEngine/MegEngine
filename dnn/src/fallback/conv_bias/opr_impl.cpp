/**
 * \file dnn/src/fallback/conv_bias/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/fallback/convolution/opr_impl.h"
#include "src/common/algo_chooser.h"
#include "src/common/metahelper.h"
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/algos.h"
#include "src/fallback/conv_bias/conv1x1/algos.h"
#include "src/fallback/conv_bias/im2col/algos.h"
#include "src/fallback/conv_bias/opr_impl.h"
#include "src/naive/convolution/algorithms.h"
#include "src/naive/handle.h"

#include <cstring>

using namespace megdnn;
using namespace fallback;

size_t megdnn::fallback::get_format_pack_size(param::ConvBias::Format format) {
    switch (format) {
        case param::ConvBias::Format::NCHW44:
        case param::ConvBias::Format::NCHW4:
            return 4_z;
        case param::ConvBias::Format::NCHW88:
            return 8_z;
        default:
            return 1_z;
    }
}

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
#if MEGDNN_X86
//! As we haven't direct conv for int8x8x16 yet, if we disable gemv here, it may
//! fallback to naive implementation, which may cause performance very low, so
//! here we just enable im2col for gemv in x86 backend.
//! FIXME: remove it when we add direct conv support for int8x8x16
#else
            if (algo->algoset() ==
                MatrixMulImpl::AlgoBase::AlgoSet::ALGO_TYPE_GEMV) {
                continue;
            }
#endif

            for (size_t ohw_tile_size : {192, 384, 96, 48, 24}) {
                refhold.emplace_back(new AlgoIm2col(
                        static_cast<MatrixMulImpl::AlgoBase*>(algo),
                        ohw_tile_size));
                all_algos.emplace_back(refhold.back().get());
            }
            for (size_t oc_tile_size : {24, 48}) {
                refhold.emplace_back(new AlgoConv1x1(
                        static_cast<MatrixMulImpl::AlgoBase*>(algo),
                        oc_tile_size));
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
                        _megdnn_tensor_out dst,
                        const PreprocessedFilter* preprocessed_filter,
                        _megdnn_workspace workspace) {
    check_exec(src.layout, filter.layout, bias.layout, z.layout, dst.layout,
               workspace.size);
    auto fparam = make_ncb_kern_param(src, filter, bias, dst, workspace);
    ConvBiasImpl::Algorithm* algo = get_algorithm(fparam, workspace.size);
    if (!is_naive_algo(algo) &&
        ncb_algo_get_workspace(algo, fparam) <= workspace.size) {
        exec_with_ncb_kern(fparam, algo);
    } else {
        naive::ConvBiasForwardImpl::exec(src, filter, bias, z, dst,
                                         preprocessed_filter, workspace);
    }
}

size_t ConvBiasImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst,
        const PreprocessedFilter* preprocessed_filter) {
    auto fparam = make_ncb_kern_size_param(src, filter, bias, dst);
    ConvBiasImpl::Algorithm* algo = get_algorithm(fparam);
    if (is_naive_algo(algo)) {
        return naive::ConvBiasForwardImpl::get_workspace_in_bytes(
                src, filter, bias, z, dst, preprocessed_filter);
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
        param().format == Param::Format::NCHW44 ||
        param().format == Param::Format::NCHW ||
        param().format == Param::Format::NCHW_WINOGRAD ||
        param().format == Param::Format::NCHW88_WINOGRAD ||
        param().format == Param::Format::NCHW44_WINOGRAD) {
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
        param().format == Param::Format::NCHW88_WINOGRAD ||
        param().format == Param::Format::NCHW44_WINOGRAD) {
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
    for (auto&& kernel : ncb_kerns) {
        auto run = [kernel, param](size_t index, size_t thread_id) {
            CpuNDRange ndrange_id(kernel.global_size, index);
            kernel.kern(param, {thread_id, ndrange_id});
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

namespace megdnn {
namespace fallback {

template <typename T>
const T* ConvBiasImpl::NCBKernParam::src(size_t batch_id, size_t group_pack_id,
                                         size_t channel_pack_id,
                                         size_t group_pack_size,
                                         size_t channel_pack_size) const {
    size_t batch_offset = batch_id * inp_bs * src_type.size();
    size_t group_offset = group_pack_size * group_pack_id * filter_meta.icpg *
                          isz[0] * isz[1] * src_type.size();
    size_t channel_offset = channel_pack_size * channel_pack_id * isz[0] *
                            isz[1] * src_type.size();
    return reinterpret_cast<T*>(reinterpret_cast<ptrdiff_t>(src_ptr) +
                                batch_offset + group_offset + channel_offset);
}

template <typename T>
const T* ConvBiasImpl::NCBKernParam::filter(size_t group_pack_id,
                                            size_t pack_group_size) const {
    size_t group_offset = 0_z;
    switch (filter_meta.format) {
        case Param::Format::NCHW: {
            group_offset = pack_group_size * group_pack_id * filter_meta.icpg *
                           filter_meta.ocpg * filter_meta.spatial[0] *
                           filter_meta.spatial[1] * filter_type.size();
            break;
        }
        case Param::Format::NCHW88: {
            size_t group = filter_meta.group;
            size_t icpg = filter_meta.icpg;
            size_t ocpg = filter_meta.ocpg;
            //! four format of weight layout
            //! 1. {oc/8, ic/8, fh, fw, 8, 8},
            //! 2. {g, oc/8, ic/8, fh, fw, 8, 8},
            //! 3. {g/8, fh, fw, 1, 1, 8}, 4. {oc/8, fh, fw, ic, 8}
            megdnn_assert((icpg % 8 == 0 && ocpg % 8 == 0) ||
                                  (group % 8 == 0 && icpg == 1 && ocpg == 1 &&
                                   pack_group_size > 1) ||
                                  (group == 1 && ocpg % 8 == 0),
                          "The filter shepe is not right of nchw88");
            group_offset = pack_group_size * group_pack_id * filter_meta.icpg *
                           filter_meta.ocpg * filter_meta.spatial[0] *
                           filter_meta.spatial[1] * filter_type.size();

            break;
        }
        case Param::Format::NCHW44: {
            size_t group = filter_meta.group;
            size_t icpg = filter_meta.icpg;
            size_t ocpg = filter_meta.ocpg;
            //! four format of weight layout
            //! 1. {oc/4, ic/4, fh, fw, 4, 4},
            //! 2. {g, oc/4, ic/4, fh, fw, 4, 4},
            //! 3. {g/4, fh, fw, 1, 1, 4}, 4. {oc/4, fh, fw, ic, 4}
            megdnn_assert((icpg % 4 == 0 && ocpg % 4 == 0) ||
                                  (group % 4 == 0 && icpg == 1 && ocpg == 1 &&
                                   pack_group_size > 1) ||
                                  (group == 1 && ocpg % 4 == 0),
                          "The filter shepe is not right of nchw44");
            group_offset = pack_group_size * group_pack_id * filter_meta.icpg *
                           filter_meta.ocpg * filter_meta.spatial[0] *
                           filter_meta.spatial[1] * filter_type.size();

            break;
        }
        case ConvBiasImpl::Param::Format::NCHW_WINOGRAD:
        case ConvBiasImpl::Param::Format::NCHW44_WINOGRAD:
        case ConvBiasImpl::Param::Format::NCHW88_WINOGRAD: {
            //! four format of weight layout
            //! 1. {g, alpha, alpha, ocpg/8, icpg/8, 8, 8}
            //! 2. {alpha, alpha, ocpg/8, icpg/8, 8, 8}
            //! 3. {g, alpha, alpha, oc, ic, 8, 8}
            //! 4. {alpha, alpha, oc, ic}
            group_offset = pack_group_size * group_pack_id * filter_meta.icpg *
                           filter_meta.ocpg *
                           (filter_meta.spatial[0] + output_block_size - 1) *
                           (filter_meta.spatial[1] + output_block_size - 1) *
                           filter_type.size();
            break;
        }
        default:
            megdnn_assert(0, "other filter format is not support yet");
    }
    return reinterpret_cast<T*>(reinterpret_cast<ptrdiff_t>(filter_ptr) +
                                group_offset);
}

template <typename T>
const T* ConvBiasImpl::NCBKernParam::bias(size_t batch_id, size_t group_pack_id,
                                          size_t channel_pack_id,
                                          size_t group_pack_size,
                                          size_t channel_pack_size) const {
    size_t batch_offset = 0_z;
    size_t group_offset = 0_z;
    size_t channel_offset = 0_z;
    if (bias_mode == BiasMode::BIAS) {
        batch_offset = batch_id * bias_bs * bias_type.size();
        group_offset = group_pack_size * group_pack_id * filter_meta.ocpg *
                       osz[0] * osz[1] * bias_type.size();
        channel_offset = channel_pack_size * channel_pack_id * osz[0] * osz[1] *
                         bias_type.size();
    } else if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        group_offset = group_pack_size * group_pack_id * filter_meta.ocpg *
                       bias_type.size();
        channel_offset = channel_pack_size * channel_pack_id * bias_type.size();
    }
    return reinterpret_cast<T*>(reinterpret_cast<ptrdiff_t>(bias_ptr) +
                                batch_offset + group_offset + channel_offset);
}

template <typename T>
T* ConvBiasImpl::NCBKernParam::dst(size_t batch_id, size_t group_pack_id,
                                   size_t channel_pack_id,
                                   size_t group_pack_size,
                                   size_t channel_pack_size) const {
    size_t batch_offset = batch_id * out_bs * dst_type.size();
    size_t group_offset = group_pack_size * group_pack_id * filter_meta.ocpg *
                          osz[0] * osz[1] * dst_type.size();
    size_t channel_offset = channel_pack_size * channel_pack_id * osz[0] *
                            osz[1] * dst_type.size();
    return reinterpret_cast<T*>(reinterpret_cast<ptrdiff_t>(dst_ptr) +
                                batch_offset + group_offset + channel_offset);
}

#define INST(T)                                                      \
    template const T* ConvBiasImpl::NCBKernParam::src<T>(            \
            size_t batch_id, size_t group_id, size_t channel_id,     \
            size_t group_pack_size, size_t channel_pack_size) const; \
    template const T* ConvBiasImpl::NCBKernParam::bias<T>(           \
            size_t batch_id, size_t group_id, size_t channel_id,     \
            size_t group_pack_size, size_t channel_pack_size) const; \
    template const T* ConvBiasImpl::NCBKernParam::filter<T>(         \
            size_t group_id, size_t group_pack_size) const;          \
    template T* ConvBiasImpl::NCBKernParam::dst<T>(                  \
            size_t batch_id, size_t group_id, size_t channel_id,     \
            size_t group_pack_size, size_t channel_pack_size) const;

#define INST_DT(d) INST(DTypeTrait<d>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(INST_DT)
INST(void)
#undef INST
#undef INST_DT
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
