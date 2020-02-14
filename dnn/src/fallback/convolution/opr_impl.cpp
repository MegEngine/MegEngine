/**
 * \file dnn/src/fallback/convolution/opr_impl.cpp
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
#include "src/fallback/convolution/algos.h"
#include "src/fallback/convolution/run_conv.h"
#include "src/naive/convolution/helper.h"
#include "src/naive/handle.h"

#include "midout.h"

#include <cstring>

MIDOUT_DECL(megdnn_fb_conv_float)
MIDOUT_DECL(megdnn_fb_convbwd_float)

using namespace megdnn;
using namespace fallback;

namespace {
class NaiveConvolutionBackwardData final
        : public megdnn::ConvolutionBackwardData::Algorithm {
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "NCBD"; }
};
NaiveConvolutionBackwardData naive_conv_backward_data;
uint8_t fallback_deconv_algo_type_storage;
uint8_t fallback_conv_algo_type_storage;

template <typename T>
void incr_ptr(T*& dst, ptrdiff_t delta) {
    dst = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(dst) + delta);
}
}  // namespace

class ConvolutionImpl::AlgoPack : NonCopyableObj {
    AlgoFallback algo_fallback;
    AlgoNaive algo_naive;
    SmallVector<std::unique_ptr<AlgoBase>> refhold;

public:
    AlgoPack() {
        static CpuOprDelegationStorage<1> storage;
        auto conv_bias_opr = storage.get<ConvBias, 0>();
        auto&& conv_bias_algo =
                static_cast<ConvBiasImpl*>(conv_bias_opr)->algo_pack();
        for (auto&& algorithm : conv_bias_algo) {
            // fallback algo
            refhold.emplace_back(new AlgoDefault(
                    static_cast<ConvBiasImpl*>(conv_bias_opr), algorithm));
            all_algos.emplace_back(refhold.back().get());
        }

        all_algos.emplace_back(&algo_fallback);
        all_algos.emplace_back(&algo_naive);
    }
    SmallVector<AlgoBase*> all_algos;
};

void* const ConvolutionImpl::sm_fallback_conv_algo_type =
        &fallback_conv_algo_type_storage;

SmallVector<ConvolutionImpl::AlgoBase*> ConvolutionImpl::algo_pack() {
    static AlgoPack sl_algo_pack;
    return sl_algo_pack.all_algos;
}
bool ConvolutionImpl::is_naive_algo(ConvolutionImpl::Algorithm* algo) {
    return algo == nullptr || strcmp(algo->name(), "DEFAULT") == 0;
}
void ConvolutionImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                           _megdnn_tensor_out dst,
                           _megdnn_workspace workspace) {
    auto fparam = make_ncb_kern_param(src, filter, dst, workspace);
    ConvolutionImpl::Algorithm* algo = get_algorithm(fparam, workspace.size);
    if (!is_naive_algo(algo) &&
        ncb_algo_get_workspace(algo, fparam) <= workspace.size) {
        exec_with_ncb_kern(fparam, algo);
    } else {
        naive::ConvolutionForwardImpl::exec(src, filter, dst, workspace);
    }
}

size_t ConvolutionImpl::get_workspace_in_bytes(const TensorLayout& src,
                                               const TensorLayout& filter,
                                               const TensorLayout& dst) {
    auto fparam = make_ncb_kern_size_param(src, filter, dst);
    Algorithm* algo = get_algorithm(fparam);
    if (is_naive_algo(algo)) {
        return naive::ConvolutionForwardImpl::get_workspace_in_bytes(
                src, filter, dst);
    } else {
        return ncb_algo_get_workspace(algo, fparam);
    }
}

std::vector<ConvolutionImpl::Algorithm*> ConvolutionImpl::get_all_algorithms(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst) {
    auto fparam = make_ncb_kern_size_param(src, filter, dst);
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
    auto fparam = make_ncb_kern_size_param(src, filter, dst);
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
        const TensorLayout& dst) {
    auto safe_u32 = [](size_t v) -> uint32_t {
        megdnn_assert(v <= std::numeric_limits<uint32_t>::max(),
                      "value too large: %zu", v);
        return v;
    };
    size_t spatial_pos;
    if (param().format == Param::Format::NCHW88 ||
        param().format == Param::Format::NCHW8 ||
        param().format == Param::Format::NCHW4) {
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
            nr_threads};
}

ConvolutionImpl::NCBKernParam ConvolutionImpl::make_ncb_kern_param(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    NCBKernParam ret;
    static_cast<NCBKernSizeParam&>(ret) =
            make_ncb_kern_size_param(src.layout, filter.layout, dst.layout);
    ret.src_ptr = src.raw_ptr;
    ret.filter_ptr = filter.raw_ptr;
    ret.dst_ptr = dst.raw_ptr;
    ret.workspace_ptr = workspace.raw_ptr;
    ret.workspace_size = workspace.size;
    return ret;
}

void ConvolutionImpl::exec_with_ncb_kern(const NCBKernParam& param,
                                         Algorithm* algo) {
    auto kerns = ncb_algo_dispatch_kern(algo, param);
    size_t src_batch_stride = param.inp_bs * param.src_type.size();
    size_t dst_batch_stride = param.out_bs * param.dst_type.size();
    auto group = param.filter_meta.group;
    auto fallback_handle = handle();
    for (auto kernel : kerns) {
        megdnn_assert(param.filter_meta.format == Param::Format::NCHW ||
                      param.filter_meta.format == Param::Format::NHWC ||
                      "invalid conv format");

        ptrdiff_t istrd = 0, fstrd = 0, ostrd = 0;
        fstrd = param.filter_meta.icpg * param.filter_meta.ocpg *
                param.filter_meta.spatial[0] * param.filter_meta.spatial[1] *
                param.filter_type.size();
        istrd = param.filter_meta.icpg * param.src_type.size();
        ostrd = param.filter_meta.ocpg * param.dst_type.size();
        if (param.filter_meta.format == Param::Format::NCHW) {
            istrd *= param.isz[0] * param.isz[1];
            ostrd *= param.osz[0] * param.osz[1];
        } else {
            // must be NHWC. No action performed.
        }
        auto run = [=](size_t index, size_t thread_id) {
            auto copy_param = param;
            CpuNDRange ndrange_id(kernel.global_size, index);
            size_t group_id = ndrange_id[0];
            size_t batch_id = ndrange_id[1];
            megdnn_assert(group_id < group,
                          "The group id should smaller than gruop");
            //! The kernel ptr point to batch index
            incr_ptr(copy_param.src_ptr,
                     group_id * istrd + batch_id * src_batch_stride);
            incr_ptr(copy_param.filter_ptr, group_id * fstrd);
            incr_ptr(copy_param.dst_ptr,
                     group_id * ostrd + batch_id * dst_batch_stride);
            kernel.kern(copy_param, {thread_id, ndrange_id});
        };
        static_cast<naive::HandleImpl*>(fallback_handle)
                ->dispatch_kern(run, kernel.global_size.total_size());
    }
}

ConvolutionImpl::Algorithm* ConvolutionImpl::get_algorithm_heuristic_with_ncb(
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

std::vector<ConvolutionImpl::Algorithm*>
ConvolutionImpl::get_all_algorithms_with_ncb(const NCBKernSizeParam& param) {
    std::vector<Algorithm*> ret;
    std::vector<Algorithm*> prefer_algos;
    for (auto&& i : algo_pack()) {
        if (i->usable(this, param, AlgoSelectionStrategy::FULL_RUN)) {
            if (i->is_preferred(this, param)) {
                prefer_algos.push_back(i);
            } else {
                ret.push_back(i);
            }
        }
    }
    std::reverse(prefer_algos.begin(), prefer_algos.end());
    //! Prefer algo inserted from begin
    ret.insert(ret.begin(), prefer_algos.begin(), prefer_algos.end());
    return ret;
}

ConvolutionImpl::Algorithm* ConvolutionImpl::get_algorithm(
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

const char* ConvolutionImpl::get_algorithm_set_name() const {
    // fallback version 0
    return "F0";
}

/* ===================== ConvolutionBackwardData ===================== */

void* const ConvolutionBackwardDataImpl::sm_fallback_deconv_algo_type =
        &fallback_deconv_algo_type_storage;

struct ConvolutionBackwardDataImpl::AlgoPack {
    AlgoDirect direct;
    AlgoMatrixMul matmul;
};
ConvolutionBackwardDataImpl::AlgoPack ConvolutionBackwardDataImpl::sm_algo_pack;

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
    auto algo = get_algorithm(p1g);
    auto kptr = ncb_1g_dispatch_kern(algo, p1g);
    if (algo == &naive_conv_backward_data || group == 1) {
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
        return ncb_1g_get_workspace(get_algorithm(p1g), p1g);
    }
    return ncb_1g_get_workspace(get_algorithm(param), param);
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
    if (algo->type() == sm_fallback_deconv_algo_type) {
        return static_cast<AlgoBase*>(algo)->get_workspace(this, param);
    }
    megdnn_assert(algo == &naive_conv_backward_data);
    return 0;
}

ConvolutionBackwardDataImpl::ncb_kern_t
ConvolutionBackwardDataImpl::ncb_1g_dispatch_kern(
        Algorithm* algo, const NCBKernSizeParam& param) {
    megdnn_assert(param.filter_meta.group == 1);

    if (algo->type() == sm_fallback_deconv_algo_type) {
        return static_cast<AlgoBase*>(algo)->dispatch_kern(this, param);
    }

    if (algo == &naive_conv_backward_data) {
#define cb(_dt)                                                    \
    do {                                                           \
        if (param.filter_type.enumv() == DTypeTrait<_dt>::enumv) { \
            MIDOUT_BEGIN(megdnn_fb_convbwd_float,                  \
                         midout_iv(DTypeTrait<_dt>::enumv)) {      \
                using ctype = DTypeTrait<_dt>::ctype;              \
                return kern_naive<ctype, ctype, ctype>;            \
            }                                                      \
            MIDOUT_END();                                          \
        }                                                          \
    } while (0);
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb
#define cb(dt_src, dt_dst)                                            \
    do {                                                              \
        if (param.diff_type.enumv() == DTypeTrait<dt_src>::enumv &&   \
            param.filter_type.enumv() == DTypeTrait<dt_src>::enumv && \
            param.grad_type.enumv() == DTypeTrait<dt_dst>::enumv) {   \
            return kern_naive<DTypeTrait<dt_src>::ctype,              \
                              DTypeTrait<dt_src>::ctype,              \
                              DTypeTrait<dt_dst>::ctype>;             \
        }                                                             \
    } while (0);
        cb(dtype::Int8, dtype::Int32) cb(dtype::Quantized8Asymm,
                                         dtype::QuantizedS32)
                cb(dtype::QuantizedS8, dtype::QuantizedS32) megdnn_throw(
                        "unsupported data type on ConvolutionBackwardData");
#undef cb
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
    ret.reserve(2);
    ret.push_back(&naive_conv_backward_data);

    // insert from lowest to highest preference
    AlgoBase* cand[2] = {nullptr};

    if (param.filter_meta.group == 1 && param.filter_meta.dilation[0] == 1 &&
        param.filter_meta.dilation[1] == 1) {
        // we currently only have non-dilated algos
        if (param.filter_type.enumv() == DTypeEnum::Float32) {
            if (is_matrix_mul_preferred(param)) {
                cand[0] = &sm_algo_pack.direct;
                cand[1] = &sm_algo_pack.matmul;
            } else {
                cand[0] = &sm_algo_pack.matmul;
                cand[1] = &sm_algo_pack.direct;
            }
        } else {
            cand[0] = &sm_algo_pack.matmul;
        }
    }
    for (auto i : cand) {
        if (i && i->usable(this, param)) {
            ret.push_back(i);
        }
    }

    std::reverse(ret.begin(), ret.end());
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
ConvolutionBackwardDataImpl::get_algorithm(const NCBKernSizeParam& param) {
    if (auto set = execution_policy().algorithm) {
        return set;
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
