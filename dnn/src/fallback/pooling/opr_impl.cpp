/**
 * \file dnn/src/fallback/pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/fallback/pooling/opr_impl.h"
#include "src/common/algo_chooser.h"
#include "src/common/metahelper.h"
#include "src/fallback/pooling/gi/algo.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_pooling)

using namespace megdnn;
using namespace fallback;

class PoolingImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;
    AlgoGiFilterxModexStride1 algo_gi_filterx_modex_stride1;
    AlgoGiFilter2ModexStride2 algo_gi_filter2_modex_stride2;
    AlgoGiFilter3MaxStride2 algo_gi_filter3_max_stride2;
    AlgoGiFilter3AverageStride2 algo_gi_filter3_average_stride2;
    AlgoGiFilter4MaxStride2 algo_gi_filter4_max_stride2;
    AlgoGiFilter5MaxStride2 algo_gi_filter5_max_stride2;
    AlgoGiFp32ModexStridexNCHW44 algo_gi_fp32_modex_stridex_nchw44;
    AlgoFallback algo_fallback;

public:
    AlgoPack() {
        all_algos.emplace_back(&algo_gi_filterx_modex_stride1);
        all_algos.emplace_back(&algo_gi_filter2_modex_stride2);
        all_algos.emplace_back(&algo_gi_filter3_max_stride2);
        all_algos.emplace_back(&algo_gi_filter3_average_stride2);
        all_algos.emplace_back(&algo_gi_filter4_max_stride2);
        all_algos.emplace_back(&algo_gi_filter5_max_stride2);
        all_algos.emplace_back(&algo_gi_fp32_modex_stridex_nchw44);
        all_algos.emplace_back(&algo_fallback);

        for (auto&& algo : all_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
    }
    SmallVector<AlgoBase*> all_algos;
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

PoolingImpl::AlgoPack PoolingImpl::sm_algo_pack;

PoolingImpl::PoolingKernSizeParam PoolingImpl::make_pooling_kern_szie_param(
        fallback::PoolingImpl* opr, const TensorLayout& src, const TensorLayout& dst) {
    auto safe_u32 = [](size_t v) -> uint32_t {
        megdnn_assert(
                v <= std::numeric_limits<uint32_t>::max(), "value too large: %zu", v);
        return v;
    };
    return {safe_u32(src.shape[0]),
            safe_u32(src.shape[1]),
            {{safe_u32(src.shape[2]), safe_u32(src.shape[3])}},
            {{safe_u32(dst.shape[2]), safe_u32(dst.shape[3])}},
            {{safe_u32(opr->param().pad_h), safe_u32(opr->param().pad_w)}},
            {{safe_u32(opr->param().window_h), safe_u32(opr->param().window_w)}},
            {{safe_u32(opr->param().stride_h), safe_u32(opr->param().stride_w)}},
            src.dtype,
            dst.dtype,
            opr->handle(),
            opr->param().format,
            opr->param().mode};
};

PoolingImpl::PoolingKernParam PoolingImpl::make_pooling_kern_param(
        fallback::PoolingImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    PoolingKernParam ret;
    static_cast<PoolingKernSizeParam&>(ret) =
            make_pooling_kern_szie_param(opr, src.layout, dst.layout);
    ret.src_ptr = src.get_ref_ptr();
    ret.dst_ptr = dst.get_ref_ptr();
    ret.workspace_ptr = workspace.raw_ptr;
    ret.workspace_size = workspace.size;
    return ret;
};

MEGDNN_DEF_GET_ALGO_FROM_DESC(PoolingImpl);

std::vector<Algorithm*> PoolingImpl::get_all_algorithms(
        const TensorLayout& src, const TensorLayout& dst) {
    auto param = make_pooling_kern_szie_param(this, src, dst);
    std::vector<Algorithm*> ret;
    ret.reserve(algo_pack().all_algos.size());
    for (auto i : algo_pack().all_algos) {
        if (i->usable(param)) {
            ret.push_back(i);
        }
    }
    return ret;
}

size_t PoolingImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    TensorLayoutArray layouts{src, dst};
    AlgorithmCache::Key key{this->handle(), this->get_opr_type(),
                            layouts.data(), layouts.size(),
                            &this->param(), sizeof(this->param())};
    auto rst = AlgorithmCache::instance().get(key);
    if (rst.policy.algo.valid()) {
        return rst.workspace;
    }

    auto param = make_pooling_kern_szie_param(this, src, dst);
    auto algo = static_cast<AlgoBase*>(fallback::PoolingImpl::get_algorithm_heuristic(
            src, dst, std::numeric_limits<size_t>::max(), AlgoAttribute::DEFAULT,
            AlgoAttribute::DEFAULT));
    if (!is_fallback_non_gi_algo(algo)) {
        size_t fallback_gi_workspace = 0;

        //! When multi-thread, every thread has its own workspace
        size_t nr_threads = static_cast<naive::HandleImpl*>(handle())
                                    ->megcore_dispatcher()
                                    ->nr_threads();
        if (param.src_type.category() == DTypeCategory::FLOAT &&
            param.filter[0] == param.filter[1] &&
            (param.filter[0] == 3 || param.filter[0] == 5) &&
            param.format == Param::Format::NCHW &&
            (param.mode == Mode::MAX ||
             (param.mode == Mode::AVERAGE && param.filter[0] == 3)) &&
            param.stride[0] == 2 && param.stride[1] == 2 && param.isz[0] >= 2 &&
            param.isz[1] >= 2) {
            WorkspaceBundle ws = get_bundle(param);
            fallback_gi_workspace = ws.total_size_in_bytes() * nr_threads;
        }

        return fallback_gi_workspace;
    } else {
        auto naive_worksapce =
                naive::PoolingForwardImpl::get_workspace_in_bytes(src, dst);
        return naive_worksapce;
    }
}

void PoolingImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    auto param = make_pooling_kern_param(this, src, dst, workspace);
    auto algo = static_cast<AlgoBase*>(fallback::PoolingImpl::get_algorithm_heuristic(
            src.layout, dst.layout, std::numeric_limits<size_t>::max(),
            AlgoAttribute::DEFAULT, AlgoAttribute::DEFAULT));
    if (!is_fallback_non_gi_algo(algo)) {
        algo->exec(param);
    } else {
        exec_fallback(src, dst, workspace);
    }
}

std::vector<Algorithm*> PoolingImpl::get_all_algorithms_safe(
        const TensorLayout& src, const TensorLayout& dst) {
    auto ret_safe = get_all_algorithms(src, dst);
    megdnn_assert(!ret_safe.empty(), "no usable pooling fwd algorithm");
    return ret_safe;
}

Algorithm* PoolingImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& dst,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    MEGDNN_MARK_USED_VAR(workspace_limit_in_bytes);

    auto param = make_pooling_kern_szie_param(this, src, dst);
    for (auto&& iter : sm_algo_pack.all_algos) {
        if (iter->is_available_attribute(param, positive_attr, negative_attr)) {
            return iter;
        }
    }
    megdnn_throw(ssprintf(
            "require algorithm with attribute(%s) and without "
            "attribute(%s), but can't get suitable algo.\n",
            Algorithm::attribute_str(positive_attr).c_str(),
            Algorithm::attribute_str(negative_attr).c_str()));
    return nullptr;
}
//! fallback not gi imp
namespace megdnn {
namespace fallback {
namespace pooling {

void w3x3_w1x1_1d(const float* src, float* dst, size_t I, size_t O, size_t P) {
    const float* __restrict src_ = src;
    float* __restrict dst_ = dst;
    if (P == 0) {
    } else if (P == 1) {
        dst_[0] = std::max(src_[0], src_[1]);
    } else if (P == 2) {
        dst_[0] = src_[0];
        dst_[1] = std::max(src_[0], src_[1]);
    }
    for (size_t o = P; o + P < O; ++o) {
        size_t i = o - P;
        dst_[o] = std::max(std::max(src_[i], src_[i + 1]), src_[i + 2]);
    }
    if (P == 0) {
    } else if (P == 1) {
        dst_[O - 1] = std::max(src_[I - 1], src_[I - 2]);
    } else if (P == 2) {
        dst_[O - 1] = src_[I - 1];
        dst_[O - 2] = std::max(src_[I - 1], src_[I - 2]);
    }
}

void w3x3_s1x1(
        const float* src, float* dst, size_t IH, size_t IW, size_t OH, size_t OW,
        size_t PH, size_t PW) {
    // Let tmp[i][j] = max(src[i][j'], src[i][j'+1], ..., src[i][j'+WW-1]),
    // where (i, j') is the corresponding src pixel coordinate for
    // dst pixel coordinate (i, j).
    // cache[] stores lines of tmp in a sliding-window way.
    // cache[0] denotes the line that is currently being processed.
    // The length of each line is OW.
    std::vector<float*> cache(3, nullptr);
    auto shuffle = [&cache]() {
        auto len = cache.size();
        auto ptr = cache.data();
        auto last = cache.back();
        std::memmove(ptr + 1, ptr, sizeof(float*) * (len - 1));
        cache[0] = last;
    };
    for (auto& ptr : cache) {
        ptr = new float[OW];
        megdnn_assert(ptr, "new failed (possibly lack of memory?)");
    }
    // initialize all lines with the least optimized val (-infinity)
    for (auto ptr : cache) {
        std::fill(ptr, ptr + OW, -std::numeric_limits<float>::max());
    }
    // init situation where oh == -1
    {
        int oh = -1;
        // rb for right bracket
        int ih_rb = oh - PH + 3;
        for (int ih = 0; ih < ih_rb; ++ih) {
            shuffle();
            w3x3_w1x1_1d(src + ih * IW, cache[0], IW, OW, PW);
        }
    }
    for (int oh = 0; oh < static_cast<int>(OH); ++oh) {
        shuffle();
        int ih = oh - PH + 3 - 1;
        if (ih >= static_cast<int>(IH)) {
            std::fill(cache[0], cache[0] + OW, -std::numeric_limits<float>::max());
        } else {
            w3x3_w1x1_1d(src + ih * IW, cache[0], IW, OW, PW);
        }
        float* __restrict dst_ = dst;
        for (size_t ow = 0; ow < OW; ++ow) {
            float res = std::max(cache[0][ow], std::max(cache[1][ow], cache[2][ow]));
            dst_[oh * OW + ow] = res;
        }
    }
    // free
    for (auto ptr : cache) {
        delete[] ptr;
    }
}

void w2x2_s2x2_int8(
        const int8_t* src, int8_t* dst, size_t IH, size_t IW, size_t OH, size_t OW) {
    megdnn_ignore(IH);
    for (size_t ih = 0; ih < OH * 2; ++ih) {
        size_t oh = ih >> 1;
        const int8_t* __restrict sptr = src + ih * IW;
        int8_t* __restrict dptr = dst + oh * OW;
        if (ih & 1) {
            for (size_t ow = 0; ow < OW; ++ow) {
                dptr[ow] = std::max(dptr[ow], std::max(sptr[ow * 2], sptr[ow * 2 + 1]));
            }
        } else {
            for (size_t ow = 0; ow < OW; ++ow) {
                dptr[ow] = std::max(sptr[ow * 2], sptr[ow * 2 + 1]);
            }
        }
    }
}

void w2x2_s2x2_avg_int8(
        const int8_t* src, int8_t* dst, size_t IH, size_t IW, size_t OH, size_t OW) {
    megdnn_ignore(IH);
    for (size_t oh = 0; oh < OH; ++oh) {
        size_t ih = oh * 2;
        const int8_t* __restrict sptr0 = src + (ih + 0) * IW;
        const int8_t* __restrict sptr1 = src + (ih + 1) * IW;
        int8_t* __restrict dptr = dst + oh * OW;
        for (size_t ow = 0; ow < OW; ++ow) {
            size_t iw = ow * 2;
            int32_t v00 = sptr0[iw + 0], v01 = sptr0[iw + 1], v10 = sptr1[iw + 0],
                    v11 = sptr1[iw + 1];
            dptr[ow] = (v00 + v01 + v10 + v11) / 4;
        }
    }
}

}  // namespace pooling
}  // namespace fallback
}  // namespace megdnn

void PoolingImpl::exec_w3x3_s1x1(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, const Param& param) {
    auto N = src.layout.shape[0], C = src.layout.shape[1];
    auto IH = src.layout.shape[2], IW = src.layout.shape[3];
    auto OH = dst.layout.shape[2], OW = dst.layout.shape[3];
    for (size_t nc = 0; nc < N * C; ++nc) {
        pooling::w3x3_s1x1(
                src.ptr<dt_float32>() + nc * IH * IW,
                dst.ptr<dt_float32>() + nc * OH * OW, IH, IW, OH, OW, param.pad_h,
                param.pad_w);
    }
}

void PoolingImpl::exec_w2x2_s2x2_int8(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto N = src.layout.shape[0], C = src.layout.shape[1];
    auto IH = src.layout.shape[2], IW = src.layout.shape[3];
    auto OH = dst.layout.shape[2], OW = dst.layout.shape[3];
    for (size_t nc = 0; nc < N * C; ++nc) {
        pooling::w2x2_s2x2_int8(
                src.ptr<dt_int8>() + nc * IH * IW, dst.ptr<dt_int8>() + nc * OH * OW,
                IH, IW, OH, OW);
    }
}

void PoolingImpl::exec_w2x2_s2x2_avg_int8(
        _megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto N = src.layout.shape[0], C = src.layout.shape[1];
    auto IH = src.layout.shape[2], IW = src.layout.shape[3];
    auto OH = dst.layout.shape[2], OW = dst.layout.shape[3];
    for (size_t nc = 0; nc < N * C; ++nc) {
        pooling::w2x2_s2x2_avg_int8(
                src.ptr<dt_int8>() + nc * IH * IW, dst.ptr<dt_int8>() + nc * OH * OW,
                IH, IW, OH, OW);
    }
}

void PoolingImpl::exec_fallback(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    Param param = this->param();
    check_exec(src.layout, dst.layout, workspace.size);
    if (src.layout.dtype == dtype::Float32() && param.format == Param::Format::NCHW &&
        param.mode == Mode::MAX && param.window_h == 3 && param.window_w == 3 &&
        param.stride_h == 1 && param.stride_w == 1 && param.pad_h <= 2 &&
        param.pad_w <= 2) {
        MIDOUT_BEGIN(megdnn_fallback_pooling, midout_iv(0)) {
            MEGDNN_DISPATCH_CPU_KERN_OPR(exec_w3x3_s1x1(src, dst, param));
        }
        MIDOUT_END();
        return;
    }
    // regular int conv case
    if (src.layout.dtype == dtype::Int8() && param.mode == Mode::MAX &&
        param.format == Param::Format::NCHW && param.window_h == 2 &&
        param.window_w == 2 && param.stride_h == 2 && param.stride_w == 2 &&
        param.pad_h == 0 && param.pad_w == 0) {
        MIDOUT_BEGIN(megdnn_fallback_pooling, midout_iv(1)) {
            MEGDNN_DISPATCH_CPU_KERN_OPR(exec_w2x2_s2x2_int8(src, dst));
        }
        MIDOUT_END();
        return;
    }
    // int8 2x2 AVERAGE case
    if (src.layout.dtype == dtype::Int8() && param.mode == Mode::AVERAGE &&
        param.format == Param::Format::NCHW && param.window_h == 2 &&
        param.window_w == 2 && param.stride_h == 2 && param.stride_w == 2 &&
        param.pad_h == 0 && param.pad_w == 0) {
        MIDOUT_BEGIN(megdnn_fallback_pooling, midout_iv(2)) {
            MEGDNN_DISPATCH_CPU_KERN_OPR(exec_w2x2_s2x2_avg_int8(src, dst));
        }
        MIDOUT_END();
        return;
    }
    // fallback to naive
    naive::PoolingForwardImpl::exec(src, dst, workspace);
}

// vim: syntax=cpp.doxygen
