/**
 * \file dnn/src/arm_common/pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/pooling/opr_impl.h"
#include "src/arm_common/pooling/algo.h"
#include "src/common/metahelper.h"
#include "src/common/algo_chooser.h"

using namespace megdnn;
using namespace arm_common;

class PoolingImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;
    AlgoFilterxModexStride1 algo_filterx_modex_stride1;
    AlgoFilter2ModexStride2 algo_filter2_modex_stride2;
    AlgoFilter3MaxStride2 algo_filter3_max_stride2;
    AlgoFilter3AverageStride2 algo_filter3_average_stride2;
    AlgoFilter4MaxStride2 algo_filter4_max_stride2;
    AlgoFilter5MaxStride2 algo_filter5_max_stride2;
    AlgoInt8Filter2MaxStride2 algo_int8_filter2_max_stride2;
    AlgoInt8Filter3MaxStride2 algo_int8_filter3_max_stride2;
    AlgoFilter2ModexStridexNCHW44 algo_filter2_modex_stridex_nchw4;
    AlgoFilter3ModexStridexNCHW44 algo_filter3_modex_stridex_nchw4;
    AlgoFilter4ModexStridexNCHW44 algo_filter4_modex_stridex_nchw4;
    AlgoFilter5ModexStridexNCHW44 algo_filter5_modex_stridex_nchw4;
    AlgoFp32ModexStridexNCHW44 algo_fp32_modex_stridex_nchw44;
    AlgoFallback algo_fallback;

public:
    AlgoPack() {
        all_algos.emplace_back(&algo_filterx_modex_stride1);
        all_algos.emplace_back(&algo_filter2_modex_stride2);
        all_algos.emplace_back(&algo_filter3_max_stride2);
        all_algos.emplace_back(&algo_filter3_average_stride2);
        all_algos.emplace_back(&algo_filter4_max_stride2);
        all_algos.emplace_back(&algo_filter5_max_stride2);
        all_algos.emplace_back(&algo_int8_filter2_max_stride2);
        all_algos.emplace_back(&algo_int8_filter3_max_stride2);
        all_algos.emplace_back(&algo_filter3_modex_stridex_nchw4);
        all_algos.emplace_back(&algo_filter2_modex_stridex_nchw4);
        all_algos.emplace_back(&algo_filter4_modex_stridex_nchw4);
        all_algos.emplace_back(&algo_filter5_modex_stridex_nchw4);
        all_algos.emplace_back(&algo_fp32_modex_stridex_nchw44);
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
        fallback::PoolingImpl* opr, const TensorLayout& src,
        const TensorLayout& dst) {
    auto safe_u32 = [](size_t v) -> uint32_t {
        megdnn_assert(v <= std::numeric_limits<uint32_t>::max(),
                      "value too large: %zu", v);
        return v;
    };
    return {safe_u32(src.shape[0]),
            safe_u32(src.shape[1]),
            {{safe_u32(src.shape[2]), safe_u32(src.shape[3])}},
            {{safe_u32(dst.shape[2]), safe_u32(dst.shape[3])}},
            {{safe_u32(opr->param().pad_h), safe_u32(opr->param().pad_w)}},
            {{safe_u32(opr->param().window_h),
              safe_u32(opr->param().window_w)}},
            {{safe_u32(opr->param().stride_h),
              safe_u32(opr->param().stride_w)}},
            src.dtype,
            dst.dtype,
            opr->handle(),
            opr->param().format,
            opr->param().mode};
};

PoolingImpl::PoolingKernParam PoolingImpl::make_pooling_kern_param(
        fallback::PoolingImpl* opr, _megdnn_tensor_in src,
        _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    PoolingKernParam ret;
    static_cast<PoolingKernSizeParam&>(ret) =
            make_pooling_kern_szie_param(opr, src.layout, dst.layout);
    ret.src_ptr = src.raw_ptr;
    ret.dst_ptr = dst.raw_ptr;
    ret.workspace_ptr = workspace.raw_ptr;
    ret.workspace_size = workspace.size;
    return ret;
};

size_t PoolingImpl::get_workspace_in_bytes(const TensorLayout& src,
                                           const TensorLayout& dst) {
    auto param = make_pooling_kern_szie_param(this, src, dst);
    auto algo = get_algorithm(this, src, dst);
    if (!is_fallback_algo(algo)) {
        size_t arm_common_workspace = 0;

        //! When multi-thread, every thread has its own workspace
        size_t nr_threads = static_cast<naive::HandleImpl*>(handle())
                                    ->megcore_dispatcher()
                                    ->nr_threads();
        if ((param.src_type.category() == DTypeCategory::FLOAT ||
             param.src_type == dtype::Int8{} ||
             param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
             param.src_type.enumv() == DTypeEnum::Quantized8Asymm) &&
            param.filter[0] == param.filter[1] &&
            (param.filter[0] == 3 || param.filter[0] == 5) &&
            param.format == Param::Format::NCHW &&
            (param.mode == Mode::MAX ||
             (param.mode == Mode::AVERAGE && param.filter[0] == 3)) &&
            param.stride[0] == 2 && param.stride[1] == 2 && param.isz[0] >= 2 &&
            param.isz[1] >= 2) {
            WorkspaceBundle ws = get_bundle(param);
            arm_common_workspace = ws.total_size_in_bytes() * nr_threads;
        }

        if ((param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
             param.src_type.enumv() == DTypeEnum::Int8) &&
            (param.format == param::Pooling::Format::NCHW44)) {
            WorkspaceBundle ws = get_bundle_nchw44(param);
            arm_common_workspace = ws.total_size_in_bytes() * nr_threads;
        }
        return arm_common_workspace;
    } else {
        auto fallback_worksapce =
                fallback::PoolingImpl::get_workspace_in_bytes(src, dst);
        return fallback_worksapce;
    }
}

void PoolingImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                       _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    auto param = make_pooling_kern_param(this, src, dst, workspace);
    auto algo = get_algorithm(this, src.layout, dst.layout);
    if (!is_fallback_algo(algo)) {
        algo->exec(param);
    } else {
        fallback::PoolingImpl::exec(src, dst, workspace);
    }
}

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
    megdnn_assert(!ret.empty(), "no usable pooling fwd algorithm");
    return ret;
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
    megdnn_throw(
            ssprintf("require algorithm with attribute(%s) and without "
                     "attribute(%s), but can't get suitable algo.\n",
                     Algorithm::attribute_str(positive_attr).c_str(),
                     Algorithm::attribute_str(negative_attr).c_str()));
    return nullptr;
}

// vim: syntax=cpp.doxygen
