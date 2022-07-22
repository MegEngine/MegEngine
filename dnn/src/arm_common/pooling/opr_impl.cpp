#include "src/arm_common/pooling/opr_impl.h"
#include "src/arm_common/pooling/algo.h"
#include "src/common/algo_chooser.h"
#include "src/common/metahelper.h"
#include "src/common/opr_delegate.h"

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
        all_algos.emplace_back(&algo_fallback);

        for (auto&& algo : all_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
    }
    SmallVector<AlgoBase*> all_algos;
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

PoolingImpl::AlgoPack PoolingImpl::sm_algo_pack;
namespace {
TensorLayout merge_hw_layout(TensorLayout src) {
    src.ndim -= 1;
    src.shape[2] = src.shape[2] * src.shape[3];
    src.stride[2] = src.stride[3];
    for (size_t i = 3; i < src.ndim; ++i) {
        src.shape[i] = src.shape[i + 1];
        src.stride[i] = src.stride[i + 1];
    }
    return src;
}
std::pair<TensorND, TensorND> get_gloabl_pooling_reduce_tensor(
        const TensorND& src, const TensorND& dst) {
    auto reduce_src_layout = merge_hw_layout(src.layout);
    auto reduce_dst_layout = merge_hw_layout(dst.layout);
    return std::make_pair<TensorND, TensorND>(
            {src.raw_ptr(), reduce_src_layout}, {dst.raw_ptr(), reduce_dst_layout});
}
std::unique_ptr<Reduce> get_global_pooling_reduce_opr(
        Handle* handle, const PoolingImpl::PoolingKernSizeParam& param) {
    std::unique_ptr<Reduce> opr;
    if (handle) {
        opr = handle->create_operator<Reduce>();
    } else {
        opr = inplace_cpu_handle()->create_operator<Reduce>();
    }
    param::Reduce reduce_param;
    reduce_param.axis = 2;
    if (param.mode == PoolingImpl::Param::Mode::MAX) {
        reduce_param.mode = param::Reduce::Mode::MAX;
    } else {
        megdnn_assert(
                param.mode == PoolingImpl::Param::Mode::AVERAGE ||
                param.mode == PoolingImpl::Param::Mode::AVERAGE_COUNT_EXCLUDE_PADDING);
        reduce_param.mode = param::Reduce::Mode::MEAN;
    }
    opr->param() = reduce_param;
    return opr;
}
bool is_global_pooling_reduce(PoolingImpl::PoolingKernSizeParam& param) {
    bool fmt_ok = param.format == PoolingImpl::Param::Format::NCHW ||
                  param.format == PoolingImpl::Param::Format::NCHW44 ||
                  param.format == PoolingImpl::Param::Format::NCHW88;
    bool size_ok = param.filter[0] == param.isz[0] && param.filter[1] == param.isz[1] &&
                   param.padding[0] == 0 && param.padding[1] == 0 &&
                   param.osz[0] == 1 && param.osz[1] == 1;
    bool dtype_ok = param.src_type == param.dst_type &&
                    param.src_type.enumv() != DTypeEnum::Int8;
    return fmt_ok && size_ok && dtype_ok;
}

}  // namespace
size_t PoolingImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    auto param = make_pooling_kern_szie_param(this, src, dst);
    bool fwd_reduce = is_global_pooling_reduce(param);
    if (fwd_reduce) {
        TensorND src_tensor{nullptr, src};
        TensorND dst_tensor{nullptr, dst};
        auto reduce_tensor = get_gloabl_pooling_reduce_tensor(src_tensor, dst_tensor);
        auto&& opr = get_global_pooling_reduce_opr(nullptr, param);
        auto reduce_need = opr->get_workspace_in_bytes(
                reduce_tensor.first.layout, reduce_tensor.second.layout);
        return reduce_need;
    }

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

void PoolingImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    auto param = make_pooling_kern_param(this, src, dst, workspace);

    bool fwd_reduce = is_global_pooling_reduce(param);
    if (fwd_reduce) {
        auto global_pooling_fwd = [=]() {
            auto reduce_tensor = get_gloabl_pooling_reduce_tensor(src, dst);
            auto&& opr = get_global_pooling_reduce_opr(nullptr, param);
            opr->exec(reduce_tensor.first, reduce_tensor.second, workspace);
        };
        MEGDNN_DISPATCH_CPU_KERN_OPR(global_pooling_fwd());
        return;
    }

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
    return ret;
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

// vim: syntax=cpp.doxygen
