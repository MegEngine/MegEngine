#pragma once
#include <unordered_map>
#include "megdnn/oprs/base.h"
#include "src/fallback/pooling/opr_impl.h"

namespace megdnn {
namespace arm_common {

class PoolingImpl final : public fallback::PoolingImpl {
private:
    //! TODO: remove
    //! AlgoFilterxModexStride1/AlgoFilter2ModexStride2
    //! AlgoFilter3AverageStride2/AlgoFilter4MaxStride2/AlgoFilter5MaxStride2
    //! after imp gi with float16 and int8 support to dnn/src/fallback/pooling/opr_impl.h
    class AlgoFilterxModexStride1;
    class AlgoFilter2ModexStride2;
    class AlgoFilter3MaxStride2;
    class AlgoFilter3AverageStride2;
    class AlgoFilter4MaxStride2;
    class AlgoFilter5MaxStride2;
    class AlgoInt8Filter2MaxStride2;
    class AlgoInt8Filter3MaxStride2;
    class AlgoFilter2ModexStridexNCHW44;
    class AlgoFilter3ModexStridexNCHW44;
    class AlgoFilter4ModexStridexNCHW44;
    class AlgoFilter5ModexStridexNCHW44;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    class AlgoFilterxModexStridexNCHW88;
#endif
    class AlgoFallback;
    class AlgoPack;
    static AlgoPack sm_algo_pack;

public:
    using fallback::PoolingImpl::PoolingImpl;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override;

    static size_t constexpr MAX_SPATIAL_DIM = 2;

    using PoolingKernSizeParam = fallback::PoolingImpl::PoolingKernSizeParam;

    using PoolingKernParam = fallback::PoolingImpl::PoolingKernParam;

    class AlgoBase : public detail::Algorithm {
    public:
        enum class AlgoType : uint32_t {
            ARM_FilterxModexStride1,
            ARM_Filter2ModexStride2,
            ARM_Filter3MaxStride2,
            ARM_Filter3AverageStride2,
            ARM_Filter4MaxStride2,
            ARM_Filter5MaxStride2,
            ARM_Int8Filter2MaxStride2,
            ARM_Int8Filter3MaxStride2,
            ARM_Filter2ModexStridexNCHW44,
            ARM_Filter3ModexStridexNCHW44,
            ARM_Filter4ModexStridexNCHW44,
            ARM_Filter5ModexStridexNCHW44,
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            ARM_Fp16FilterxModexStridexNCHW88,
#endif
            ARM_Fp32ModexStridexNCHW44,
            ARM_Fallback
        };

        using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;
        AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::ARM_COMMON; }
        virtual ~AlgoBase() = default;
        virtual bool usable(const PoolingKernSizeParam& param) const = 0;
        virtual void exec(const PoolingKernParam& param) const = 0;

        uint32_t type() const override { return INVALID_ALGO_TYPE; };
        bool is_available_attribute(
                const PoolingKernSizeParam& param,
                const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
                const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) {
            return contain_attribute_all(positive_attr) &&
                   !contain_attribute_any(negative_attr) && usable(param);
        }
    };

    const char* get_algorithm_set_name() const override {
        return "ARM_POOLING_FORWARD";
    }

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc&) override;

    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& dst) override;
    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& src, const TensorLayout& dst) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& src, const TensorLayout& dst,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& src, const TensorLayout& dst,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) {
        return get_algorithm_heuristic(
                       src, dst, workspace_limit_in_bytes, positive_attr, negative_attr)
                ->info();
    }

    static const AlgoPack& algo_pack() { return sm_algo_pack; }

    bool is_fallback_algo(Algorithm* algo) {
        return strcmp(algo->name(), "FALLBACK_POOLING") == 0;
    }
};
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
