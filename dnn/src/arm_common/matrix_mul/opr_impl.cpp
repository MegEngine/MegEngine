#include "src/arm_common/matrix_mul/opr_impl.h"
#include "src/arm_common/matrix_mul/algos.h"
#include "src/common/metahelper.h"

using namespace megdnn;
using namespace arm_common;

class MatrixMulImpl::AlgoPack : NonCopyableObj {
    AlgoInt8x8x16 int8x8x16;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    AlgoF16Gemv f16gemv;
#endif
    AlgoInt8x8x32Gemv int8x8x32_gemv;
    AlgoInt8x8x32GemvMK4 int8x8x32_gemv_mk4;
#if MGB_ENABLE_DOT
    AlgoInt8x8x32GemvMK4Dot int8x8x32_gemv_mk4_dot;
    AlgoInt8x8x32GevmDot int8x8x32_gevm_dot;
    AlgoInt8x8x32GevmN32K4Dot int8x8x32_gevm_n32k4_dot;
#endif
    AlgoGevm gevm;

    SmallVector<fallback::MatrixMulImpl::AlgoBase*> m_all_algos;
    fallback::MatrixMulImpl::AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack() {
        m_all_algos.emplace_back(&int8x8x16);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        m_all_algos.emplace_back(&f16gemv);
#endif
#if MGB_ENABLE_DOT
        m_all_algos.emplace_back(&int8x8x32_gemv_mk4_dot);
        m_all_algos.emplace_back(&int8x8x32_gevm_dot);
        m_all_algos.emplace_back(&int8x8x32_gevm_n32k4_dot);
#endif
        m_all_algos.emplace_back(&int8x8x32_gemv);
        m_all_algos.emplace_back(&int8x8x32_gemv_mk4);
        m_all_algos.emplace_back(&gevm);

        for (auto&& algo : m_all_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
    }

    const SmallVector<fallback::MatrixMulImpl::AlgoBase*>& all_algos() const {
        return m_all_algos;
    }
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

const MatrixMulImpl::AlgoPack& MatrixMulImpl::algo_pack() {
    static AlgoPack algo_pack;
    return algo_pack;
}

MEGDNN_FB_DEF_GET_ALGO_FROM_DESC(MatrixMulImpl)

SmallVector<fallback::MatrixMulImpl::AlgoBase*> MatrixMulImpl::get_all_packed_algo() {
    static AlgoPack s_algo_pack;
    auto&& algos = fallback::MatrixMulImpl::get_all_packed_algo();
    algos.insert(
            algos.begin(), algo_pack().all_algos().begin(),
            algo_pack().all_algos().end());
    return std::move(algos);
}

// vim: syntax=cpp.doxygen
