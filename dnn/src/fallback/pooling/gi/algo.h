#pragma once
#include "src/common/utils.h"
#include "src/fallback/pooling/opr_impl.h"

#include "pooling_helper.h"

#include "src/naive/handle.h"
#include "src/naive/pooling/opr_impl.h"

namespace megdnn {
namespace fallback {

using AlgoBase = PoolingImpl::AlgoBase;

class PoolingImpl::AlgoGiFilterxModexStride1 final : public AlgoBase {
public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; };
    const char* name() const override { return "GI_POOLING_STRIDE1"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
    MEGDNN_DECL_ALGO_TYPE(GI_FilterxModexStride1)
};

class PoolingImpl::AlgoGiFilter2ModexStride2 final : public AlgoBase {
public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; };
    const char* name() const override { return "GI_POOLING_STRIDE2"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
    MEGDNN_DECL_ALGO_TYPE(GI_Filter2ModexStride2)
};
class PoolingImpl::AlgoGiFilter3MaxStride2 final : public AlgoBase {
public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; };
    const char* name() const override { return "GI_POOLING_FILTER3_MAX"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
    MEGDNN_DECL_ALGO_TYPE(GI_Filter3MaxStride2)
};

class PoolingImpl::AlgoGiFilter3AverageStride2 final : public AlgoBase {
public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; };
    const char* name() const override { return "GI_POOLING_FILTER3_AVERAGE"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
    MEGDNN_DECL_ALGO_TYPE(GI_Filter3AverageStride2)
};

class PoolingImpl::AlgoGiFilter4MaxStride2 final : public AlgoBase {
public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; };
    const char* name() const override { return "GI_POOLING_FILTER4_MAX"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
    MEGDNN_DECL_ALGO_TYPE(GI_Filter4MaxStride2)
};

class PoolingImpl::AlgoGiFilter5MaxStride2 final : public AlgoBase {
public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; };
    const char* name() const override { return "GI_POOLING_FILTER5_MAX"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
    MEGDNN_DECL_ALGO_TYPE(GI_Filter5MaxStride2)
};

class PoolingImpl::AlgoGiFp32ModexStridexNCHW44 final : public AlgoBase {
public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; };
    const char* name() const override { return "GI_POOLING_FP32_MODEX_STRIDEX_NCHW44"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
    MEGDNN_DECL_ALGO_TYPE(GI_Fp32ModexStridexNCHW44)
};

class PoolingImpl::AlgoFallback final : public AlgoBase {
public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; };
    const char* name() const override { return "FALLBACK_NOT_GI_POOLING"; }
    bool usable(const PoolingKernSizeParam&) const override { return true; }
    void exec(const PoolingKernParam& /*param*/) const override {
        megdnn_assert(false, "code issue happened!!");
    }
    MEGDNN_DECL_ALGO_TYPE(FallbackNotGI)
};
WorkspaceBundle get_bundle(const PoolingImpl::PoolingKernSizeParam&);

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
