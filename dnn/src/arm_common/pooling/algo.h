/**
 * \file dnn/src/arm_common/pooling/algo.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "src/arm_common/pooling/opr_impl.h"
#include "src/arm_common/pooling/pooling_helper.h"
#include "src/common//utils.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace arm_common {

using AlgoBase = PoolingImpl::AlgoBase;

class PoolingImpl::AlgoFilterxModexStride1 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_STRIDE1"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

class PoolingImpl::AlgoFilter2ModexStride2 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_STRIDE2"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};
class PoolingImpl::AlgoFilter3MaxStride2 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_FILTER3_MAX"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

class PoolingImpl::AlgoFilter3AverageStride2 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_FILTER3_AVERAGE"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

class PoolingImpl::AlgoFilter4MaxStride2 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_FILTER4_MAX"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

class PoolingImpl::AlgoFilter5MaxStride2 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_FILTER5_MAX"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

class PoolingImpl::AlgoInt8Filter2MaxStride2 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_INT8_FILTER2X2"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

class PoolingImpl::AlgoInt8Filter3MaxStride2 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_INT8_FILTER3X3"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

class PoolingImpl::AlgoFilter3MaxStride2NCHW44 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_FILTER3_MAX_STRIDE2_NCHW44"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

class PoolingImpl::AlgoFilter3MaxStride1NCHW44 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_FILTER3_MAX_STRIDE1_NCHW44"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

class PoolingImpl::AlgoFilter2MaxStridexNCHW44 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_FILTER2_MAX_STRIDEX_NCHW44"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

class PoolingImpl::AlgoFilter4MaxStridexNCHW44 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_FILTER4_MAX_STRIDEX_NCHW44"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

class PoolingImpl::AlgoFilter5MaxStridexNCHW44 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_POOLING_FILTER5_MAX_STRIDEX_NCHW44"; }
    bool usable(const PoolingKernSizeParam& param) const override;
    void exec(const PoolingKernParam& param) const override;
};

WorkspaceBundle get_bundle(const PoolingImpl::PoolingKernSizeParam& param);

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen

