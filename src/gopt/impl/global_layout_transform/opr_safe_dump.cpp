/**
 * \file src/gopt/impl/global_layout_transform/opr_safe_dump.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./opr_safe_dump.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_manip.h"

#include "midout.h"
MIDOUT_DECL(megbrain_opr_safe_dump)
#define MIDOUT_B(...) MIDOUT_BEGIN(megbrain_opr_safe_dump, __VA_ARGS__) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace opr;

namespace {
template <typename Param>
void write_param(std::string& data, const Param& param) {
    megdnn::Algorithm::serialize_write_pod(param, data);
}

template <>
void write_param(std::string& /* data */, const DType& /* dtype */) {}

template <class Opr>
struct OprDumpImpl {
    static std::string dump(const cg::OperatorNodeBase* opr_) {
        MIDOUT_B(Opr)
        auto&& opr = opr_->cast_final_safe<Opr>();
        std::string data;
        write_param(data, opr.param());
        return data;
        MIDOUT_E
    }
};

#define INST(_Opr)                                                     \
    template <>                                                        \
    struct OprDumpImpl<_Opr> {                                         \
        static std::string dump(const cg::OperatorNodeBase* opr_) {    \
            MIDOUT_B(_Opr)                                             \
            auto&& opr = opr_->cast_final_safe<_Opr>();                \
            std::string data;                                          \
            write_param(data, opr.param());                            \
            using ExecutionPolicy = megdnn::param::ExecutionPolicy;    \
            ExecutionPolicy policy{                                    \
                    opr.execution_policy_transient().strategy,         \
                    opr.execution_policy_transient().workspace_limit}; \
            write_param(data, policy);                                 \
            return data;                                               \
            MIDOUT_E                                                   \
        }                                                              \
    };
INST(Convolution);
INST(ConvBiasForward);
INST(ConvolutionBackwardData);
INST(PoolingForward);
#undef INST
}  // namespace

namespace mgb {
namespace gopt {
namespace intl {

std::string opr_safe_dump(const cg::OperatorNodeBase* opr) {
#define cb(_Opr)                                   \
    if (opr->dyn_typeinfo() == _Opr::typeinfo()) { \
        return OprDumpImpl<_Opr>::dump(opr);       \
    } else
    FOREACH_SUPPORTED_OPR(cb) {
        mgb_throw(InternalError, "unsupported operator(got:%s)",
                  opr->dyn_typeinfo()->name);
    }
#undef cb
}

}  // namespace intl
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen
