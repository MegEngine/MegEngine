/**
 * \file imperative/src/include/megbrain/imperative/ops/batch_norm.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/imperative/op_def.h"
#include "megbrain/utils/hash.h"

namespace mgb::imperative {

class BatchNorm : public OpDefImplBase<BatchNorm> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
public:
    using Param = opr::BatchNorm::Param;

    Param::ParamDim param_dim;
    Param::FwdMode fwd_mode;
    double epsilon;
    double avg_factor;
    float scale;
    float bias;

    BatchNorm() = default;
    
    BatchNorm(const Param::ParamDim& param_dim_, const Param::FwdMode& fwd_mode_, 
              double epsilon_, double avg_factor_, float scale_, float bias_)
            : param_dim(param_dim_),
              fwd_mode(fwd_mode_),
              epsilon(epsilon_),
              avg_factor(avg_factor_),
              scale(scale_),
              bias(bias_) {}

    size_t hash() const override {
        XXHash xxhash{};
        auto append = [&xxhash](auto field){
            auto hash_val = HashTrait<decltype(field)>::eval(field);
            xxhash.update(reinterpret_cast<void*>(&hash_val), sizeof(hash_val));
        };
        append(param_dim); 
        append(fwd_mode); 
        append(epsilon);
        append(avg_factor);
        append(scale);
        append(bias);
        return xxhash.digest();
    }

    bool is_same_st(const Hashable& rhs_) const override {
        auto&& rhs = static_cast<const BatchNorm&>(rhs_);
        return rhs.param_dim == param_dim
            && rhs.fwd_mode == fwd_mode
            && rhs.epsilon == epsilon
            && rhs.avg_factor == avg_factor
            && rhs.scale == scale
            && rhs.bias == bias;
    }

};

} // namespace mgb::imperative
