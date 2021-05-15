/**
 * \file imperative/src/impl/interpreter/tensor_info.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/physical_tensor.h"
#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/utils/to_string.h"

namespace mgb::imperative {

namespace interpreter::intl {

enum EvictType {
    NONE = 0,
    SWAP = 1,
    DROP = 2,
};

/*!
 * \brief an identifier to specify a component of evicted tensors
 * 
 * Each component tracks the sum of the compute costs of its elements, with the
 * union of two components having the sum of each constituent cost.
 */
struct DsuNode {
    DsuNode(double _t): t(_t) {}
    
    std::shared_ptr<DsuNode> parent;

    bool is_root() {
        return !bool(parent);
    }
    
    double t;
};

struct TensorInfo;
using TensorInfoPtr = std::shared_ptr<TensorInfo>;

struct TensorInfo {
    enum Prop {
        Device, Shape, DType, DevValue, HostValue
    };

    uint64_t id;
    TensorPtr ptr;
    LogicalTensorDesc desc;

    double compute_time;
    size_t memory;
    double last_used_time;
    
    // FIXME: broken by drop
    bool value_fetched = false;
    bool invalid = false;
    bool allow_delete = false;

    EvictType evict_type = NONE;

    HostTensorND h_value;

    // reserved for auto drop
    size_t pinned = 0;
    size_t recompute_times = 0;
    size_t ref_cnt = 0;
    std::shared_ptr<DsuNode> dsu_ptr;

    struct ComputePath {
        uint64_t id;
        std::shared_ptr<OpDef> op;
        SmallVector<TensorInfo*> inputs;
        SmallVector<TensorInfo*> unique_inputs;
        SmallVector<TensorInfo*> outputs;

        size_t ref_cnt() {
            return outputs.size() - std::count(outputs.begin(), outputs.end(), nullptr);
        }

        static ComputePath* make(uint64_t id, std::shared_ptr<OpDef> op, SmallVector<TensorInfo*> inputs, SmallVector<TensorInfo*> outputs) {
            auto* path = new TensorInfo::ComputePath();
            path->id = id;
            path->op = op;
            path->inputs = inputs;
            path->outputs = outputs;
            // dedup
            SmallVector<TensorInfo*> unique_inputs = inputs;
            std::sort(unique_inputs.begin(), unique_inputs.end());
            unique_inputs.erase(std::unique(unique_inputs.begin(), unique_inputs.end()), unique_inputs.end());
            path->unique_inputs = unique_inputs;
            // attach users
            for (auto input: unique_inputs) {
                input->users.push_back(path);
            }
            // attach producer
            for (auto output: outputs) {
                output->producer = path;
            }
            // update ref_cnt
            for (auto input: inputs) {
                input->ref_cnt += outputs.size();
            }
            return path;
        }
    }* producer = nullptr;
  
    double eval_func(double cost, double free_mem, double cur_time,
                     double param_cost, double param_mem, double param_time, double param_recompute_times) {
        return pow(cost + 1e-3, param_cost) * pow(param_recompute_times, (double)recompute_times)
               / (pow((memory + free_mem) / 1024.0 / 1024.0, param_mem) * pow((double)(cur_time - last_used_time + 1e-3), param_time));
    }

    void pin() {
        ++pinned;
    }

    void unpin() {
        --pinned;
    }

    void detach_producer() {
        if (!producer) {
            return;
        }
        auto output = std::find(producer->outputs.begin(), producer->outputs.end(), this);
        mgb_assert(output != producer->outputs.end());
        *output = nullptr;
        if (producer->ref_cnt() == 0) {
            for (auto* input: producer->unique_inputs) {
                input->users.erase(std::find(input->users.begin(), input->users.end(), producer));
            }
            delete producer;
        }
        producer = nullptr;
    }

    bool size_exceeds_thd(size_t thd) {
        return memory > thd;
    }

    SmallVector<ComputePath*> users;
};
}

template <>
struct ToStringTrait<interpreter::intl::TensorInfo::Prop>{
    using TensorInfo = interpreter::intl::TensorInfo;

    std::string operator()(TensorInfo::Prop prop) const {
        switch(prop) {
        case TensorInfo::DType:
            return "dtype";
        case TensorInfo::DevValue:
            return "dev_value";
        case TensorInfo::Device:
            return "device";
        case TensorInfo::HostValue:
            return "host_value";
        case TensorInfo::Shape:
            return "shape";
        default:
            return "unknown";
        }
    }
};

}
