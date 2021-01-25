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

struct TensorInfo;
using TensorInfoPtr = std::shared_ptr<TensorInfo>;

struct TensorInfo {
    enum Prop {
        Device, Shape, DType, DevValue, HostValue
    };

    uint64_t id;
    TensorPtr ptr;
    LogicalTensorDesc desc;

    // FIXME: broken by drop
    bool value_fetched = false;
    bool invalid = false;
    bool allow_delete = false;

    EvictType evict_type = NONE;

    HostTensorND h_value;

    // reserved for auto drop
    size_t pinned = 0;
    size_t recompute_times = 0;

    struct ComputePath {
        std::shared_ptr<OpDef> op;
        SmallVector<TensorInfo*> inputs;
        SmallVector<TensorInfo*> unique_inputs;
        SmallVector<TensorInfo*> outputs;

        size_t ref_cnt() {
            return outputs.size() - std::count(outputs.begin(), outputs.end(), nullptr);
        }

        static ComputePath* make(std::shared_ptr<OpDef> op, SmallVector<TensorInfo*> inputs, SmallVector<TensorInfo*> outputs) {
            auto* path = new TensorInfo::ComputePath();
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
            return path;
        }
    }* producer = nullptr;

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
