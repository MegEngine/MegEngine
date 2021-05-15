#pragma once

#include <set>
#include <any>
#include <typeindex>

#include "megbrain/tensor.h"

#include "./events.h"

namespace mgb::imperative::profiler {

struct ProfileDeviceState {
    int64_t index;
    CompNode device;
    std::shared_ptr<CompNode::Event> base_event;
    uint64_t base_time; //in ns
};

struct ProfileWorkerState {

};

struct ProfileTensorState {
    uint64_t id;
    TensorLayout layout;
    CompNode device;
    std::string name;
    uint64_t produced = 0;
    uint64_t living_time = 0;

    size_t size_in_bytes() const {
        if (!layout.dtype.valid()) {
            return 0;
        }
        return layout.dtype.size(layout.total_nr_elems());
    }
};

struct ProfileStaticsState {
    size_t op_enqueue_count = 0;
    size_t op_execute_count = 0;
    size_t wait_value_count = 0;
    size_t wait_shape_count = 0;
    size_t exception_count = 0;
    size_t infer_shape_valid_count = 0;
    size_t infer_shape_invalid_count = 0;
    size_t alive_tensor_count = 0;
    size_t produce_tensor_count = 0;
    size_t erase_tensor_count = 0;
    size_t wait_prop_count = 0;
    size_t redundant_tensor_count = 0;
};

struct ProfileOperatorState {
    uint64_t id;
    std::string name;
    OpParams params;
    SmallVector<uint64_t> inputs;
    SmallVector<uint64_t> outputs;
    CompNode device;

    uint64_t host_begin;
    uint64_t host_end;
    std::shared_ptr<CompNode::Event> device_begin;
    std::shared_ptr<CompNode::Event> device_end;
};

struct ProfileThreadState {
    std::thread::id tid;
    int64_t index;
    std::vector<std::string> scope_stack;
};

template <typename TProp>
struct ProfileTensorPropPair {
    uint64_t id;
    TProp value;

    bool operator<(const ProfileTensorPropPair& lhs) const {
        return value == lhs.value ? id < lhs.id : value < lhs.value;
    }

    bool operator==(const ProfileTensorPropPair& lhs) const {
        return id == lhs.id && value == lhs.value;
    }

    bool operator>(const ProfileTensorPropPair& lhs) const {
        return value == lhs.value ? id > lhs.id : value > lhs.value;
    }
};

using ProfileTensorSizePair = ProfileTensorPropPair<size_t>;
using ProfileTensorProducedPair = ProfileTensorPropPair<uint64_t>;

struct GeneralTensorEvent {
    uint64_t tensor_id;
    std::type_index type;
};

struct ProfileState {
    std::unordered_map<uint64_t, ProfileTensorState> tensors;
    std::unordered_map<uint64_t, ProfileOperatorState> operators;
    std::unordered_map<std::string, uint64_t> tensor_name_counter;
    std::set<ProfileTensorSizePair> tensors_by_size;
    std::set<ProfileTensorSizePair> tensors_by_produced;
    ProfileWorkerState worker;
    ProfileStaticsState statics;
    std::unordered_map<std::thread::id, ProfileThreadState> threads;
    CompNode::UnorderedMap<ProfileDeviceState> devices;

    ProfileThreadState& operator[](std::thread::id tid) {
        if (threads.count(tid) == 0) {
            threads[tid].tid = tid;
            threads[tid].index = threads.size();
        }
        return threads[tid];
    }

    ProfileDeviceState& operator[](CompNode device) {
        if (devices.count(device) == 0) {
            devices[device].device = device;
            devices[device].index = devices.size();
        }
        return devices[device];
    }

    std::vector<uint64_t> top_k_tensor_in_device(CompNode device, size_t k) {
        std::vector<uint64_t> results;
        for (auto iter = tensors_by_size.rbegin(); iter != tensors_by_size.rend(); ++iter) {
            if (!k) {
                break;
            }
            if (tensors[iter->id].device == device) {
                results.push_back(iter->id);
                --k;
            }
        }
        return results;
    }

    std::string concat_scope(std::thread::id tid) {
        auto& scope_stack = threads[tid].scope_stack;
        if (scope_stack.empty()) {
            return {};
        }
        std::string result = scope_stack[0];
        for (size_t i = 1; i < scope_stack.size(); ++i) {
            result += "::";
            result += scope_stack[i];
        }
        return result;
    }
};

}
