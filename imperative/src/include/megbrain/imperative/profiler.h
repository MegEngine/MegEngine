/**
 * \file imperative/src/include/megbrain/imperative/profiler.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/comp_node.h"
#include "megbrain/graph/event.h"
#include "megbrain/utils/json.h"
#include "megbrain/utils/timer.h"

#include "megbrain/imperative/op_def.h"

namespace mgb {
namespace imperative {

class ProfilerPrivate;

using OpDefPrinter = thin_function<std::string(const OpDef&)>;

class Profiler {
private:
    std::unique_ptr<ProfilerPrivate> m_private;

public:
    enum EventKind { OprBegin, OprEnd };

public:
    Profiler();
    Profiler(const std::string& path);
    ~Profiler();
    void enable();
    void disable();
    void dump();
    void dump(const std::string& path);
    void record_host(size_t id, std::string name, EventKind type,
                     double host_time);
    void record_device(size_t id, std::string name, EventKind type,
                       double host_time, CompNode comp_node);
    double get_device_time(CompNode::Event& event);
    size_t get_dump_count();
    std::unique_ptr<CompNode::Event> create_event(CompNode comp_node);
    double get_host_time_now();
    std::string print_op(const OpDef& def);
};
}  // namespace imperative
}  // namespace mgb
