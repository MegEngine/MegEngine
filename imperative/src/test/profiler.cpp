/**
 * \file imperative/src/test/profiler.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"

#include "../impl/profiler/events.h"
#include "megbrain/imperative/profiler.h"

using namespace mgb;
using namespace cg;
using namespace imperative;

namespace mgb {
void imperative_log_profile(const char* message);
}

TEST(TestProfiler, ImperativeLogProfile) {
    imperative::Profiler::start_profile();
    imperative_log_profile("XXX");
    auto results = imperative::Profiler::collect();
    imperative::Profiler::stop_profile();
    mgb_assert(results.entries.size() == 2);
    auto* event_start = results.entries[0].data.as<profiler::CustomEvent>();
    auto* event_finish = results.entries[1].data.as<profiler::CustomFinishEvent>();
    mgb_assert(event_start && event_start->title == "XXX");
    mgb_assert(event_finish && event_finish->title == "XXX");
    mgb_assert(results.entries[0].time < results.entries[1].time);
    mgb_assert(results.entries[0].id < results.entries[1].id);
}
