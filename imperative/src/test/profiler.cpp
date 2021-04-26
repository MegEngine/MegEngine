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

#include "megbrain/imperative/profiler.h"
#include "../impl/profiler/events.h"

using namespace mgb;
using namespace cg;
using namespace imperative;

namespace mgb { void imperative_log_profile(const char* message); }

TEST(TestProfiler, ImperativeLogProfile) {
    imperative::Profiler::start_profile();
    imperative_log_profile("XXX");
    auto results = imperative::Profiler::collect();
    imperative::Profiler::stop_profile();
    mgb_assert(results.size() == 2);
    auto* event_start = std::any_cast<profiler::CustomEvent>(&results[0].second.data);
    auto* event_finish = std::any_cast<profiler::CustomFinishEvent>(&results[1].second.data);
    mgb_assert(event_start && event_start->title == "XXX");
    mgb_assert(event_finish && event_finish->title == "XXX");
    mgb_assert(results[0].second.time < results[1].second.time);
    mgb_assert(results[0].second.id < results[1].second.id);
}
