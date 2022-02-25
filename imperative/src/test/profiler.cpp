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
    auto& event_start = results.entries[0].data.cast<profiler::CustomEvent>();
    auto& event_finish = results.entries[1].data.cast<profiler::CustomFinishEvent>();
    mgb_assert(event_start.title == "XXX");
    mgb_assert(event_finish.title == "XXX");
    mgb_assert(results.entries[0].time < results.entries[1].time);
    mgb_assert(results.entries[0].id < results.entries[1].id);
}
