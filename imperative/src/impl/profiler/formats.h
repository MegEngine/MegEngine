#pragma once

#include <unordered_set>

#include "megbrain/imperative/profiler.h"

#include "./states.h"

namespace mgb::imperative::profiler {

void dump_chrome_timeline(std::string filename, Profiler::bundle_t result);

void dump_memory_flow(std::string filename, Profiler::bundle_t result);

}  // namespace mgb::imperative::profiler
