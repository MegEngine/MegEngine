#include <typeindex>

#include "megbrain/imperative/transformation.h"
#include "megbrain/imperative/utils/debug.h"
#include "megbrain/imperative/value.h"

namespace mgb::imperative::debug {

const char* get_type_name(const std::type_info& type) {
    return type.name();
}

const char* get_type_name(const std::type_index& type) {
    return type.name();
}

void notify_event(const char* event) {}

void watch_value(ValueRef value) {
    value.watch();
}

}  // namespace mgb::imperative::debug