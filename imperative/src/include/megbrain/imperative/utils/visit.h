#pragma once

#include <vector>

#include "megbrain/utils/small_vector.h"

namespace mgb::imperative {

template <typename... TVisitors>
class Visitor : public TVisitors... {
public:
    using TVisitors::operator()...;
};

}  // namespace mgb::imperative
