/**
 * \file dnn/src/common/algo_base.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/algo_base.h"
#include "src/common/utils.h"

using namespace megdnn;

#define FOREACH_ALGO_ATTRIBUTE(cb) \
    cb(DEFAULT)                    \
    cb(REPRODUCIBLE)               \
    cb(NAIVE)

namespace {
inline const char* attr_str(const AlgoAttribute& attr) {
#define cb(attr)              \
    case AlgoAttribute::attr: \
        return #attr;
    switch (attr) { FOREACH_ALGO_ATTRIBUTE(cb) }
#undef cb
    return "UNKNOWN";
}
}  // namespace

std::string Algorithm::attribute_str(const Attribute& attr) {
    std::string ret;
    uint32_t attr_val = static_cast<uint32_t>(attr);
    while(attr_val) {
        uint32_t mask = ~(attr_val & (attr_val - 1));
        Attribute sub_attr = static_cast<Attribute>(mask & attr_val);
        if (!ret.empty()) {
            ret.append(" | ");
        }
        ret.append(attr_str(sub_attr));
        attr_val = attr_val & (attr_val - 1);
    }
    if (ret.empty()) {
        ret = "DEFAULT";
    }
    return ret;
}

bool Algorithm::contain_attribute_all(const Attribute& attr) const {
    return attr == static_cast<Attribute>(attribute() & attr);
}

bool Algorithm::contain_attribute_any(const Attribute& attr) const {
    return static_cast<bool>(attribute() & attr);
}

void Algorithm::check_attribute(const Attribute& positive_attr,
                                const Attribute& negative_attr) const {
    megdnn_assert(contain_attribute_all(positive_attr) &&
                          !contain_attribute_any(negative_attr),
                  "require algorithm with attribute(%s) and without "
                  "attribute(%s), but get"
                  "algorithm(%s) with attribute(%s) ",
                  Algorithm::attribute_str(positive_attr).c_str(),
                  Algorithm::attribute_str(negative_attr).c_str(), name(),
                  Algorithm::attribute_str(attribute()).c_str());
}

// vim: syntax=cpp.doxygen
