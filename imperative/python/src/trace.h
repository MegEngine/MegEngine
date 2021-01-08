/**
 * \file imperative/python/src/trace.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./tensor.h"
#include <stdexcept>

namespace mgb::imperative::python {

class TraceReadError : public std::exception {
public:
    explicit TraceReadError(const char * m) : message{m} {}
    const char * what() const noexcept override {return message.c_str();}
private:
    std::string message = "";
};


apply_result_t apply_trace(ApplyContext& ctx);

} // namespace mgb::imperative::python
