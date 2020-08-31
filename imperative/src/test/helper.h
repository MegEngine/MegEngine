/**
 * \file imperative/src/test/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <variant>

#include "megbrain/imperative.h"
#include "megbrain/test/helper.h"

namespace mgb {
namespace imperative {

class OprChecker {
public:
     using InputSpec = std::variant<HostTensorND, TensorShape>;
     OprChecker(std::shared_ptr<OpDef> opdef);
     void run(std::vector<InputSpec> inp_shapes);
private:
     std::shared_ptr<OpDef> m_op;
};

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
