/**
 * \file imperative/src/test/helper.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
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
