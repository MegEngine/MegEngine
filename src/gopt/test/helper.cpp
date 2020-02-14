/**
 * \file src/gopt/test/helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"

#include "megbrain/serialization/sereg.h"
#include "megbrain/opr/utility.h"

using namespace mgb;

namespace {

MGB_DEFINE_OPR_CLASS(OprReaderForTest, opr::intl::ForwardInputToOutput) // {
public:
    OprReaderForTest(VarNode* input, const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar input,
                          const OperatorNodeConfig& config = {});
};

cg::OperatorNodeBase* opr_shallow_copy_opr_reader_for_test(
        const serialization::OprShallowCopyContext &ctx,
        const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
        const OperatorNodeConfig &config) {
    mgb_assert(inputs.size() == 1);
    return OprReaderForTest::make(inputs[0], config).node()->owner_opr();
}

MGB_REG_OPR_SHALLOW_COPY(OprReaderForTest,
                         opr_shallow_copy_opr_reader_for_test);

}  // anonymous namespace

MGB_DYN_TYPE_OBJ_FINAL_IMPL(OprReaderForTest);

OprReaderForTest::OprReaderForTest(VarNode* input,
                                   const OperatorNodeConfig& config)
        : Super(input->owner_graph(), config, "opr_reader", {input}) {
    add_input({input});
    add_output(None);
}

SymbolVar OprReaderForTest::make(SymbolVar input,
                                 const OperatorNodeConfig& config) {
    return input.insert_single_output_opr<OprReaderForTest>(input.node(),
                                                            config);
}

SymbolVar mgb::opr_reader_for_test(SymbolVar x) {
    return OprReaderForTest::make(x);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
