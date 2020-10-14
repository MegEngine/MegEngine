/**
 * \file src/core/test/comp_node_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/comp_node.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/test/helper.h"
#include "megbrain/opr/io.h"

using namespace mgb;

namespace mgb {
namespace comp_node_test {

template <typename Opr>
HostTensorND eval_conv(const std::shared_ptr<HostTensorND>& src,
                       const std::shared_ptr<HostTensorND>& filter,
                       const typename Opr::Param& param = {}) {
    auto graph = ComputingGraph::make();
    graph->options().log_level = 0;
    SymbolVar x = opr::Host2DeviceCopy::make(*graph, src);
    SymbolVar y = opr::Host2DeviceCopy::make(*graph, filter);
    SymbolVar z = Opr::make(x, y, param);
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();

    host_z.sync();
    return host_z;
}

template <typename Opr>
HostTensorND eval_conv_cpu(const HostTensorND& xv, const HostTensorND& fv,
                           const typename Opr::Param& param = {}) {
    auto cn = CompNode::load("cpux");
    auto src = std::make_shared<HostTensorND>(cn, xv.layout()),
         filter = std::make_shared<HostTensorND>(cn, fv.layout());
    memcpy(src->raw_ptr(), xv.raw_ptr(), xv.layout().span().dist_byte());
    memcpy(filter->raw_ptr(), fv.raw_ptr(), fv.layout().span().dist_byte());
    return eval_conv<Opr>(src, filter, param);
}

//! test CompNodeSeqRecorder
namespace seq_rec {

// clang-format off
#define MGB_FOREACH_COMP_NODE_SEQ_REC_TEST(cb)                                 \
    cb(basic) cb(basic_level2) cb(basic_fake_exec) cb(dyn_elemwise)            \
    cb(dyn_elemwise_fake_exec)                                                 \
    cb(level2) cb(level2_multi_holder) cb(level2_share_storage)                \
    cb(level2_exec_check) cb(sync_from_func) cb(cb_non_contig)                 \
    cb(shape_dep_const_shape) cb(multi_recorder_run)
// clang-format on

#define def_tags(name) \
    struct name {};
MGB_FOREACH_COMP_NODE_SEQ_REC_TEST(def_tags);
#undef def_tags

//! run CompNodeSeqRecorder tests
template <typename tag>
void run(CompNode cn);

#define t(n) n,
using test_types = ::testing::Types<MGB_FOREACH_COMP_NODE_SEQ_REC_TEST(t) void>;
#undef t

}  // namespace seq_rec
}  // namespace comp_node_test
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
