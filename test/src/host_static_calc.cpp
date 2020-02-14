/**
 * \file test/src/host_static_calc.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief static calculating on host to check opr correctness
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/test/host_static_calc.h"

void mgb::elemwise_static_calc(opr::Elemwise::Mode mode,
        HostTensorND &dest, const std::vector<HostTensorND>& inputs) {
#if defined(ANDROID) || defined(IOS) || defined(__arm__)
    static opr::intl::UniqPtrWithCN<megdnn::Elemwise> opr_impl;
    static std::mutex mtx;
    MGB_LOCK_GUARD(mtx);
#else
    static thread_local opr::intl::UniqPtrWithCN<megdnn::Elemwise> opr_impl;
#endif
    auto cn = CompNode::default_cpu();
    if (!opr_impl) {
        opr_impl = opr::intl::create_megdnn_opr<megdnn::Elemwise>(cn);
    }
    DeviceTensorND dev_dest{cn};
    SmallVector<DeviceTensorND> dev_inp(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++ i) {
        dev_inp[i].comp_node(cn).copy_from(inputs[i]);
    }
    opr::Elemwise::perform(mode, dev_dest, dev_inp, opr_impl);
    dest.copy_from(dev_dest);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}


