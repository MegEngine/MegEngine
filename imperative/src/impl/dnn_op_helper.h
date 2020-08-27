/**
 * \file src/core/include/megbrain/imperative.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/comp_node_env.h"
#include "megbrain/comp_node.h"

using namespace megdnn;

namespace mgb {
namespace imperative {

/*!
 * \brief A struct for safely calling DNN oprs
 * In some cases, op may be released before the complete of the execution
 * This destructor will prevent this
 */
template<typename Opr>
struct DnnOprCaller {
    CompNode cn;
    DeviceTensorND dev_tensor;
    Workspace workspace;
    std::unique_ptr<Opr> op;

    DnnOprCaller(CompNode cn): cn(cn) {
        auto&& handle = MegDNNHandle::get(
                                CompNodeEnv::from_comp_node(cn)).handle();
        op = handle->create_operator<Opr>();
    }

    megdnn::Workspace create_workspace(TensorLayout layout) {
        dev_tensor = Tensor::make(layout, cn)->dev_tensor();
        workspace = megdnn::Workspace(dev_tensor.raw_ptr(), 
                                      dev_tensor.storage().size());
        return workspace;
    }
    
    ~DnnOprCaller() {
        using DT = CompNode::DeviceType;
        if (cn.device_type() == DT::CPU && cn != CompNode::default_cpu()) {
            CompNodeEnv::from_comp_node(cn).cpu_env().dispatch(
                [p = op.release()] { delete p; }
            );
        }
    }
};

} // namespace imperative
} // namespace mgb