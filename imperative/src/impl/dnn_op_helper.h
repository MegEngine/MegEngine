/**
 * \file imperative/src/impl/dnn_op_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/comp_node_env.h"
#include "megbrain/comp_node.h"
#include "megbrain/imperative/physical_tensor.h"

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

    DnnOprCaller(CompNode cn): cn(cn), op(create_operator(cn)) {}

    static std::unique_ptr<Opr> create_operator(CompNode cn) {
        auto&& handle = MegDNNHandle::get(
                                CompNodeEnv::from_comp_node(cn)).handle();
        return handle->create_operator<Opr>();
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

template <size_t OSize>
class MegDNNDynOutMallocImpl final: public megdnn::DynOutMallocPolicy {
    using Output = std::array<TensorPtr, OSize>;

    CompNode m_cn;
    Output m_out;

public:
    MegDNNDynOutMallocImpl(CompNode cn): m_cn{cn} {}

    megdnn::TensorND alloc_output(
            size_t id, DType dtype, const TensorShape &shape,
            void *user_data) override {
        TensorLayout m_layout(shape, dtype);
        m_out[id] = Tensor::make(m_layout, m_cn);
        return m_out[id]->dev_tensor().as_megdnn();
    }

    void* alloc_workspace(size_t sz, void *user_data) override {
        return m_cn.alloc_device(sz);
    }

    void free_workspace(void *ptr, void *user_data) override {
        m_cn.free_device(ptr);
    }

    TensorPtr at(size_t id) {
        return m_out[id];
    }
};

} // namespace imperative
} // namespace mgb
