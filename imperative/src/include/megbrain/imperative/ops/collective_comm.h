/**
 * \file src/core/include/megbrain/imperative.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/imperative/op_def.h"
#include "megbrain/opr/param_defs.h"

namespace mgb {
namespace imperative {

class CollectiveComm : public OpDefImplBase<CollectiveComm> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    CollectiveComm() = default;
    CollectiveComm(const std::string& key_, size_t nr_devices_,
                      uint32_t rank_, bool is_root_, bool local_grad_,
                      const std::string& addr_, uint32_t port_,
                      const megdnn::param::CollectiveComm::Mode& mode_,
                      const DType& dtype_, const std::string& backend_,
                      const std::string& comp_node_)
            : key(key_),
              nr_devices(nr_devices_),
              rank(rank_),
              is_root(is_root_),
              local_grad(local_grad_),
              addr(addr_),
              port(port_),
              mode(mode_),
              dtype(dtype_),
              backend(backend_),
              comp_node(comp_node_) {}
    std::string key;
    size_t nr_devices;
    uint32_t rank;
    bool is_root;
    bool local_grad;
    std::string addr;
    uint32_t port;
    megdnn::param::CollectiveComm::Mode mode;
    DType dtype;
    std::string backend;
    std::string comp_node;
};

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
