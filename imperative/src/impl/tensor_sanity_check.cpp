/**
 * \file src/core/impl/imperative/tensor_sanity_check.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/imperative/tensor_sanity_check.h"

#include "./op_trait.h"

namespace mgb {

namespace imperative {


TensorChecksumCalc::ChecksumResult TensorChecksumCalc::calc(TensorPtr ptr) {
    auto&& dt = ptr->dev_tensor();
    if (!dt.layout().total_nr_elems()) {
        static ChecksumResult empty_checksum;
        return empty_checksum;
    }

    auto span = dt.layout().span();
    megdnn::TensorND tensor;
    tensor.raw_ptr = dt.raw_ptr() + span.low_byte;
    tensor.layout.init_contiguous_stride({span.dist_byte()});
    tensor.layout.dtype = dtype::Byte();

    DeviceTensorStorage* workspace;
    {
        MGB_LOCK_GUARD(m_workspace_mtx);
        workspace = &m_workspace[std::this_thread::get_id()]
                             .storage[ptr->comp_node()];
    }
    auto comp_node = ptr->comp_node();
    comp_node.activate();
    auto opr = opr::intl::get_megdnn_global_opr<megdnn::Checksum>(comp_node);
    auto workspace_reqsize = opr->get_workspace_in_bytes(tensor.layout);
    workspace->comp_node(ptr->comp_node()).ensure_size(workspace_reqsize);

    megdnn::Workspace mwk;
    if (workspace_reqsize)
        mwk = {workspace->ptr(), workspace_reqsize};

    return opr->exec(tensor, mwk);
}


class TensorSanityCheckImpl {
public:
    std::vector<std::tuple<OpTrait*, std::unique_ptr<ApplyOnPhysicalTensor>>>
            hook_list;
    std::unordered_map<TensorPtr, TensorChecksumCalc::ChecksumResult>
            tensor2chksum; // TODO: may increase device memory overhead
    TensorSanityCheckImpl() {
        m_calc = std::make_unique<TensorChecksumCalc>();
    }
    bool check(TensorPtr p);
private:
    std::unique_ptr<TensorChecksumCalc> m_calc;
};

bool TensorSanityCheckImpl::check(TensorPtr p) {
    auto&& it = tensor2chksum.find(p);
    auto&& chksum = m_calc->calc(p);
    if (it == tensor2chksum.end()) {
        tensor2chksum[p] = chksum;
        return true;
    }
    return it->second == chksum;
}

void TensorSanityCheck::enable() {
    CompNode::sync_all();
    OpTrait::for_each_trait([this](OpTrait& trait) {
        auto backup = std::make_unique<ApplyOnPhysicalTensor>(
            std::move(trait.apply_on_physical_tensor));
        trait.apply_on_physical_tensor = [this, backup = backup.get()] (
                const OpDef& def, const SmallVector<TensorPtr>& inputs) {
            for (auto&& i: inputs) {
                if (!m_checker->check(i)) {
                    mgb_throw(TensorChecksumCalc::Error,
                            "tensor modified before exec %s", print_op(def).c_str());
                }
            }
            auto output = (*backup)(def, inputs);
            for (auto&& i: output) {
                mgb_assert(m_checker->check(i));
            }
            for (auto&& i: inputs) {
                if (!m_checker->check(i)) {
                    mgb_throw(TensorChecksumCalc::Error,
                            "tensor modified after exec %s", print_op(def).c_str());
                }
            }
            return output;
        };
        m_checker->hook_list.push_back({&trait, std::move(backup)});
    });
}

void TensorSanityCheck::disable() {
    for (auto&& hook : m_checker->hook_list) {
        std::get<0>(hook)->apply_on_physical_tensor = 
                std::move(*std::get<1>(hook));
    }
    m_checker->tensor2chksum.clear();
    m_checker->hook_list.clear();
}

TensorSanityCheck::TensorSanityCheck() {
    m_checker = std::make_unique<TensorSanityCheckImpl>();
}

TensorSanityCheck::~TensorSanityCheck () {
}

std::string TensorSanityCheck::print_op(const OpDef& def){
    auto* opr_attr = def.try_cast_final<const OprAttr>();
    if(opr_attr){
        return std::string("OprAttr:") + opr_attr->type;
    }
    return def.dyn_typeinfo()->name;
}

} // namespace imperative
} // namespace mgb