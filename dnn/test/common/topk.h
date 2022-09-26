#pragma once

#include "megdnn/handle.h"
#include "megdnn/oprs/general.h"
#include "test/common/opr_proxy.h"

namespace megdnn {
namespace test {

template <>
struct OprProxy<TopK> {
private:
    int m_k = 0;
    WorkspaceWrapper m_workspace;

public:
    OprProxy() = default;
    OprProxy(int k) : m_k{k} {}
    void init(TopK*, const TensorNDArray&) {}

    void deduce_layout(TopK* opr, TensorLayoutArray& layouts) {
        if (layouts.size() == 3) {
            opr->deduce_layout(m_k, layouts[0], layouts[1], layouts[2]);
        } else {
            megdnn_assert(layouts.size() == 2);
            TensorLayout l;
            opr->deduce_layout(m_k, layouts[0], layouts[1], l);
        }
    }

    void exec(TopK* opr, const TensorNDArray& tensors) {
        if (!m_workspace.valid()) {
            m_workspace = {opr->handle(), 0};
        }
        if (tensors.size() == 3) {
            m_workspace.update(opr->get_workspace_in_bytes(
                    m_k, tensors[0].layout, tensors[1].layout, tensors[2].layout));
            opr->exec(m_k, tensors[0], tensors[1], tensors[2], m_workspace.workspace());
        } else {
            m_workspace.update(opr->get_workspace_in_bytes(
                    m_k, tensors[0].layout, tensors[1].layout, {}));
            opr->exec(m_k, tensors[0], tensors[1], {}, m_workspace.workspace());
        }
    }
};

template <typename Dtype>
void run_topk_test(Handle* handle);

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
