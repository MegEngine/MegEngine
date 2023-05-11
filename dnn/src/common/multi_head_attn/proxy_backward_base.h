#pragma once
#include "megdnn/dtype.h"

#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "megdnn/oprs/linalg.h"
#include "megdnn/oprs/nn.h"
#include "src/common/multi_head_attn/helper.h"
#include "src/common/utils.h"

namespace megdnn {

namespace multi_head_attn {

struct MHABackwardProxyBase {
    MHABackwardProxyBase() {}
    virtual ~MHABackwardProxyBase() = default;

    /********************** function member **********************/
    template <typename T>
    void exec_internal(MHA_PROXY_BACKWARD_EXEC_PARAM);
    void exec(MHA_PROXY_BACKWARD_EXEC_PARAM);

    // lambda
#define cb(DType)                                          \
    virtual void move_scaler_to_device(                    \
            Handle* handle, DTypeTrait<DType>::ctype* dst, \
            DTypeTrait<DType>::ctype* src) = 0;
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb

    size_t get_workspace_in_bytes(MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM);
    size_t get_mask_reservespace_in_bytes(MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM);
    size_t get_othr_reservespace_in_bytes(MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM);

    void layout_refill(MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM);
    bool layout_ismatch(MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM);
    WorkspaceBundle get_mask_reservespace_bundle(
            MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM, void* ptr = nullptr);
    WorkspaceBundle get_othr_reservespace_bundle(
            MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM, void* ptr = nullptr);
    WorkspaceBundle get_workspace_bundle(
            MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM, void* ptr = nullptr);

    /************************ data member ************************/
    std::unique_ptr<MatrixMulForward> m_matmul_opr;
    std::unique_ptr<BatchedMatrixMul> m_bmatmul_opr;
    std::unique_ptr<AddUpdate> m_add_opr;
    std::unique_ptr<Elemwise> m_elem_opr;
    std::unique_ptr<Reduce> m_reduce_opr;
    std::unique_ptr<SoftmaxBackward> m_softmaxbw_opr;
    std::unique_ptr<Dropout> m_dropout_opr;
    std::unique_ptr<DropoutBackward> m_dropoutbw_opr;
    std::unique_ptr<Relayout> m_relayout_opr;

    // metadata
    size_t m_sizeof_datatype;
    megdnn::DTypeEnum m_datatype;
    size_t m_wq_off, m_wk_off, m_wv_off, m_wo_off;
    size_t m_bq_off, m_bk_off, m_bv_off, m_bo_off;
    size_t m_head, m_embed_size, m_ksize, m_vsize, m_qproj_size, m_kproj_size,
            m_vproj_size, m_oproj_size;
    bool m_qbias, m_kbias, m_vbias, m_obias;

    // out = dropout(out)
    TensorLayout m_mask2_layout;
    TensorLayout m_grad_drop2_layout;
    size_t m_grad_drop2_workspacesize;
    TensorLayout m_grad_out_layout;

    // out = z @ wo + bo
    TensorLayout m_wo_layout, m_bo_layout;
    TensorLayout m_grad_z_layout, m_grad_wo_layout, m_grad_bo_layout;
    size_t m_grad_z_workspacesize, m_grad_wo0_workspacesize, m_grad_wo1_workspacesize,
            m_grad_bo0_workspacesize, m_grad_bo1_workspacesize;

    // z = nz
    TensorLayout m_grad_nz_layout;

    // nz = ny @ nv
    TensorLayout m_grad_nv_layout, m_grad_ny_layout;
    size_t m_grad_nv_workspacesize, m_grad_ny_workspacesize;

    // ny = dropout(ny)
    TensorLayout m_mask1_layout;
    TensorLayout m_grad_drop1_layout;
    size_t m_grad_drop1_workspacesize;

    // ny = softmax(nx)
    TensorLayout m_grad_nx_layout;
    size_t m_grad_nx_workspacesize;

    // nx = nq @ nk
    TensorLayout m_grad_nq_layout, m_grad_nk_layout;
    size_t m_grad_nq_workspacesize, m_grad_nk_workspacesize;

    // nq, nk, nv = q, k, v
    TensorLayout m_grad_q_layout, m_grad_k_layout, m_grad_v_layout;

    // q = qin @ wq + bq
    TensorLayout m_wq_layout, m_bq_layout;
    TensorLayout m_grad_qin_layout, m_grad_wq_layout, m_grad_bq_layout;
    size_t m_grad_qin_workspacesize, m_grad_wq0_workspacesize, m_grad_wq1_workspacesize,
            m_grad_bq0_workspacesize, m_grad_bq1_workspacesize;
    size_t m_grad_qin_reduce_workspacesize, m_grad_kin_reduce_workspacesize,
            m_grad_vin_reduce_workspacesize;

    // k = kin @ wk + bk
    TensorLayout m_wk_layout, m_bk_layout;
    TensorLayout m_grad_kin_layout, m_grad_wk_layout, m_grad_bk_layout;
    size_t m_grad_kin_workspacesize, m_grad_wk0_workspacesize, m_grad_wk1_workspacesize,
            m_grad_bk0_workspacesize, m_grad_bk1_workspacesize;

    // v = vin @ wv + bv
    TensorLayout m_wv_layout, m_bv_layout;
    TensorLayout m_grad_vin_layout, m_grad_wv_layout, m_grad_bv_layout;
    size_t m_grad_vin_workspacesize, m_grad_wv0_workspacesize, m_grad_wv1_workspacesize,
            m_grad_bv0_workspacesize, m_grad_bv1_workspacesize;
};

}  // namespace multi_head_attn
}  // namespace megdnn
