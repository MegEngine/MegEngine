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

struct MHAForwardProxyBase {
    MHAForwardProxyBase() {}
    virtual ~MHAForwardProxyBase() = default;

    /********************** function member **********************/
    template <typename T>
    void exec_internal(MHA_PROXY_FORWARD_EXEC_PARAM);
    void exec(MHA_PROXY_FORWARD_EXEC_PARAM);

    // lambda
#define cb(DType)                                          \
    virtual void move_scaler_to_device(                    \
            Handle* handle, DTypeTrait<DType>::ctype* dst, \
            DTypeTrait<DType>::ctype* src) = 0;
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb

    void deduce_layout(MHA_PROXY_FORWARD_LAYOUT_PARAM);
    size_t get_workspace_in_bytes(MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM);
    size_t get_mask_reservespace_in_bytes(MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM);
    size_t get_othr_reservespace_in_bytes(MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM);

    void layout_refill(MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM);
    bool layout_ismatch(MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM);
    WorkspaceBundle get_mask_reservespace_bundle(
            MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM, void* ptr = nullptr);
    WorkspaceBundle get_othr_reservespace_bundle(
            MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM, void* ptr = nullptr);
    WorkspaceBundle get_workspace_bundle(
            MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM, void* ptr = nullptr);

    /************************ data member ************************/
    std::unique_ptr<MatrixMulForward> m_matmul_opr;
    std::unique_ptr<BatchedMatrixMul> m_bmatmul_opr;
    std::unique_ptr<AddUpdate> m_add_opr;
    std::unique_ptr<Elemwise> m_elem_opr;
    std::unique_ptr<Softmax> m_softmax_opr;
    std::unique_ptr<Dropout> m_dropout_opr;
    std::unique_ptr<Relayout> m_relayout_opr;
    std::unique_ptr<RepeatForward> m_repeat_opr;
    // std::unique_ptr<ConcatForward> m_concat_opr;
    // std::unique_ptr<Fill> m_fill_opr;

    // metadata
    size_t m_sizeof_datatype;
    megdnn::DTypeEnum m_datatype;
    size_t m_wq_off, m_wk_off, m_wv_off, m_wo_off;
    size_t m_bq_off, m_bk_off, m_bv_off, m_bo_off;
    size_t m_heads, m_embed_size, m_ksize, m_vsize, m_qproj_size, m_kproj_size,
            m_vproj_size, m_oproj_size;
    bool m_qbias, m_kbias, m_vbias, m_obias;

    // add bias_kv
    // TensorLayout m_nbias_k_layout, m_nbias_v_layout;
    // size_t m_bias_k_repeat_workspacesize, m_bias_v_repeat_workspacesize;
    // add zero_attn
    // TensorLayout m_zero_k_layout, m_zero_v_layout;
    // TensorLayout m_added_bias_zero_k_layout, m_added_bias_zero_v_layout;
    // size_t m_added_bias_zero_k_workspacesize, m_added_bias_zero_v_workspacesize;

    // q/k/v = matmul(qu/ky/va, wq/wk/wv, bq/bk/bv)
    // nq/nk/nv = dimshuffle(q/k/v) (norm to multihead)
    TensorLayout m_wq_layout, m_wk_layout, m_wv_layout;
    TensorLayout m_bq_layout, m_bk_layout, m_bv_layout;
    TensorLayout m_q_layout, m_k_layout, m_v_layout;
    TensorLayout m_nq_layout, m_nk_layout, m_nv_layout;
    size_t m_q_workspacesize, m_k_workspacesize, m_v_workspacesize;
    size_t m_q_head_repeat_workspacesize, m_k_head_repeat_workspacesize,
            m_v_head_repeat_workspacesize;

    // nx = matmul(nq, nk)
    // ny = softmax(nx), ny_layout = m_nx_layout;
    // ny = dropout(ny), dropout1_layout = m_nx_layout;
    TensorLayout m_nx_layout;
    TensorLayout m_mask1_layout;
    size_t m_nx_workspacesize, m_softmax_workspacesize, m_dropout1_workspacesize;

    // nz = matmul(ny, v)
    // z = dimshuffle(nz) (multihead to norm)
    TensorLayout m_nz_layout;
    TensorLayout m_z_layout;
    size_t m_nz_workspacesize;

    // out = matmul(z, wo, bo)
    // out = dropout(out), dropout2_layout = m_out_layout;
    TensorLayout m_wo_layout, m_bo_layout;
    TensorLayout m_out_layout;
    TensorLayout m_mask2_layout;
    size_t m_out_workspacesize, m_dropout2_workspacesize;
};
}  // namespace multi_head_attn
}  // namespace megdnn
