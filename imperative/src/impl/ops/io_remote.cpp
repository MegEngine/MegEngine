#include "megbrain/imperative/ops/io_remote.h"
#include "megbrain_build_config.h"

#if MGB_ENABLE_OPR_MM
#include <algorithm>
#include <functional>
#include <numeric>
#include "../blob_manager_impl.h"
#include "../op_trait.h"
#include "megbrain/imperative/proxy_graph_detail.h"
#include "megbrain/opr/io_remote.h"
#include "megbrain/opr/megray_helper.h"
#include "megbrain/opr/mm_handler.h"
#endif  // MGB_ENABLE_OPR_MM
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/proxy_graph_detail.h"
namespace mgb {
namespace imperative {

#if MGB_ENABLE_OPR_MM
namespace {
cg::OperatorNodeBase* apply_on_var_node_remote_send(
        const OpDef& def, const VarNodeArray& inputs) {
    auto&& send = def.cast_final_safe<RemoteSend>();
    auto group_client = std::make_shared<opr::GroupClientProxy>(
            ssprintf("%s:%d", send.addr.data(), send.port));
    auto&& graph = inputs[0]->owner_graph();

    OperatorNodeConfig config{send.make_name()};
    cg::OperatorNodeBase* opr =
            graph->insert_opr(std::make_unique<mgb::opr::RemoteSend>(
                    send.key, inputs[0], group_client, true, send.backend, config));
    return opr;
}

cg::OperatorNodeBase* apply_on_var_node_remote_recv(
        const OpDef& def, const VarNodeArray& inputs) {
    auto&& recv = def.cast_final_safe<RemoteRecv>();
    OperatorNodeConfig config{recv.cn};
    config.name(recv.make_name());
    auto group_client = std::make_shared<opr::GroupClientProxy>(
            ssprintf("%s:%d", recv.addr.data(), recv.port));
    auto&& graph = inputs[0]->owner_graph();
    mgb_assert(!recv.shape.empty());
    TensorShape shape;
    for (auto&& dim : recv.shape) {
        shape[shape.ndim++] = dim;
    }
    return graph->insert_opr(std::make_unique<mgb::opr::RemoteRecv>(
            recv.key, inputs[0], *graph, group_client, config, shape, recv.dtype,
            recv.backend));
}

TensorPtr megray_recv_tensor(
        std::shared_ptr<MegRay::Communicator> megray_comm, TensorLayout& layout,
        CompNode cn, uint32_t rank_from) {
    auto out = Tensor::make(layout, cn);
    auto dnn_out = out->dnn_tensor();
    auto megray_ctx = mgb::opr::get_megray_context(cn);
    size_t data_size = layout.total_nr_elems();
    auto status = megray_comm->recv(
            dnn_out.raw_ptr(), data_size, mgb::opr::get_megray_dtype(layout.dtype),
            rank_from, megray_ctx);
    mgb_assert(status == MegRay::MEGRAY_OK, "MegRay recv failed");
    return out;
}

void megray_send_tensor(
        std::shared_ptr<MegRay::Communicator> megray_comm, const TensorPtr& src,
        uint32_t rank_to) {
    auto&& tensor = src->dev_tensor();
    auto&& ishp = src->shape();
    size_t data_size = ishp.total_nr_elems();
    auto megray_ctx = mgb::opr::get_megray_context(src->comp_node());
    auto status = megray_comm->send(
            src->dev_tensor().raw_ptr(), data_size,
            mgb::opr::get_megray_dtype(src->layout().dtype), rank_to, megray_ctx);
    mgb_assert(status == MegRay::MEGRAY_OK, "MegRay send failed");
}

TensorLayout create_layout(const std::vector<int32_t>& shape, DType dtype) {
    TensorShape tshape;
    tshape.ndim = shape.size();
    mgb_assert(tshape.ndim <= TensorLayout::MAX_NDIM);
    std::copy(shape.begin(), shape.end(), tshape.shape);
    return TensorLayout(tshape, dtype);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible_remote_send(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    auto&& dtype = input_descs[0].layout.dtype;
    auto&& cn = input_descs[0].comp_node;
    return {{{TensorLayout({0}, dtype), cn}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor_remote_send(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<RemoteSend>();
    auto megray_comm = mgb::opr::BatchSendRecvHelper::getInstance()->get(
            std::string("init_all_cards"));
    if (!megray_comm) {
        return proxy_graph_detail::apply_on_physical_tensor(
                def, inputs, output_descs, validated);
    }
    mgb_assert(megray_comm != nullptr);
    megray_send_tensor(megray_comm, inputs[0], op.rank_to);
    TensorLayout layout({0}, inputs[0]->dtype());
    return {Tensor::make(layout, inputs[0]->comp_node())};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible_remote_recv(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    auto& op = def.cast_final_safe<RemoteRecv>();
    return {{{create_layout(op.shape, op.dtype), op.cn}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor_remote_recv(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<RemoteRecv>();
    auto layout = create_layout(op.shape, op.dtype);
    auto megray_comm = mgb::opr::BatchSendRecvHelper::getInstance()->get(
            std::string("init_all_cards"));
    if (!megray_comm) {
        return proxy_graph_detail::apply_on_physical_tensor(
                def, inputs, output_descs, validated);
    }
    auto&& out = megray_recv_tensor(megray_comm, layout, op.cn, op.rank_from);
    return {out};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    for (size_t i; i < inputs.size(); i++) {
        layout_checker[i] = [](const TensorLayout& layout) {
            return layout.is_contiguous();
        };
    }
    return layout_checker;
}

OP_TRAIT_REG(RemoteSend, RemoteSend, mgb::opr::RemoteSend)
        .apply_on_var_node(apply_on_var_node_remote_send)
        .apply_on_physical_tensor(apply_on_physical_tensor_remote_send)
        .infer_output_attrs_fallible(infer_output_attrs_fallible_remote_send)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();

OP_TRAIT_REG(RemoteRecv, RemoteRecv, mgb::opr::RemoteRecv)
        .apply_on_var_node(apply_on_var_node_remote_recv)
        .apply_on_physical_tensor(apply_on_physical_tensor_remote_recv)
        .infer_output_attrs_fallible(infer_output_attrs_fallible_remote_recv)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();

SmallVector<TensorPtr> apply_on_physical_tensor_batch_send_recv(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<BatchSendRecvOp>();
    auto megray_comm = mgb::opr::BatchSendRecvHelper::getInstance()->get(
            std::string("init_all_cards"));
    mgb_assert(megray_comm != nullptr);
    megray_comm->group_start();
    SmallVector<TensorPtr> outputs;
    size_t ind = 0;
    for (auto&& op_ : op.op_list) {
        if (op_->same_type<RemoteSend>()) {
            auto&& send_op = op_->cast_final_safe<RemoteSend>();
            auto&& tensor = inputs[ind];
            megray_send_tensor(megray_comm, tensor, send_op.rank_to);
            ind++;
        } else {
            mgb_assert(op_->same_type<RemoteRecv>());
            auto&& recv_op = op_->cast_final_safe<RemoteRecv>();
            auto layout = create_layout(recv_op.shape, recv_op.dtype);
            auto&& out = megray_recv_tensor(
                    megray_comm, layout, recv_op.cn, recv_op.rank_from);
            outputs.push_back(out);
        }
    }
    megray_comm->group_end();
    return outputs;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool>
infer_output_attrs_fallible_batch_send_recv(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    auto& op = def.cast_final_safe<BatchSendRecvOp>();
    SmallVector<LogicalTensorDesc> output_descs;
    for (auto&& op_ : op.op_list) {
        if (op_->same_type<RemoteRecv>()) {
            auto&& recv_op = op_->cast_final_safe<RemoteRecv>();
            output_descs.push_back(
                    {create_layout(recv_op.shape, recv_op.dtype), recv_op.cn});
        }
    }
    return {output_descs, true};
}

OP_TRAIT_REG(BatchSendRecvOp, BatchSendRecvOp)
        .apply_on_physical_tensor(apply_on_physical_tensor_batch_send_recv)
        .infer_output_attrs_fallible(infer_output_attrs_fallible_batch_send_recv)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace

#endif  // MGB_ENABLE_OPR_MM
MGB_DYN_TYPE_OBJ_FINAL_IMPL(BatchSendRecvOp);

}  // namespace imperative
}  // namespace mgb
