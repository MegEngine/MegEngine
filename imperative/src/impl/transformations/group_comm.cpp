#include "megbrain/imperative/transformations/group_comm.h"
#include "megbrain/imperative/blob_manager.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/io_remote.h"
namespace mgb {
namespace imperative {

ValueRefList GroupCommTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    for (auto inp : inputs) {
        mgb_assert(
                !inp.is(m_value_type), "Can not use PlaceholderValue as apply input");
    }
    if (auto* apply_op = op.as<ApplyOp>()) {
        if (apply_op->op().same_type<RemoteSend>()) {
            auto&& send_op = apply_op->op().cast_final_safe<RemoteSend>();
            if (send_op.key[0] == 'b') {
                send_inputs.push_back(inputs[0]);
                record_ops.push_back(send_op.shared_from_this());
                return {};
            }
        }
        if (apply_op->op().same_type<RemoteRecv>()) {
            auto&& recv_op = apply_op->op().cast_final_safe<RemoteRecv>();
            if (recv_op.key[0] == 'b') {
                record_ops.push_back(recv_op.shared_from_this());
                auto rst = m_value_type.make();
                recv_tensors.push_back(rst);
                auto outputs = ValueRefList(1);
                outputs[0] = rst;
                return outputs;
            }
        }
        return imperative::apply(op, inputs);
    } else {
        return imperative::apply(op, inputs);
    }
}

ValueRefList GroupCommTransformation::execute_batch_op() {
    auto batch_op = BatchSendRecvOp::make(record_ops);
    auto outputs = imperative::apply(*batch_op, send_inputs);
    return outputs;
}

void GroupCommTransformation::on_unregister() noexcept {
    auto rst = execute_batch_op();
    mgb_assert(rst.size() == recv_tensors.size());
    for (size_t i = 0; i < rst.size(); i++) {
        auto v = recv_tensors[i].lock();
        if (v != ValueRef::nil) {
            v.reset(rst[i]);
        }
    }
}

GroupCommTransformation::~GroupCommTransformation() {
    for (auto&& recv : recv_tensors) {
        mgb_assert(
                recv.lock() == ValueRef::nil,
                "Some PlaceholderValues are not reset after GroupCommTransformation "
                "destroyed！");
    };
}

}  // namespace imperative
}  // namespace mgb