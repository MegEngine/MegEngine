#include "megbrain/imperative/dispatch.h"

#include "megbrain/imperative/utils/debug.h"
#include "megbrain/imperative/utils/helper.h"
#include "megbrain/imperative/utils/map.h"

namespace mgb {
namespace imperative {
namespace {

ValueRefList apply_release(const Operator& op, Span<ValueRef> inputs) {
    auto& context = Transformation::get_context();
    size_t& depth = context.next_transformation;
    mgb_assert(depth < context.transformations.size());
    auto& transformation = *context.transformations[depth++];
    CleanupGuard _{[&] { --depth; }};
    return transformation.apply_transformation(op, inputs);
}

MGB_NOINLINE ValueRefList apply_debug(const Operator& op, Span<ValueRef> inputs) {
    auto& context = Transformation::get_context();
    size_t& depth = context.next_transformation;
    mgb_assert(depth < context.transformations.size());
    static const char tabs[] = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
    const char* prefix = tabs + (sizeof(tabs) / sizeof(char)) - depth - 1;
    mgb_log_debug(
            "%s apply %s to %s", prefix, op.to_string().c_str(),
            imperative::to_string(inputs).c_str());
    ValueRefList result;
    auto& transformation = *context.transformations[depth++];
    CleanupGuard _{[&] { --depth; }};
    result = transformation.apply_transformation(op, inputs);
    mgb_log_debug(
            "%s returns %s", prefix,
            imperative::to_string(Span<ValueRef>(result)).c_str());
    return result;
}

}  // namespace

ValueRefList apply(const Operator& op, Span<ValueRef> inputs) {
    static bool debug = MGB_GETENV("MGE_LOG_OP_DISPATCH");
    if (mgb_unlikely(debug)) {
        return apply_debug(op, inputs);
    } else {
        return apply_release(op, inputs);
    }
}

ValueRefList apply(const OpDef& def, Span<ValueRef> inputs) {
    return imperative::apply(ApplyOp{def}, inputs);
}

ValueRefList apply(const Subgraph& graph, Span<ValueRef> inputs) {
    auto apply_functor = [](std::shared_ptr<OpDef> op, Span<ValueRef> inputs, size_t) {
        auto outputs = imperative::apply(*op, inputs);
        return SmallVector<ValueRef>(outputs.begin(), outputs.end());
    };
    auto make_const = [](TensorPtr constant) -> ValueRef {
        auto host_value = constant->get_value();
        auto device_value = constant->dev_tensor();
        mgb_assert(
                host_value.layout().is_contiguous() &&
                device_value.layout().is_contiguous());
        ValueShape shape;
        // FIXME: assume Tensor with shape {1} is scalar
        if (!constant->shape().is_scalar()) {
            shape = ValueShape::from(constant->shape());
        }
        return imperative::apply(
                CreateTensor(
                        CreateTensor::Const, constant->comp_node(), constant->dtype(),
                        shape),
                HostStorage::make(host_value.storage()),
                DeviceStorage::make(device_value.storage()))[0];
    };
    auto outputs = graph.apply(inputs, apply_functor, make_const);
    return ValueRefList{outputs.begin(), outputs.end()};
}

}  // namespace imperative
}  // namespace mgb
