#include "megbrain/opr/custom_opnode.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace serialization {

void custom_dumper(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
    auto&& custom_op = opr.cast_final_safe<opr::CustomOpNode>();

    std::string op_type = custom_op.op_type();
    ctx.dump_buf_with_len(op_type.c_str(), op_type.size());

    uint32_t tag = custom_op.param_tag();
    ctx.dump_buf_with_len(&tag, sizeof(tag));

    std::string bytes = custom_op.param().to_bytes();
    ctx.dump_buf_with_len(bytes.c_str(), bytes.size());
}

mgb::cg::OperatorNodeBase* custom_loader(
        OprLoadContext& ctx, const cg::VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    std::string op_type = ctx.load_buf_with_len();
    auto* op_manager = custom::CustomOpManager::inst();
    auto op = op_manager->find(op_type);

    std::string tag_str = ctx.load_buf_with_len();
    uint32_t tag = *reinterpret_cast<const uint32_t*>(tag_str.c_str());
    mgb_assert(
            tag == op->param_info().tag(),
            "Wrong Param TAG of Op %s, should be %u, but load %u\n", op_type.c_str(),
            op->param_info().tag(), tag);

    custom::Param param(op->param_info());
    std::string bytes = ctx.load_buf_with_len();
    param.from_bytes(bytes);
    return opr::CustomOpNode::make(op, inputs, param, config)[0]->owner_opr();
}

}  // namespace serialization
}  // namespace mgb

#define CUSTOM_OP_SEREG_REG(cls)                              \
    namespace {                                               \
    struct _OprReg##cls {                                     \
        static void entry() {                                 \
            MGB_SEREG_OPR_INTL_CALL_ADD(                      \
                    cls, ::mgb::serialization::custom_dumper, \
                    ::mgb::serialization::custom_loader);     \
        }                                                     \
    };                                                        \
    }                                                         \
    MGB_SEREG_OPR_INTL_CALL_ENTRY(cls, _OprReg##cls)

#define CUSTOM_OP_SEREG_REG_V2(cls, _version_min, _version_max)                 \
    namespace {                                                                 \
    struct _OprRegV2##cls {                                                     \
        static void entry() {                                                   \
            MGB_SEREG_OPR_INTL_CALL_ADD_V2(                                     \
                    cls, ::mgb::serialization::custom_dumper,                   \
                    ::mgb::serialization::custom_loader, nullptr, _version_min, \
                    _version_max);                                              \
        }                                                                       \
    };                                                                          \
    }                                                                           \
    MGB_SEREG_OPR_INTL_CALL_ENTRY_V2(cls, _OprRegV2##cls)

using namespace mgb;
using CustomOpNode = opr::CustomOpNode;
CUSTOM_OP_SEREG_REG(CustomOpNode);

CUSTOM_OP_SEREG_REG_V2(CustomOpNode, 2, CURRENT_VERSION);
