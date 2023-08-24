#pragma once

#include "megbrain/common.h"

#if MGB_CUSTOM_OP

#include "megbrain/custom/custom.h"
#include "megbrain/custom/manager.h"
#include "megbrain/imperative/op_def.h"

namespace mgb {
namespace imperative {

class CustomOpDef : public OpDefImplBase<CustomOpDef> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
    const std::shared_ptr<const custom::CustomOp> m_op;
    custom::Param m_param;

public:
    CustomOpDef(const std::shared_ptr<const custom::CustomOp>& op);
    CustomOpDef(
            const std::shared_ptr<const custom::CustomOp>& op, const custom::Param&);

    void param(const custom::Param&);
    custom::Param& param(void);
    custom::Param param(void) const;
    size_t input_num(void) const;
    size_t output_num(void) const;
    std::string name(void) const;
    custom::RunTimeId runtime_id(void) const;
    const std::shared_ptr<const custom::CustomOp>& impl(void) const;

    void compute(
            std::shared_ptr<SmallVector<DeviceTensorND>>,
            std::shared_ptr<SmallVector<DeviceTensorND>>) const;
    std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs(
            const SmallVector<TensorPtr>& inputs) const;
    std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs(
            const SmallVector<LogicalTensorDesc>&) const;
};

class CustomOpDefFactory {
    custom::CustomOpManager* ops;
    CustomOpDefFactory();

public:
    PREVENT_COPY_AND_ASSIGN(CustomOpDefFactory);

    static CustomOpDefFactory* inst(void);
    static bool is_custom_op(const OpDef& op);

    std::vector<std::string> op_list(void) const;

    std::shared_ptr<OpDef> create_opdef(const std::string&) const;
    std::shared_ptr<OpDef> create_opdef(const custom::RunTimeId&) const;
    std::shared_ptr<OpDef> create_opdef(const std::string&, const custom::Param&) const;
    std::shared_ptr<OpDef> create_opdef(
            const custom::RunTimeId&, const custom::Param&) const;
};

}  // namespace imperative
}  // namespace mgb

#endif
