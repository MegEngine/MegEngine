#pragma once
#if MGB_ENABLE_FBS_SERIALIZATION
#include "megbrain/comp_node_env.h"
#include "megbrain/opr/dnn/softmax.h"
#include "megbrain/opr/io.h"
#include "megbrain/serialization/oss_opr_load_dump.h"
#include "megbrain/serialization/sereg.h"

#include "megbrain/serialization/internal/mgb_cpp_opr_generated.h"
#include "megbrain/serialization/internal/schema_v2_generated.h"

namespace mgb {
namespace serialization {

template <>
struct OprLoadDumpImplV2<opr::ImmutableTensor, 0> {
    using Opr = opr::ImmutableTensor;

    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        using Meth = OprDumpContext::TensorWriteMethod;
        auto&& opr = opr_.cast_final_safe<Opr>();
        ctx.dump_tensor(
                {}, HostTensorND{}.copy_from(opr.value()).sync(),
                Meth::VALUE_ANONYMOUS);
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        auto& fbs_ctx = CAST_TO_FBS_V2_CTX(ctx);
        mgb_assert(inputs.empty());
        auto fopr = reinterpret_cast<const fbs::v2::Operator*>(
                fbs_ctx.get_current_opr_data());
        if (fopr->tensors() && fopr->tensors()->size() > 0) {
            auto val = fbs_ctx.load_tensor();
            return Opr::make(fbs_ctx.graph(), *val, config).node()->owner_opr();
        } else {
            mgb_throw(SerializationError, "ImmutableTensor load with no tensor data.");
        }
    }
};

template <>
struct OprLoadDumpImplV2<opr::Host2DeviceCopy, 0> {
    using Opr = opr::Host2DeviceCopy;
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        auto&& opr = opr_.cast_final_safe<Opr>();
        ctx.write_param(opr.param());

        using Meth = OprDumpContext::TensorWriteMethod;
        ctx.dump_tensor(
                opr.name(), *opr.host_data(),
                opr.param().dump_default_value ? Meth::VALUE_INPUT : Meth::META_INPUT);
    }
    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        mgb_assert(inputs.empty());
        auto& fbs_ctx = CAST_TO_FBS_V2_CTX(ctx);
        auto param = fbs_ctx.read_param<Opr::Param>(0);
        auto tensor = fbs_ctx.load_tensor();
        return Opr::make(fbs_ctx.graph(), tensor, param, config).node()->owner_opr();
    }
};

template <>
struct OprLoadDumpImplV2<opr::SharedDeviceTensorWithFormat, 0> {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        using Meth = OprDumpContext::TensorWriteMethod;
        auto&& opr = opr_.cast_final_safe<opr::SharedDeviceTensorWithFormat>();
        HostTensorND val;
        val.copy_from(opr.get_dev_tensor()).sync();
        ctx.dump_tensor({}, val, Meth::VALUE_ANONYMOUS);
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        mgb_assert(inputs.empty());
        auto val = ctx.load_tensor();
        auto dev_val =
                std::make_shared<DeviceTensorND>(val->comp_node(), val->layout());
        dev_val->copy_from_fixlayout(*val);
        auto out_var =
                opr::SharedDeviceTensorWithFormat::make(ctx.graph(), dev_val, config);
        dev_val->sync();
        return out_var.node()->owner_opr();
    }
};

template <>
struct OprLoadDumpImplV2<opr::MultipleDeviceTensorHolder, 0> {
    using Opr = opr::MultipleDeviceTensorHolder;

    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        using Meth = OprDumpContext::TensorWriteMethod;
        auto&& opr = opr_.cast_final_safe<Opr>();
        uint32_t nr_val = opr.values().size();
        for (uint32_t i = 0; i < nr_val; ++i) {
            HostTensorND val;
            val.copy_from(*opr.values()[i]).sync();
            ctx.dump_tensor(opr.output(i)->name(), val, Meth::VALUE_SHARED);
        }
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        mgb_assert(inputs.empty());
        auto& fbs_ctx = CAST_TO_FBS_V2_CTX(ctx);
        auto fopr = reinterpret_cast<const fbs::v2::Operator*>(
                fbs_ctx.get_current_opr_data());
        uint32_t nr = 0;
        if (fopr && fopr->tensors()) {
            nr = fopr->tensors()->size();
        }
        Opr::ValueArray values(nr);
        for (auto&& i : values) {
            i = ctx.load_tensor_shared();
        }
        return Opr::make(ctx.graph(), std::move(values), config)[0].node()->owner_opr();
    }
};

template <>
struct OprLoadDumpImplV2<opr::MultipleDeviceTensorWithFormatHolder, 0> {
    using Opr = opr::MultipleDeviceTensorWithFormatHolder;

    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        using Meth = OprDumpContext::TensorWriteMethod;
        auto&& opr = opr_.cast_final_safe<Opr>();
        uint32_t nr_val = opr.values().size();
        for (uint32_t i = 0; i < nr_val; ++i) {
            HostTensorND val;
            auto value = *opr.values()[i];
            val.copy_from(value).sync();
            ctx.dump_tensor(opr.output(i)->name(), val, Meth::VALUE_SHARED);
        }
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        mgb_assert(inputs.empty());
        auto& fbs_ctx = CAST_TO_FBS_V2_CTX(ctx);
        auto fopr = reinterpret_cast<const fbs::v2::Operator*>(
                fbs_ctx.get_current_opr_data());
        uint32_t nr = 0;
        if (fopr && fopr->tensors()) {
            nr = fopr->tensors()->size();
        }
        Opr::ValueArray values(nr);
        for (auto&& i : values) {
            i = ctx.load_tensor_shared();
            //! set tensor format
            TensorLayout layout_with_format = i->layout();

            if (i->storage().comp_node().mem_node() ==
                CompNode::default_cpu().mem_node()) {
                mgb_assert(
                        i->storage().ptr(),
                        "storage should not be nullptr if mem_node is "
                        "default_cpu");
                HostTensorND src{i->storage().comp_node(), layout_with_format};
                src.copy_from_fixlayout(*i).sync();
                *i = DeviceTensorND::make_proxy(src);
            } else {
                //! actually only layout of this tensor will be used later, see
                //! src/serialization/impl/batched_device_value_loader.cpp:49. But we
                //! have no way to reset layout only, so just construct a invalid
                //! storage instead
                auto size = layout_with_format.span().dist_byte();
                DeviceTensorStorage storage;
                storage.reset(i->comp_node(), size, nullptr);
                i->reset(storage, layout_with_format);
            }
        }
        return Opr::make(ctx.graph(), std::move(values), config)[0].node()->owner_opr();
    }
};

}  // namespace serialization

namespace opr {
#define SERGE_OPR_V2_NO_CONVERTER(_cls, _arity) \
    MGB_SEREG_OPR_V2(_cls, _arity, nullptr, VERSION_2, CURRENT_VERSION);

SERGE_OPR_V2_NO_CONVERTER(ImmutableTensor, 0);
SERGE_OPR_V2_NO_CONVERTER(Host2DeviceCopy, 0);
SERGE_OPR_V2_NO_CONVERTER(SharedDeviceTensorWithFormat, 0);
SERGE_OPR_V2_NO_CONVERTER(MultipleDeviceTensorWithFormatHolder, 0);
SERGE_OPR_V2_NO_CONVERTER(MultipleDeviceTensorHolder, 0);
}  // namespace opr
}  // namespace mgb

#endif

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
