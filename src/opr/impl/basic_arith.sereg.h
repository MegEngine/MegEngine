#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/internal/param_tag_defs.h"
#include "megbrain/serialization/helper.h"
#include "megbrain/serialization/sereg.h"

#if MGB_ENABLE_FBS_SERIALIZATION
#include "megbrain/serialization/internal/dtype_generated.h"
#include "megbrain/serialization/internal/flatbuffers_helper.h"
#include "megbrain/serialization/internal/mgb_cpp_opr_generated.h"
#endif

namespace mgb {
namespace serialization {
namespace opr_add_update {
struct PersistentDTypeScalar {
    dt_byte storage[sizeof(dt_int32)];
    DTypeEnum dtype;

    PersistentDTypeScalar(const DTypeScalar& s) : dtype{s.dtype().enumv()} {
        memcpy(storage, s.storage(), sizeof(storage));
    }

    PersistentDTypeScalar(const dt_byte* storage, DTypeEnum dtype) : dtype(dtype) {
        memcpy(this->storage, storage, sizeof(this->storage));
    }

    DTypeScalar restore() const {
        return DTypeScalar::make_from_raw(DType::from_enum(dtype), storage);
    }
};

struct PersistentAddUpdateParam {
    static constexpr uint32_t TAG = opr::param_tag::ADD_UPDATE;
    PersistentDTypeScalar alpha, beta, bias;

    PersistentAddUpdateParam(const opr::AddUpdate::Param& p)
            : alpha{*p.alpha}, beta{*p.beta}, bias{*p.bias} {}

    PersistentAddUpdateParam(
            PersistentDTypeScalar alpha, PersistentDTypeScalar beta,
            PersistentDTypeScalar bias)
            : alpha(alpha), beta(beta), bias(bias) {}

    operator opr::AddUpdate::Param() const {
        auto s = [](const PersistentDTypeScalar& v) {
            return std::make_shared<DTypeScalar>(v.restore());
        };
        return {s(alpha), s(beta), s(bias)};
    }
};

}  // namespace opr_add_update

// Old SerializedDType used in MegBrain 7.22.0 - 7.23.1
// Should be kept as-is even if there are new dtypes.
struct SerializedDTypeV1 {
    static constexpr uint32_t TAG = megdnn::param::FakeSerializedDType::TAG;
    DTypeEnum enumv;
    union {
        megdnn::DTypeParam<dtype::Quantized8Asymm> Quantized8Asymm;
        megdnn::DTypeParam<dtype::QuantizedS8> QuantizedS8;
        megdnn::DTypeParam<dtype::QuantizedS32> QuantizedS32;
    } param;

    operator DType() const {
        switch (enumv) {
#define cb(_dt)          \
    case DTypeEnum::_dt: \
        return DType::from_enum(enumv);
            MEGDNN_FOREACH_DTYPE_NAME(cb)
#undef cb
            case DTypeEnum::Quantized8Asymm:
                return dtype::Quantized8Asymm{param.Quantized8Asymm};
            case DTypeEnum::QuantizedS8:
                return dtype::QuantizedS8{param.QuantizedS8};
            case DTypeEnum::QuantizedS32:
                return dtype::QuantizedS32{param.QuantizedS32};
            default:
                mgb_assert(
                        false, "unexpected old serialized dtype: invalid enumv %d",
                        static_cast<uint32_t>(enumv));
        }
    }
};
template <>
struct OprPersistentParam<opr::AddUpdate> {
    using Param = opr_add_update::PersistentAddUpdateParam;
};

#if MGB_ENABLE_FBS_SERIALIZATION
namespace fbs {
using namespace opr_add_update;
template <>
struct ParamConverter<PersistentAddUpdateParam> {
    using FlatBufferType = fbs::param::MGBAddUpdate;
    static PersistentAddUpdateParam to_param(const FlatBufferType* fb) {
        auto c = [](const auto* s) -> PersistentDTypeScalar {
            return {reinterpret_cast<const dt_byte*>(s->storage()->data()),
                    intl::convert_dtype_to_megdnn(s->dtype())};
        };
        return {c(fb->alpha()), c(fb->beta()), c(fb->bias())};
    }
    static flatbuffers::Offset<FlatBufferType> to_flatbuffer(
            flatbuffers::FlatBufferBuilder& builder,
            const PersistentAddUpdateParam& p) {
        auto c = [](const PersistentDTypeScalar& v) {
            auto res =
                    param::PersistentDTypeScalar(intl::convert_dtype_to_fbs(v.dtype));
            memcpy(res.mutable_storage()->data(), v.storage, sizeof(v.storage));
            return res;
        };
        auto alpha = c(p.alpha), beta = c(p.beta), bias = c(p.bias);
        return param::CreateMGBAddUpdate(builder, &alpha, &beta, &bias);
    }
};
template <>
struct ParamConverter<megdnn::DType> {
    using FlatBufferType = fbs::DType;
    static megdnn::DType to_param(const fbs::DType* fb) {
        return fbs::intl::load_dtype(fb);
    }
    static flatbuffers::Offset<fbs::DType> to_flatbuffer(
            flatbuffers::FlatBufferBuilder& builder, megdnn::DType dtype) {
        return fbs::intl::build_dtype(builder, dtype);
    }
};
template <>
struct ParamConverter<SerializedDTypeV1> {
    using FlatBufferType = SerializedDTypeV1;
    static SerializedDTypeV1 to_param(const FlatBufferType* fb) {
        mgb_assert(
                false,
                "You are calling SerializedDTypeV1 in flatbuffer, you should not call "
                "here, this code is just to avoid compiling errors, but not be used in "
                "flatbuffer.");
    }
};
};  // namespace fbs
#endif

template <>
struct OprMaker<opr::Elemwise, 0> : public OprMakerVariadic<opr::Elemwise> {};

template <>
struct OprMaker<opr::Reduce, 0> {
    using Opr = opr::Reduce;
    using Param = Opr::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& inputs, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        SymbolVar target_shape;
        if (inputs.size() == 1) {
            mgb_throw_if(
                    param.axis < -megdnn::param::OptionalAxisV1::MAX_NDIM ||
                            param.axis >= megdnn::param::OptionalAxisV1::MAX_NDIM,
                    MegBrainError, "DIM error");
        } else {
            mgb_assert(inputs.size() == 2);
            target_shape = inputs[1];
        }
        return Opr::make(inputs[0], param, target_shape, config).node()->owner_opr();
    }
};

}  // namespace serialization

namespace opr {
cg::OperatorNodeBase* opr_shallow_copy_add_update(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    mgb_assert(inputs.size() == 2);
    auto&& opr = opr_.cast_final_safe<AddUpdate>();
    return AddUpdate::make(inputs[0], inputs[1], opr.param(), config)
            .node()
            ->owner_opr();
}

MGB_SEREG_OPR(Elemwise, 0);
MGB_SEREG_OPR(PowC, 1);
MGB_SEREG_OPR(AddUpdate, 2);
MGB_REG_OPR_SHALLOW_COPY(AddUpdate, opr_shallow_copy_add_update);

//! current reduce version
using ReduceV2 = opr::Reduce;
MGB_SEREG_OPR(ReduceV2, 0);
}  // namespace opr
using TypeCvtV2 = opr::TypeCvt;
MGB_SEREG_OPR(TypeCvtV2, 1);

}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
