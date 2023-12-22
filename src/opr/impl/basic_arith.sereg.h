#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/internal/param_tag_defs.h"
#include "megbrain/opr/io.h"
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
struct OprLoadDumpImplV2<opr::Elemwise, 0> {
    using Opr = opr::Elemwise;
    using PersisParam = opr::Elemwise::Param;
    using PersisElemwseiParam = opr::Elemwise::Param;
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        ctx.write_param<PersisParam>(opr.cast_final_safe<Opr>().param());
    }

    static cg::OperatorNodeBase* replace_opr(
            cg::OperatorNodeBase* opr, const VarNodeArray& inputs) {
        auto mode = opr->cast_final_safe<Opr>().param().mode;
        auto astype = [](VarNode* inp, VarNode* ref) {
            return opr::TypeCvt::make(inp, ref->dtype()).node();
        };
        auto make_const = [](DTypeScalar val, VarNode* ref) {
            return opr::ImmutableTensor::make(
                           *ref->owner_graph(), val, ref->comp_node())
                    .node();
        };
        auto float_half = DTypeScalar(static_cast<megdnn::dt_float32>(0.5));
        auto float_one = DTypeScalar(static_cast<megdnn::dt_float32>(1.0));
        auto float_two = DTypeScalar(static_cast<megdnn::dt_float32>(2.0));
        auto float_zero = DTypeScalar(static_cast<megdnn::dt_float32>(0.0));
        auto float_six = DTypeScalar(static_cast<megdnn::dt_float32>(6.0));
        auto float_three = DTypeScalar(static_cast<megdnn::dt_float32>(3.0));
        if (PersisParam::Mode::SQRT == mode) {
            auto elemwise_mode = PersisParam::Mode::POW;
            auto half_var = make_const(float_half, inputs[0]);
            if (inputs[0]->dtype() != half_var->dtype()) {
                half_var = astype(half_var, inputs[0]);
            }
            return opr::Elemwise::make({inputs[0], half_var}, elemwise_mode)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::SQUARE == mode) {
            auto elemwise_mode = PersisParam::Mode::POW;
            auto two_var = make_const(float_two, inputs[0]);
            if (inputs[0]->dtype() != two_var->dtype()) {
                two_var = astype(two_var, inputs[0]);
            }
            return opr::Elemwise::make({inputs[0], two_var}, elemwise_mode)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::TAN == mode) {
            auto sin = opr::Elemwise::make({inputs[0]}, PersisParam::Mode::SIN).node();
            auto cos = opr::Elemwise::make({inputs[0]}, PersisParam::Mode::COS).node();
            return opr::Elemwise::make({sin, cos}, PersisParam::Mode::TRUE_DIV)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::COSH == mode) {
            auto half_var = make_const(float_half, inputs[0]);
            if (inputs[0]->dtype() != half_var->dtype())
                half_var = astype(half_var, inputs[0]);
            auto expx = opr::Elemwise::make({inputs[0]}, PersisParam::Mode::EXP).node();
            auto negatex =
                    opr::Elemwise::make({inputs[0]}, PersisParam::Mode::NEGATE).node();
            auto expnegatex =
                    opr::Elemwise::make({negatex}, PersisParam::Mode::EXP).node();
            return opr::Elemwise::make(
                           {half_var,
                            opr::Elemwise::make(
                                    {expx, expnegatex}, PersisParam::Mode::ADD)
                                    .node()},
                           PersisParam::Mode::MUL)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::SINH == mode) {
            auto inp = inputs[0];
            auto two_var = make_const(float_two, inputs[0]);
            auto half_var = make_const(float_half, inputs[0]);
            auto one_var = make_const(float_one, inputs[0]);
            if (inp->dtype() != two_var->dtype()) {
                two_var = astype(two_var, inputs[0]);
                half_var = astype(half_var, inputs[0]);
                one_var = astype(one_var, inputs[0]);
            }
            auto u = opr::Elemwise::make({inp}, PersisParam::Mode::EXPM1).node();
            auto tmp1 =
                    opr::Elemwise::make({u, half_var}, PersisParam::Mode::MUL).node();
            auto uadd1 =
                    opr::Elemwise::make({u, one_var}, PersisParam::Mode::ADD).node();
            auto uadd2 =
                    opr::Elemwise::make({u, two_var}, PersisParam::Mode::ADD).node();
            auto tmp2 = opr::Elemwise::make({tmp1, uadd1}, PersisParam::Mode::TRUE_DIV)
                                .node();
            return opr::Elemwise::make({tmp2, uadd2}, PersisParam::Mode::MUL)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::ASINH == mode) {
            auto inp = inputs[0];
            auto two_var = make_const(float_two, inp);
            auto half_var = make_const(float_half, inp);
            auto one_var = make_const(float_one, inp);
            if (inp->dtype() != two_var->dtype()) {
                two_var = astype(two_var, inputs[0]);
                half_var = astype(half_var, inputs[0]);
                one_var = astype(one_var, inputs[0]);
            }
            auto inp2 =
                    opr::Elemwise::make({inp, two_var}, PersisParam::Mode::POW).node();
            auto inp2add1 =
                    opr::Elemwise::make({inp2, one_var}, PersisParam::Mode::ADD).node();
            auto inp2add1sqrt =
                    opr::Elemwise::make({inp2add1, half_var}, PersisParam::Mode::POW)
                            .node();
            auto tmp = opr::Elemwise::make({inp, inp2add1sqrt}, PersisParam::Mode::ADD)
                               .node();
            return opr::Elemwise::make({tmp}, PersisElemwseiParam::Mode::LOG)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::ACOSH == mode) {
            auto inp = inputs[0];
            auto two_var = make_const(float_two, inp);
            auto half_var = make_const(float_half, inp);
            auto one_var = make_const(float_one, inp);
            if (inp->dtype() != two_var->dtype()) {
                two_var = astype(two_var, inputs[0]);
                half_var = astype(half_var, inputs[0]);
                one_var = astype(one_var, inputs[0]);
            }
            auto inp2 =
                    opr::Elemwise::make({inp, two_var}, PersisParam::Mode::POW).node();
            auto inp2sub1 =
                    opr::Elemwise::make({inp2, one_var}, PersisParam::Mode::SUB).node();
            auto inp2sub1sqrt =
                    opr::Elemwise::make({inp2sub1, half_var}, PersisParam::Mode::POW)
                            .node();
            auto tmp = opr::Elemwise::make({inp, inp2sub1sqrt}, PersisParam::Mode::ADD)
                               .node();
            return opr::Elemwise::make({tmp}, PersisElemwseiParam::Mode::LOG)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::ATANH == mode) {
            auto inp = inputs[0];
            auto two_var = make_const(float_two, inp);
            auto one_var = make_const(float_one, inp);
            if (inp->dtype() != two_var->dtype()) {
                two_var = astype(two_var, inputs[0]);
                one_var = astype(one_var, inputs[0]);
            }
            auto tmp1 =
                    opr::Elemwise::make({two_var, inp}, PersisParam::Mode::MUL).node();
            auto tmp2 =
                    opr::Elemwise::make({one_var, inp}, PersisParam::Mode::SUB).node();
            auto tmp3 = opr::Elemwise::make({tmp1, tmp2}, PersisParam::Mode::TRUE_DIV)
                                .node();
            auto log1p = opr::Elemwise::make({tmp3}, PersisParam::Mode::LOG1P).node();
            return opr::Elemwise::make({log1p, two_var}, PersisParam::Mode::TRUE_DIV)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::CLIP == mode) {
            auto tmp =
                    opr::Elemwise::make({inputs[0], inputs[1]}, PersisParam::Mode::MAX)
                            .node();
            return opr::Elemwise::make({tmp, inputs[2]}, PersisParam::Mode::MIN)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::SIGN == mode) {
            auto zero_var = make_const(float_zero, inputs[0]);
            zero_var = astype(zero_var, inputs[0]);
            auto tmp1 =
                    opr::Elemwise::make({inputs[0], zero_var}, PersisParam::Mode::LT)
                            .node();
            auto tmp2 =
                    opr::Elemwise::make({zero_var, inputs[0]}, PersisParam::Mode::LT)
                            .node();
            return opr::Elemwise::make({tmp1, tmp2}, PersisParam::Mode::SUB)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::HSIGMOID == mode) {
            auto six_var = make_const(float_six, inputs[0]);
            auto zero_var = make_const(float_zero, inputs[0]);
            auto three_var = make_const(float_three, inputs[0]);
            if (inputs[0]->dtype() != six_var->dtype()) {
                six_var = astype(six_var, inputs[0]);
                zero_var = astype(zero_var, inputs[0]);
                three_var = astype(three_var, inputs[0]);
            }
            auto tmp1 =
                    opr::Elemwise::make({inputs[0], three_var}, PersisParam::Mode::ADD)
                            .node();
            auto tmp2 = opr::Elemwise::make({tmp1, zero_var}, PersisParam::Mode::MAX)
                                .node();
            auto tmp3 =
                    opr::Elemwise::make({tmp2, six_var}, PersisParam::Mode::MIN).node();
            return opr::Elemwise::make({tmp3, six_var}, PersisParam::Mode::TRUE_DIV)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::RELU6 == mode) {
            auto six_var = make_const(float_six, inputs[0]);
            auto zero_var = make_const(float_zero, inputs[0]);
            six_var = astype(six_var, inputs[0]);
            zero_var = astype(zero_var, inputs[0]);
            auto max_0 =
                    opr::Elemwise::make({inputs[0], zero_var}, PersisParam::Mode::MAX)
                            .node();
            return opr::Elemwise::make({max_0, six_var}, PersisParam::Mode::MIN)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::PRELU == mode) {
            auto zero_var = make_const(float_zero, inputs[0]);
            auto inp = inputs[0];
            auto weight = inputs[1];
            if (inp->dtype() != zero_var->dtype()) {
                zero_var = astype(zero_var, inp);
            }
            auto min_0 =
                    opr::Elemwise::make({inp, zero_var}, PersisParam::Mode::MIN).node();
            auto max_0 =
                    opr::Elemwise::make({inp, zero_var}, PersisParam::Mode::MAX).node();
            return opr::Elemwise::make(
                           {min_0, weight, max_0}, PersisParam::Mode::FUSE_MUL_ADD3)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::SOFTPLUS == mode) {
            auto inp = inputs[0];
            auto abs = opr::Elemwise::make({inp}, PersisParam::Mode::ABS).node();
            auto neg_abs = opr::Elemwise::make({abs}, PersisParam::Mode::NEGATE).node();
            auto exp = opr::Elemwise::make({neg_abs}, PersisParam::Mode::EXP).node();
            auto oup0 = opr::Elemwise::make({exp}, PersisParam::Mode::LOG1P).node();
            auto oup1 = opr::Elemwise::make({inp}, PersisParam::Mode::RELU).node();
            return opr::Elemwise::make({oup0, oup1}, PersisParam::Mode::ADD)
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::LOGSIGMOID == mode) {
            auto inp = inputs[0];
            auto abs = opr::Elemwise::make({inp}, PersisParam::Mode::ABS).node();
            auto neg_abs = opr::Elemwise::make({abs}, PersisParam::Mode::NEGATE).node();
            auto exp = opr::Elemwise::make({neg_abs}, PersisParam::Mode::EXP).node();
            auto oup0 = opr::Elemwise::make({exp}, PersisParam::Mode::LOG1P).node();
            auto neg_inp = opr::Elemwise::make({inp}, PersisParam::Mode::NEGATE).node();
            auto oup1 = opr::Elemwise::make({neg_inp}, PersisParam::Mode::RELU).node();
            auto oup = opr::Elemwise::make({oup0, oup1}, PersisParam::Mode::ADD).node();
            return opr::Elemwise::make({oup}, PersisParam::Mode::NEGATE)
                    .node()
                    ->owner_opr();
        }
        return opr;
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        return OprMaker<opr::Elemwise, 0>::make(
                ctx.read_param<PersisParam>(), inputs, ctx.graph(), config);
    }
};
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

MGB_SEREG_OPR_V1_WITH_CONVERTER(
        Elemwise, 0,
        (mgb::serialization::OprLoadDumpImplV2<opr::Elemwise, 0>::replace_opr),
        nullptr);
MGB_SEREG_OPR_V2_HASH_WITHOUT_TAIL_0(
        Elemwise, 0,
        (mgb::serialization::OprLoadDumpImplV2<opr::Elemwise, 0>::replace_opr),
        VERSION_1, VERSION_1);
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
