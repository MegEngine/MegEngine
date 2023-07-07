#pragma once

#include <cstddef>

#include "megbrain/common.h"
#include "megbrain/exception.h"
#include "megbrain/imperative/basic_operators.h"
#include "megbrain/imperative/basic_values.h"
#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/operator.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/transformation.h"
#include "megbrain/imperative/utils/helper.h"
#include "megbrain/imperative/utils/span.h"
#include "megbrain/imperative/value.h"
#include "megbrain/tensor.h"
#include "megdnn/thin/small_vector.h"

namespace mgb {
namespace imperative {

class ComplexTensor final : public ObjectValue<ComplexTensor> {
private:
    ValueRef m_real;
    ValueRef m_imag;

public:
    ComplexTensor(ValueRef real, ValueRef imag) : m_real(real), m_imag(imag) {}

    std::string to_string() const override {
        return ssprintf(
                "ComplexTensor{m_real=%s, m_imag=%s}", m_real.to_string().c_str(),
                m_imag.to_string().c_str());
    }

    DTypeValue::ref_t dtype() const {
        auto dtype = m_real.dtype();
        mgb_assert(dtype == m_imag.dtype());
        return dtype;
    }

    const ValueRef& real() const { return m_real; }

    const ValueRef imag() const { return m_imag; }

    /**
     * \brief clear all states of this value
     *
     */
    void clear() override {
        m_real = {};
        m_imag = {};
    }
};

class CreateComplex final : public OperatorImpl<CreateComplex> {
public:
    std::string to_string() const override { return "CreateComplex"; }

    std::string raw_type() const override { return "CreateComplex"; }
};

class GetReal final : public OperatorImpl<GetReal> {
public:
    std::string to_string() const override { return "GetReal"; }

    std::string raw_type() const override { return "GetReal"; }
};

class GetImag final : public OperatorImpl<GetImag> {
public:
    std::string to_string() const override { return "GetImag"; }

    std::string raw_type() const override { return "GetImag"; }
};

class ComplexTransformation final : public Transformation {
private:
    ObjectType<ComplexTensor> m_complex_type{"Complex"};

public:
    std::string name() const override { return "ComplexTransformation"; }

    HostTensorND make_complex_tensor(HostTensorND real, HostTensorND imag) {
        mgb_assert(real.shape().eq_shape(imag.shape()));
        mgb_assert(
                real.dtype() == dtype::Float32() && imag.dtype() == dtype::Float32());
        mgb_assert(real.comp_node() == imag.comp_node());
        HostTensorND complex{real.comp_node(), real.shape(), dtype::Complex64()};
        TensorShape f32_shape = complex.shape();
        f32_shape[f32_shape.ndim++] = 2;
        TensorLayout f32_layout = {f32_shape, dtype::Float32()};
        f32_layout.init_contiguous_stride();
        HostTensorND f32{complex.comp_node(), f32_layout};
        f32.storage(complex.storage());
        TensorLayout real_layout = f32_layout;
        real_layout.ndim--;
        TensorLayout imag_layout = real_layout;
        // mgb_assert(!real_layout.is_contiguous());
        // mgb_assert(!imag_layout.is_contiguous());
        f32.sub(SubTensorSpec::make_from_layout(real_layout)).copy_from_fixlayout(real);
        f32.sub(SubTensorSpec::make_from_offset_elem(imag_layout, 1))
                .copy_from_fixlayout(imag);
        return complex;
    }

    ValueRefList apply_complex_mask(
            const ApplyOp& apply_op, Span<ValueRef> inputs, Span<bool> mask) {
        ValueRefList real_list(inputs.size());
        ValueRefList imag_list(inputs.size());
        bool any_complex = false;
        bool all_complex = true;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (auto* complex = inputs[i].as(m_complex_type)) {
                mgb_assert(mask[i], "unexpected complex");
                any_complex = true;
                real_list[i] = complex->real();
                imag_list[i] = complex->imag();
            } else {
                real_list[i] = inputs[i];
                if (mask[i]) {
                    all_complex = false;
                } else {
                    imag_list[i] = inputs[i];
                }
            }
        }
        if (!any_complex) {
            // no complex
            return imperative::apply(apply_op, real_list);
        } else {
            // all complex
            mgb_assert(all_complex, "only serval inputs are complex");
            auto reals = imperative::apply(apply_op, real_list);
            auto imags = imperative::apply(apply_op, imag_list);
            mgb_assert(reals.size() == imags.size());
            ValueRefList results(reals.size());
            for (size_t i = 0; i < results.size(); ++i) {
                results[i] = m_complex_type.make(reals[i], imags[i]);
            }
            return results;
        }
    }

    ValueRefList apply_complex_real(const ApplyOp& apply_op, Span<ValueRef> inputs) {
        ValueRefList real_list(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (auto* complex = inputs[i].as(m_complex_type)) {
                real_list[i] = complex->real();
            } else {
                real_list[i] = inputs[i];
            }
        }
        return imperative::apply(apply_op, real_list);
    }

    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override {
        if (auto* create_complex = op.as<CreateComplex>()) {
            auto [real, imag] = inputs.as_array<2>();
            auto dtype_real = real.dtype();
            auto dtype_imag = imag.dtype();
            mgb_assert(
                    *dtype_real == *dtype_imag, "dtype mismatch: %s vs %s",
                    dtype_real->name(), dtype_imag->name());
            return {m_complex_type.make(real, imag)};
        } else if (auto* create_tensor = op.as<CreateTensor>()) {
            if (create_tensor->dtype().is_complex()) {
                auto args = create_tensor->parse(inputs);
                mgb_assert(!args.device);
                auto& host = *args.host;
                // reinterpret_cast to f32
                mgb_assert(host.layout().is_physical_contiguous());
                mgb_assert(host.dtype() == dtype::Complex64());
                TensorShape f32_shape = host.shape();
                f32_shape[f32_shape.ndim++] = 2;
                TensorLayout f32_layout = {f32_shape, dtype::Float32()};
                HostTensorND f32_host = {host.comp_node(), f32_layout};
                f32_host.storage(host.storage());
                // take real slice and imag slice
                auto real_layout = f32_layout;
                real_layout[real_layout.ndim - 1] = 1;
                auto imag_layout = real_layout;
                auto real_host =
                        f32_host.sub(SubTensorSpec::make_from_layout(real_layout));
                auto imag_host = f32_host.sub(
                        SubTensorSpec::make_from_offset_elem(imag_layout, 1));
                // copy into continuous
                real_layout.init_contiguous_stride();
                imag_layout.init_contiguous_stride();
                auto real_value = HostTensorND{create_tensor->device(), real_layout};
                real_value.copy_from_fixlayout(real_host);
                auto imag_value = HostTensorND{create_tensor->device(), imag_layout};
                imag_value.copy_from_fixlayout(imag_host);
                // create real and imag
                auto real = imperative::apply(
                        CreateTensor(
                                create_tensor->kind(), create_tensor->device(),
                                real_layout),
                        HostStorage::make(real_value.storage()))[0];
                auto imag = imperative::apply(
                        CreateTensor(
                                create_tensor->kind(), create_tensor->device(),
                                imag_layout),
                        HostStorage::make(imag_value.storage()))[0];
                return {m_complex_type.make(real, imag)};
            } else {
                return imperative::apply(op, inputs);
            }
        }
        bool any_complex = false;
        for (auto&& input : inputs) {
            if (input.is(m_complex_type)) {
                any_complex = true;
                break;
            }
        }
        if (!any_complex) {
            return imperative::apply(op, inputs);
        }
        if (auto* apply_op = op.as<ApplyOp>()) {
            // TODO: handle apply op
            // see https://zhuanlan.zhihu.com/p/627536105
            if (auto* elemwise = apply_op->op().try_cast_final<Elemwise>()) {
                switch (elemwise->mode) {
                    case Elemwise::Mode::MUL: {
                        auto* complex_a = inputs[0].as(m_complex_type);
                        auto* complex_b = inputs[1].as(m_complex_type);
                        auto& mul = *apply_op;
                        if (complex_a && complex_b) {
                            auto add = Elemwise::make(Elemwise::Mode::ADD);
                            auto sub = Elemwise::make(Elemwise::Mode::SUB);
                            auto real = imperative::apply(
                                    *sub,
                                    imperative::apply(
                                            mul, complex_a->real(),
                                            complex_b->real())[0],
                                    imperative::apply(
                                            mul, complex_a->imag(),
                                            complex_b->imag())[0])[0];
                            auto imag = imperative::apply(
                                    *add,
                                    imperative::apply(
                                            mul, complex_a->real(),
                                            complex_b->imag())[0],
                                    imperative::apply(
                                            mul, complex_a->imag(),
                                            complex_b->real())[0])[0];
                            return {m_complex_type.make(real, imag)};
                        } else if (complex_a) {
                            auto real = imperative::apply(
                                    mul, complex_a->real(), inputs[1])[0];
                            auto imag = imperative::apply(
                                    mul, complex_a->imag(), inputs[1])[0];
                            return {m_complex_type.make(real, imag)};
                        } else if (complex_b) {
                            auto real = imperative::apply(
                                    mul, complex_b->real(), inputs[0])[0];
                            auto imag = imperative::apply(
                                    mul, complex_b->imag(), inputs[0])[0];
                            return {m_complex_type.make(real, imag)};
                        } else {
                            mgb_assert(0);
                        }
                    }
                    case Elemwise::Mode::ADD:
                    case Elemwise::Mode::SUB: {
                        bool mask[2] = {true, true};
                        return apply_complex_mask(*apply_op, inputs, {mask, 2});
                    }
                    case Elemwise::Mode::NEGATE: {
                        bool mask[1] = {true};
                        return apply_complex_mask(*apply_op, inputs, {mask, 1});
                    }
                    default: {
                        mgb_assert(0, "unsupported elemwise mode");
                    }
                }
            } else if (auto* reshape = apply_op->op().try_cast_final<Reshape>()) {
                SmallVector<bool> mask(inputs.size(), false);
                mask[0] = true;
                return apply_complex_mask(*apply_op, inputs, mask);
            } else if (auto* subtensor = apply_op->op().try_cast_final<Subtensor>()) {
                SmallVector<bool> mask(inputs.size(), false);
                mask[0] = true;
                return apply_complex_mask(*apply_op, inputs, mask);
            } else if (auto* get_shape = apply_op->op().try_cast_final<GetVarShape>()) {
                return apply_complex_real(*apply_op, inputs);
            } else {
                mgb_assert(0, "unsupported operator");
            }
        } else if (auto* get_attr = op.as<GetAttr>()) {
            // TODO: handle get attr
            auto&& input = inputs[0].as_ref(m_complex_type);
            switch (get_attr->attr()) {
                case GetAttr::DType:
                    switch (input->dtype()->enumv()) {
                        case DTypeEnum::Float32: {
                            return {DTypeValue::make(dtype::Complex64())};
                        }
                        default:
                            mgb_assert(
                                    0, "unsupported dtype %s", input->dtype()->name());
                    }
                case GetAttr::Device:
                case GetAttr::Shape:
                    return imperative::apply(op, input->real());
                case GetAttr::Value: {
                    auto complex = make_complex_tensor(
                            input->real().numpy()->as_nd(),
                            input->imag().numpy()->as_nd());
                    return {HostValue::make(complex)};
                }
                default:
                    mgb_throw(
                            MegBrainError, "unsupported %s for complex",
                            get_attr->to_string().c_str());
            }
        } else if (auto* as_real = op.as<GetReal>()) {
            auto&& input = inputs[0].as_ref(m_complex_type);
            return {input->real()};
        } else if (auto* as_real = op.as<GetImag>()) {
            auto&& input = inputs[0].as_ref(m_complex_type);
            return {input->imag()};
        }
        mgb_throw(
                MegBrainError, "unsupported op for complex: %s",
                op.to_string().c_str());
    }

    ValueRef unwrap(ValueRef value) override {
        mgb_assert(!value.is(m_complex_type), "cannot unwrap complex value");
        return value;
    }
};

}  // namespace imperative
}  // namespace mgb
