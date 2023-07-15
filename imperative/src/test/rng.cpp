#include "megbrain/imperative/ops/rng.h"
#include "./helper.h"

using namespace mgb;
using namespace imperative;
using namespace imperative::rng;

template <typename Op, typename... Args>
void check_rng_basic(Args&&... args) {
    for (auto&& tshape : {TensorShape{2, 3, 4, 5}, {3, 4, 5, 6}, {2333}})
        for (auto&& cn : {CompNode::load("xpu0"), CompNode::load("xpu1")}) {
            Handle h = new_handle(cn, 123);
            auto op = Op::make(std::forward<Args>(args)..., h);
            DeviceTensorND tshape_dev;
            cg::copy_shape_to_tensor_value(tshape_dev, tshape);
            SmallVector<TensorPtr> inputs = {Tensor::make(tshape_dev)};
            SmallVector<LogicalTensorDesc> input_descs;
            input_descs.push_back(
                    {inputs[0]->layout(), inputs[0]->comp_node(),
                     inputs[0]->dev_tensor()});
            auto [output_descs, validated] =
                    OpDef::infer_output_attrs_fallible(*op, input_descs);
            auto outputs = OpDef::apply_on_physical_tensor(
                    *op, inputs, output_descs, validated);
            ASSERT_TRUE(outputs[0]->layout().eq_shape(tshape));
            ASSERT_TRUE(cn == outputs[0]->comp_node());
            // sync before delete handle
            for (auto&& p : outputs) {
                p->get_value();
            }
            delete_handle(h);
        }
}

template <typename Op, typename... Args>
void check_rng_with_input_basic(
        const CompNode& cn, const SmallVector<TensorPtr>& inputs, Args&&... args) {
    Handle h = new_handle(cn, 123);
    auto op = Op::make(std::forward<Args>(args)..., h);
    SmallVector<LogicalTensorDesc> input_descs;
    for (auto&& i : inputs) {
        input_descs.push_back({i->layout(), i->comp_node(), i->dev_tensor()});
    }
    auto [output_descs, validated] =
            OpDef::infer_output_attrs_fallible(*op, input_descs);
    auto outputs =
            OpDef::apply_on_physical_tensor(*op, inputs, output_descs, validated);
    ASSERT_TRUE(outputs[0]->layout().eq_shape(inputs[0]->shape()));
    ASSERT_TRUE(cn == outputs[0]->comp_node());
    // sync before delete handle
    for (auto&& p : outputs) {
        p->get_value();
    }
    delete_handle(h);
}

TEST(TestImperative, PoissonRNGBasic) {
    REQUIRE_XPU(2);
    for (auto&& cn : {CompNode::load("xpu0"), CompNode::load("xpu1")}) {
        TensorShape shape{5, 3000};
        HostTensorND lam{cn, shape, dtype::Float32()};
        auto lam_ptr = lam.ptr<float>();
        for (int i = 0; i < 5 * 3000; ++i)
            lam_ptr[i] = 2;
        SmallVector<TensorPtr> inputs{Tensor::make(lam)};
        check_rng_with_input_basic<PoissonRNG>(cn, inputs, 123);
    }
}

TEST(TestImperative, BetaRNGBasic) {
    REQUIRE_XPU(2);
    for (auto&& cn : {CompNode::load("xpu0"), CompNode::load("xpu1")}) {
        TensorShape shape{5, 3000};
        HostTensorND alpha{cn, shape, dtype::Float32()},
                beta{cn, shape, dtype::Float32()};
        auto lam_ptr = alpha.ptr<float>(), beta_ptr = beta.ptr<float>();
        for (int i = 0; i < 5 * 3000; ++i)
            lam_ptr[i] = 2, beta_ptr[i] = 2;
        SmallVector<TensorPtr> inputs{Tensor::make(alpha), Tensor::make(beta)};
        check_rng_with_input_basic<BetaRNG>(cn, inputs, 123);
    }
}

TEST(TestImperative, GammaRNGBasic) {
    REQUIRE_XPU(2);
    for (auto&& cn : {CompNode::load("xpu0"), CompNode::load("xpu1")}) {
        TensorShape size{5, 3000};
        HostTensorND shape{cn, size, dtype::Float32()},
                scale{cn, size, dtype::Float32()};
        auto shape_ptr = shape.ptr<float>(), scale_ptr = scale.ptr<float>();
        for (int i = 0; i < 5 * 3000; ++i)
            shape_ptr[i] = 2, scale_ptr[i] = 2;
        SmallVector<TensorPtr> inputs{Tensor::make(shape), Tensor::make(scale)};
        check_rng_with_input_basic<GammaRNG>(cn, inputs, 123);
    }
}

TEST(TestImperative, UniformRNGBasic) {
    REQUIRE_XPU(2);
    check_rng_basic<UniformRNG>(123, dtype::Float32());
}

TEST(TestImperative, GaussianRNGBasic) {
    REQUIRE_XPU(2);
    check_rng_basic<GaussianRNG>(123, 2.f, 3.f, dtype::Float32());
}

TEST(TestImperative, PermutationRNGBasic) {
    REQUIRE_XPU(2);
    check_rng_basic<PermutationRNG>(123, dtype::Int32());
}

TEST(TestImperative, ExponentialRNGBasic) {
    REQUIRE_XPU(2);
    for (auto&& cn : {CompNode::load("xpu0"), CompNode::load("xpu1")}) {
        TensorShape shape{5, 3000};
        HostTensorND scale{cn, shape, dtype::Float32()};
        auto scale_ptr = scale.ptr<float>();
        for (int i = 0; i < 5 * 3000; ++i)
            scale_ptr[i] = 2;
        SmallVector<TensorPtr> inputs{Tensor::make(scale)};
        check_rng_with_input_basic<ExponentialRNG>(cn, inputs, 123);
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
