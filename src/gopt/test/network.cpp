#include "./network.h"
#include "megbrain/opr/tensor_manip.h"

using namespace mgb;

SymbolVar Network::add_conv(
        SymbolVar f, size_t output_channels, KernSize kern_size, DType out_dtype,
        bool has_relu, Stride stride, Padding padding) {
    static int weight_idx = 0;
    static int bias_idx = 0;

    size_t input_channels = f.node()->shape()[1];
    auto weight = add_cvar(
            ssprintf("w%d", weight_idx).c_str(),
            {output_channels, input_channels, kern_size[0], kern_size[1]});
    auto bias = add_cvar(ssprintf("b%d", bias_idx).c_str(), {1, output_channels, 1, 1});
    if (out_dtype.category() == DTypeCategory::QUANTIZED) {
        weight = add_type_cvt(weight, out_dtype);
        bias = add_type_cvt(bias, dtype::QuantizedS32{1.f});
    }
    opr::ConvBias::Param param;
    param.stride_h = stride[0], param.stride_w = stride[1];
    param.pad_h = padding[0], param.pad_w = padding[1];
    if (has_relu) {
        param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;
    } else {
        param.nonlineMode = opr::ConvBias::Param::NonlineMode::IDENTITY;
    }

    SymbolVar conv;
    if (out_dtype.category() == DTypeCategory::QUANTIZED) {
        conv = opr::ConvBias::make(
                f, weight, bias, param, {}, OperatorNodeConfig{out_dtype});
    } else {
        conv = opr::ConvBias::make(f, weight, bias, param, {});
    }
    weight_idx++;
    bias_idx++;
    return conv;
}

SymbolVar Network::add_group_conv(
        SymbolVar f, size_t output_channels, size_t groups, KernSize kern_size,
        DType out_dtype, bool has_relu, Stride stride, Padding padding) {
    static int weight_idx = 0;
    static int bias_idx = 0;

    size_t input_channels = f.node()->shape()[1];
    auto weight = add_cvar(
            ssprintf("w%d", weight_idx).c_str(),
            {groups, output_channels / groups, input_channels / groups, kern_size[0],
             kern_size[1]});
    auto bias = add_cvar(ssprintf("b%d", bias_idx).c_str(), {1, output_channels, 1, 1});
    if (out_dtype.category() == DTypeCategory::QUANTIZED) {
        weight = add_type_cvt(weight, out_dtype);
        bias = add_type_cvt(bias, dtype::QuantizedS32{1.f});
    }
    opr::ConvBias::Param param;
    param.sparse = opr::ConvBias::Param::Sparse::GROUP;
    param.stride_h = stride[0], param.stride_w = stride[1];
    param.pad_h = padding[0], param.pad_w = padding[1];
    if (has_relu) {
        param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;
    } else {
        param.nonlineMode = opr::ConvBias::Param::NonlineMode::IDENTITY;
    }

    weight_idx++;
    bias_idx++;
    SymbolVar conv;
    if (out_dtype.category() == DTypeCategory::QUANTIZED) {
        conv = opr::ConvBias::make(
                f, weight, bias, param, {}, OperatorNodeConfig{out_dtype});
    } else {
        conv = opr::ConvBias::make(f, weight, bias, param, {});
    }
    weight_idx++;
    bias_idx++;
    return conv;
}

SymbolVar Network::add_deconv(
        SymbolVar f, size_t ratio, size_t output_channels, DType out_dtype) {
    static int weight_idx = 0;
    size_t kernel = ratio * 2 - ratio % 2;
    size_t pad = ratio / 2;

    size_t input_channels = f.node()->shape()[1];
    auto weight = add_cvar(
            ssprintf("w%d", weight_idx).c_str(),
            {input_channels, output_channels, kernel, kernel});

    if (out_dtype.category() == DTypeCategory::QUANTIZED) {
        weight = add_type_cvt(weight, out_dtype);
    }
    opr::ConvolutionBackwardData::Param param;
    param.stride_h = param.stride_w = ratio;
    param.pad_h = param.pad_w = pad;

    auto deconv = opr::ConvolutionBackwardData::make(
            weight, f, param, {}, OperatorNodeConfig{out_dtype});
    weight_idx++;
    return deconv;
}

SymbolVar Network::add_elemwise(
        const SymbolVarArray inps, DType out_dtype, opr::Elemwise::Param::Mode mode) {
    using ElemMode = opr::Elemwise::Param::Mode;
    using MultiMode = opr::ElemwiseMultiType::Param::Mode;
    static const ThinHashMap<ElemMode, MultiMode> map = {
            {ElemMode::ADD, MultiMode::QADD},
            {ElemMode::FUSE_ADD_RELU, MultiMode::QFUSE_ADD_RELU}};
    if (out_dtype.category() == DTypeCategory::QUANTIZED) {
        MultiMode alter_mode = map.at(mode);
        return opr::ElemwiseMultiType::make(
                inps, {alter_mode}, OperatorNodeConfig{out_dtype});
    } else {
        return opr::Elemwise::make(inps, mode);
    }
}

SymbolVar Network::add_pooling(
        SymbolVar f, Window window, Stride stride, Padding padding,
        opr::Pooling::Param::Mode mode) {
    opr::Pooling::Param param;
    param.window_h = window[0], param.window_w = window[1];
    param.stride_h = stride[0], param.stride_w = stride[1];
    param.pad_h = padding[0], param.pad_w = padding[1];
    param.mode = mode;
    return opr::Pooling::make(f, param);
}

SymbolVar Network::add_type_cvt(SymbolVar f, DType out_dtype) {
    return opr::TypeCvt::make(f, out_dtype);
}

SymbolVar Network::add_concat(SymbolVar f, SymbolVar g, int axis) {
    return opr::Concat::make({f, g}, axis);
}

SymbolVar Network::add_dimshuffle(SymbolVar f, std::vector<int> pattern) {
    return opr::Dimshuffle::make(f, pattern);
}

SymbolVar Network::add_axisaddremove(SymbolVar f) {
    return opr::AxisAddRemove::make(
            f, {{opr::AxisAddRemove::AxisDesc::Method::REMOVE, {0}}});
}

SymbolVar Network::add_subtensor(SymbolVar f) {
    using AIdx = opr::indexing::AxisIndexer;
    return opr::Subtensor::make(
            f, {AIdx::make_interval(0, f.make_scalar(0), None, None)});
}

SymbolVar Network::add_reshape(SymbolVar f) {
    auto shp = opr::GetVarShape::make(f);
    return opr::Reshape::make(f, shp);
}

SymbolVar Network::add_broadcast(SymbolVar f) {
    auto shp = opr::GetVarShape::make(f);
    return opr::Broadcast::make(f, shp);
}

SymbolVar Network::add_copy(SymbolVar f) {
    return opr::Copy::make(f);
}

SymbolVar mgb::create_block(
        Network& network, SymbolVar f_in, size_t stride, size_t num_outputs1,
        bool has_proj, DType out_dtype) {
    auto proj = f_in;
    if (has_proj) {
        proj = network.add_conv(
                f_in, num_outputs1, {1, 1}, out_dtype, false, {stride, stride});
    }

    auto f = network.add_conv(
            f_in, num_outputs1, {3, 3}, out_dtype, true, {stride, stride}, {1, 1});

    f = network.add_conv(f, num_outputs1, {3, 3}, out_dtype, true, {1, 1}, {1, 1});

    f = network.add_elemwise({f, proj}, out_dtype, opr::Elemwise::Mode::FUSE_ADD_RELU);
    return f;
}

SymbolVar mgb::make_resnet18(Network& network, size_t batch, DType out_dtype) {
    auto data = network.add_var("data", {batch, 4, 224, 224});
    if (out_dtype.category() == DTypeCategory::QUANTIZED)
        data = network.add_type_cvt(data, dtype::QuantizedS8{1.f});
    auto first = out_dtype;
    if (out_dtype.category() == DTypeCategory::QUANTIZED)
        first = dtype::QuantizedS8{1.f};
    auto f = network.add_conv(data, 64, {7, 7}, first, true, {2, 2}, {3, 3});
    if (out_dtype.enumv() == DTypeEnum::QuantizedS4 ||
        out_dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        f = network.add_type_cvt(f, out_dtype);
    }
    f = network.add_pooling(f, {3, 3}, {2, 2}, {1, 1});

    using Vector = SmallVector<size_t, 4>;
    Vector stages = {2, 2, 2, 2};
    Vector mid_outputs = {64, 128, 256, 512};
    Vector enable_stride = {0, 1, 1, 1};
    for (size_t i = 0; i < 4; ++i) {
        auto s = stages[i];
        auto o = mid_outputs[i];
        auto es = enable_stride[i];
        for (size_t j = 0; j < s; ++j) {
            size_t stride = !es || j > 0 ? 1 : 2;
            bool has_proj = j > 0 ? false : true;
            f = create_block(network, f, stride, o, has_proj, out_dtype);
        }
    }
    f = network.add_pooling(
            f, {7, 7}, {7, 7}, {0, 0}, opr::Pooling::Param::Mode::AVERAGE);

    f = network.add_type_cvt(f, dtype::Float32());
    return f;
}

namespace {
SymbolVarArray make_pyramids(Network& network, size_t batch, DType out_dtype) {
    SymbolVarArray pyramids;
    auto data = network.add_var("data", {batch, 3, 256, 256});
    data = data + (-128.f);
    if (out_dtype.category() == DTypeCategory::QUANTIZED)
        data = network.add_type_cvt(data, dtype::QuantizedS8{1.f});
    auto first = out_dtype;
    if (out_dtype.category() == DTypeCategory::QUANTIZED)
        first = dtype::QuantizedS8{1.f};
    auto f = network.add_conv(data, 16, {3, 3}, first, true, {2, 2}, {1, 1});
    f = network.add_conv(f, 16, {3, 3}, first, true, {1, 1}, {1, 1});
    f = network.add_conv(f, 32, {3, 3}, first, true, {2, 2}, {1, 1});
    if (out_dtype.enumv() == DTypeEnum::QuantizedS4 ||
        out_dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        f = network.add_type_cvt(f, out_dtype);
    }

    using Vector = SmallVector<size_t, 4>;
    Vector stages = {3, 6, 6, 3};
    Vector mid_outputs = {32, 64, 128, 256};
    Vector enable_stride = {0, 1, 1, 1};
    for (size_t i = 0; i < 4; ++i) {
        auto s = stages[i];
        auto o = mid_outputs[i];
        auto es = enable_stride[i];
        for (size_t j = 0; j < s; ++j) {
            size_t stride = !es || j > 0 ? 1 : 2;
            bool has_proj = j > 0 ? false : true;
            f = create_block(network, f, stride, o, has_proj, out_dtype);
        }
        pyramids.push_back(f);
    }

    for (size_t i = 0; i < pyramids.size(); ++i) {
        pyramids[i] = network.add_type_cvt(pyramids[i], first);
    }
    return pyramids;
}

SymbolVarArray fusion_pyramids_feature(
        Network& network, SymbolVarArray pyramids, size_t fpn_conv_channels) {
    bool touch = false;
    SymbolVar x;
    SymbolVarArray fpn;
    for (int i = 5; i >= 3; --i) {
        auto f = network.add_conv(
                pyramids[i - 2], fpn_conv_channels, {1, 1}, dtype::QuantizedS8{1.f},
                false, {1, 1}, {0, 0});
        if (!touch) {
            x = f;
            touch = true;
        } else {
            x = network.add_deconv(x, 2, 16, dtype::QuantizedS8{1.f});
            x = network.add_elemwise(
                    {x, f}, dtype::QuantizedS8{1.f}, opr::Elemwise::Mode::ADD);
        }
        fpn.push_back(x);
    }

    x = fpn[0];
    for (int i = 6; i < 8; ++i) {
        x = network.add_conv(
                x, fpn_conv_channels, {3, 3}, dtype::QuantizedS8{1.f}, true, {2, 2},
                {1, 1});
    }
    return fpn;
}
}  // namespace

SymbolVarArray mgb::make_det(Network& network, size_t batch, DType out_dtype) {
    SymbolVarArray outputs;
    auto pyramids = make_pyramids(network, batch, out_dtype);
    auto fpn_hv = fusion_pyramids_feature(network, pyramids, 16);
    auto fpn_plate = fusion_pyramids_feature(network, pyramids, 16);
    outputs.insert(outputs.end(), fpn_hv.begin(), fpn_hv.end());
    outputs.insert(outputs.end(), fpn_plate.begin(), fpn_plate.end());
    return outputs;
}

SymbolVar mgb::bottleneck(
        Network& network, SymbolVar f, size_t input_channels, size_t channels, size_t t,
        size_t stride, DType out_dtype) {
    size_t in_channels = f.node()->shape()[1];
    SymbolVar x = f;
    if (t != 1) {
        x = network.add_conv(
                f, input_channels * t, {1, 1}, out_dtype, true, {1, 1}, {0, 0});
    }
    x = network.add_group_conv(
            x, input_channels * t, input_channels * t, {3, 3}, out_dtype, true,
            {stride, stride}, {1, 1});
    x = network.add_conv(x, channels, {1, 1}, out_dtype, false, {1, 1}, {0, 0});
    if (stride == 1 && in_channels == channels)
        x = f + x;
    return x;
}

SymbolVar mgb::bottleneck_group(
        Network& network, SymbolVar f, size_t input_channels, size_t channels,
        size_t stages, size_t s, size_t t, DType out_dtype) {
    SymbolVar x = f;
    for (size_t i = 0; i < stages; ++i) {
        size_t stride = i == 0 ? s : 1;
        x = bottleneck(network, x, input_channels, channels, t, stride, out_dtype);
        input_channels = channels;
    }
    return x;
}

namespace {
size_t make_divisible(size_t v, size_t divisor) {
    size_t min_value = divisor;
    size_t new_v = std::max(min_value, (v + divisor / 2) / divisor * divisor);
    if (new_v < 0.9 * v)
        new_v += divisor;
    return new_v;
}
}  // namespace

SymbolVar mgb::make_mobilenet_v2(Network& network, size_t batch, DType out_dtype) {
    auto data = network.add_var("data", {batch, 3, 224, 224});
    if (out_dtype.category() == DTypeCategory::QUANTIZED) {
        data = network.add_type_cvt(data, dtype::QuantizedS8{1.f});
    }
    constexpr size_t round_nearest = 8;
    auto x = network.add_conv(
            data, make_divisible(32, round_nearest), {3, 3}, out_dtype, true, {2, 2},
            {1, 1});
    x = bottleneck(network, x, 32, make_divisible(16, round_nearest), 1, 1, out_dtype);
    x = bottleneck_group(
            network, x, 16, make_divisible(24, round_nearest), 2, 2, 6, out_dtype);
    x = bottleneck_group(
            network, x, 24, make_divisible(32, round_nearest), 3, 2, 6, out_dtype);
    x = bottleneck_group(
            network, x, 32, make_divisible(64, round_nearest), 4, 2, 6, out_dtype);
    x = bottleneck_group(
            network, x, 64, make_divisible(96, round_nearest), 3, 1, 6, out_dtype);
    x = bottleneck_group(
            network, x, 96, make_divisible(160, round_nearest), 3, 2, 6, out_dtype);
    x = bottleneck_group(
            network, x, 160, make_divisible(320, round_nearest), 1, 1, 6, out_dtype);
    x = network.add_conv(
            x, make_divisible(1280, round_nearest), {1, 1}, out_dtype, true, {1, 1},
            {0, 0});
    if (out_dtype.category() == DTypeCategory::QUANTIZED) {
        x = network.add_type_cvt(x, dtype::Float32());
    }
    return x;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
