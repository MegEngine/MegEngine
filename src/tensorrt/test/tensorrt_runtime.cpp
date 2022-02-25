#include "megbrain/comp_node_env.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"
#include "megbrain/utils/debug.h"

#if MGB_ENABLE_TENSOR_RT

#include "make_trt_net.h"
#include "megbrain/tensorrt/tensorrt_opr.h"
#include "megbrain/tensorrt/tensorrt_runtime_opr.h"

#include <fstream>
#include <random>

using namespace mgb;
using namespace nvinfer1;

template <typename T>
using TensorRTUniquePtr = intl::TensorRTUniquePtr<T>;

TEST(TestOprTensorRT, RuntimeBasic) {
    REQUIRE_GPU(1);
    intl::SimpleTensorRTNetwork net;
    auto make_trt = [&net]() {
        auto p = net.create_trt_network(false);
        TensorRTUniquePtr<INetworkDefinition> trt_net{p.second, {}};
        TensorRTUniquePtr<IBuilder> builder{p.first, {}};
        builder->setMaxBatchSize(5);
#if NV_TENSOR_RT_VERSION >= 6001
        TensorRTUniquePtr<IBuilderConfig> build_config{builder->createBuilderConfig()};
        TensorRTUniquePtr<ICudaEngine> cuda_engine{
                builder->buildEngineWithConfig(*trt_net, *build_config)};
#else
        TensorRTUniquePtr<ICudaEngine> cuda_engine{builder->buildCudaEngine(*trt_net)};
#endif
        TensorRTUniquePtr<IHostMemory> mem{cuda_engine->serialize(), {}};
        return TensorRTRuntimeOpr::make(mem->data(), mem->size(), {net.x})[0];
    };
    auto y2 = make_trt();

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile(
            {make_callback_copy(net.y, host_z1), make_callback_copy(y2, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 5e-4);
}

TEST(TestOprTensorRT, RuntimeBasicBatched) {
    REQUIRE_GPU(1);
    intl::BatchedTensorRTNetwork net;
    auto make_trt = [&net]() {
        auto p = net.create_trt_network(false);
        TensorRTUniquePtr<INetworkDefinition> trt_net{p.second, {}};
        TensorRTUniquePtr<IBuilder> builder{p.first, {}};
        builder->setMaxBatchSize(5);
#if NV_TENSOR_RT_VERSION >= 6001
        TensorRTUniquePtr<IBuilderConfig> build_config{builder->createBuilderConfig()};
        TensorRTUniquePtr<ICudaEngine> cuda_engine{
                builder->buildEngineWithConfig(*trt_net, *build_config)};
#else
        TensorRTUniquePtr<ICudaEngine> cuda_engine{builder->buildCudaEngine(*trt_net)};
#endif
        TensorRTUniquePtr<IHostMemory> mem{cuda_engine->serialize(), {}};
        auto nx = opr::Broadcast::make(
                net.x, {1, net.x.shape()[0], net.x.shape()[1], net.x.shape()[2]});
        return TensorRTRuntimeOpr::make(mem->data(), mem->size(), {nx})[0];
    };
    auto y2 = make_trt();

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile(
            {make_callback_copy(net.y, host_z1), make_callback_copy(y2, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 5e-4);
}

TEST(TestOprTensorRT, ConcatRuntimeBasic) {
    REQUIRE_GPU(1);
    intl::ConcatConvTensorRTNetwork net;

    SymbolVar y2;
    {
        auto p = net.create_trt_network(false);
        TensorRTUniquePtr<INetworkDefinition> trt_net{p.second, {}};
        TensorRTUniquePtr<IBuilder> builder{p.first, {}};
        builder->setMaxBatchSize(5);
#if NV_TENSOR_RT_VERSION >= 6001
        TensorRTUniquePtr<IBuilderConfig> build_config{builder->createBuilderConfig()};
        auto cuda_engine = builder->buildEngineWithConfig(*trt_net, *build_config);
#else
        auto cuda_engine = builder->buildCudaEngine(*trt_net);
#endif
        TensorRTUniquePtr<IHostMemory> mem{cuda_engine->serialize(), {}};

        FILE* fout = fopen(output_file("trt_cuda_engine").c_str(), "wb");
        auto wr = fwrite(mem->data(), 1, mem->size(), fout);
        mgb_assert(wr == mem->size());
        fclose(fout);

        y2 = TensorRTRuntimeOpr::make(
                TensorRTRuntimeOpr::to_shared_ptr_engine(cuda_engine), {},
                {net.x0, net.x1})[0];
    }

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile(
            {make_callback_copy(net.y, host_z1), make_callback_copy(y2, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
}

TEST(TestOprTensorRT, RuntimeChangeBatchSize) {
    REQUIRE_GPU(1);
    intl::SimpleTensorRTNetwork net;
    auto make_trt = [&net]() {
        auto p = net.create_trt_network(false);
        TensorRTUniquePtr<INetworkDefinition> trt_net{p.second, {}};
        TensorRTUniquePtr<IBuilder> builder{p.first, {}};
        builder->setMaxBatchSize(10);
#if NV_TENSOR_RT_VERSION >= 6001
        TensorRTUniquePtr<IBuilderConfig> build_config{builder->createBuilderConfig()};
        TensorRTUniquePtr<ICudaEngine> cuda_engine{
                builder->buildEngineWithConfig(*trt_net, *build_config)};
#else
        TensorRTUniquePtr<ICudaEngine> cuda_engine{builder->buildCudaEngine(*trt_net)};
#endif
        TensorRTUniquePtr<IHostMemory> mem{cuda_engine->serialize(), {}};
        return TensorRTRuntimeOpr::make(mem->data(), mem->size(), {net.x})[0];
    };
    auto y2 = make_trt();

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile(
            {make_callback_copy(net.y, host_z1), make_callback_copy(y2, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 5e-4);
    *net.host_x = *net.gen({1, 23, 28, 28});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 5e-4);
    *net.host_x = *net.gen({10, 23, 28, 28});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 5e-4);
}

#if NV_TENSOR_RT_VERSION >= 6001
TEST(TestOprTensorRT, IOFormatFree) {
    size_t N = 1, C = 3, H = 7, W = 7;
    REQUIRE_GPU(1);
    TensorRTUniquePtr<IBuilder> builder{
            createInferBuilder(TensorRTOpr::Logger::instance()), {}};
    nvinfer1::NetworkDefinitionCreationFlags flags;
    ::memset(&flags, 0, sizeof(nvinfer1::NetworkDefinitionCreationFlags));
    flags = 1 << static_cast<int>(
                    nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TensorRTUniquePtr<INetworkDefinition> network{builder->createNetworkV2(flags), {}};
    auto cast = [](size_t i) { return static_cast<int>(i); };
    ITensor* data = network->addInput(
            "data", DataType::kINT8, Dims4{cast(N), cast(C), cast(H), cast(W)});
    TensorFormats formats = 1 << static_cast<int>(nvinfer1::TensorFormat::kCHW4);
    data->setAllowedFormats(formats);
    data->setDynamicRange(-127.f * 1.2f, 127.f * 1.2f);
    HostTensorGenerator<> fgen;
    auto mean = fgen({N, C, H, W});
    Weights mean_weights{DataType::kFLOAT, nullptr, 0};
    mean_weights.values = mean->raw_ptr();
    mean_weights.count = N * C * H * W;
    auto constant = network->addConstant(
            Dims4{cast(N), cast(C), cast(H), cast(W)}, mean_weights);
    auto out = network->addElementWise(
            *network->getInput(0), *constant->getOutput(0), ElementWiseOperation::kSUB);
    out->getOutput(0)->setDynamicRange(-127.f * 2.3f, 127.f * 2.3f);
    network->markOutput(*out->getOutput(0));
    network->getInput(0)->setType(DataType::kINT8);
    network->getOutput(0)->setType(DataType::kFLOAT);
    network->getOutput(0)->setAllowedFormats(
            1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    TensorRTUniquePtr<IBuilderConfig> build_config{builder->createBuilderConfig()};
    build_config->setFlag(BuilderFlag::kINT8);
    build_config->setFlag(BuilderFlag::kSTRICT_TYPES);
    TensorRTUniquePtr<ICudaEngine> cuda_engine{
            builder->buildEngineWithConfig(*network, *build_config)};
    TensorRTUniquePtr<IHostMemory> mem{cuda_engine->serialize(), {}};

    HostTensorGenerator<dtype::Int8> gen;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp, const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name), dtype);
    };
    auto x = mkvar("x", {N, C, H, W}, dtype::QuantizedS8(1.2f));
    auto fx = opr::TypeCvt::make(x, dtype::Float32());
    auto wval = opr::SharedDeviceTensor::make(*graph, *mean).rename("mean");
    auto z = fx - wval;
    HostTensorND y1;
    auto func1 = graph->compile({make_callback_copy(z, y1)});
    func1->execute();

    TensorShape shp{N, 1, H, W};
    auto host =
            std::make_shared<HostTensorND>(x.node()->comp_node(), x.node()->dtype());
    host->resize(shp);
    auto ptr = host->raw_ptr();
    size_t size_bytes = TensorLayout{shp, x.node()->dtype()}.span().dist_byte();
    std::memset(ptr, 0, size_bytes);
    auto padding = opr::ImmutableTensor::make(*graph, *host);
    x = opr::Concat::make({x, padding}, 1);

    auto nchw2nchw4 = [](SymbolVar x) {
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make({sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
        auto y0 = opr::Reshape::make(x, tshp);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});
        return y1;
    };
    x = nchw2nchw4(x);
    auto trt = TensorRTRuntimeOpr::make(mem->data(), mem->size(), {x})[0];
    HostTensorND y2;
    auto func2 = graph->compile({make_callback_copy(trt, y2)});
    func2->execute();
    MGB_ASSERT_TENSOR_EQ(y1, y2);
}

TEST(TestOprTensorRT, FlattenConcatPlugin) {
    REQUIRE_GPU(1);
    intl::ReshapeConcatTensorRTNetwork net;
    auto make_trt = [&net]() {
        auto p = net.create_trt_network(false);
        TensorRTUniquePtr<INetworkDefinition> trt_net{p.second, {}};
        TensorRTUniquePtr<IBuilder> builder{p.first, {}};
        builder->setMaxBatchSize(5);
#if NV_TENSOR_RT_VERSION >= 6001
        TensorRTUniquePtr<IBuilderConfig> build_config{builder->createBuilderConfig()};
        TensorRTUniquePtr<ICudaEngine> cuda_engine{
                builder->buildEngineWithConfig(*trt_net, *build_config)};
#else
        TensorRTUniquePtr<ICudaEngine> cuda_engine{builder->buildCudaEngine(*trt_net)};
#endif
        TensorRTUniquePtr<IHostMemory> mem{cuda_engine->serialize(), {}};
        return TensorRTRuntimeOpr::make(mem->data(), mem->size(), {net.x0, net.y0})[0];
    };
    auto z2 = make_trt();

    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = net.graph->compile(
            {make_callback_copy(net.z, host_z1), make_callback_copy(z2, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_z1, host_z2);
}
#endif

TEST(TestOprTensorRT, ICudaEngine) {
    REQUIRE_GPU(1);
    CompNode::load("xpu0").activate();
    std::ifstream engineFile("model.trt", std::ios::binary);
    if (!engineFile)
        return;

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
        return;

    std::shared_ptr<ComputingGraph> graph;
    graph = ComputingGraph::make();

    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x0, host_y0;
    host_x0 = gen({2, 3, 375, 500});
    host_y0 = gen({2, 1, 1, 3});

    SymbolVar x0 = Host2DeviceCopy::make(*graph, host_x0);
    SymbolVar y0 = Host2DeviceCopy::make(*graph, host_y0);

    auto z = TensorRTRuntimeOpr::make(engineData.data(), fsize, {x0, y0})[0];
    HostTensorND host_z;

    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
}

#if NV_TENSOR_RT_VERSION >= 6001
TEST(TestOprTensorRT, RuntimeDynamicShape) {
    REQUIRE_GPU(1);
    intl::DynamicShapeTensorRTNetwork net1{2, 23, 14, 14};
#if NV_TENSOR_RT_VERSION >= 7200
    intl::DynamicShapeTensorRTNetwork net2{4, 23, 24, 24};
#else
    intl::DynamicShapeTensorRTNetwork net2{3, 23, 10, 10};
#endif

    auto make_trt = [](intl::DynamicShapeTensorRTNetwork& net) {
        TensorRTUniquePtr<ICudaEngine> cuda_engine = net.create_trt_network();
        TensorRTUniquePtr<IHostMemory> mem{cuda_engine->serialize(), {}};
        return TensorRTRuntimeOpr::make(mem->data(), mem->size(), {net.x});
    };

    HostTensorND host_z1, host_z2;

    auto y1 = make_trt(net1);
    auto func1 = net1.graph->compile(
            {make_callback_copy(net1.y1, host_z1), make_callback_copy(y1[0], host_z2)});
    func1->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 5e-4);

    auto y2 = make_trt(net2);
    auto func2 = net2.graph->compile(
            {make_callback_copy(net2.y1, host_z1), make_callback_copy(y2[0], host_z2)});
    func2->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 5e-4);
}
#endif

#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
