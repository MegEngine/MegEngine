#include "megbrain/comp_node_env.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/io.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/test/helper.h"

#if MGB_CAMBRICON
#if CNRT_MAJOR_VERSION >= 5

#include "megbrain/cambricon/magicmind_runtime_opr.h"

#include "interface_builder.h"
#include "interface_network.h"

using namespace mgb;
using namespace opr;
using namespace magicmind;

namespace {
template <typename T>
void gen_rand_data(std::vector<T>& data, size_t num_elems, size_t scale) {
    unsigned int seed = time(0);
    data.resize(num_elems);
    for (size_t i = 0; i < num_elems; ++i) {
        data[i] =
                static_cast<T>((rand_r(&seed) % (scale * 1000)) / 1000.0 - scale / 2.0);
    }
}

template <typename T>
void get_min_max(std::vector<T>& data, double& min, double& max) {
    min = *std::min_element(data.begin(), data.end());
    max = *std::max_element(data.begin(), data.end());
}

void cast_data_type(
        std::vector<float>& input, void* output, size_t size, cnrtDataType_t input_type,
        cnrtDataType_t output_type, double& min, double& max) {
    cnrtQuantizedParam_t param = NULL;
    if (output_type == CNRT_INT8 || output_type == CNRT_INT16) {
        get_min_max(input, min, max);
        int bitwidth = 8;
        if (output_type == CNRT_INT8) {
            bitwidth = 8;
        } else if (output_type == CNRT_INT16) {
            bitwidth = 16;
        }
        auto par_tmp = magicmind::RangeToUniformQuantParamWithQuantAlg(
                {min, max}, bitwidth, "symmetric");
        auto par = magicmind::UniformToNormalCast(par_tmp);
        MGB_CNRT_CHECK(cnrtCreateQuantizedParam(&param, par.pos, par.scale, 0));
    }
    MGB_CNRT_CHECK(cnrtCastDataType(
            reinterpret_cast<void*>(input.data()), input_type, output, output_type,
            size, param));
}
cnrtDataType_t convert_data_type(magicmind::DataType dtype) {
    static const std::unordered_map<magicmind::DataType, cnrtDataType_t> dtype_map = {
#define cb(dt_mm_, dt_cnrt_) {magicmind::DataType::dt_mm_, CNRT_##dt_cnrt_}
            cb(QINT8, INT8),      cb(QINT16, INT16),    cb(INT8, INT8),
            cb(INT16, INT16),     cb(INT32, INT32),     cb(UINT8, UINT8),
            cb(FLOAT16, FLOAT16), cb(FLOAT32, FLOAT32),
    };
    auto it = dtype_map.find(dtype);
    if (it != dtype_map.end())
        return it->second;
    else {
        mgb_assert(
                false, "unsupported magicmind dtype(%u).",
                static_cast<uint32_t>(dtype));
    }
}

///! taken from src/jit/impl/utils.cpp
void replace_all_pairs_inplace(
        std::string& text,
        const std::vector<std::pair<std::string, std::string>>& replace) {
    using str = std::string;
    auto repl_one = [&text](const str& from, const str& to) {
        mgb_assert(!from.empty());
        size_t pos = 0;
        while ((pos = text.find(from, pos)) != str::npos) {
            text.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    for (auto&& i : replace) {
        repl_one(i.first, i.second);
    }
}

class MMNetwork {
public:
    template <typename T>
    using MagicMindUniquePtr = magicmind_intl::MagicMindUniquePtr<T>;
    using IModelPtr = MagicMindRuntimeOpr::IModelPtr;
    using IContextPtr = MagicMindRuntimeOpr::IContextPtr;
    using IEnginePtr = MagicMindRuntimeOpr::IEnginePtr;

    const CompNode& cn_;
    magicmind::DataType op_datatype_;
    IModelPtr model_;
    bool graph_shape_mutable_;
    bool built_;

    template <typename T>
    static MagicMindUniquePtr<T> make_mm_unique_ptr(T* ptr) {
        return {ptr, magicmind_intl::MagicMindDeleter<T>()};
    }

    MMNetwork(
            const CompNode& cn, magicmind::DataType op_datatype,
            bool graph_shape_mutable = false)
            : cn_{cn},
              op_datatype_{op_datatype},
              model_{nullptr},
              graph_shape_mutable_{graph_shape_mutable},
              built_{false} {}
    void build() {
        auto&& cnrt_env = CompNodeEnv::from_comp_node(cn_).cnrt_env();
        cnrt_env.activate();
        constexpr int ni = 16, ci = 64, hi = 32, wi = 32;
        constexpr int no = 16, co = 64, ho = 32, wo = 32;
        constexpr int kh = 3, kw = 3;
        constexpr int stride_h = 1, stride_w = 1;
        constexpr int pad_h = 1, pad_w = 1;
        magicmind::Dims input_dim{{ni, hi, wi, ci}};
        magicmind::Dims filter_dim{{co, kh, kw, ci}};
        magicmind::Dims bias_dim{{co}};
        magicmind::Dims add_dim{{no, ho, wo, co}};
        magicmind::DataType output_datatype = magicmind::DataType::FLOAT32;

        // init
        auto builder = make_mm_unique_ptr(magicmind::CreateIBuilder());
        auto config = make_mm_unique_ptr(magicmind::CreateIBuilderConfig());
        std::string user_json_config = R"(
{
    "graph_shape_mutable": {{GRAPH_SHAPE_MUTABLE}},  
    "precision_config": {
      "precision_mode": "qint8_mixed_float32"
    }
}
)";
        replace_all_pairs_inplace(
                user_json_config,
                {{"{{GRAPH_SHAPE_MUTABLE}}", graph_shape_mutable_ ? "true" : "false"}});
        config->ParseFromString(user_json_config);
        auto network = make_mm_unique_ptr(magicmind::CreateINetwork());
        magicmind::Range filter_range = {0.0f, 0.0f};
        // create input tensor
        auto init_tensor = [](magicmind::ITensor* tensor, const std::string& name,
                              const Dims& input_dim) {
            magicmind::Range input_range = {0.0f, 0.0f};
            std::vector<float> temp_buffer;
            gen_rand_data(temp_buffer, input_dim.GetElementCount(), 256);
            get_min_max(temp_buffer, input_range.min, input_range.max);
            MM_CHECK(tensor->SetDynamicRange(input_range, false));
            tensor->SetTensorName(name);
        };
        auto input_tensor = network->AddInput(op_datatype_, input_dim);
        init_tensor(input_tensor, "x", input_dim);
        auto add_tensor = network->AddInput(output_datatype, add_dim);
        init_tensor(add_tensor, "add", add_dim);
        // create filter tensor
        magicmind::ITensor* filter_tensor = nullptr;
        {
            std::vector<float> filter_buf;
            gen_rand_data(filter_buf, filter_dim.GetElementCount(), 1);
            std::vector<uint8_t> filter_buf_intx;
            filter_buf_intx.resize(
                    filter_dim.GetElementCount() *
                    magicmind::DataTypeSize(op_datatype_));
            cast_data_type(
                    filter_buf, reinterpret_cast<void*>(filter_buf_intx.data()),
                    filter_dim.GetElementCount(), CNRT_FLOAT32,
                    convert_data_type(op_datatype_), filter_range.min,
                    filter_range.max);
            auto filter = network->AddIConstNode(
                    op_datatype_, filter_dim,
                    reinterpret_cast<void*>(filter_buf_intx.data()));
            filter_tensor = filter->GetOutput(0);
            filter_tensor->SetDynamicRange(filter_range, false);
        }

        // create bias tensor
        magicmind::ITensor* bias_tensor = nullptr;
        {
            std::vector<float> bias_buf;
            gen_rand_data(bias_buf, bias_dim.GetElementCount(), 1);
            std::vector<uint8_t> bias_buf_floatx;
            if (output_datatype == magicmind::DataType::FLOAT16) {
                bias_buf_floatx.resize(
                        bias_dim.GetElementCount() *
                        magicmind::DataTypeSize(output_datatype));
                double min = 0., max = 0.;
                cast_data_type(
                        bias_buf, reinterpret_cast<void*>(bias_buf_floatx.data()),
                        bias_dim.GetElementCount(), CNRT_FLOAT32,
                        convert_data_type(output_datatype), min, max);
                auto bias = network->AddIConstNode(
                        output_datatype, bias_dim,
                        reinterpret_cast<void*>(bias_buf_floatx.data()));
                bias_tensor = bias->GetOutput(0);
            } else {
                auto bias = network->AddIConstNode(
                        output_datatype, bias_dim,
                        reinterpret_cast<void*>(bias_buf.data()));
                bias_tensor = bias->GetOutput(0);
            }
        }

        // x   w bias
        //  \ /   |
        //   |   /
        //   conv
        //    |
        //   relu ------ out1
        //     \  add
        //      \ /
        //       |
        //      out2

        // create conv + relu node
        auto conv = network->AddIConvNode(input_tensor, filter_tensor, bias_tensor);
        MM_CHECK(conv->SetStride(stride_h, stride_w));
        MM_CHECK(conv->SetPad(pad_h, pad_w, pad_h, pad_w));
        MM_CHECK(conv->SetDilation(1, 1));
        MM_CHECK(conv->SetPaddingMode(magicmind::IPaddingMode::EXPLICIT));
        auto conv_output = conv->GetOutput(0);
        // conv output tensor datatype should be set same with bias tensor
        MM_CHECK(conv->SetOutputType(0, output_datatype));
        // relu output tensor datatype will be same with input tensor
        auto relu =
                network->AddIActivationNode(conv_output, magicmind::IActivation::RELU);
        MM_CHECK(relu->SetOutputType(0, output_datatype));
        relu->GetOutput(0)->SetTensorName("out1");

        // set outputs nodes
        MM_CHECK(network->MarkOutput(relu->GetOutput(0)));

        // create elemwise add
        auto add = network->AddIElementwiseNode(
                relu->GetOutput(0), add_tensor, magicmind::IElementwise::ADD);
        add->GetOutput(0)->SetTensorName("out2");
        MM_CHECK(network->MarkOutput(add->GetOutput(0)));

        // create model
        model_ = {
                builder->BuildModel("model", network.get(), config.get()),
                magicmind_intl::MagicMindDeleter<magicmind::IModel>()};
        mgb_assert(model_ != nullptr);

        built_ = true;
    }

    const IModelPtr& get_inference_model() {
        if (!built_)
            build();
        return model_;
    }

    std::string get_serialized_model(bool serialize_to_file) {
        if (!built_)
            build();
        size_t size = 0;
        MM_CHECK(model_->GetSerializedModelSize(&size));
        std::string buf;
        buf.resize(size);
        MM_CHECK(model_->SerializeToMemory(reinterpret_cast<void*>(buf.data()), size));
        model_.reset();
        model_ = std::move(MagicMindRuntimeOpr::make_model_ptr(CreateIModel()));
        model_->DeserializeFromMemory(reinterpret_cast<void*>(buf.data()), size);
        if (serialize_to_file) {
            std::string fname = ssprintf(
                    "./output/MagicMindRuntimeOprTest.%s.mlu",
                    graph_shape_mutable_ ? "GraphShapeMutable"
                                         : "GraphShapeImmutableBatch");
            model_->SerializeToFile(fname.c_str());
        }
        return buf;
    }

    void infer_model(
            const std::vector<void*>& inputs, const std::vector<void*>& outputs,
            const std::vector<Dims>& input_dims) {
        if (!built_)
            build();
        auto&& cnrt_env = CompNodeEnv::from_comp_node(cn_).cnrt_env();
        cnrt_env.activate();
        auto engine = make_mm_unique_ptr(model_->CreateIEngine());
        mgb_assert(engine != nullptr);
        auto context = make_mm_unique_ptr(engine->CreateIContext());
        mgb_assert(context != nullptr);

        // create and get irttensor from context
        std::vector<magicmind::IRTTensor*> input_tensors;
        std::vector<magicmind::IRTTensor*> output_tensors;
        MM_CHECK(CreateInputTensors(context.get(), &input_tensors));
        MM_CHECK(CreateOutputTensors(context.get(), &output_tensors));
        MM_CHECK(FindIRTTensorByName(input_tensors, "x")->SetDimensions(input_dims[0]));
        MM_CHECK(FindIRTTensorByName(input_tensors, "add")
                         ->SetDimensions(input_dims[1]));
        MM_CHECK(context->InferOutputShape(input_tensors, output_tensors));
        MM_CHECK(FindIRTTensorByName(input_tensors, "x")->SetData(inputs[0]));
        MM_CHECK(FindIRTTensorByName(input_tensors, "add")->SetData(inputs[1]));
        MM_CHECK(FindIRTTensorByName(output_tensors, "out1")->SetData(outputs[0]));
        MM_CHECK(FindIRTTensorByName(output_tensors, "out2")->SetData(outputs[1]));

        auto&& queue = cnrt_env.queue;
        cnrtNotifier_t start, end;
        MGB_CNRT_CHECK(cnrtNotifierCreate(&start));
        MGB_CNRT_CHECK(cnrtNotifierCreate(&end));
        MGB_CNRT_CHECK(cnrtPlaceNotifier(start, queue));

        constexpr size_t runs = 50;
        for (size_t i = 0; i < runs; ++i) {
            MM_CHECK(context->Enqueue(input_tensors, output_tensors, queue));
        }

        MGB_CNRT_CHECK(cnrtPlaceNotifier(end, queue));
        MGB_CNRT_CHECK(cnrtQueueSync(queue));
        float time = 0.f;
        MGB_CNRT_CHECK(cnrtNotifierDuration(start, end, &time));
        printf("inference time = %.2fs\n", time / static_cast<float>(runs) * 1e-3);
        MGB_CNRT_CHECK(cnrtNotifierDestroy(&start));
        MGB_CNRT_CHECK(cnrtNotifierDestroy(&end));
        for (auto&& i : input_tensors)
            i->Destroy();
        for (auto&& o : output_tensors)
            o->Destroy();
    }
};
}  // namespace

TEST(TestMagicMindRuntimeOpr, Basic) {
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    MMNetwork network(cn, magicmind::DataType::FLOAT32, false);
    size_t dtype_size = magicmind::DataTypeSize(magicmind::DataType::FLOAT32);

    // prepare parameter for addpad and conv
    const int ni = 16, ci = 64, hi = 32, wi = 32;
    const int no = 16, co = 64, ho = 32, wo = 32;

    // count tensor nums
    int conv_input_count = ni * hi * wi * ci;
    int relu_output_count = no * ho * wo * co;

    // prepare cpu origin data
    std::vector<float> conv_input_cpu_data;
    gen_rand_data(conv_input_cpu_data, conv_input_count, 256);
    std::vector<float> add_input_cpu_data;
    gen_rand_data(add_input_cpu_data, relu_output_count, 256);
    std::vector<float> relu_output_cpu_data(relu_output_count);
    std::vector<float> add_output_cpu_data(relu_output_count);

    auto mlu_deleter = [](void* p) { MGB_CNRT_CHECK(cnrtFree(p)); };
    void* conv_input_mlu_ptr;
    void* add_input_mlu_ptr;
    void* relu_output_mlu_ptr;
    void* add_output_mlu_ptr;

    // malloc mlu mem for fusion input and output
    MGB_CNRT_CHECK(cnrtMalloc(&conv_input_mlu_ptr, conv_input_count * dtype_size));
    MGB_CNRT_CHECK(cnrtMalloc(&add_input_mlu_ptr, relu_output_count * sizeof(float)));
    MGB_CNRT_CHECK(cnrtMalloc(&relu_output_mlu_ptr, relu_output_count * sizeof(float)));
    MGB_CNRT_CHECK(cnrtMalloc(&add_output_mlu_ptr, relu_output_count * sizeof(float)));

    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(
            conv_input_mlu_ptr, conv_input_cpu_data.data(),
            conv_input_count * dtype_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    MGB_CNRT_CHECK(cnrtMemcpy(
            add_input_mlu_ptr, add_input_cpu_data.data(),
            relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV));
    std::unique_ptr<void, decltype(mlu_deleter)> conv_input_holder{
            conv_input_mlu_ptr, mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> add_input_holder{
            add_input_mlu_ptr, mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> relu_output_holder{
            relu_output_mlu_ptr, mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> add_output_holder{
            add_output_mlu_ptr, mlu_deleter};

    network.infer_model(
            {conv_input_mlu_ptr, add_input_mlu_ptr},
            {relu_output_mlu_ptr, add_output_mlu_ptr},
            {Dims{{ni, hi, wi, ci}}, Dims{{no, ho, wo, co}}});

    // result memory copy cnml->cpu
    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(
            relu_output_cpu_data.data(), relu_output_mlu_ptr,
            relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
    MGB_CNRT_CHECK(cnrtMemcpy(
            add_output_cpu_data.data(), add_output_mlu_ptr,
            relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));

    auto buf = network.get_serialized_model(false);
    auto x = std::make_shared<HostTensorND>(
            cn, TensorLayout{{ni, hi, wi, ci}, dtype::Float32()});
    auto add = std::make_shared<HostTensorND>(
            cn, TensorLayout{{no, ho, wo, co}, dtype::Float32()});
    std::memcpy(
            reinterpret_cast<void*>(x->ptr<dt_float32>()), conv_input_cpu_data.data(),
            conv_input_count * sizeof(float));
    std::memcpy(
            reinterpret_cast<void*>(add->ptr<dt_float32>()), add_input_cpu_data.data(),
            relu_output_count * sizeof(float));
    auto graph = ComputingGraph::make();
    auto x_ = opr::Host2DeviceCopy::make(*graph, x);
    auto add_ = opr::Host2DeviceCopy::make(*graph, add);
    auto outs = opr::MagicMindRuntimeOpr::make(
            reinterpret_cast<const void*>(buf.data()), buf.size(), {x_, add_});
    auto out1 = outs[0];
    auto out2 = outs[1];
    HostTensorND o1(cn, {no, ho, wo, co}, dtype::Float32());
    HostTensorND o2(cn, {no, ho, wo, co}, dtype::Float32());
    auto func = graph->compile(
            {make_callback_copy(out1, o1), make_callback_copy(out2, o2)});
    func->execute();
    HostTensorND o1_mm(cn, {no, ho, wo, co}, dtype::Float32()),
            o2_mm(cn, {no, ho, wo, co}, dtype::Float32());
    std::memcpy(
            o1_mm.ptr<float>(), relu_output_cpu_data.data(),
            relu_output_count * sizeof(float));
    std::memcpy(
            o2_mm.ptr<float>(), add_output_cpu_data.data(),
            relu_output_count * sizeof(float));
    MGB_ASSERT_TENSOR_NEAR(o1, o1_mm, 1e-4);
    MGB_ASSERT_TENSOR_NEAR(o2, o2_mm, 1e-4);
}

TEST(TestMagicMindRuntimeOpr, InputQInt8) {
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    MMNetwork network(cn, magicmind::DataType::QINT8, false);
    size_t dtype_size = magicmind::DataTypeSize(magicmind::DataType::QINT8);

    // prepare parameter for addpad and conv
    const int ni = 16, ci = 64, hi = 32, wi = 32;
    const int no = 16, co = 64, ho = 32, wo = 32;

    // count tensor nums
    int conv_input_count = ni * hi * wi * ci;
    int relu_output_count = no * ho * wo * co;

    // prepare cpu origin data
    std::vector<int8_t> conv_input_cpu_data;
    gen_rand_data(conv_input_cpu_data, conv_input_count, 256);
    std::vector<float> add_input_cpu_data;
    gen_rand_data(add_input_cpu_data, relu_output_count, 256);
    std::vector<float> relu_output_cpu_data(relu_output_count);
    std::vector<float> add_output_cpu_data(relu_output_count);

    auto mlu_deleter = [](void* p) { MGB_CNRT_CHECK(cnrtFree(p)); };
    void* conv_input_mlu_ptr;
    void* add_input_mlu_ptr;
    void* relu_output_mlu_ptr;
    void* add_output_mlu_ptr;

    // malloc mlu mem for fusion input and output
    MGB_CNRT_CHECK(cnrtMalloc(&conv_input_mlu_ptr, conv_input_count * dtype_size));
    MGB_CNRT_CHECK(cnrtMalloc(&add_input_mlu_ptr, relu_output_count * sizeof(float)));

    MGB_CNRT_CHECK(cnrtMalloc(&relu_output_mlu_ptr, relu_output_count * sizeof(float)));
    MGB_CNRT_CHECK(cnrtMalloc(&add_output_mlu_ptr, relu_output_count * sizeof(float)));
    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(
            conv_input_mlu_ptr, conv_input_cpu_data.data(),
            conv_input_count * dtype_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    MGB_CNRT_CHECK(cnrtMemcpy(
            add_input_mlu_ptr, add_input_cpu_data.data(),
            relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV));
    std::unique_ptr<void, decltype(mlu_deleter)> conv_input_holder{
            conv_input_mlu_ptr, mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> add_input_holder{
            add_input_mlu_ptr, mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> relu_output_holder{
            relu_output_mlu_ptr, mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> add_output_holder{
            add_output_mlu_ptr, mlu_deleter};

    network.infer_model(
            {conv_input_mlu_ptr, add_input_mlu_ptr},
            {relu_output_mlu_ptr, add_output_mlu_ptr},
            {Dims{{ni, hi, wi, ci}}, Dims{{no, ho, wo, co}}});

    // result memory copy cnml->cpu
    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(
            relu_output_cpu_data.data(), relu_output_mlu_ptr,
            relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
    MGB_CNRT_CHECK(cnrtMemcpy(
            add_output_cpu_data.data(), add_output_mlu_ptr,
            relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));

    auto buf = network.get_serialized_model(false);
    auto x = std::make_shared<HostTensorND>(
            cn, TensorLayout{{ni, hi, wi, ci}, dtype::QuantizedS8{1.f}});
    auto add = std::make_shared<HostTensorND>(
            cn, TensorLayout{{no, ho, wo, co}, dtype::Float32()});
    std::memcpy(
            reinterpret_cast<void*>(x->raw_ptr()), conv_input_cpu_data.data(),
            conv_input_count * sizeof(int8_t));
    std::memcpy(
            reinterpret_cast<void*>(add->ptr<dt_float32>()), add_input_cpu_data.data(),
            relu_output_count * sizeof(float));
    auto graph = ComputingGraph::make();
    auto x_ = opr::Host2DeviceCopy::make(*graph, x);
    auto add_ = opr::Host2DeviceCopy::make(*graph, add);
    auto outs = opr::MagicMindRuntimeOpr::make(
            reinterpret_cast<const void*>(buf.data()), buf.size(), {x_, add_});
    auto out1 = outs[0];
    auto out2 = outs[1];
    HostTensorND o1(cn, {no, ho, wo, co}, dtype::Float32());
    HostTensorND o2(cn, {no, ho, wo, co}, dtype::Float32());
    auto func = graph->compile(
            {make_callback_copy(out1, o1), make_callback_copy(out2, o2)});
    func->execute();
    HostTensorND o1_mm(cn, {no, ho, wo, co}, dtype::Float32()),
            o2_mm(cn, {no, ho, wo, co}, dtype::Float32());
    std::memcpy(
            o1_mm.ptr<float>(), relu_output_cpu_data.data(),
            relu_output_count * sizeof(float));
    std::memcpy(
            o2_mm.ptr<float>(), add_output_cpu_data.data(),
            relu_output_count * sizeof(float));
    MGB_ASSERT_TENSOR_NEAR(o1, o1_mm, 1e-4);
    MGB_ASSERT_TENSOR_NEAR(o2, o2_mm, 1e-4);
}

TEST(TestMagicMindRuntimeOpr, GraphShapeMutable) {
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    MMNetwork network(cn, magicmind::DataType::FLOAT32, true);
    size_t dtype_size = magicmind::DataTypeSize(magicmind::DataType::FLOAT32);

    auto check = [&](magicmind::Dims input_dim, magicmind::Dims output_dim) {
        // prepare parameter for addpad and conv
        const int ni = input_dim[0], ci = input_dim[1], hi = input_dim[2],
                  wi = input_dim[3];
        const int no = output_dim[0], co = output_dim[1], ho = output_dim[2],
                  wo = output_dim[3];

        // count tensor nums
        int conv_input_count = ni * hi * wi * ci;
        int relu_output_count = no * ho * wo * co;

        // prepare cpu origin data
        std::vector<float> conv_input_cpu_data;
        gen_rand_data(conv_input_cpu_data, conv_input_count, 256);
        std::vector<float> add_input_cpu_data;
        gen_rand_data(add_input_cpu_data, relu_output_count, 256);
        std::vector<float> relu_output_cpu_data(relu_output_count);
        std::vector<float> add_output_cpu_data(relu_output_count);

        auto mlu_deleter = [](void* p) { MGB_CNRT_CHECK(cnrtFree(p)); };
        void* conv_input_mlu_ptr;
        void* add_input_mlu_ptr;
        void* relu_output_mlu_ptr;
        void* add_output_mlu_ptr;

        // malloc mlu mem for fusion input and output
        MGB_CNRT_CHECK(cnrtMalloc(&conv_input_mlu_ptr, conv_input_count * dtype_size));
        MGB_CNRT_CHECK(
                cnrtMalloc(&add_input_mlu_ptr, relu_output_count * sizeof(float)));
        MGB_CNRT_CHECK(
                cnrtMalloc(&relu_output_mlu_ptr, relu_output_count * sizeof(float)));
        MGB_CNRT_CHECK(
                cnrtMalloc(&add_output_mlu_ptr, relu_output_count * sizeof(float)));

        // memory copy cpu->mlu
        MGB_CNRT_CHECK(cnrtMemcpy(
                conv_input_mlu_ptr, conv_input_cpu_data.data(),
                conv_input_count * dtype_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
        MGB_CNRT_CHECK(cnrtMemcpy(
                add_input_mlu_ptr, add_input_cpu_data.data(),
                relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV));
        std::unique_ptr<void, decltype(mlu_deleter)> conv_input_holder{
                conv_input_mlu_ptr, mlu_deleter};
        std::unique_ptr<void, decltype(mlu_deleter)> add_input_holder{
                add_input_mlu_ptr, mlu_deleter};
        std::unique_ptr<void, decltype(mlu_deleter)> relu_output_holder{
                relu_output_mlu_ptr, mlu_deleter};
        std::unique_ptr<void, decltype(mlu_deleter)> add_output_holder{
                add_output_mlu_ptr, mlu_deleter};

        network.infer_model(
                {conv_input_mlu_ptr, add_input_mlu_ptr},
                {relu_output_mlu_ptr, add_output_mlu_ptr},
                {Dims{{ni, hi, wi, ci}}, Dims{{no, ho, wo, co}}});

        // result memory copy cnml->cpu
        // memory copy cpu->mlu
        MGB_CNRT_CHECK(cnrtMemcpy(
                relu_output_cpu_data.data(), relu_output_mlu_ptr,
                relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
        MGB_CNRT_CHECK(cnrtMemcpy(
                add_output_cpu_data.data(), add_output_mlu_ptr,
                relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));

        auto buf = network.get_serialized_model(true);
        auto mkshp = [](int n, int c, int h, int w) {
            size_t nz = n, cz = c, hz = h, wz = w;
            return TensorShape{nz, hz, wz, cz};
        };
        auto mkly = [](int n, int c, int h, int w, DType dtype) {
            size_t nz = n, cz = c, hz = h, wz = w;
            return TensorLayout{{nz, hz, wz, cz}, dtype};
        };
        auto x = std::make_shared<HostTensorND>(
                cn, mkly(ni, ci, hi, wi, dtype::Float32()));
        auto add = std::make_shared<HostTensorND>(
                cn, mkly(no, co, ho, wo, dtype::Float32()));
        std::memcpy(
                reinterpret_cast<void*>(x->ptr<dt_float32>()),
                conv_input_cpu_data.data(), conv_input_count * sizeof(float));
        std::memcpy(
                reinterpret_cast<void*>(add->ptr<dt_float32>()),
                add_input_cpu_data.data(), relu_output_count * sizeof(float));
        auto graph = ComputingGraph::make();
        auto x_ = opr::Host2DeviceCopy::make(*graph, x);
        auto add_ = opr::Host2DeviceCopy::make(*graph, add);
        auto outs = opr::MagicMindRuntimeOpr::make(
                reinterpret_cast<const void*>(buf.data()), buf.size(), {x_, add_});
        auto out1 = outs[0];
        auto out2 = outs[1];
        HostTensorND o1(cn, mkshp(no, co, ho, wo), dtype::Float32());
        HostTensorND o2(cn, mkshp(no, co, ho, wo), dtype::Float32());
        auto func = graph->compile(
                {make_callback_copy(out1, o1), make_callback_copy(out2, o2)});
        func->execute();
        func->execute();
        HostTensorND o1_mm(cn, mkshp(no, co, ho, wo), dtype::Float32()),
                o2_mm(cn, mkshp(no, co, ho, wo), dtype::Float32());
        std::memcpy(
                o1_mm.ptr<float>(), relu_output_cpu_data.data(),
                relu_output_count * sizeof(float));
        std::memcpy(
                o2_mm.ptr<float>(), add_output_cpu_data.data(),
                relu_output_count * sizeof(float));
        MGB_ASSERT_TENSOR_NEAR(o1, o1_mm, 1e-4);
        MGB_ASSERT_TENSOR_NEAR(o2, o2_mm, 1e-4);
    };
    check(Dims{{1, 64, 32, 32}}, Dims{{1, 64, 32, 32}});
    check(Dims{{32, 64, 32, 32}}, Dims{{32, 64, 32, 32}});
    check(Dims{{7, 64, 16, 16}}, Dims{{7, 64, 16, 16}});
}

TEST(TestMagicMindRuntimeOpr, Serialization) {
    using namespace serialization;
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    MMNetwork network(cn, magicmind::DataType::FLOAT32, true);
    auto buf = network.get_serialized_model(false);

    // prepare parameter for addpad and conv
    const int ni = 1, ci = 64, hi = 32, wi = 32;
    const int no = 1, co = 64, ho = 32, wo = 32;
    auto x = std::make_shared<HostTensorND>(
            cn, TensorLayout{{ni, hi, wi, ci}, dtype::Float32()});
    auto add = std::make_shared<HostTensorND>(
            cn, TensorLayout{{no, ho, wo, co}, dtype::Float32()});
    auto graph = ComputingGraph::make();
    auto x_ = opr::Host2DeviceCopy::make(*graph, x);
    auto add_ = opr::Host2DeviceCopy::make(*graph, add);
    auto outs = opr::MagicMindRuntimeOpr::make(
            reinterpret_cast<const void*>(buf.data()), buf.size(), {x_, add_});
    auto out1 = outs[0];
    auto out2 = outs[1];
    auto fname = output_file("model_magicmind.mgb");
    auto dump = [&]() {
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()));
        auto rst = dumper->dump({out1, out2});
        ASSERT_EQ(rst.outputs.size(), 2u);
    };
    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()));
        auto rst = loader->load();
        ASSERT_EQ(rst.output_var_list.size(), 2u);
    };
    dump();
    load();
}

TEST(TestMagicMindRuntimeOpr, Profiling) {
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    MMNetwork network(cn, magicmind::DataType::FLOAT32, true);
    auto buf = network.get_serialized_model(false);
    const int ni = 8, ci = 64, hi = 32, wi = 32;
    const int no = 8, co = 64, ho = 32, wo = 32;

    HostTensorGenerator<dtype::Float32, RandomDistribution::GAUSSIAN> gen(0, 1);
    auto x = gen({ni, hi, wi, ci}, cn);
    auto add = gen({no, ho, wo, co}, cn);

    auto graph = ComputingGraph::make();
    GraphProfiler profiler{graph.get()};
    auto x_ = opr::Host2DeviceCopy::make(*graph, x);
    auto add_ = opr::Host2DeviceCopy::make(*graph, add);
    auto outs = opr::MagicMindRuntimeOpr::make(
            reinterpret_cast<const void*>(buf.data()), buf.size(), {x_, add_});
    auto out1 = outs[0];
    auto out2 = outs[1];
    graph->options().var_sanity_check_first_run = false;
    HostTensorND o1(cn, {no, ho, wo, co}, dtype::Float32());
    HostTensorND o2(cn, {no, ho, wo, co}, dtype::Float32());
    auto func = graph->compile(
            {make_callback_copy(out1, o1), make_callback_copy(out2, o2)});
    func->execute();
    profiler.to_json_full(func.get())
            ->writeto_fpath(output_file("magicmind_runtime_opr_profile.json"));
}

TEST(TestMagicMindRuntimeOpr, CrossCNCopy) {
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    MMNetwork network(cn, magicmind::DataType::FLOAT32, false);
    size_t dtype_size = magicmind::DataTypeSize(magicmind::DataType::FLOAT32);

    // prepare parameter for addpad and conv
    const int ni = 16, ci = 64, hi = 32, wi = 32;
    const int no = 16, co = 64, ho = 32, wo = 32;

    // count tensor nums
    int conv_input_count = ni * hi * wi * ci;
    int relu_output_count = no * ho * wo * co;

    // prepare cpu origin data
    std::vector<float> conv_input_cpu_data;
    gen_rand_data(conv_input_cpu_data, conv_input_count, 256);
    std::vector<float> add_input_cpu_data;
    gen_rand_data(add_input_cpu_data, relu_output_count, 256);
    std::vector<float> relu_output_cpu_data(relu_output_count);
    std::vector<float> add_output_cpu_data(relu_output_count);

    auto mlu_deleter = [](void* p) { MGB_CNRT_CHECK(cnrtFree(p)); };
    void* conv_input_mlu_ptr;
    void* add_input_mlu_ptr;
    void* relu_output_mlu_ptr;
    void* add_output_mlu_ptr;

    // malloc mlu mem for fusion input and output
    MGB_CNRT_CHECK(cnrtMalloc(&conv_input_mlu_ptr, conv_input_count * dtype_size));
    MGB_CNRT_CHECK(cnrtMalloc(&add_input_mlu_ptr, relu_output_count * sizeof(float)));
    MGB_CNRT_CHECK(cnrtMalloc(&relu_output_mlu_ptr, relu_output_count * sizeof(float)));
    MGB_CNRT_CHECK(cnrtMalloc(&add_output_mlu_ptr, relu_output_count * sizeof(float)));

    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(
            conv_input_mlu_ptr, conv_input_cpu_data.data(),
            conv_input_count * dtype_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    MGB_CNRT_CHECK(cnrtMemcpy(
            add_input_mlu_ptr, add_input_cpu_data.data(),
            relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV));
    std::unique_ptr<void, decltype(mlu_deleter)> conv_input_holder{
            conv_input_mlu_ptr, mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> add_input_holder{
            add_input_mlu_ptr, mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> relu_output_holder{
            relu_output_mlu_ptr, mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> add_output_holder{
            add_output_mlu_ptr, mlu_deleter};

    network.infer_model(
            {conv_input_mlu_ptr, add_input_mlu_ptr},
            {relu_output_mlu_ptr, add_output_mlu_ptr},
            {Dims{{ni, hi, wi, ci}}, Dims{{no, ho, wo, co}}});

    // result memory copy cnml->cpu
    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(
            relu_output_cpu_data.data(), relu_output_mlu_ptr,
            relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
    MGB_CNRT_CHECK(cnrtMemcpy(
            add_output_cpu_data.data(), add_output_mlu_ptr,
            relu_output_count * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));

    auto cn_cpu = CompNode::load("cpu0");
    auto buf = network.get_serialized_model(false);
    auto x = std::make_shared<HostTensorND>(
            cn_cpu, TensorLayout{{ni, hi, wi, ci}, dtype::Float32()});
    auto add = std::make_shared<HostTensorND>(
            cn_cpu, TensorLayout{{no, ho, wo, co}, dtype::Float32()});
    std::memcpy(
            reinterpret_cast<void*>(x->ptr<dt_float32>()), conv_input_cpu_data.data(),
            conv_input_count * sizeof(float));
    std::memcpy(
            reinterpret_cast<void*>(add->ptr<dt_float32>()), add_input_cpu_data.data(),
            relu_output_count * sizeof(float));
    auto graph = ComputingGraph::make();
    auto x_ = opr::Host2DeviceCopy::make(*graph, x, {cn_cpu});
    auto add_ = opr::Host2DeviceCopy::make(*graph, add, {cn_cpu});
    x_ = opr::Copy::make(x_, {cn});
    add_ = opr::Copy::make(add_, {cn});
    auto outs = opr::MagicMindRuntimeOpr::make(
            reinterpret_cast<const void*>(buf.data()), buf.size(), {x_, add_});
    auto out1 = outs[0];
    auto out2 = outs[1];
    HostTensorND o1(CompNode::default_cpu(), {no, ho, wo, co}, dtype::Float32());
    HostTensorND o2(CompNode::default_cpu(), {no, ho, wo, co}, dtype::Float32());
    auto func = graph->compile(
            {make_callback_copy(out1, o1), make_callback_copy(out2, o2)});
    func->execute();
    HostTensorND o1_mm(cn, {no, ho, wo, co}, dtype::Float32()),
            o2_mm(cn, {no, ho, wo, co}, dtype::Float32());
    std::memcpy(
            o1_mm.ptr<float>(), relu_output_cpu_data.data(),
            relu_output_count * sizeof(float));
    std::memcpy(
            o2_mm.ptr<float>(), add_output_cpu_data.data(),
            relu_output_count * sizeof(float));
    MGB_ASSERT_TENSOR_NEAR(o1, o1_mm, 1e-4);
    MGB_ASSERT_TENSOR_NEAR(o2, o2_mm, 1e-4);
}

#endif
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
