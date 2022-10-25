#include "megbrain/opr/nn_int.h"
#if MGB_ENABLE_FBS_SERIALIZATION

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/softmax.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/test/helper.h"

using namespace mgb;
using namespace serialization;

#define GET_OUTPUT_FILE(format) \
    output_file(ssprintf("TestSerializer2.line%d.V%d", __LINE__, (int)format))

namespace {
void test_graph_load_dump(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);

    auto orig_id = -1;
    auto dump = [&]() {
        auto cn = CompNode::load("cpu0");
        auto graph = ComputingGraph::make();
        auto x = opr::ImmutableTensor::make(*graph, 1926.0817f, {cn});
        x.rename("varz");
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        auto rst = dumper->dump({x});
        ASSERT_EQ(rst.nr_opr, 1);
        ASSERT_EQ(rst.inputs.size(), 0);
        ASSERT_EQ(rst.outputs.size(), 1);
        ASSERT_EQ(rst.params.size(), 0);
        orig_id = x.node()->id();
        mgb_log("%zu of %zu", rst.tensor_value_bytes, rst.tot_bytes);
    };
    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
        ASSERT_EQ(rst.tensor_map.size(), 0);
        ASSERT_EQ(rst.output_var_list.size(), 1);
        ASSERT_EQ(rst.output_var_map.size(), 1);
        ASSERT_EQ(rst.output_var_map_id.size(), 1);
        ASSERT_EQ(rst.output_var_map.count("varz"), 1);
        ASSERT_EQ(rst.output_var_map_id.count(orig_id), 1);

        HostTensorND host_x;

        auto func =
                rst.graph_compile({make_callback_copy(rst.output_var_list[0], host_x)});
        func->execute().wait();
        EXPECT_NEAR(*host_x.ptr<float>(), 1926.0817f, 1e-6);
    };
    dump();
    load();
}

void test_multi_graph_dump_load(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);

    auto dump = [&]() {
        auto cn = CompNode::load("cpu0");
        auto graph = ComputingGraph::make();
        auto x = opr::ImmutableTensor::make(*graph, 1926.0817f, {cn});
        x.rename("varz");
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        // dump twice
        dumper->dump({x});
        dumper->dump({x});
    };
    auto load = [&]() {
        GraphLoader::LoadConfig load_config = {};
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        // load twice
        loader->load(load_config, false);
        loader = GraphLoader::make(loader->reset_file(), loader->format());
        loader->load(load_config, false);
    };

    dump();
    load();
}

void test_metadata(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3};

    auto dump = [&]() {
        auto cn = CompNode::load("xpu0");
        auto host_x = std::make_shared<HostTensorND>(cn, shape),
             host_y = std::make_shared<HostTensorND>(cn, shape);
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
             y = opr::Host2DeviceCopy::make(*graph, host_y, {"y"});
        using Mode = opr::Elemwise::Mode;
        auto z = opr::Elemwise::make({x, y}, Mode::ADD, {"add(x, y)"});

        Metadata metadata;
        metadata.user_info = "TEST_METADATA";
        metadata.has_user_info = true;

        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        auto rst = dumper->dump({z.rename("z")}, {}, metadata);
    };

    auto load = [&]() {
        HostTensorGenerator<> gen;
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
        auto metadata = rst.metadata;
        int cmp = strcmp(metadata.user_info.c_str(), "TEST_METADATA");
        EXPECT_EQ(cmp, 0);
    };

    dump();
    load();
}

void test_serializer_APlusB(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3};

    auto dump = [&]() {
        auto cn = CompNode::load("xpu0");
        auto host_x = std::make_shared<HostTensorND>(cn, shape),
             host_y = std::make_shared<HostTensorND>(cn, shape);
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
             y = opr::Host2DeviceCopy::make(*graph, host_y, {"y"});

        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        // test dump duplicated
        auto rst = dumper->dump({(x + y).rename("z"), x + y});
        ASSERT_EQ(2u, rst.outputs.size());
    };

    auto load = [&]() {
        HostTensorGenerator<> gen;
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
        auto xv = rst.tensor_map.at("x");
        auto yv = rst.tensor_map.at("y");
        ASSERT_EQ(shape, xv->shape());
        ASSERT_EQ(shape, yv->shape());
        *xv = *gen(shape);
        *yv = *gen(shape);
        HostTensorND host_z, host_z_expect;
        host_z_expect.copy_from(*xv);
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++i)
            host_z_expect.ptr<float>()[i] += yv->ptr<float>()[i];
        auto func = rst.graph_compile(
                {make_callback_copy(rst.output_var_map.at("z"), host_z)});
        func->execute();
        MGB_ASSERT_TENSOR_EQ(host_z_expect, host_z);
    };

    dump();
    load();
}

void test_serializer_APlusB_param(GraphDumpFormat format) {
    auto cns = load_multiple_xpus(2);
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3};

    HostTensorGenerator<> gen;
    auto bias = std::make_shared<DeviceTensorND>();
    auto bias_hv = gen(shape, cns[0]);
    bias->copy_from(*bias_hv);

    {
        // dump
        auto host_x = std::make_shared<HostTensorND>(cns[0], shape);
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
             y = opr::SharedDeviceTensor::make(*graph, bias, {"y"});

        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        GraphDumper::DumpConfig config;
        config.keep_param_name = true;
        dumper->dump({(x + y).rename("z")}, config);
    }
    auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);

    auto load = [&](CompNode dest_cn) {
        auto dest_cn_loc = dest_cn.locator_logical();
        auto rst = loader->load({[&](CompNode::Locator& loc) { loc = dest_cn_loc; }});
        auto xv = rst.tensor_map.at("x");
        ASSERT_EQ(1u, rst.tensor_map.size());
        ASSERT_EQ(shape, xv->shape());
        *xv = *gen(shape, cns[0]);
        HostTensorND host_z, host_z_expect;
        host_z_expect.copy_from(*xv);
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++i)
            host_z_expect.ptr<float>()[i] += bias_hv->ptr<float>()[i];
        auto func = rst.graph_compile(
                {make_callback_copy(rst.output_var_map.at("z"), host_z)});
        func->execute();
        MGB_ASSERT_TENSOR_EQ(host_z_expect, host_z);
    };

    load(cns[0]);
    auto&& shmap = loader->shared_tensor_name_map();
    ASSERT_EQ(1u, shmap.at("y")->size());
    load(cns[0].change_stream(1));
    ASSERT_EQ(1u, shmap.at("y")->size());
    load(cns[1]);
    ASSERT_EQ(1u + (cns[1].mem_node() != cns[0].mem_node()), shmap.at("y")->size());
}

void test_serializer_immutable(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3};

    auto dump = [&]() {
        auto cn = CompNode::load("xpu0");
        auto host_x = std::make_shared<HostTensorND>(cn, shape);
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"});
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        dumper->dump({(x + 1.f).rename("y")});
    };

    auto load = [&]() {
        HostTensorGenerator<> gen;
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
        auto xv = rst.tensor_map.at("x");
        ASSERT_EQ(shape, xv->shape());
        *xv = *gen(shape);
        HostTensorND host_y, host_y_expect;
        host_y_expect.copy_from(*xv);
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++i)
            host_y_expect.ptr<float>()[i] += 1;
        auto func = rst.graph_compile(
                {make_callback_copy(rst.output_var_map.at("y"), host_y)});
        func->execute();
        MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y);
    };

    dump();
    load();
}

void test_serializer_custom_loader(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3};

    int load_nr_null_ptr = 0, load_nr_call = 0;
    std::vector<HostTensorND> saved_val;

    auto tensor_value_dumper = [&saved_val](
                                       OutputFile& fout,
                                       const cg::OperatorNodeBase& opr,
                                       const HostTensorND& tensor) {
        size_t idx = saved_val.size();
        saved_val.emplace_back();
        saved_val.back().copy_from(tensor);
        fout.write(&idx, sizeof(idx));
    };
    auto tensor_value_loader = [&saved_val, &load_nr_null_ptr, &load_nr_call](
                                       void* ptr, const TensorLayout& layout,
                                       InputFile& fin) {
        ++load_nr_call;
        size_t idx;
        if (!ptr) {
            load_nr_null_ptr++;
            fin.skip(sizeof(idx));
            return;
        }
        fin.read(&idx, sizeof(idx));
        auto&& val = saved_val.at(idx);
        ASSERT_TRUE(val.layout().eq_layout(layout));
        memcpy(ptr, val.raw_ptr(), layout.span().high_byte);
    };

    auto dump = [&]() {
        auto cn = CompNode::load("xpu0");
        auto host_x = std::make_shared<HostTensorND>(cn, shape);
        HostTensorND y_val(cn, {1});
        y_val.ptr<float>()[0] = 2.3f;
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
             y = opr::SharedDeviceTensor::make(*graph, y_val),
             z = ((x + 1.f) * y).rename("z");
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        GraphDumpConfig config;
        config.tensor_value_dumper = tensor_value_dumper;
        dumper->dump({z}, config);
    };
    dump();

    GraphLoadConfig config;
    config.tensor_value_loader = tensor_value_loader;
    auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
    auto load = [&]() {
        HostTensorGenerator<> gen;
        auto rst = loader->load(config);
        auto xv = rst.tensor_map.at("x");
        ASSERT_EQ(shape, xv->shape());
        *xv = *gen(shape);
        HostTensorND host_y, host_y_expect;
        host_y_expect.copy_from(*xv);
        auto py = host_y_expect.ptr<float>();
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++i) {
            py[i] = (py[i] + 1.f) * 2.3f;
        }
        auto func = rst.graph_compile(
                {make_callback_copy(rst.output_var_map.at("z"), host_y)});
        func->execute();
        MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y);
    };

    load();
    load();
    ASSERT_EQ(2u, saved_val.size());
    if (GraphDumpFormat::FLATBUFFERS_V2 != format) {
        ASSERT_EQ(2, load_nr_null_ptr);  // immutable tensor is also shared
        ASSERT_EQ(4, load_nr_call);
    }
}

void test_serializer_many_io_var(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    constexpr size_t NR_VARS = 32;
    auto dump = [&]() {
        auto graph = ComputingGraph::make();
        SymbolVarArray xs;
        cg::OperatorNodeConfig::CompNodeArray y_comp_nodes;
        for (size_t i = 0; i < NR_VARS; ++i) {
            CompNode::Locator loc;
            loc.type = CompNode::DeviceType::CPU;
            loc.device = 0;
            loc.stream = i;
            auto cn = CompNode::load(loc);
            auto host_x = std::make_shared<HostTensorND>(cn, TensorShape{1});
            xs.push_back(opr::Host2DeviceCopy::make(*graph, host_x, std::to_string(i)));

            loc.device = 1;
            y_comp_nodes.push_back(CompNode::load(loc));
        }
        auto con = opr::Concat::make(xs, 0, CompNode::load("cpu2")) * 2 + 1;
        auto ys = opr::Split::make(
                con,
                opr::Split::Options::make_partition(
                        con, 0, std::vector<size_t>(NR_VARS, 1)),
                OperatorNodeConfig{}.comp_node_arr(y_comp_nodes));

        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        auto rst = dumper->dump(ys);
    };

    auto load = [&]() {
        HostTensorGenerator<> gen;
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
        ASSERT_EQ(NR_VARS, rst.output_var_list.size());
        ComputingGraph::OutputSpec out_spec(NR_VARS);
        std::vector<HostTensorND> host_ys(NR_VARS);
        for (size_t i = 0; i < NR_VARS; ++i) {
            auto y = rst.output_var_list[i];
            auto loc = y.node()->comp_node().locator_logical();
            ASSERT_EQ(1, loc.device);
            ASSERT_EQ(static_cast<int>(i), loc.stream);
            out_spec[i] = make_callback_copy(y, host_ys[i]);

            auto&& inp = rst.tensor_map.at(std::to_string(i));
            inp->resize({1}).ptr<float>()[0] = i;
        }
        auto func = rst.graph_compile(out_spec);
        func->execute();
        for (size_t i = 0; i < NR_VARS; ++i) {
            auto&& val = host_ys[i];
            ASSERT_EQ(TensorShape{1}, val.shape());
            ASSERT_EQ(static_cast<float>(i * 2 + 1), val.ptr<float>()[0]);
        }
    };

    dump();
    load();
}

void test_serializer_remove_set_grad(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3};

    auto dump = [&]() {
        auto cn = CompNode::load("xpu0");
        auto host_x = std::make_shared<HostTensorND>(cn, shape),
             host_y = std::make_shared<HostTensorND>(cn, shape);
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
             y = opr::Host2DeviceCopy::make(*graph, host_y, {"y"});

        auto sg = [](SymbolVar var) {
            return opr::SetGrad::make(var, opr::SetGrad::zero_grad);
        };

        // SetGrad as output
        auto z0 = sg(x + y);
        // SetGrad as internal
        auto z1 = sg(x) + sg(sg(y));

        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        dumper->dump({z0, z1});
    };

    auto load = [&]() {
        HostTensorGenerator<> gen;
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
        auto xv = rst.tensor_map.at("x");
        auto yv = rst.tensor_map.at("y");
        ASSERT_EQ(shape, xv->shape());
        ASSERT_EQ(shape, yv->shape());
        *xv = *gen(shape);
        *yv = *gen(shape);
        HostTensorND host_z0, host_z1, host_z_expect;
        host_z_expect.copy_from(*xv);
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++i)
            host_z_expect.ptr<float>()[i] += yv->ptr<float>()[i];
        ASSERT_EQ(2u, rst.output_var_list.size());
        auto func = rst.graph_compile(
                {{make_callback_copy(rst.output_var_list[0], host_z0)},
                 {make_callback_copy(rst.output_var_list[1], host_z1)}});
        func->execute();
        MGB_ASSERT_TENSOR_EQ(host_z_expect, host_z0);
        MGB_ASSERT_TENSOR_EQ(host_z_expect, host_z1);
    };

    dump();
    load();
}

void test_serializer_multiple_param(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    std::vector<std::shared_ptr<DeviceTensorND>> values;
    auto add_value = [&](int stream, int ndim, DType dtype) {
        CompNode::Locator loc;
        loc.type = CompNode::DeviceType::CPU;
        loc.device = 0;
        loc.stream = stream;
        auto cn = CompNode::load(loc);

        TensorShape shp;
        shp.ndim = ndim;
        for (int i = 0; i < ndim; ++i)
            shp[i] = i + 1;

        auto cur = std::make_shared<DeviceTensorND>(cn, shp, dtype);
        uint8_t* ptr = reinterpret_cast<uint8_t*>(cur->raw_ptr());
        for (size_t i = 0, it = cur->layout().span().dist_byte(); i < it; ++i) {
            ptr[i] = i;
        }

        values.push_back(cur);
        return cur;
    };
    auto dump = [&]() {
        auto graph = ComputingGraph::make();
        int stream = 0;
        auto mkvar = [&](int ndim, DType dtype) {
            auto dv = add_value(stream++, ndim, dtype);
            auto var = opr::SharedDeviceTensor::make(*graph, dv);
            var = opr::TypeCvt::make(
                    opr::reduce_sum(var, var.make_scalar(1)), dtype::Int32());
            var = opr::Copy::make(var, CompNode::load("cpu1"));
            return var;
        };
        auto x = mkvar(1, dtype::Float32());
        for (size_t ndim = 1; ndim <= TensorShape::MAX_NDIM; ++ndim) {
#define cb(_dt) x = x + mkvar(ndim, _dt());
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
        }
        ASSERT_GT(values.size(), 8u);
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        dumper->dump({x});
    };

    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        ASSERT_THROW(loader->shared_tensor_id_map(), MegBrainError);
        loader->load();
        auto&& got = loader->shared_tensor_id_map();
        ASSERT_EQ(2 * values.size(), got.size());
        for (size_t i = 0; i < values.size(); ++i) {
            ASSERT_EQ(1u, got[i].second.size());
            auto &&vi = *values[i], &&gi = *got[2 * i].second.begin()->second;
            ASSERT_EQ(vi.shape(), gi.shape());
            ASSERT_EQ(vi.comp_node(), gi.comp_node());
            ASSERT_EQ(vi.dtype(), gi.dtype());
            ASSERT_EQ(
                    0,
                    memcmp(vi.raw_ptr(), gi.raw_ptr(), vi.layout().span().dist_byte()));
        }
    };

    dump();
    load();
}

void test_serializer_const_var_shape(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3};
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});

    {
        // dump
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"});
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        dumper->dump({x + 1.f});
    }

    auto run_and_check = [&](const GraphLoadConfig& config) {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load(config);
        rst.tensor_map.at("x")->copy_from(*host_x);
        auto y = rst.output_var_list[0];
        ASSERT_EQ(shape, y.shape());
        auto infer_type = y.node()->owner_graph()
                                  ->static_infer_manager()
                                  .get_infer_type(y.node())
                                  .shape;
        if (config.const_var_shape) {
            ASSERT_EQ(cg::static_infer::InferType::CONST, infer_type);
        } else {
            ASSERT_EQ(cg::static_infer::InferType::RT_STATIC, infer_type);
        }
        HostTensorND host_y, host_y_expect;
        host_y_expect.copy_from(*host_x);
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++i)
            host_y_expect.ptr<float>()[i] += 1;
        auto func = rst.graph_compile({make_callback_copy(y, host_y)});
        func->execute();
        MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y);

        if (config.const_var_shape) {
            rst.tensor_map.at("x")->resize({4, 5});
            ASSERT_THROW(func->execute(), MegBrainError);
        }
    };

    for (bool const_shape : {false, true}) {
        GraphLoadConfig config;
        config.const_var_shape = const_shape;
        run_and_check(config);
    };

    // test const shape with tensor modifier
    {
        int nr_tensor = 0, nr_mod = 0;
        shape = {7, 6};
        *host_x = *gen(shape);
        GraphLoadConfig config;
        config.const_var_shape = true;
        config.tensor_modifier = [&](const std::string& name, bool has_value,
                                     HostTensorND& tensor) {
            ++nr_tensor;
            if (!has_value) {
                ASSERT_EQ("x", name);
                tensor.resize(shape);
                ++nr_mod;
            }
        };
        run_and_check(config);
        ASSERT_EQ(1, nr_tensor);  // immutable tensor is shared tensor
        ASSERT_EQ(1, nr_mod);
    }
}

void test_serializer_const_var_shape_output_name(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3};
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});

    {
        // dump
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
             y = opr::GetVarShape::make(x) + 1;
        y.rename("out");
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        dumper->dump({y});
    }

    {
        // load
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        GraphLoadConfig config;
        config.const_var_shape = true;
        auto rst = loader->load(config);
        ASSERT_EQ(1u, rst.tensor_map.count("x"));
        auto y = rst.output_var_map.at("out");
        ASSERT_TRUE(y.node()->owner_opr()->same_type<opr::ImmutableTensor>());
    }
}

void test_serializer_priority(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3};

    auto dump = [&](bool keep_pri) {
        auto cn = CompNode::load("xpu0");
        auto host_x = std::make_shared<HostTensorND>(cn, shape),
             host_y = std::make_shared<HostTensorND>(cn, shape);
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"}) + 1,
             y = opr::Host2DeviceCopy::make(*graph, host_y, {"y"}) + 1;

        set_priority(x, 1);
        set_priority(y, 2);

        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        GraphDumper::DumpConfig config;
        if (keep_pri) {
            config.keep_opr_priority = true;
        }
        dumper->dump({x * y}, config);
    };

    auto load = [&](bool has_pri) {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
        VarNode *x, *y;
        unpack_vector(rst.output_var_list.front().node()->owner_opr()->input(), x, y);
        auto get_pri = [](VarNode* var) {
            return var->owner_opr()->node_prop().attribute().priority;
        };
        int xpri = get_pri(x), ypri = get_pri(y);
        if (has_pri) {
            ASSERT_EQ(1, xpri);
            ASSERT_EQ(2, ypri);
        } else {
            ASSERT_EQ(0, xpri);
            ASSERT_EQ(0, ypri);
        }
    };

    dump(false);
    load(false);

    dump(true);
    load(true);
}

void test_serializer_multiple_params(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    HostTensorGenerator<> gen;
    std::vector<std::shared_ptr<HostTensorND>> tensors{
            gen({2, 3}), gen({1}), gen({3, 2}), gen({1, 1})};

    auto dump = [&]() {
        auto graph = ComputingGraph::make();
        SymbolVarArray outputs;
        for (auto&& i : tensors) {
            outputs.push_back(opr::SharedDeviceTensor::make(*graph, *i));
        }
        GraphDumper::make(OutputFile::make_fs(fname.c_str()), format)->dump(outputs);
    };

    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
        ASSERT_EQ(tensors.size(), rst.output_var_list.size());
        for (size_t i = 0; i < tensors.size(); ++i) {
            HostTensorND got;
            got.copy_from(rst.output_var_list[i]
                                  .node()
                                  ->owner_opr()
                                  ->cast_final_safe<opr::SharedDeviceTensor>()
                                  .get_dev_tensor())
                    .sync();
            MGB_ASSERT_TENSOR_EQ(*tensors[i], got);
        }
    };

    dump();
    load();
}

void test_serializer_paramerized_dtype(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3, 3};
    dtype::Quantized8Asymm dtype(0.01f, (uint8_t)123);

    auto dump = [&]() {
        auto cn = CompNode::load("cpu0");
        auto host_x = std::make_shared<HostTensorND>(cn, shape, dtype);
        for (size_t i = 0; i < host_x->layout().span().dist_elem(); i++) {
            host_x->ptr<dt_quint8>()[i] = dt_quint8(static_cast<uint8_t>(i & 255));
        }
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"});
        auto rst = opr::Dimshuffle::make(x, {1, 2, 0});
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        dumper->dump({rst});
    };

    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
        ASSERT_EQ(rst.output_var_list.size(), 1u);
        EXPECT_EQ(rst.output_var_list.front().node()->dtype(), dtype);
    };

    dump();
    load();
}

void test_serializer_operator_name(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3};

    auto dump = [&]() {
        auto cn = CompNode::load("xpu0");
        auto host_x = std::make_shared<HostTensorND>(cn, shape),
             host_y = std::make_shared<HostTensorND>(cn, shape);
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
             y = opr::Host2DeviceCopy::make(*graph, host_y, {"y"});
        using Mode = opr::Elemwise::Mode;
        auto z = opr::Elemwise::make({x, y}, Mode::ADD, {"add(x, y)"});

        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        auto rst = dumper->dump({z.rename("z")});
    };

    auto load = [&]() {
        HostTensorGenerator<> gen;
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
        auto z = rst.output_var_map.at("z");
        auto op_name = z.node()->owner_opr()->cname();
        int cmp = strcmp(op_name, "add(x, y)");
        EXPECT_EQ(cmp, 0);
    };

    dump();
    load();
}

void test_serializer_has_output_dtype(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);

    HostTensorGenerator<> gen;

    auto graph = ComputingGraph::make();

    auto gen_tensor = [&](const TensorShape& shape, const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shape)), dtype);
    };

    auto dump = [&]() {
        auto x = gen_tensor({20, 4, 56, 56}, dtype::QuantizedS8(0.5f));
        auto w = gen_tensor({4, 4, 1, 1}, dtype::QuantizedS8(0.1f));
        auto b = gen_tensor({1, 4, 1, 1}, dtype::QuantizedS32(0.05f));
        opr::ConvBias::Param param;
        auto y0 = opr::ConvBias::make(
                x, w, b, param, {}, OperatorNodeConfig{dtype::QuantizedS32(0.05f)});
        auto y1 = opr::ConvBias::make(
                x, w, b, param, {}, OperatorNodeConfig{dtype::QuantizedS8(0.3f)});
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        dumper->dump({y0, y1});
    };

    auto check = [](const serialization::GraphLoader::LoadResult& rst, size_t idx,
                    const DType& expected_dtype) {
        auto&& dtype =
                rst.output_var_list[idx].node()->owner_opr()->config().output_dtype();
        ASSERT_TRUE(dtype.valid());
        ASSERT_EQ(dtype, expected_dtype);
    };

    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
        ASSERT_EQ(rst.output_var_list.size(), 2u);
        check(rst, 0, dtype::QuantizedS32(0.05f));
        check(rst, 1, dtype::QuantizedS8(0.3f));
    };

    dump();
    load();
}

void test_serializer_log_exp(GraphDumpFormat format) {
    auto fname = GET_OUTPUT_FILE(format);
    TensorShape shape{2, 3};
    using Mode = opr::Elemwise::Mode;
    bool inplace_opt = true;
    auto dump = [&]() {
        auto cn = CompNode::load("xpu0");
        auto host_x = std::make_shared<HostTensorND>(cn, shape);
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++i)
            host_x->ptr<float>()[i] = 0.0;  // To avoid NAN
        auto graph = ComputingGraph::make();
        if (!inplace_opt)
            graph->options().graph_opt_level = 0;
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"});
        auto y = opr::Elemwise::make({x}, Mode::EXP);
        auto z = opr::Elemwise::make({y}, Mode::LOG);

        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()), format);
        auto rst = dumper->dump({z.rename("z"), z});
        size_t expected_nr_opr = inplace_opt ? 1 : 3;
        ASSERT_EQ(expected_nr_opr, rst.nr_opr);
    };

    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()), format);
        auto rst = loader->load();
    };

    dump();
    load();

    inplace_opt = !inplace_opt;
    dump();
    load();
}

void test_serializer_memshare(GraphDumpFormat format) {
    std::vector<uint8_t> buf;
    HostTensorGenerator<> gen;
    constexpr size_t SIZE = 127;
    auto xval = gen({SIZE}, "cpu0"), bval = gen({1}, "cpu0");

    auto dump = [&]() {
        auto graph = ComputingGraph::make();
        auto x0 = opr::SharedDeviceTensor::make(*graph, *xval).rename("x0");
        auto x1 = opr::SharedDeviceTensor::make(*graph, *xval).rename("x1");
        auto x2 = opr::SharedDeviceTensor::make(*graph, *xval).rename("x2");
        auto x3 = opr::SharedDeviceTensor::make(*graph, *xval).rename("x3");
        auto i4 = opr::ImmutableTensor::make(*graph, *xval).rename("i4");
        auto i5 = opr::ImmutableTensor::make(*graph, *xval).rename("i5");
        auto b = opr::SharedDeviceTensor::make(*graph, *bval).rename("b");
        auto dumper = GraphDumper::make(OutputFile::make_vector_proxy(&buf), format);
        dumper->dump({((x0 + x1) + b) + (x2 + x3) + i4 + i5, x0, i4});
    };

    HostTensorND expected;
    expected.copy_from(*xval);
    for (size_t i = 0; i < SIZE; ++i) {
        auto&& v = expected.ptr<float>()[i];
        v = v * 6 + bval->ptr<float>()[0];
    }

    std::vector<uint8_t> buf_al;
    auto load = [&](bool share) {
        std::unique_ptr<InputFile> fin;
        if (share) {
            buf_al.resize(buf.size());
            memcpy(buf_al.data(), buf.data(), buf.size());

            fin = InputFile::make_mem_proxy(
                    std::shared_ptr<void>{std::shared_ptr<void>{}, buf_al.data()},
                    buf.size());
        } else {
            fin = InputFile::make_mem_proxy(buf.data(), buf.size());
        }
        auto loader = GraphLoader::make(std::move(fin), format);
        auto rst = loader->load();
        auto x = rst.output_var_map.at("x0");
        auto i4 = rst.output_var_map.at("i4");
        auto&& opr = x.node()->owner_opr()->cast_final_safe<opr::SharedDeviceTensor>();
        auto&& opr_imm =
                i4.node()->owner_opr()->cast_final_safe<opr::ImmutableTensor>();
        HostTensorND val;
        auto func =
                rst.graph_compile({make_callback_copy(rst.output_var_list[0], val)});
        func->execute();
        return std::make_pair(
                val, std::vector<DeviceTensorND>{*opr.dev_data(), opr_imm.value()});
    };

    auto in_range = [](const std::vector<uint8_t>& buf, DeviceTensorND& dv) {
        auto p0 = reinterpret_cast<uint8_t*>(dv.raw_ptr()),
             p1 = reinterpret_cast<uint8_t*>(p0 + dv.layout().span().high_byte);
        return buf.data() <= p0 && p1 <= buf.data() + buf.size();
    };

    for (bool share : {false, true}) {
        buf.clear();
        dump();
        auto get = load(share);
        MGB_ASSERT_TENSOR_EQ(*xval, HostTensorND{}.copy_from(get.second[0]).sync());
        MGB_ASSERT_TENSOR_EQ(expected, get.first);
        ASSERT_EQ(share, in_range(buf_al, get.second[0]));
        ASSERT_EQ(share, in_range(buf_al, get.second[1]));
    }
}

}  // namespace

TEST(TestSerializer2, GraphDumpLoad) {
    test_graph_load_dump(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, MultiGraphDumpLoad) {
    test_multi_graph_dump_load(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, Metadata) {
    test_metadata(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, APlusB) {
    test_serializer_APlusB(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, APlusBParam) {
    test_serializer_APlusB_param(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, Immutable) {
    test_serializer_immutable(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, CustomLoader) {
    test_serializer_custom_loader(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, ManyIOVars) {
    test_serializer_many_io_var(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, RemoveSetGrad) {
    test_serializer_remove_set_grad(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, MultipleParamNDIMDTypeCompNode) {
    test_serializer_multiple_param(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, ConstVarShape) {
    test_serializer_const_var_shape(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, ConstVarShapeOutputName) {
    test_serializer_const_var_shape_output_name(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, Priority) {
    test_serializer_priority(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, MultipleParams) {
    test_serializer_multiple_params(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, ParamerizedDType) {
    test_serializer_paramerized_dtype(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, OperatorName) {
    test_serializer_operator_name(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, HasOutputDtype) {
    test_serializer_has_output_dtype(GraphDumpFormat::FLATBUFFERS);
}

TEST(TestSerializer2, LOGEXP) {
    test_serializer_log_exp(GraphDumpFormat::FLATBUFFERS);
}

/******************** Flatbuffer V2 Test **********************/

TEST(TestSerializer2, GraphDumpLoadV2) {
    test_graph_load_dump(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, MultiGraphDumpLoadV2) {
    test_multi_graph_dump_load(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, MetadataV2) {
    test_metadata(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, APlusBV2) {
    test_serializer_APlusB(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, APlusBParamV2) {
    test_serializer_APlusB_param(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, ImmutableV2) {
    test_serializer_immutable(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, ManyIOVarsV2) {
    test_serializer_many_io_var(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, CustomLoaderV2) {
    test_serializer_custom_loader(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, RemoveSetGradV2) {
    test_serializer_remove_set_grad(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, MultipleParamNDIMDTypeCompNodeV2) {
    test_serializer_multiple_param(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, ConstVarShapeV2) {
    test_serializer_const_var_shape(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, ConstVarShapeOutputNameV2) {
    test_serializer_const_var_shape_output_name(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, MultipleParamsV2) {
    test_serializer_multiple_params(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, ParamerizedDTypeV2) {
    test_serializer_paramerized_dtype(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, PriorityV2) {
    test_serializer_priority(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, OperatorNameV2) {
    test_serializer_operator_name(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, HasOutputDtypeV2) {
    test_serializer_has_output_dtype(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, LOGEXPV2) {
    test_serializer_log_exp(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, ShareMemv2) {
    test_serializer_memshare(GraphDumpFormat::FLATBUFFERS_V2);
}

TEST(TestSerializer2, TestSoftMaxLoadDump) {
    auto fname = GET_OUTPUT_FILE(GraphDumpFormat::FLATBUFFERS_V2);
    TensorShape shape{2, 3};

    auto cn = CompNode::load("xpu0");
    std::shared_ptr<HostTensorND> host =
            std::make_shared<HostTensorND>(cn, shape, dtype::Float32{});
    HostTensorND dst_truth;
    for (int i = 0; i < 6; i++) {
        host->ptr<float>()[i] = i;
    }
    auto dump = [&]() {
        auto graph = ComputingGraph::make();
        auto h2d = opr::Host2DeviceCopy::make(*graph, host);
        auto x = opr::Softmax::make(h2d, {1}, {});
        x.rename("softmax_out");
        auto func = graph->compile({make_callback_copy(x, dst_truth)});
        auto dumper = GraphDumper::make(
                OutputFile::make_fs(fname.c_str()), GraphDumpFormat::FLATBUFFERS_V2);
        auto rst = dumper->dump({x});
        func->execute().wait();
        //! if convert to reduce and elemwise, nr_opr is 6
        // ASSERT_EQ(rst.nr_opr, 6);
        ASSERT_EQ(rst.nr_opr, 2);
        ASSERT_EQ(rst.inputs.size(), 1);
        ASSERT_EQ(rst.outputs.size(), 1);
        ASSERT_EQ(rst.params.size(), 0);
    };
    auto load = [&]() {
        auto loader = GraphLoader::make(
                InputFile::make_fs(fname.c_str()), GraphDumpFormat::FLATBUFFERS_V2);
        auto rst = loader->load();
        ASSERT_EQ(rst.tensor_map.size(), 1);
        ASSERT_EQ(rst.output_var_list.size(), 1);
        ASSERT_EQ(rst.output_var_map.size(), 1);
        ASSERT_EQ(rst.output_var_map.count("softmax_out"), 1);

        HostTensorND host_x;
        auto func =
                rst.graph_compile({make_callback_copy(rst.output_var_list[0], host_x)});
        rst.tensor_map.begin()->second->copy_from(*host).sync();
        func->execute().wait();
        for (int i = 0; i < 6; i++) {
            EXPECT_NEAR(host_x.ptr<float>()[i], dst_truth.ptr<float>()[i], 1e-6);
        }
    };
    dump();
    load();
}

TEST(TestSerializer2, TestElemwiseMultiTypeLoadDump) {
    auto fname = GET_OUTPUT_FILE(GraphDumpFormat::FLATBUFFERS_V2);
    TensorShape shape{3};
    auto cn = CompNode::load("xpu0");
    std::shared_ptr<HostTensorND> host0 =
            std::make_shared<HostTensorND>(cn, shape, dtype::Float32{});
    std::shared_ptr<HostTensorND> host1 =
            std::make_shared<HostTensorND>(cn, shape, dtype::Float32{});
    HostTensorND dst_truth;
    host0->ptr<float>()[0] = 2;
    host0->ptr<float>()[1] = 2;
    host0->ptr<float>()[2] = -1;
    host1->ptr<float>()[0] = 1;
    host1->ptr<float>()[1] = 2;
    host1->ptr<float>()[2] = 3;

    auto dump = [&](opr::ElemwiseMultiType::Param::Mode mode, size_t nr_opr) {
        auto graph = ComputingGraph::make();
        OperatorNodeConfig config;
        config.name("input0");
        auto h2d0 = opr::Host2DeviceCopy::make(*graph, host0, config);
        config.name("input1");
        auto h2d1 = opr::Host2DeviceCopy::make(*graph, host1, config);

        auto x = opr::ElemwiseMultiType::make(
                {h2d0, h2d1}, {mode}, OperatorNodeConfig{dtype::Bool()});
        x.rename("out");
        auto func = graph->compile({make_callback_copy(x, dst_truth)});
        auto dumper = GraphDumper::make(
                OutputFile::make_fs(fname.c_str()), GraphDumpFormat::FLATBUFFERS_V2);
        auto rst = dumper->dump({x});
        func->execute().wait();
        ASSERT_EQ(rst.nr_opr, nr_opr);
    };
    auto load = [&]() {
        auto loader = GraphLoader::make(
                InputFile::make_fs(fname.c_str()), GraphDumpFormat::FLATBUFFERS_V2);
        auto rst = loader->load();
        ASSERT_EQ(rst.tensor_map.size(), 2);
        ASSERT_EQ(rst.output_var_map.count("out"), 1);

        HostTensorND host_x;
        auto func =
                rst.graph_compile({make_callback_copy(rst.output_var_list[0], host_x)});
        for (auto& input : rst.tensor_map) {
            if (input.first == "input0") {
                input.second->copy_from(*host0).sync();
            } else if (input.first == "input1") {
                input.second->copy_from(*host1).sync();
            }
        }
        func->execute().wait();
        for (int i = 0; i < 3; i++) {
            EXPECT_EQ(host_x.ptr<bool>()[i], dst_truth.ptr<bool>()[i]);
        }
    };
    dump(opr::ElemwiseMultiType::Param::Mode::EQ, 4);
    load();
    dump(opr::ElemwiseMultiType::Param::Mode::LT, 4);
    load();
    dump(opr::ElemwiseMultiType::Param::Mode::LEQ, 4);
    load();
    dump(opr::ElemwiseMultiType::Param::Mode::NEQ, 5);
    load();

    auto dump_single_input = [&](opr::ElemwiseMultiType::Param::Mode mode,
                                 size_t nr_opr) {
        auto graph = ComputingGraph::make();
        auto h2d0 = opr::Host2DeviceCopy::make(*graph, host0);
        auto x = opr::ElemwiseMultiType::make(
                {h2d0}, {mode}, OperatorNodeConfig{dtype::Bool()});
        x.rename("out");
        auto func = graph->compile({make_callback_copy(x, dst_truth)});
        auto dumper = GraphDumper::make(
                OutputFile::make_fs(fname.c_str()), GraphDumpFormat::FLATBUFFERS_V2);
        auto rst = dumper->dump({x});
        func->execute().wait();
        ASSERT_EQ(rst.nr_opr, nr_opr);
    };
    auto load_single_input = [&]() {
        auto loader = GraphLoader::make(
                InputFile::make_fs(fname.c_str()), GraphDumpFormat::FLATBUFFERS_V2);
        auto rst = loader->load();
        ASSERT_EQ(rst.tensor_map.size(), 1);
        ASSERT_EQ(rst.output_var_map.count("out"), 1);

        HostTensorND host_x;
        auto func =
                rst.graph_compile({make_callback_copy(rst.output_var_list[0], host_x)});
        rst.tensor_map.begin()->second->copy_from(*host0).sync();
        func->execute().wait();
        for (int i = 0; i < 3; i++) {
            EXPECT_EQ(host_x.ptr<bool>()[i], dst_truth.ptr<bool>()[i]);
        }
    };
    host0->ptr<float>()[2] = INFINITY;
    dump_single_input(opr::ElemwiseMultiType::Param::Mode::ISINF, 4);
    load_single_input();
    host0->ptr<float>()[2] = NAN;
    dump_single_input(opr::ElemwiseMultiType::Param::Mode::ISNAN, 4);
    load_single_input();
}

#endif
