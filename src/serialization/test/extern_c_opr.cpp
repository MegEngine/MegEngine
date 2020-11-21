/**
 * \file src/serialization/test/extern_c_opr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <memory>
#include "megbrain/graph/extern_copr_api.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/extern_c_opr_io.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/test/helper.h"
#include "megbrain/utils/debug.h"

using namespace mgb;
using namespace serialization;

namespace {

DType dtype_c2cpp(MGBDType dtype) {
    switch (dtype) {
        case MGB_DTYPE_UINT8:
            return dtype::Uint8{};
        case MGB_DTYPE_INT32:
            return dtype::Int32{};
        case MGB_DTYPE_FLOAT32:
            return dtype::Float32{};
#if !MEGDNN_DISABLE_FLOAT16
        case MGB_DTYPE_FLOAT16:
            return dtype::Float16{};
#endif
        default:
            mgb_throw(SerializationError, "bad dtype value: %d",
                      static_cast<int>(dtype));
    }
}

const void* prev_desc_buf_addr;
size_t prev_desc_buf_size;

//! a custom opr to compute x + bias
template <MGBDType out_dtype = MGB_DTYPE_FLOAT32>
class MGBOprDescImpl {
    struct UserData {
        float bias;
    };
    static UserData* user_data(const MGBOprDesc* self) {
        return static_cast<UserData*>(self->user_data);
    }

    static void release(MGBOprDesc* self) {
        delete user_data(self);
        delete self;
        --nr_inst;
    }

    static size_t hash(const MGBOprDesc* self) {
        return mgb::hash<float>(user_data(self)->bias);
    }

    static int is_same(const MGBOprDesc* self, const MGBOprDesc* rhs) {
        return user_data(self)->bias == user_data(rhs)->bias;
    }

    static void execute(const MGBOprDesc* self, const MGBTensor* input,
                        const MGBTensor* output) {
        if (self->dynamic_param) {
            auto device_id = self->dynamic_param->device_id;
            mgb_assert(0 == device_id || 8 == device_id);
        }
        bool use_extern_input =
                (self->dynamic_param && self->dynamic_param->nr_input > 0)
                        ? true
                        : false;
        bool use_extern_output =
                (self->dynamic_param && self->dynamic_param->nr_output > 0)
                        ? true
                        : false;

        auto&& i = input[0].layout;
        auto&& o = output[0].layout;
        mgb_assert(i.shape.ndim == 1 && o.shape.ndim == 1 &&
                   i.shape.shape[0] == o.shape.shape[0]);
        mgb_assert(i.dtype == MGB_DTYPE_FLOAT32 && o.dtype == out_dtype);
        auto input_p = static_cast<float*>(input[0].data);
        if (use_extern_input)
            input_p = static_cast<float*>(
                    self->dynamic_param->input[0].device_ptr);
        auto bias = user_data(self)->bias;
        if (out_dtype == MGB_DTYPE_FLOAT32) {
            auto output_p = static_cast<float*>(output[0].data);
            if (use_extern_output)
                output_p = static_cast<float*>(
                        self->dynamic_param->output[0].device_ptr);
            for (size_t x = 0; x < i.shape.shape[0]; ++x) {
                output_p[x] = input_p[x] + bias;
            }
        } else if (MEGDNN_FLOAT16_SELECT(out_dtype == MGB_DTYPE_FLOAT16,
                                         false)) {
#if !MEGDNN_DISABLE_FLOAT16
            auto output_p = static_cast<dt_float16*>(output[0].data);
            for (size_t x = 0; x < i.shape.shape[0]; ++x) {
                output_p[x] = input_p[x] + bias;
            }
#endif
        } else {
            mgb_assert(out_dtype == MGB_DTYPE_INT32);
            auto output_p = static_cast<int32_t*>(output[0].data);
            for (size_t x = 0; x < i.shape.shape[0]; ++x) {
                output_p[x] = input_p[x] + bias;
            }
        }
    }

    static void infer_shape(const MGBOprDesc*, const MGBTensorShape* input,
                            MGBTensorShape* output) {
        output[0] = input[0];
    }

    static void infer_dtype(const struct MGBOprDesc* self,
                            const MGBDType* input, MGBDType* output) {
        output[0] = out_dtype;
    }

    static const char* name() {
        return out_dtype == MGB_DTYPE_FLOAT32
                       ? "bias_adder_f23"
                       : (out_dtype == MGB_DTYPE_INT32 ? "bias_adder_int32"
                                                       : "bias_addr_float16");
    }

public:
    static int nr_inst;
    static MGBOprDesc* make(float bias) {
        ++nr_inst;
        auto ud = std::make_unique<UserData>();
        ud->bias = bias;
        auto desc = std::make_unique<MGBOprDesc>();
        mgb_init_opr_desc(desc.get(), 1, name());
        desc->user_data = ud.release();
#define s(n) desc->n = &MGBOprDescImpl::n;
        MGB_OPR_DESC_FOREACH_MEM_FN(s);
#undef s
        if (out_dtype != MGB_DTYPE_FLOAT32) {
            desc->infer_dtype = infer_dtype;
        }
        return desc.release();
    }
};
template <MGBDType out_dtype>
int MGBOprDescImpl<out_dtype>::nr_inst = 0;

template <MGBDType out_dtype = MGBDType::MGB_DTYPE_FLOAT32>
class MGBOprLoaderImpl {
    static MGBOprDesc* create_desc(size_t nr_input, const void* buf,
                                   size_t buf_len) {
        mgb_assert(buf_len == sizeof(float));
        prev_desc_buf_addr = buf;
        prev_desc_buf_size = buf_len;
        float fv;
        memcpy(&fv, buf, buf_len);
        return MGBOprDescImpl<out_dtype>::make(fv);
    }

public:
    static MGBOprLoader make() { return {name(), &create_desc}; }

    static const char* name() {
        return out_dtype == MGB_DTYPE_FLOAT32
                       ? "bias_adder_dump"
                       : (out_dtype == MGB_DTYPE_INT32 ? "bias_adder_dump_i32"
                                                       : "bias_adder_dump_f16");
    }
};

template <MGBDType out_dtype>
class MGBOprLoaderReg {
public:
    MGBOprLoaderReg() {
        auto api = mgb_get_extern_c_opr_api();
        auto loader = MGBOprLoaderImpl<out_dtype>::make();
        auto succ = api->register_loader(&loader);
        mgb_assert(succ);
    }
};
MGBOprLoaderReg<MGB_DTYPE_FLOAT32> loader_reg_f32;
MGBOprLoaderReg<MGB_DTYPE_INT32> loader_reg_i32;
#if !MEGDNN_DISABLE_FLOAT16
MGBOprLoaderReg<MGB_DTYPE_FLOAT16> loader_reg_f16;
#endif

std::vector<uint8_t> create_graph_dump(float bias, float extra_scale,
                                       float sleep, MGBDType dtype) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1}, "cpux");
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    if (sleep)
        x = opr::Sleep::make(x, sleep);
    x = opr::ExternCOprRunner::make_placeholder(
                {x}, {TensorShape{1}},
                dtype == MGB_DTYPE_FLOAT32
                        ? "bias_adder_dump:test"
                        : (dtype == MGB_DTYPE_INT32 ? "bias_adder_dump_i32"
                                                    : "bias_adder_dump_f16"),
                &bias, sizeof(bias), {}, {dtype_c2cpp(dtype)})
                ->output(0);
    if (extra_scale)
        x = x * extra_scale;

    std::vector<uint8_t> ret;
    auto dumper = GraphDumper::make(OutputFile::make_vector_proxy(&ret));
    dumper->dump({x});
    return ret;
}

void check_dump_by_compute(std::unique_ptr<serialization::InputFile> input_file,
                           CompNode cn, MGBDType dtype, float bias,
                           float scale) {
    GraphLoadConfig config;
    config.comp_node_mapper = [loc = cn.locator()](CompNode::Locator & t) {
        t = loc;
    };
    auto loader = GraphLoader::make(std::move(input_file));
    auto load_ret = loader->load(config);
    load_ret.graph->options().var_sanity_check_first_run = false;
    SymbolVar y;
    unpack_vector(load_ret.output_var_list, y);

    HostTensorGenerator<> gen;
    auto host_x = load_ret.tensor_map.begin()->second;
    *host_x = *gen({23}, cn);
    HostTensorND y_expect;
    y_expect.copy_from(*host_x);
    {
        auto py = y_expect.ptr<float>();
        for (int i = 0; i < 23; ++i) {
            auto t = py[i] + bias;
            if (dtype == MGB_DTYPE_INT32) {
                t = int(t);
#if !MEGDNN_DISABLE_FLOAT16
            } else if (dtype == MGB_DTYPE_FLOAT16) {
                t = dt_float16(t);
#endif
            }
            py[i] = t * scale;
        }
    }

    HostTensorND host_y;
    auto func = load_ret.graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
}

void check_dump_by_compute_with_param(
        std::unique_ptr<serialization::InputFile> input_file, CompNode cn,
        MGBDType dtype, float bias, std::shared_ptr<ExternCOprParam> param) {
    GraphLoadConfig config;
    config.comp_node_mapper = [loc = cn.locator()](CompNode::Locator& t) {
        t = loc;
    };
    auto loader = GraphLoader::make(std::move(input_file));
    auto load_ret = loader->load(config);
    load_ret.graph->options().var_sanity_check_first_run = false;
    SymbolVar y;
    unpack_vector(load_ret.output_var_list, y);

    HostTensorGenerator<> gen;
    auto host_x = load_ret.tensor_map.begin()->second;
    *host_x = *gen({23}, cn);
    HostTensorND y_expect;
    y_expect.copy_from(*host_x);
    {
        auto py = y_expect.ptr<float>();
        float* extern_input_device_ptr = nullptr;
        if (param->nr_input && param->input && param->input->device_ptr) {
            extern_input_device_ptr =
                    static_cast<float*>(param->input->device_ptr);
        }
        for (int i = 0; i < 23; ++i) {
            float t = 0;
            //! this test code is run before config_extern_c_opr_dynamic_param
            //! so we need double child member ptr is valid or not
            if (param->nr_input && param->input && param->input->device_ptr) {
                t = extern_input_device_ptr[i] + bias;
            } else {
                t = py[i] + bias;
            }
            if (dtype == MGB_DTYPE_INT32) {
                t = int(t);
#if !MEGDNN_DISABLE_FLOAT16
            } else if (dtype == MGB_DTYPE_FLOAT16) {
                t = dt_float16(t);
#endif
            }
            py[i] = t;
        }
    }

    HostTensorND host_y;
    auto func = load_ret.graph->compile({make_callback_copy(y, host_y)});
    config_extern_c_opr_dynamic_param(func, param);
    func->execute();
    if (param->nr_output) {
        auto ph = host_y.ptr<float>();
        auto outp = static_cast<float*>(param->output->device_ptr);
        for (int i = 0; i < 23; ++i) {
            ph[i] = outp[i];
        }
    }
    MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
}

void run_compute_test(CompNode cn, MGBDType dtype) {
    float bias = 1.2, scale = -2.1;
    auto graph_dump = create_graph_dump(bias, scale, 0.3, dtype);
    check_dump_by_compute(
            InputFile::make_mem_proxy(graph_dump.data(), graph_dump.size()), cn,
            dtype, bias, scale);
}
void run_compute_test_with_param(CompNode cn, MGBDType dtype,
                                 std::shared_ptr<ExternCOprParam> param) {
    float bias = 1.2, scale = 0;
    auto graph_dump = create_graph_dump(bias, scale, 0.3, dtype);
    check_dump_by_compute_with_param(
            InputFile::make_mem_proxy(graph_dump.data(), graph_dump.size()), cn,
            dtype, bias, param);
}
}  // namespace

TEST(TestExternCOpr, ExternCOprParam) {
    //! same with check_dump_by_compute_with_param
    constexpr int input_output_size = 23;
    auto c_opr_param = std::make_shared<ExternCOprParam>();
    MGBTensorLayout input_layput, output_layput;
    ExternDeviceTensor input, output;
    float* input_device_ptr = (float*)malloc(input_output_size * sizeof(float));
    float* output_device_ptr =
            (float*)malloc(input_output_size * sizeof(float));

    auto reset = [&] {
        memset(c_opr_param.get(), 0, sizeof(ExternCOprParam));
        memset(&input_layput, 0, sizeof(MGBTensorLayout));
        memset(&input, 0, sizeof(ExternDeviceTensor));
        memset(&output_layput, 0, sizeof(MGBTensorLayout));
        memset(&output, 0, sizeof(ExternDeviceTensor));
        memset(input_device_ptr, 0, input_output_size * sizeof(float));
        memset(output_device_ptr, 0, input_output_size * sizeof(float));

        for (size_t i = 0; i < input_output_size; i++) {
            input_device_ptr[i] = i;
        }
    };

    auto run_test = [&] {
        run_compute_test_with_param(CompNode::load("cpux"), MGB_DTYPE_FLOAT32,
                                    c_opr_param);
    };

    auto init_param = [&] {
        reset();
        c_opr_param->nr_input = 1;
        input_layput.shape = {1, {input_output_size}};
        input.layout = input_layput;
        input.device_ptr = input_device_ptr;
        c_opr_param->input = &input;

        c_opr_param->nr_output = 1;
        output_layput.shape = {1, {input_output_size}};
        output.layout = output_layput;
        output.device_ptr = output_device_ptr;
        c_opr_param->output = &output;
    };

    //! run with null param
    reset();
    run_test();

    //! run with full param
    init_param();
    run_test();

    //! run with a right index
    init_param();
    c_opr_param->extern_c_opr_dump_name = "bias_adder_dump:test";
    run_test();

    //! set a wrong index
    init_param();
    c_opr_param->extern_c_opr_dump_name = "bias_adder_dump";
    ASSERT_THROW(run_test(), MegBrainError);

    //! set a wrong index
    init_param();
    c_opr_param->extern_c_opr_dump_name = "sdfsdfs";
    ASSERT_THROW(run_test(), MegBrainError);

    //! set wrong input
    init_param();
    c_opr_param->input = nullptr;
    ASSERT_THROW(run_test(), MegBrainError);

    //! set wrong nr_input
    init_param();
    c_opr_param->nr_input = 3;
    ASSERT_THROW(run_test(), MegBrainError);

    //! set wrong input device_ptr
    init_param();
    c_opr_param->input->device_ptr = nullptr;
    ASSERT_THROW(run_test(), MegBrainError);

    //! set wrong input shape
    init_param();
    c_opr_param->input->layout.shape.shape[0] = input_output_size - 2;
    ASSERT_THROW(run_test(), MegBrainError);

    //! set wrong output
    init_param();
    c_opr_param->output = nullptr;
    ASSERT_THROW(run_test(), MegBrainError);

    //! set wrong nr_output
    init_param();
    c_opr_param->nr_output = 3;
    ASSERT_THROW(run_test(), MegBrainError);

    //! set wrong output device_ptr
    init_param();
    c_opr_param->output->device_ptr = nullptr;
    ASSERT_THROW(run_test(), MegBrainError);

    //! set wrong output shape
    init_param();
    c_opr_param->output->layout.shape.shape[0] = input_output_size - 2;
    ASSERT_THROW(run_test(), MegBrainError);

    //! set wrong dtype(test MGB_DTYPE_FLOAT32)
    init_param();
    c_opr_param->input[0].layout.dtype = MGB_DTYPE_INT32;
    ASSERT_THROW(run_test(), MegBrainError);

    //! test only device_id
    reset();
    c_opr_param->device_id = 8;
    run_test();

    //! free
    free(input_device_ptr);
    free(output_device_ptr);
}

TEST(TestExternCOpr, CPUCompute) {
    run_compute_test(CompNode::load("cpux"), MGB_DTYPE_FLOAT32);
}

TEST(TestExternCOpr, GPUCompute) {
    REQUIRE_GPU(1);
    run_compute_test(CompNode::load("gpux"), MGB_DTYPE_FLOAT32);
}

TEST(TestExternCOpr, CPUComputeMultiDtype) {
    run_compute_test(CompNode::load("cpux"), MGB_DTYPE_INT32);
#if !MEGDNN_DISABLE_FLOAT16
    run_compute_test(CompNode::load("cpux"), MGB_DTYPE_FLOAT16);
#endif
}

TEST(TestExternCOpr, Register) {
    auto api = mgb_get_extern_c_opr_api();
    ASSERT_TRUE(api->unregister_loader("bias_adder_dump"));
    ASSERT_FALSE(api->unregister_loader("bias_adder_dump"));
    auto loader = MGBOprLoaderImpl<MGB_DTYPE_FLOAT32>::make();
    ASSERT_TRUE(api->register_loader(&loader));
    ASSERT_FALSE(api->register_loader(&loader));
}

TEST(TestExternCOpr, Dedup) {
    ASSERT_EQ(0, MGBOprDescImpl<>::nr_inst);
    {
        HostTensorGenerator<> gen;
        auto host_x = gen({1});
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x);
        auto make_opr = [x](float bias) {
            std::string name = "test";
            return opr::ExternCOprRunner::make_from_desc(
                    name, {x.node()}, MGBOprDescImpl<>::make(bias));
        };
        auto y0 = make_opr(0.5), y1 = make_opr(0.6), y2 = make_opr(0.5);
        ASSERT_EQ(y0, y2);
        ASSERT_NE(y0, y1);
        ASSERT_EQ(2, MGBOprDescImpl<>::nr_inst);
    }
    ASSERT_EQ(0, MGBOprDescImpl<>::nr_inst);
}


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
