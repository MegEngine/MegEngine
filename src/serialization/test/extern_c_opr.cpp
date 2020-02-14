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
        auto&& i = input[0].layout;
        auto&& o = output[0].layout;
        mgb_assert(i.shape.ndim == 1 && o.shape.ndim == 1 &&
                   i.shape.shape[0] == o.shape.shape[0]);
        mgb_assert(i.dtype == MGB_DTYPE_FLOAT32 && o.dtype == out_dtype);
        auto pi = static_cast<float*>(input[0].data);
        auto bias = user_data(self)->bias;
        if (out_dtype == MGB_DTYPE_FLOAT32) {
            auto po = static_cast<float*>(output[0].data);
            for (size_t x = 0; x < i.shape.shape[0]; ++x) {
                po[x] = pi[x] + bias;
            }
        } else if (MEGDNN_FLOAT16_SELECT(out_dtype == MGB_DTYPE_FLOAT16,
                                         false)) {
#if !MEGDNN_DISABLE_FLOAT16
            auto po = static_cast<dt_float16*>(output[0].data);
            for (size_t x = 0; x < i.shape.shape[0]; ++x) {
                po[x] = pi[x] + bias;
            }
#endif
        } else {
            mgb_assert(out_dtype == MGB_DTYPE_INT32);
            auto po = static_cast<int32_t*>(output[0].data);
            for (size_t x = 0; x < i.shape.shape[0]; ++x) {
                po[x] = pi[x] + bias;
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
    x = serialization::ExternCOprRunner::make_placeholder(
                {x}, {TensorShape{1}},
                dtype == MGB_DTYPE_FLOAT32
                        ? "bias_adder_dump"
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

void run_compute_test(CompNode cn, MGBDType dtype) {
    float bias = 1.2, scale = -2.1;
    auto graph_dump = create_graph_dump(bias, scale, 0.3, dtype);
    check_dump_by_compute(
            InputFile::make_mem_proxy(graph_dump.data(), graph_dump.size()), cn,
            dtype, bias, scale);
}
}  // namespace

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
            return ExternCOprRunner::make_from_desc(
                    {x.node()}, MGBOprDescImpl<>::make(bias));
        };
        auto y0 = make_opr(0.5), y1 = make_opr(0.6), y2 = make_opr(0.5);
        ASSERT_EQ(y0, y2);
        ASSERT_NE(y0, y1);
        ASSERT_EQ(2, MGBOprDescImpl<>::nr_inst);
    }
    ASSERT_EQ(0, MGBOprDescImpl<>::nr_inst);
}


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
