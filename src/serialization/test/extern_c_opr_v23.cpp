/**
 * \file src/serialization/test/extern_c_opr_v23.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./extern_c_opr_v23.h"

#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/serialization/extern_c_opr_io.h"
#include "megbrain/test/helper.h"

using namespace mgb;
using namespace serialization;

namespace {

//! a custom opr to compute x + bias
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
                        MGBTensor* output) {
        auto&& i = input[0].layout;
        auto&& o = output[0].layout;
        mgb_assert(i.shape.ndim == 1 && o.shape.ndim == 1 &&
                   i.shape.shape[0] == o.shape.shape[0]);
        mgb_assert(i.dtype == MGB_DTYPE_FLOAT32 &&
                   o.dtype == MGB_DTYPE_FLOAT32);
        auto pi = static_cast<float*>(input[0].data),
             po = static_cast<float*>(output[0].data);
        auto bias = user_data(self)->bias;
        for (size_t x = 0; x < i.shape.shape[0]; ++x) {
            po[x] = pi[x] + bias;
        }
    }

    static void infer_shape(const MGBOprDesc*, const MGBTensorShape* input,
                            MGBTensorShape* output) {
        output[0] = input[0];
    }

public:
    static int nr_inst;
    static MGBOprDesc* make(float bias) {
        ++nr_inst;
        auto ud = std::make_unique<UserData>();
        ud->bias = bias;
        auto desc = std::make_unique<MGBOprDesc>();
        desc->nr_input = desc->nr_output = 1;
        desc->type_name = "bias_adder";
        desc->user_data = ud.release();
#define s(n) desc->n = &MGBOprDescImpl::n;
        MGB_OPR_DESC_FOREACH_MEM_FN(s);
#undef s
        return desc.release();
    }
};
int MGBOprDescImpl::nr_inst = 0;

class MGBOprLoaderImpl {
    static MGBOprDesc* create_desc(size_t nr_input, const void* buf,
                                   size_t buf_len) {
        mgb_assert(buf_len == sizeof(float));
        float fv;
        memcpy(&fv, buf, buf_len);
        return MGBOprDescImpl::make(fv);
    }
public:
    static MGBOprLoader make() { return {"bias_adder_dump_v23", &create_desc}; }
};

class MGBOprLoaderReg {
public:
    MGBOprLoaderReg() {
        auto api = mgb_get_extern_c_opr_api();
        auto loader = MGBOprLoaderImpl::make();
        auto succ = api->register_loader(&loader);
        mgb_assert(succ);
    }
};
MGBOprLoaderReg loader_reg;

std::vector<uint8_t> create_graph_dump(float bias, float extra_scale,
                                       float sleep) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1}, "cpux");
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    if (sleep)
        x = opr::Sleep::make(x, sleep);
    x = opr::ExternCOprRunner::make_placeholder(
                {x}, {TensorShape{1}}, "bias_adder_dump_v23", &bias, sizeof(bias))
                ->output(0);
    if (extra_scale)
        x = x * extra_scale;

    std::vector<uint8_t> ret;
    auto dumper = GraphDumper::make(OutputFile::make_vector_proxy(&ret));
    dumper->dump({x});
    return ret;
}

void run_compute_test(CompNode cn) {
    float bias = 1.2, scale = -2.1;
    auto graph_dump = create_graph_dump(bias, scale, 0.3);
    GraphLoadConfig config;
    config.comp_node_mapper = [loc = cn.locator()](CompNode::Locator & t) {
        t = loc;
    };
    auto loader = GraphLoader::make(
            InputFile::make_mem_proxy(graph_dump.data(), graph_dump.size()));
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
        for (int i = 0; i < 23; ++i)
            py[i] = (py[i] + bias) * scale;
    }

    HostTensorND host_y;
    auto func = load_ret.graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
}
}  // namespace

TEST(TestExternCOprV23, CPUCompute) {
    run_compute_test(CompNode::load("cpux"));
}

TEST(TestExternCOprV23, GPUCompute) {
    REQUIRE_GPU(1);
    run_compute_test(CompNode::load("gpux"));
}

TEST(TestExternCOprV23, Register) {
    auto api = mgb_get_extern_c_opr_api();
    ASSERT_TRUE(api->unregister_loader("bias_adder_dump_v23"));
    ASSERT_FALSE(api->unregister_loader("bias_adder_dump_v23"));
    auto loader = MGBOprLoaderImpl::make();
    ASSERT_TRUE(api->register_loader(&loader));
    ASSERT_FALSE(api->register_loader(&loader));
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
