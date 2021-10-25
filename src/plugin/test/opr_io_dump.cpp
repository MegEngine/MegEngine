/**
 * \file src/plugin/test/opr_io_dump.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_io_dump_text_out.h"

#include "megbrain/test/helper.h"

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/plugin/opr_io_dump.h"
#include "megbrain/utils/debug.h"

#include <fstream>
#include <sstream>

using namespace mgb;

namespace {

using PluginMaker =
        thin_function<std::unique_ptr<OprIODumpBase>(ComputingGraph*, int level)>;
using ResultChecker = thin_function<void()>;

void run_test(CompNode cn, const PluginMaker& plugin_maker) {
    // use a predefiend seed because we have hard-coded the expected outputs
    HostTensorGenerator<> gen{0.f, 1.f, /*seed*/ 23};
    std::shared_ptr<HostTensorND> host_x;

    auto make_expect = [&host_x]() {
        HostTensorND ret{host_x->comp_node(), host_x->dtype()};
        auto x = host_x->ptr<float>(), p = ret.resize(host_x->shape()).ptr<float>();
        auto shp1 = host_x->shape(1);
        for (size_t i = 0, it = host_x->shape().total_nr_elems(); i < it; ++i) {
            p[i] = (x[i] >= 0.f ? x[i] : 0.f) * (x[i % shp1] + 2.f);
        }
        return ret;
    };
    for (size_t record : {0, 1, 2}) {
        host_x = gen({2, 3}, cn);
        auto graph = ComputingGraph::make();
        graph->options().var_sanity_check_first_run = false;
        graph->options().comp_node_seq_record_level = record;
        graph->options().graph_opt_level = 0;
        auto sync = (record != 1);
        auto plug = plugin_maker(graph.get(), record);

        // make a non-contiguous value, also introduce some shape dependencies
        auto sub_brd = [](SymbolVar x) {
            using S = opr::Subtensor;
            auto zero = x.make_scalar(0), one = x.make_scalar(1), xshp = x.symshape();
            return S::make(x, {S::AxisIndexer::make_interval(0, zero, one, None)})
                    .broadcast(xshp);
        };

        // write in primitive oprs to ensure stable opr ordering across
        // compilers
        auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
             two = x.make_scalar_dt(2), sub = sub_brd(x) + two, xrelu = opr::relu(x),
             y = xrelu * sub;

        // set stable names so the test can be used when opr naming is disabled
        auto cb_rename = [](cg::OperatorNodeBase* opr) {
            opr->name(ssprintf("opr%zu", opr->id()));
            for (auto i : opr->output()) {
                i->name(ssprintf("var%zu", i->id()));
            }
        };
        cg::DepOprIter{cb_rename}.add(y);

        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y, sync)});
        if (record == 2) {
            ComputingGraph::assert_destroy(graph);
        }
        func->execute();
        if (!sync) {
            func->wait();
        }
        plug->flush_lazy();
        MGB_ASSERT_TENSOR_EQ(make_expect(), host_y);

        if (record == 2) {
            host_x->copy_from(*gen(host_x->shape(), cn));
        } else {
            // change ptr
            *host_x = *gen(host_x->shape(), cn);
        }
        func->execute();
        if (!sync) {
            func->wait();
        }
        MGB_ASSERT_TENSOR_EQ(make_expect(), host_y);
        for (int i = 0; i < 2; ++i) {
            host_x->copy_from(*gen(host_x->shape(), cn));
            func->execute();
            if (!sync) {
                func->wait();
            }
            MGB_ASSERT_TENSOR_EQ(make_expect(), host_y);
        }

        if (record != 2) {
            // change shape
            *host_x = *gen({5, 4}, cn);
            if (record == 1) {
                ASSERT_THROW(func->execute(), MegBrainError);
            } else {
                func->execute();
                MGB_ASSERT_TENSOR_EQ(make_expect(), host_y);
            }
        }
    }
}

void run_test(const PluginMaker& plugin_maker, const ResultChecker& result_checker) {
    for (size_t i = 1; i < CompNode::NR_DEVICE_TYPE; ++i) {
        auto type = static_cast<CompNode::DeviceType>(i);
        if (!check_device_type_avaiable(type))
            continue;
        if (CompNode::get_device_count(type)) {
            auto cn = CompNode::load({type, -1, 0});
            if (cn.contain_flag(CompNode::Flag::SUPPORT_RECORDER)) {
                run_test(cn, plugin_maker);
                ASSERT_FALSE(::testing::Test::HasFailure())
                        << "failed for comp node " << cn.to_string();
                result_checker();
                ASSERT_FALSE(::testing::Test::HasFailure())
                        << "failed for comp node " << cn.to_string();
            }
        }
    }
}

std::vector<std::string> getlines(std::istream& inp, size_t skip_head = 0) {
    std::vector<std::string> ret;
    for (std::string line; std::getline(inp, line);) {
        if (skip_head) {
            --skip_head;
        } else {
            ret.emplace_back(std::move(line));
        }
    }
    return ret;
}

}  // anonymous namespace
#if MGB_VERBOSE_TYPEINFO_NAME
TEST(TestOprIODump, Text) {
    auto fname_base = output_file("test_opr_iodump");
    std::array<std::string, 3> fnames;
    auto make_plugin = [&](ComputingGraph* graph, int level) {
        fnames.at(level) = ssprintf("%s-%d.txt", fname_base.c_str(), level);
        auto ret = std::make_unique<TextOprIODump>(graph, fnames[level].c_str());
        ret->print_addr(false);
        return ret;
    };

    auto check_result = [&]() {
        for (int level = 0; level < 3; ++level) {
            std::ifstream inp_get{fnames[level]};
            std::istringstream inp_expect{EXPECTED_TEXT_OUT_REC[level]};

            auto lines_get = getlines(inp_get), lines_expect = getlines(inp_expect, 1);
            ASSERT_EQ(lines_expect.size(), lines_get.size());
            for (size_t i = 0; i < lines_expect.size(); ++i) {
                ASSERT_EQ(lines_expect[i], lines_get[i]) << "fail on line " << i;
            }
        }
        for (auto&& i : fnames) {
            // clear the content to test if next run does not produce any output
            debug::write_to_file(i.c_str(), "Lorem ipsum");
        }
    };

    run_test(make_plugin, check_result);
}
#endif

TEST(TestOprIODump, StdErr) {
    MGB_MARK_USED_VAR(EXPECTED_TEXT_OUT_REC);
    HostTensorGenerator<> gen;
    auto host_x = gen({5});
    auto host_y = gen({5});

    auto graph = ComputingGraph::make();
    std::shared_ptr<FILE> sp(stdout, [](FILE*) {});
    auto plugin = std::make_unique<TextOprIODump>(graph.get(), sp);

    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto y = opr::Host2DeviceCopy::make(*graph, host_y);
    auto z = x + y;

    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
}

TEST(TestOprIODump, Binary) {
    auto fname = output_file("");
    auto make_plugin = [&](ComputingGraph* graph, int level) {
        return std::make_unique<BinaryOprIODump>(graph, fname);
    };
    run_test(make_plugin, []() {});
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
