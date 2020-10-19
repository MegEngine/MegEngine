/**
 * \file src/gopt/test/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/test/helper.h"

#include "megbrain/opr/io.h"
#include "megbrain/gopt/framework.h"

namespace mgb {
    //! make an opr that reads \p x; only used for test
    SymbolVar opr_reader_for_test(SymbolVar x);

    template<class Pass>
    class TestGoptBasicArithPass: public ::testing::Test {
        protected:
            HostTensorGenerator<> gen;
            std::shared_ptr<ComputingGraph> graph = ComputingGraph::make();

            SymbolVar mkvar(const char* name, const TensorShape& shp = {1},
                            CompNode cn = CompNode::load("xpu0")) {
                return opr::Host2DeviceCopy::make(*graph, gen(shp), cn)
                        .rename(name);
            }

            SymbolVar mkcvar(const char* name, const TensorShape& shp = {1},
                             CompNode cn = CompNode::load("xpu0")) {
                return opr::SharedDeviceTensor::make(
                        *graph, *gen(shp), cn).rename(name);
            }

            template<typename ...Args>
            SymbolVarArray run_opt(
                    const SymbolVarArray &inp, Args&& ...args) {
                return gopt::GraphOptimizer{}.
                    add_pass<Pass>(std::forward<Args>(args)...).
                    apply({{inp}}).endpoint_vars();
            }

            template<bool check_ne=true, typename ...Args>
            void check(SymbolVar expect, SymbolVar inp, Args&& ...args) {
                if (check_ne) {
                    ASSERT_NE(expect.node(), inp.node());
                } else {
                    ASSERT_EQ(expect, inp);
                }
                SymbolVar get;
                unpack_vector(run_opt({inp}, std::forward<Args>(args)...),
                        get);
                ASSERT_EQ(expect, get);

                // test multiple readers
                unpack_vector(
                        gopt::GraphOptimizer{}.
                        add_pass<Pass>(std::forward<Args>(args)...).
                        apply({{inp + opr_reader_for_test(inp)}}).endpoint_vars(),
                        get);

                ASSERT_EQ(expect + opr_reader_for_test(expect), get);
            }
    };
}

#define TEST_PASS(pass, name) \
    using TestGopt##pass = TestGoptBasicArithPass<gopt::pass>; \
    TEST_F(TestGopt##pass, name)

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
