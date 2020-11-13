/**
 * \file imperative/src/test/imperative.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/opr/utility.h"
#include "megbrain/imperative/blob_manager.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/comp_node_env.h"


using namespace mgb;
using namespace cg;
using namespace imperative;

TEST(TestImperative, APlusB) {
     auto op = OprAttr::make("Elemwise");
     auto&& attr = op->cast_final_safe<OprAttr>();
     using Param = opr::Elemwise::Param;
     Param param{Param::Mode::ADD};
     attr.param.write_pod(param);
     OprChecker(op).run({TensorShape{42}, TensorShape{42}});
}

TEST(TestImperative, Convolution) {
     auto op = OprAttr::make("ConvolutionV2");
     auto&& attr = op->cast_final_safe<OprAttr>();
     using Param = opr::Convolution::Param;
     using Policy = opr::Convolution::ExecutionPolicy;
     Param param{Param::Mode::CONVOLUTION};
     Policy policy{Policy::Strategy::HEURISTIC};
     attr.param.write_pod(param);
     attr.param.write_pod(policy);
     size_t N = 4, IC = 3, OC = 8, FH = 3, FW = 3, IH = 16, IW = 16;
     OprChecker(op).run({TensorShape{N, IC, IH, IW}, TensorShape{OC, IC, FH, FW}});
}

TEST(TestImperative, Reduce) {
     auto op = OprAttr::make("ReduceV2");
     auto&& attr = op->cast_final_safe<OprAttr>();
     using Param = opr::Reduce::Param;
     Param param{Param::Mode::SUM_SQR};
     attr.param.write_pod(param);
     HostTensorND one{CompNode::load("xpu0"), {{1}, dtype::Int32()}};
     one.ptr<int>()[0] = 1;
     OprChecker(op).run({TensorShape{2, 3, 4}, one});
}

TEST(TestImperative, BatchNorm) {
     auto op = OprAttr::make("BatchNorm");
     auto&& attr = op->cast_final_safe<OprAttr>();
     using Param = opr::BatchNorm::Param;
     Param param;
     param.param_dim = Param::ParamDim::DIM_1C11;
     param.avg_factor = 0.999;
     attr.param.write_pod(param);
     size_t N=2, C=3, H=5, W=5;
     OprChecker(op).run({
          TensorShape{N, C, H, W},
          TensorShape{1, C, 1, 1},
          TensorShape{1, C, 1, 1},
          TensorShape{1, C, 1, 1},
          TensorShape{1, C, 1, 1}
     });
}

TEST(TestImperative, Concat) {
     OprAttr::Param param;
     param.write_pod(megdnn::param::Axis(0));
     OperatorNodeConfig config{CompNode::load("xpu1")};
     OprChecker(OprAttr::make("Concat", param, config))
          .run({TensorShape{200, 300}, TensorShape{300, 300}});
}

TEST(TestImperative, Split) {
     OprAttr::Param param;
     param.write_pod(megdnn::param::Axis(0));
     auto op = OprAttr::make("Split", param, OperatorNodeConfig{});
     auto cn = CompNode::load("xpu0");
     HostTensorND s1{cn, {{1}, dtype::Int32()}};
     s1.ptr<int>()[0] = 20;
     HostTensorND s2{cn, {{1}, dtype::Int32()}};
     s2.ptr<int>()[0] = 80;
     OprChecker(op).run({TensorShape{100}, s1, s2});
}

#if MGB_CUDA && MGB_ENABLE_EXCEPTION
void run_graph(size_t mem_reserved, bool enable_defrag) {
     CompNode::try_coalesce_all_free_memory();
     CompNode::finalize();

     auto cn = CompNode::load("gpux");
     cn.sync(); // wait for async init to finish

     BlobManager::inst() -> set_enable(enable_defrag);

     HostTensorGenerator<> gen;
     using TensorPtr = std::shared_ptr<Tensor>;
     TensorPtr ptr_a[100];

     size_t unit_size = mem_reserved / (100.5 * 4);
     auto host_a = gen({unit_size});
     for(int i = 0; i < 100; ++i) {
          ptr_a[i] = Tensor::make(*host_a);
     }

     // free half
     for(int i = 0; i < 100; i += 2) {
          ptr_a[i].reset();
     }

     auto op = OprAttr::make("Elemwise");
     auto&& attr = op->cast_final_safe<OprAttr>();
     using Param = opr::Elemwise::Param;
     Param param{Param::Mode::MUL};
     attr.param.write_pod(param);

     auto out = OpDef::apply_on_physical_tensor(*op, {ptr_a[1], ptr_a[99]}).at(0);

     // value before defrag
     HostTensorND host_out_before;
     host_out_before.copy_from(out->dev_tensor()).sync();

     // make defrag work
     auto e = Tensor::make(*gen({unit_size * 10}));

     // value after defrag
     HostTensorND host_out_after;
     host_out_after.copy_from(out->dev_tensor()).sync();

     // make sure defragment do not change the value
     for (size_t i = 0; i < unit_size; ++ i) {
          ASSERT_EQ(host_out_before.ptr<float>()[i], host_out_after.ptr<float>()[i]);
     }
}

TEST(TestImperative, Defragment) {
     REQUIRE_GPU(1);
     CompNode::load("gpux").activate();
     size_t reserve;
     {
          size_t free, tot;
          MGB_CUDA_CHECK(cudaMemGetInfo(&free, &tot));
          reserve = free * 0.92;
     }
     auto reserve_setting = ssprintf("b:%zu", reserve);

     auto do_run = [reserve]() {
          ASSERT_THROW(run_graph(reserve, false), MemAllocError);
          run_graph(reserve, true);
     };

     // reserve memory explicitly to avoid uncontrollable factors
     constexpr const char* KEY = "MGB_CUDA_RESERVE_MEMORY";
     auto old_value = getenv(KEY);
     setenv(KEY, reserve_setting.c_str(), 1);
     MGB_TRY {
          do_run();
     } MGB_FINALLY(
             if (old_value) {
                 setenv(KEY, old_value, 1);
             } else {
                 unsetenv(KEY);
             }
             CompNode::try_coalesce_all_free_memory();
             CompNode::finalize();
     );
}
#endif // MGB_CUDA && MGB_ENABLE_EXCEPTION

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
