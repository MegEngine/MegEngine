#include "./legacy_checker.h"
#include "megbrain/comp_node_env.h"

#include "megbrain/gopt/inference.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs/base.h"

#include <gmock/gmock.h>

#include <cmath>
#include <memory>
#include <random>

using namespace std;
using namespace mgb;

namespace {

TEST(TestOprDNN, PaddingForwardSerialization) {
    using namespace serialization;

    auto fname = output_file("PaddingForwardTest");
    auto dump = [&]() {
        opr::Padding::Param param;
        param.padding_mode = megdnn::param::Padding::PaddingMode(0);
        param.front_offset_dim0 = 3;
        param.front_offset_dim1 = 3;
        param.front_offset_dim2 = 3;
        param.front_offset_dim3 = 3;
        param.front_offset_dim4 = 0;
        param.front_offset_dim5 = 0;
        param.front_offset_dim6 = 0;
        param.back_offset_dim0 = 0;
        param.back_offset_dim1 = 0;
        param.back_offset_dim2 = 0;
        param.back_offset_dim3 = 0;
        param.back_offset_dim4 = 0;
        param.back_offset_dim5 = 0;
        param.back_offset_dim6 = 0;
        param.padding_val = 0;

        auto cn = CompNode::load("xpu");
        auto graph = ComputingGraph::make();
        HostTensorND inp_host{cn, {32, 4, 24, 24}, dtype::Float32()};
        auto inp = opr::ImmutableTensor::make(*graph, inp_host);
        auto opr = opr::PaddingForward::make(inp, param, {});
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()));
        auto rst = dumper->dump({opr});
        ASSERT_EQ(rst.outputs.size(), 1u);
    };

    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()));
        auto rst = loader->load();
        ASSERT_EQ(rst.output_var_list.size(), 1u);
    };

    dump();
    load();
}
}  // namespace