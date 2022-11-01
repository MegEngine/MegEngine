#include "megbrain/imperative/interpreter.h"
#include "../impl/interpreter/tensor_info.h"
#include "./helper.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/imperative/blob_manager.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/physical_tensor.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

using namespace mgb;
using namespace cg;
using namespace imperative;
using namespace interpreter;

TEST(TestImperative, InterpreterPut) {
    HostTensorGenerator<> gen;
    auto h0 = gen({3});
    auto&& channel = Interpreter::inst().create_channel();
    auto tensor_handle = channel->put(*h0, true);
    auto tensor_info = reinterpret_cast<intl::TensorInfo*>(tensor_handle);
    channel->sync();
    ASSERT_TRUE(tensor_info->status == intl::TensorInfo::Produced);
    //! because tensor elems is less than TensorShape::MAX_NDIM, stored it
    //! directly
    ASSERT_EQ(tensor_info->ptr->get_value().raw_ptr(), h0->raw_ptr());
    auto shape = channel->get_shape(tensor_handle);
    ASSERT_TRUE(shape.ndim == 1);
    ASSERT_TRUE(shape.total_nr_elems() == 3);

    auto h2 = gen({10});
    auto tensor_handle2 = channel->put(*h2, false);
    auto tensor_handle2_once = channel->put(*h2, false);
    channel->sync();
    ASSERT_NE(tensor_handle2_once, tensor_handle2);
    auto finded = MultiCNConstTensorCache::inst().lookup(*h2);
    ASSERT_TRUE(finded.get());
    //! Device tensor ptr is not equal host tensor ptr
    ASSERT_NE(finded->raw_ptr_not_for_readwrite(), h2->raw_ptr());
    channel->del(tensor_handle);
    channel->del(tensor_handle2);
    channel->del(tensor_handle2_once);
}

TEST(TestImperative, InterpreterApplyOp) {
    HostTensorGenerator<> gen;
    size_t add = 2, dim0 = 5, dim1 = 10;
    auto h0 = gen({1});
    h0->ptr<float>()[0] = add;
    auto h1 = gen({dim0, dim1});
    for (size_t i = 0; i < dim0 * dim1; i++) {
        h1->ptr<float>()[i] = i;
    }
    auto&& channel = Interpreter::inst().create_channel();
    auto tensor_handle0 = channel->put(*h0, false);
    auto tensor_handle1 = channel->put(*h1, false);
    SmallVector<Interpreter::Handle> inputs{tensor_handle0, tensor_handle1};

    auto op = OprAttr::make("Elemwise");
    auto&& attr = op->cast_final_safe<OprAttr>();
    using Param = opr::Elemwise::Param;
    Param param{Param::Mode::ADD};
    attr.param.write_pod(param);

    auto outputs = channel->apply_op(op, inputs);
    channel->sync();
    auto out_tensor = reinterpret_cast<intl::TensorInfo*>(outputs[0])->ptr->get_value();
    ASSERT_EQ(out_tensor.layout().ndim, 2);
    ASSERT_EQ(out_tensor.shape(0), dim0);
    ASSERT_EQ(out_tensor.shape(1), dim1);
    float* output = out_tensor.ptr<float>();
    for (size_t i = 0; i < dim0 * dim1; i++) {
        ASSERT_EQ(output[i], i + add);
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
