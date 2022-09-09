#include "./non_zero.h"
#include "./rng.h"
#include "./tensor.h"
#include "./utils.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

using Param = NonZero::Param;

std::vector<NonZeroTestcase> NonZeroTestcase::make() {
    Param param;
    std::vector<NonZeroTestcase> ret;

    TensorShape shape1{4};
    ret.push_back({param, TensorLayout{shape1, dtype::Int8()}});
    NonZeroTestcase& case1 = ret.back();
    case1.m_mem.reset(new uint8_t[case1.m_data.layout.span().dist_byte()]);
    memset(case1.m_mem.get(), 0, case1.m_data.layout.span().dist_byte());
    case1.m_data.reset_ptr(case1.m_mem.get());
    dt_int8* pt_1 = reinterpret_cast<dt_int8*>(case1.m_mem.get());
    pt_1[3] = 1;
    case1.correct_answer.push_back(3);

    TensorShape shape2{1, 1, 1, 1, 1, 1, 1};
    ret.push_back({param, TensorLayout{shape2, dtype::Float32()}});
    NonZeroTestcase& case2 = ret.back();
    case2.m_mem.reset(new uint8_t[case2.m_data.layout.span().dist_byte()]);
    memset(case2.m_mem.get(), 0, case2.m_data.layout.span().dist_byte());
    case2.m_data.reset_ptr(case2.m_mem.get());
    dt_float32* pt_2 = reinterpret_cast<dt_float32*>(case2.m_mem.get());
    pt_2[0] = 1.0;
    case2.correct_answer = {0, 0, 0, 0, 0, 0, 0};

    TensorShape shape3{0};
    ret.push_back({param, TensorLayout{shape3, dtype::Float32()}});
    NonZeroTestcase& case3 = ret.back();
    case3.m_mem.reset(new uint8_t[case3.m_data.layout.span().dist_byte()]);
    memset(case3.m_mem.get(), 0, case3.m_data.layout.span().dist_byte());
    case3.m_data.reset_ptr(case3.m_mem.get());
    case3.correct_answer = {};

    TensorShape shape4{1, 2, 3, 4, 5, 6, 7};
    ret.push_back({param, TensorLayout{shape4, dtype::Float32()}});
    NonZeroTestcase& case4 = ret.back();
    case4.m_mem.reset(new uint8_t[case4.m_data.layout.span().dist_byte()]);
    memset(case4.m_mem.get(), 0, case4.m_data.layout.span().dist_byte());
    case4.m_data.reset_ptr(case4.m_mem.get());
    dt_float32* pt_4 = reinterpret_cast<dt_float32*>(case4.m_mem.get());
    pt_4[shape4.total_nr_elems() - 1] = 1.0;
    case4.correct_answer = {0, 1, 2, 3, 4, 5, 6};

    TensorShape shape5{2, 2, 2, 2, 2, 2, 2};
    ret.push_back({param, TensorLayout{shape5, dtype::Float32()}});
    NonZeroTestcase& case5 = ret.back();
    case5.m_mem.reset(new uint8_t[case5.m_data.layout.span().dist_byte()]);
    memset(case5.m_mem.get(), 0, case5.m_data.layout.span().dist_byte());
    case5.m_data.reset_ptr(case5.m_mem.get());
    dt_float32* pt_5 = reinterpret_cast<dt_float32*>(case5.m_mem.get());
    pt_5[63] = 1.0;
    case5.correct_answer = {
            0, 1, 1, 1, 1, 1, 1,
    };

    return ret;
}

NonZeroTestcase::Result NonZeroTestcase::run_naive(NonZero* opr) {
    auto handle = opr->handle();
    DynOutMallocPolicyImpl malloc_policy(handle);
    opr->param() = m_param;

    auto workspace_size = opr->get_workspace_in_bytes(m_data.layout);
    auto workspace_ptr = malloc_policy.alloc_workspace(workspace_size, nullptr);
    auto result = opr->exec(
            m_data, {(dt_byte*)workspace_ptr, workspace_size}, &malloc_policy);
    malloc_policy.free_workspace(workspace_ptr, nullptr);

    return result;
}
NonZeroTestcase::CUDAResult NonZeroTestcase::run_cuda(NonZero* opr) {
    auto handle = opr->handle();
    DynOutMallocPolicyImpl malloc_policy(handle);
    opr->param() = m_param;
    auto data = make_tensor_h2d(handle, m_data);

    auto workspace_size = opr->get_workspace_in_bytes(m_data.layout);
    auto workspace_ptr = malloc_policy.alloc_workspace(workspace_size, nullptr);
    auto result =
            opr->exec(*data, {(dt_byte*)workspace_ptr, workspace_size}, &malloc_policy);
    malloc_policy.free_workspace(workspace_ptr, nullptr);

    return {make_tensor_d2h(handle, result)};
}

void NonZeroTestcase::Assert(
        std::vector<int>& correct_answer, int ndim, NonZeroTestcase::Result result) {
    dt_int32* data_pt = result.ptr<dt_int32>();
    ASSERT_EQ(result.layout.total_nr_elems(), correct_answer.size());
    ASSERT_EQ(ndim, result.layout.shape[0]);
    for (size_t ele_idx = 0; ele_idx < result.layout.total_nr_elems(); ele_idx++) {
        ASSERT_EQ(data_pt[ele_idx], correct_answer[ele_idx]);
    }
}