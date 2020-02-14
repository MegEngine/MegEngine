/**
 * \file src/jit/impl/nvrtc/codegen_cuda.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./codegen_cuda.h"

#include "megbrain/common.h"
#include "megbrain/jit/ast_c.h"
#include "megbrain/jit/executor_opr.h"
#include "megbrain/jit/placeholder_opr.h"
#include "megbrain/jit/utils.h"
#include "megbrain/opr/tensor_manip.h"

#include <cinttypes>

#if MGB_JIT && MGB_CUDA

using namespace mgb;
using namespace jit;
using namespace ast_c;

namespace {

using VarNode2AST = ThinHashMap<VarNode*, ASTPtr>;

const char* dtype_to_cstr(DType dtype) {
    if (dtype == dtype::Float16())
        return "__half";
    if (dtype == dtype::Float32())
        return "float";
    mgb_throw(GraphError, "unsupported output dtype %s in JIT fusion",
              dtype.name());
}

std::string gen_fastdiv_offset(size_t nr_inps) {
    std::string res = "";

    char tmp[100];
#define APPEND(fmt...)                   \
    do {                                 \
        snprintf(tmp, sizeof(tmp), fmt); \
        res += tmp;                      \
    } while (0)

    for (size_t i = 0; i < nr_inps; ++i) {
        APPEND("offset_%zu = 0;\n", i);
    }
    for (size_t i = 0; i < nr_inps; ++i) {
        APPEND("tmp_idx = global_idx;\n");
        APPEND("#pragma unroll\n");
        APPEND("for (int j = {{NDIM}} - 1; j >= 1; --j) {\n");
        APPEND("Uint32Fastdiv& shp = "
               "visitors.m[%zu].m_shape_highdim[j-1];\n",
               i);
        res += R"(
        unsigned int
            ans_for_one = tmp_idx & ~shp.m_divisor_is_not_1,
            dfix = tmp_idx + shp.m_inc_dividend,
            hi32 = __umulhi(dfix, shp.m_mul),
            ans = hi32 >> shp.m_shift,
            idx_div = (ans & shp.m_divisor_is_not_1) | ans_for_one;
        )";
        APPEND("offset_%zu += (tmp_idx - idx_div * shp.m_divisor) * "
               "visitors.m[%zu].m_stride[j];\n",
               i, i);
        APPEND("tmp_idx = idx_div;\n");
        APPEND("}\n");
        APPEND("offset_%zu += tmp_idx * visitors.m[%zu].m_stride[0];\n", i, i);
    }

#undef APPEND
    return res;
}

ASTPtr gen_data_ast(size_t input_id, const JITExecutor::Args::Data& n) {
    auto res = ssprintf("(static_cast<%s*>(data.inputs[%zu]))[offset_%zu]",
                        dtype_to_cstr(n.layout.dtype), input_id, input_id);
    return ASTPtr::make<VariableAST>(res);
}

//! generate code to access input values in the kernel
void gen_input_code(str_util::StrReplaceMap& replace_map, VarNode2AST& var2ast,
                    const JITExecutor::Args& args,
                    const PlaceholderArray& placeholders) {
    std::string decl_exps_str, assign_exps_str, decl_fastdiv_offset_str;
    for (size_t i = 0; i < args.inputs.size(); i++) {
        ASTPtr elem_var = ASTPtr::make<VariableAST>("x" + std::to_string(i));
        ASTPtr elem_val = gen_data_ast(i, args.inputs[i]);
        ASTPtr elem_decl = ASTPtr::make<DeclFloatAST>(elem_var);
        ASTPtr elem_assign = ASTPtr::make<AssignAST>(elem_var, elem_val);
        var2ast[placeholders[args.inputs[i].idx]->output(0)] = elem_var;
        decl_exps_str += elem_decl->code_gen();
        assign_exps_str += elem_assign->code_gen();

        ASTPtr offset_var =
                ASTPtr::make<VariableAST>("offset_" + std::to_string(i));
        ASTPtr offset_decl = ASTPtr::make<DeclIntAST>(offset_var);
        decl_fastdiv_offset_str += offset_decl->code_gen();
    }
    str_util::append_replace_map(
            replace_map, {{"{{DECL_fastdiv_offset}}", decl_fastdiv_offset_str},
                          {"{{DECL_EXPRS}}", decl_exps_str},
                          {"{{ASSIGN_EXPRS}}", assign_exps_str}});
}

ASTPtr gen_opr_ast(cg::OperatorNodeBase* opr, const VarNode2AST& var2ast) {
    ASTPtrArray cur_inputs;
    for (auto inp_node : opr->input()) {
        cur_inputs.push_back(var2ast.at(inp_node));
    }
    if (opr->same_type<opr::Reduce>() || opr->same_type<opr::GetVarShape>() ||
        opr->same_type<opr::Dimshuffle>()) {
        // Reduce and GetVarShape occur in grad and would be ignored
        return {cur_inputs[0]};
    }

    return opr2AST(opr, cur_inputs).at(0);
}
}  // anonymous namespace

std::pair<std::string, std::string> mgb::jit::codegen_cuda(
        const InternalGraph& internal_graph, const JITExecutor::Args& args,
        bool copy_param_to_dev) {
    std::string cuda_kernel =
            R"(
#include <cuda_fp16.h>

struct Uint32Fastdiv {
    unsigned int m_mul, m_divisor, m_divisor_is_not_1, m_inc_dividend, m_shift;

    static const unsigned int MAX_DIVIDEND = ~0u - 1;
};

template <int ndim>
struct ParamElemVisitor {
    int m_stride[ndim];

    //! m_shape_highdim[i] = original_shape[i + 1]
    Uint32Fastdiv m_shape_highdim[ndim > 1 ? ndim - 1 : 1];
    static const int NDIM = ndim;
};

struct Data {
    void* inputs[{{NR_INPS}}];
    {{OUTPUT_DTYPE}}* output;
};

struct PEVisitors {
    ParamElemVisitor<{{NDIM}}> m[{{NR_INPS}}];
};

template<typename T>
static __forceinline__ __device__ T mgb_log_sum_exp(T x, T y) {
    T a, b;
    a = x < y ? x : y;
    b = x < y ? y : x;
    return T(b + log1pf(expf(a - b)));
}

)";

    cuda_kernel += copy_param_to_dev ? R"(
extern "C" __global__ void {{KERNEL_NAME}} (Data* data_ptr, size_t num_elements, PEVisitors* visitors_ptr) {
    Data data = *data_ptr;
    PEVisitors visitors = *visitors_ptr;
)"
                                     : R"(
extern "C" __global__ void {{KERNEL_NAME}} (Data data, size_t num_elements,
 PEVisitors visitors) { )";

    cuda_kernel += R"(
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int delta = blockDim.x * gridDim.x;
    unsigned int tmp_idx;

    {{DECL_EXPRS}}
    {{INTERNAL_DECL_EXPRS}}
    {{DECL_fastdiv_offset}}

    if (global_idx < num_elements) {
        {{fastdiv_offset}}
        {{ASSIGN_EXPRS}}
        {{INTERNAL_ASSIGN_EXPRS}}
        data.output[global_idx] = {{EXP}};

        global_idx += delta;
        if (global_idx < num_elements) {
            {{fastdiv_offset}}
            {{ASSIGN_EXPRS}}
            {{INTERNAL_ASSIGN_EXPRS}}
            data.output[global_idx] = {{EXP}};

            global_idx += delta;
            if (global_idx < num_elements) {
                {{fastdiv_offset}}
                {{ASSIGN_EXPRS}}
                {{INTERNAL_ASSIGN_EXPRS}}
                data.output[global_idx] = {{EXP}};
            }
        }
    }
}
)";

    VarNode2AST var2ast;
    str_util::StrReplaceMap source_replace_map;

    // add inputs to the replace map
    gen_input_code(source_replace_map, var2ast, args,
                   internal_graph.placeholders());

    // add other oprs
    std::string internal_decl_exps_str, internal_assign_exps_str;
    size_t cur_opr_cnt = 0;
    cg::DepOprIter{[&](cg::OperatorNodeBase* opr) {
        ++cur_opr_cnt;
        if (opr->same_type<JITPlaceholder>()) {
            return;
        }
        ASTPtr elem_var =
                ASTPtr::make<VariableAST>("y" + std::to_string(cur_opr_cnt));
        ASTPtr elem_val = gen_opr_ast(opr, var2ast);
        ASTPtr elem_decl = ASTPtr::make<DeclFloatAST>(elem_var);
        ASTPtr elem_assign = ASTPtr::make<AssignAST>(elem_var, elem_val);
        var2ast[opr->output(0)] = elem_var;
        internal_decl_exps_str += elem_decl->code_gen();
        internal_assign_exps_str += elem_assign->code_gen();
    }}
            .add(internal_graph.output());

    str_util::append_replace_map(
            source_replace_map,
            {{"{{NR_INPS}}", std::to_string(args.inputs.size())},
             {"{{NDIM}}", std::to_string(args.outputs[0].layout.ndim)},
             {"{{fastdiv_offset}}", gen_fastdiv_offset(args.inputs.size())},
             {"{{INTERNAL_DECL_EXPRS}}", internal_decl_exps_str},
             {"{{INTERNAL_ASSIGN_EXPRS}}", internal_assign_exps_str},
             {"{{EXP}}", var2ast.at(internal_graph.output())->code_gen()},
             {"{{OUTPUT_DTYPE}}",
              dtype_to_cstr(args.outputs[0].layout.dtype)}});

    str_util::replace_all_pairs_inplace(cuda_kernel, source_replace_map);
    str_util::replace_all_pairs_inplace(cuda_kernel, source_replace_map);

    auto kernel_name = ssprintf(
            "jit_nvrtc_%" PRIx64,
            XXHash{}.update(cuda_kernel.data(), cuda_kernel.size()).digest());
    str_util::replace_all_pairs_inplace(cuda_kernel,
                                        {{"{{KERNEL_NAME}}", kernel_name}});

    if (ExecutableHelper::keep_interm()) {
        ExecutableHelper::get().write_file(
                kernel_name + ".cu",
                "// " + internal_graph.output()->owner_opr()->name() + "\n" +
                        cuda_kernel);
    }

    return {kernel_name, cuda_kernel};
}

#endif  // MGB_JIT && MGB_CUDA

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
