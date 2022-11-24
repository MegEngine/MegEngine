#include "./codegen_opencl.h"
#include "./utils.h"

#include "megbrain/common.h"
#include "megbrain/jit/ast_c.h"
#include "megbrain/jit/executor_opr.h"
#include "megbrain/jit/placeholder_opr.h"
#include "megbrain/jit/utils.h"
#include "megbrain/opr/tensor_manip.h"

#include <cinttypes>

#if MGB_JIT && MGB_OPENCL

using namespace mgb;
using namespace jit;
using namespace ast_c;

namespace {

using VarNode2AST = ThinHashMap<VarNode*, ASTPtr>;

//! generate code to access input values in the kernel
void gen_input_code_and_gen_input_data_update(
        str_util::StrReplaceMap& replace_map, VarNode2AST& var2ast,
        const JITExecutor::Args& args, const PlaceholderArray& placeholders,
        bool is_half) {
    std::string decl_exps_str, input_data_read_str;
    std::string read_image_func = is_half ? "read_imageh" : "read_imagef";
    std::string scaler_dec_prefix = is_half ? "__global half* x" : "__global float* x";
    auto&& b_info = get_channel_broadcast_info(args);
    for (size_t i = 0; i < args.inputs.size(); i++) {
        //! gen input args
        ASTPtr elem_var_raw =
                ASTPtr::make<VariableAST>("x_after_read" + std::to_string(i));
        ASTPtr elem_var = ASTPtr::make<VariableAST>(
                "__read_only image2d_t x" + std::to_string(i));
        ASTPtr elem_var_scalar_offset;
        if (LayoutType::SCALAR == b_info[i]) {
            elem_var = ASTPtr::make<VariableAST>(scaler_dec_prefix + std::to_string(i));
            elem_var_scalar_offset = ASTPtr::make<VariableAST>(
                    "const uint x_offset" + std::to_string(i));
        }
        var2ast[placeholders[args.inputs[i].idx]->output(0)] = elem_var_raw;
        decl_exps_str += elem_var->code_gen() + ", ";
        if (LayoutType::SCALAR == b_info[i]) {
            decl_exps_str += elem_var_scalar_offset->code_gen() + ", ";
        }

        //! gen input data update
        ASTPtr elem_var_raw_input = ASTPtr::make<VariableAST>("x" + std::to_string(i));
        elem_var_raw = ASTPtr::make<VariableAST>(
                (is_half ? "half4 x_after_read" : "float4 x_after_read") +
                std::to_string(i));
        std::string coord = "coord";
        if (LayoutType::BROADCAST == b_info[i]) {
            coord = "coord_b";
        }
        std::string read_method = read_image_func + "(" +
                                  elem_var_raw_input->code_gen() + ", " + coord + ")";
        if (LayoutType::SCALAR == b_info[i]) {
            if (is_half) {
                read_method = "(half4)(vload_half(x_offset" + std::to_string(i) +
                              ", x" + std::to_string(i) + "))";
            } else {
                read_method = "(float4)(vload(x_offset" + std::to_string(i) + ", x" +
                              std::to_string(i) + "))";
            }
        }
        ASTPtr elem_assign = ASTPtr::make<AssignAST>(
                elem_var_raw, ASTPtr::make<VariableAST>(read_method));
        input_data_read_str += elem_assign->code_gen();
    }
    str_util::append_replace_map(
            replace_map, {
                                 {"{{KERNEL_SRC_ARGS}}", decl_exps_str},
                                 {"{{ASSIGN_EXPRS}}", input_data_read_str},
                         });
}

ASTPtr gen_opr_ast(cg::OperatorNodeBase* opr, const VarNode2AST& var2ast) {
    mgb_assert(
            !opr->same_type<opr::Reduce>() && !opr->same_type<opr::GetVarShape>() &&
                    !opr->same_type<opr::Dimshuffle>() && !opr->same_type<opr::PowC>(),
            "OpenCL jit not support Reduce/GetVarShape/Dimshuffle/PowC type now");
    ASTPtrArray cur_inputs;
    for (auto inp_node : opr->input()) {
        cur_inputs.push_back(var2ast.at(inp_node));
    }

    return opr2AST(opr, cur_inputs, CompNode::DeviceType::OPENCL).at(0);
}
}  // anonymous namespace

std::pair<std::string, std::string> mgb::jit::codegen_opencl(
        const InternalGraph& internal_graph, const JITExecutor::Args& args) {
    std::string opencl_kernel = R"(
__kernel void {{KERNEL_NAME}} (
        {{KERNEL_SRC_ARGS}}
        __write_only image2d_t dst,
        __private const int global_size_dim0,
        __private const int global_size_dim1,
        __private const int wc_size,
        __private const int hb_size,
        __private const uint w_size
        ) {
    #if OPENCL_ENABLE_FP16
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #endif

    const sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP;

    int wc = get_global_id(0);
    int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
    if (wc >= global_size_dim0 || hb >= global_size_dim1)
        return;
#endif

    for (; hb < hb_size; hb += global_size_dim1) {
        for (; wc < wc_size; wc += global_size_dim0) {
            int2 coord = (int2)(wc, hb);
            int2 coord_b = (int2)(wc / w_size, 0);
            {{INTERNAL_DECL_EXPRS}}
            {{ASSIGN_EXPRS}}
            {{INTERNAL_ASSIGN_EXPRS}}
            {{WRITE_IMAGE}}(dst, coord, {{EXP}});
        }
        wc = get_global_id(0);
    }
}
    )";

    auto input_dtype = args.inputs[0].layout.dtype;
    for (size_t i = 0; i < args.inputs.size(); i++) {
        mgb_assert(
                args.inputs[i].layout.dtype == input_dtype,
                "OpenCL jit all oprs should have same dtype");
    }
    mgb_assert(
            args.outputs.size() == 1 && args.outputs[0].layout.dtype == input_dtype,
            "output size should be 1 and output dtype should be same with input");
    mgb_assert(
            dtype::Float16() == input_dtype || dtype::Float32() == input_dtype,
            "OpenCL jit dtype only support float32 or float16, %s not support",
            input_dtype.name());
    auto is_half = dtype::Float16() == input_dtype;

    VarNode2AST var2ast;
    str_util::StrReplaceMap source_replace_map;

    // add inputs to the replace map
    gen_input_code_and_gen_input_data_update(
            source_replace_map, var2ast, args, internal_graph.placeholders(), is_half);

    // add other oprs
    std::string internal_decl_exps_str, internal_assign_exps_str;
    std::string write_image_func = is_half ? "write_imageh" : "write_imagef";
    size_t cur_opr_cnt = 0;
    cg::DepOprIter{[&](cg::OperatorNodeBase* opr) {
        ++cur_opr_cnt;
        if (opr->same_type<JITPlaceholder>()) {
            return;
        }
        ASTPtr elem_var = ASTPtr::make<VariableAST>("y" + std::to_string(cur_opr_cnt));
        ASTPtr elem_val = gen_opr_ast(opr, var2ast);
        ASTPtr elem_decl = ASTPtr::make<DeclFloatAST>(
                elem_var, CompNode::DeviceType::OPENCL, is_half);
        ASTPtr elem_assign = ASTPtr::make<AssignAST>(elem_var, elem_val);
        var2ast[opr->output(0)] = elem_var;
        internal_decl_exps_str += elem_decl->code_gen();
        internal_assign_exps_str += elem_assign->code_gen();
    }}.add(internal_graph.output());

    str_util::append_replace_map(
            source_replace_map,
            {{"{{INTERNAL_DECL_EXPRS}}", internal_decl_exps_str},
             {"{{INTERNAL_ASSIGN_EXPRS}}", internal_assign_exps_str},
             {"{{WRITE_IMAGE}}", write_image_func},
             {"{{EXP}}", var2ast.at(internal_graph.output())->code_gen()}});

    str_util::replace_all_pairs_inplace(opencl_kernel, source_replace_map);
    // str_util::replace_all_pairs_inplace(opencl_kernel, source_replace_map);

    auto kernel_name = ssprintf(
            "jit_opencl_%" PRIx64,
            XXHash{}.update(opencl_kernel.data(), opencl_kernel.size()).digest());
    str_util::replace_all_pairs_inplace(
            opencl_kernel, {{"{{KERNEL_NAME}}", kernel_name}});

    return {kernel_name, opencl_kernel};
}

#endif  // MGB_JIT && MGB_OPENCL
