/**
 * \file src/jit/impl/mlir/mlir_gen.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "./mlir_gen.h"
#include "./ir/each_mode.h"
#include "./ir/types.h"

#include "megbrain/jit/mlir/ir/dialect.h"
#include "megbrain/jit/mlir/ir/utils.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/tensor_manip.h"
#include "megdnn/dtype.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/Support/raw_ostream.h>

using namespace mgb;
using namespace jit;

namespace {
class MLIRGenImpl {
public:
    MLIRGenImpl(mlir::MLIRContext& context) : m_builder(&context) {}

    std::pair<llvm::StringRef, mlir::OwningModuleRef> gen(
            const InternalGraph& internal_graph,
            const JITExecutor::Args& args) {
        mlir::ModuleOp module =
                mlir::ModuleOp::create(m_builder.getUnknownLoc());

        //! Create main routine function
        auto func_op = gen_func_op(internal_graph, args);
        module.push_back(func_op);

        if (mlir::failed(mlir::verify(module))) {
            module.emitError("module verification error");
            return {};
        }

        return {func_op.getName(), module};
    }

private:
    mlir::OpBuilder m_builder;
    llvm::ScopedHashTable<mlir::StringRef, mlir::Value> m_symbol_table;

    mlir::FuncOp gen_func_op(const InternalGraph& internal_graph,
                             const JITExecutor::Args& args) {
        llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(
                m_symbol_table);
        std::vector<mlir::Type> func_args;
        for (auto&& arg : args.inputs) {
            func_args.push_back(get_type(arg.from->layout()));
        }
        for (auto&& arg : args.outputs) {
            func_args.push_back(get_type(arg.from->layout()));
        }
        //! nr_elements
        func_args.push_back(m_builder.getIndexType());
        //! nr_threads
        func_args.push_back(m_builder.getIndexType());

        auto func_type = m_builder.getFunctionType(func_args, llvm::None);
        //! function name maybe renamed in later pass
        mlir::FuncOp func_op = mlir::FuncOp::create(m_builder.getUnknownLoc(),
                                                    "func", func_type);
        if (!func_op)
            return nullptr;

        func_op.setAttr("llvm.emit_c_interface",
                        mlir::UnitAttr::get(m_builder.getContext()));
        auto& entry_block = *func_op.addEntryBlock();
        size_t idx = 0;
        for (auto&& input : args.inputs) {
            if (mlir::failed(declare(internal_graph.placeholders()[input.idx]
                                             ->output(0)
                                             ->name(),
                                     entry_block.getArgument(idx)))) {
                return nullptr;
            }
            idx++;
        }
        for (auto&& output : args.outputs) {
            if (mlir::failed(declare(output.from->name(),
                                     entry_block.getArgument(idx)))) {
                return nullptr;
            }
            idx++;
        }

        m_builder.setInsertionPointToStart(&entry_block);

        if (mlir::failed(gen_func_body(internal_graph, args))) {
            func_op.erase();
            return nullptr;
        }

        dialect::ReturnOp return_op;
        if (!return_op) {
            m_builder.create<dialect::ReturnOp>(m_builder.getUnknownLoc());
        }
        std::string op_content = mlir_type_to_string(func_op);
        func_op.setName(
                ssprintf("jit_mlir_%" PRIx64,
                         XXHash{}.update(op_content.data(), op_content.size())
                                 .digest()));
        return func_op;
    }

    mlir::LogicalResult gen_func_body(const InternalGraph& internal_graph,
                                      const JITExecutor::Args& args) {
        llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(
                m_symbol_table);
        cg::DepOprIter{[&](cg::OperatorNodeBase* opr) {
            if (opr->same_type<JITPlaceholder>()) {
                return;
            } else if (opr->same_type<opr::ImmutableTensor>()) {
                auto imm = SymbolVar{opr->output(0)}.as_immutable_scalar();
                if (imm.valid()) {
                    auto dtype = imm->dtype();
                    float scalar_value;
                    if (dtype == dtype::Float32()) {
                        scalar_value = imm->get<float>();
                    } else {
                        mgb_throw(InternalError,
                                  "mlir backend currently only support f32 "
                                  "dtype, but got %s",
                                  dtype.name());
                    }
                    auto&& out = m_builder.create<dialect::ConstantScalarOp>(
                            m_builder.getUnknownLoc(), m_builder.getF32Type(),
                            m_builder.getF32FloatAttr(scalar_value));
                    mgb_assert(mlir::succeeded(
                            declare(opr->output(0)->name(), out)));
                }
            } else if (opr->same_type<opr::Elemwise>()) {
                auto&& out = gen_elemwise(opr->cast_final<opr::Elemwise>());
                mgb_assert(
                        mlir::succeeded(declare(opr->output(0)->name(), out)));
                return;
            } else if (opr->same_type<opr::Dimshuffle>()) {
                auto&& out = gen_dimshuffle(opr->cast_final<opr::Dimshuffle>());
                mgb_assert(
                        mlir::succeeded(declare(opr->output(0)->name(), out)));
            } else if (opr->same_type<opr::TypeCvt>()) {
                auto&& out = gen_typecvt(opr->cast_final<opr::TypeCvt>());
                mgb_assert(
                        mlir::succeeded(declare(opr->output(0)->name(), out)));
            }
        }}
                .add(internal_graph.output());
        m_builder.create<dialect::AssignOp>(m_builder.getUnknownLoc(),
                                            get(internal_graph.output()),
                                            get(args.outputs[0].from));

        return mlir::success();
    }

    mlir::Value gen_elemwise(const opr::Elemwise& opr) {
        llvm::SmallVector<mlir::Value, 4> operands;
        for (size_t i = 0; i < opr.input().size(); i++) {
            operands.push_back(get(opr.input(i)));
        }
        mlir::Type res_type = deduce_elemwise_res_type(operands);
        return m_builder.create<dialect::Elemwise>(
                m_builder.getUnknownLoc(), res_type, mlir::ValueRange(operands),
                opr.param().mode);
    }

    mlir::Value gen_typecvt(const opr::TypeCvt& opr) {
        auto itype = get(opr.input(0))
                             .getType()
                             .dyn_cast_or_null<mlir::MemRefType>();
        mgb_assert(itype, "currently only support MemRefType");
        auto output_type = megdnn_dtype_to_mlir_type(opr.param(),
                                                     m_builder.getContext());
        auto res_type = mlir::MemRefType::get(
                itype.getShape(), signless(output_type));
        auto inp_type = megdnn_dtype_to_mlir_type(opr.input(0)->dtype(),
                                                  m_builder.getContext());
        return m_builder.create<dialect::TypeCvt>(
                m_builder.getUnknownLoc(), res_type, get(opr.input(0)),
                mlir::TypeAttr::get(inp_type), opr.param());
    }

    mlir::Value gen_dimshuffle(const opr::Dimshuffle& opr) {
        auto itype = get(opr.input(0))
                             .getType()
                             .dyn_cast_or_null<mlir::MemRefType>();
        mgb_assert(itype, "the input type of Dimshuffle must be MemRefType");
        auto ishape = itype.getShape();
        auto param = opr.param();

        std::vector<int32_t> pattern;
        std::vector<int64_t> oshape;
        for (size_t i = 0; i < param.pattern_len; i++) {
            int32_t j = param.pattern[i];
            pattern.push_back(j);
            if (j < 0) {
                oshape.push_back(1);
            } else {
                oshape.push_back(ishape[j]);
            }
        }
        auto res_type = mlir::MemRefType::get(oshape, itype.getElementType());

        return m_builder.create<dialect::Dimshuffle>(
                m_builder.getUnknownLoc(), res_type, get(opr.input(0)),
                pattern);
    }

    mlir::Type get_type(const TensorLayout& layout) {
        return layout_to_mlir_type(layout, m_builder);
    }

    mlir::Value get(const VarNode* var) {
        if (auto ret = m_symbol_table.lookup(var->name())) {
            return ret;
        }
        mgb_throw(InternalError, "Unknown var: %s", var->cname());
    }

    mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
        if (m_symbol_table.count(var)) {
            return mlir::failure();
        }
        m_symbol_table.insert(var, value);
        return mlir::success();
    }
};
}  // namespace

std::pair<llvm::StringRef, mlir::OwningModuleRef> mgb::jit::mlir_gen(
        mlir::MLIRContext& context,
        const mgb::jit::InternalGraph& internal_graph,
        const mgb::jit::JITExecutor::Args& args) {
    return MLIRGenImpl(context).gen(internal_graph, args);
}

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
