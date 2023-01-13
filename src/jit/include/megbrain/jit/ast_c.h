#pragma once

#include "megbrain_build_config.h"
#if MGB_JIT

#include "megbrain/common.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/utils/small_vector.h"

namespace mgb {
namespace jit {

// a simplified AST for C source code
namespace ast_c {

class AST {
public:
    virtual ~AST() {}
    virtual std::string code_gen() = 0;
};

class ASTPtr {
    std::shared_ptr<AST> m_ptr;

    explicit ASTPtr(std::shared_ptr<AST> p) : m_ptr{std::move(p)} {}

public:
    ASTPtr() = default;

    /*!
     * \brief construct a new ASTPtr
     * \tparam T AST node type
     * \param args arguments to be passed to ctor of \p T
     */
    template <typename T, typename... Args>
    static std::enable_if_t<std::is_base_of<AST, T>::value, ASTPtr> make(
            Args&&... args) {
        return ASTPtr{std::make_shared<T>(std::forward<Args>(args)...)};
    }

    AST* operator->() const { return m_ptr.get(); }

    inline ASTPtr(int imm);
    inline ASTPtr(float imm);
    inline ASTPtr(float imm, CompNode::DeviceType cn_type, bool is_half);
};
using ASTPtrArray = SmallVector<ASTPtr>;

//! function type for generating AST nodes
using AstGenerator = thin_function<ASTPtrArray(const ASTPtrArray&, bool is_half)>;

class IntAST : public AST {
public:
    IntAST(int val) : m_val(val) {}
    inline std::string code_gen() override { return std::to_string(m_val); }

private:
    int m_val;
};

class FloatAST : public AST {
public:
    FloatAST(
            float val, CompNode::DeviceType cn_type = CompNode::DeviceType::CPU,
            bool is_half = false)
            : m_val(val), m_cn_type(cn_type), m_is_half(is_half) {}
    inline std::string code_gen() override {

        mgb_assert(!m_is_half, "code issue, only OpenCL support as half now");
        return ssprintf("float(%.12e)", m_val);
    }

private:
    float m_val;
    CompNode::DeviceType m_cn_type;
    bool m_is_half;
};

class VariableAST : public AST {
public:
    VariableAST(const std::string& name) : m_name(name) {}
    inline std::string code_gen() override { return m_name; }

private:
    std::string m_name;
};

class BinaryAST : public AST {
public:
    BinaryAST(const std::string& op, const ASTPtr& lhs, const ASTPtr& rhs)
            : m_op(op), m_lhs(lhs), m_rhs(rhs) {}
    inline std::string code_gen() override {
        return "(" + m_lhs->code_gen() + " " + m_op + " " + m_rhs->code_gen() + ")";
    }

private:
    std::string m_op;
    ASTPtr m_lhs, m_rhs;
};

class CallAST : public AST {
public:
    CallAST(std::string callee, ASTPtrArray args)
            : m_callee{std::move(callee)}, m_args{std::move(args)} {}

    inline std::string code_gen() override {
        std::string ret = m_callee + "(";
        for (uint32_t i = 0; i < m_args.size(); ++i) {
            ret += m_args[i]->code_gen();
            ret += ", ";
        }
        ret.pop_back();
        ret.pop_back();
        ret += ")";
        return ret;
    }

private:
    std::string m_callee;
    ASTPtrArray m_args;
};

class ArraySubscriptAST : public AST {
public:
    ArraySubscriptAST(const ASTPtr& lhs, const ASTPtr& rhs) : m_lhs(lhs), m_rhs(rhs) {}
    inline std::string code_gen() override {
        return m_lhs->code_gen() + "[" + m_rhs->code_gen() + "]";
    }

private:
    ASTPtr m_lhs, m_rhs;
};

//! ternary conditional opr
class Cond3AST : public AST {
    ASTPtr m_cond, m_true, m_false;

public:
    Cond3AST(ASTPtr cond, ASTPtr true_, ASTPtr false_)
            : m_cond{std::move(cond)},
              m_true{std::move(true_)},
              m_false{std::move(false_)} {}

    std::string code_gen() override {
        return "(" + m_cond->code_gen() + " ? " + m_true->code_gen() + " : " +
               m_false->code_gen() + ")";
    }
};

class DeclFloatAST : public AST {
public:
    DeclFloatAST(
            const ASTPtr& var, CompNode::DeviceType cn_type = CompNode::DeviceType::CPU,
            bool is_half = false)
            : m_var(var), m_cn_type(cn_type), m_is_half(is_half) {}
    inline std::string code_gen() override {

        mgb_assert(!m_is_half, "code issue, only OpenCL support as half now");
        return "float " + m_var->code_gen() + ";";
    }

private:
    ASTPtr m_var;
    CompNode::DeviceType m_cn_type;
    bool m_is_half;
};

class DeclIntAST : public AST {
public:
    DeclIntAST(const ASTPtr& var) : m_var(var) {}
    inline std::string code_gen() override { return "int " + m_var->code_gen() + ";"; }

private:
    ASTPtr m_var;
};

class AssignAST : public AST {
public:
    AssignAST(ASTPtr var, ASTPtr val) : m_var(var), m_val(val) {}
    inline std::string code_gen() override {
        return m_var->code_gen() + " = " + m_val->code_gen() + ";\n";
    }

private:
    ASTPtr m_var;
    ASTPtr m_val;
};

static inline ASTPtr make_call(std::string callee, ASTPtrArray args) {
    return ASTPtr::make<CallAST>(std::move(callee), std::move(args));
}

static inline ASTPtr operator+(const ASTPtr& lhs, const ASTPtr& rhs) {
    return ASTPtr::make<BinaryAST>("+", lhs, rhs);
}

static inline ASTPtr operator-(const ASTPtr& lhs, const ASTPtr& rhs) {
    return ASTPtr::make<BinaryAST>("-", lhs, rhs);
}

static inline ASTPtr operator-(const ASTPtr& lhs) {
    return make_call("-", {lhs});
}

static inline ASTPtr operator*(const ASTPtr& lhs, const ASTPtr& rhs) {
    return ASTPtr::make<BinaryAST>("*", lhs, rhs);
}

static inline ASTPtr operator/(const ASTPtr& lhs, const ASTPtr& rhs) {
    return ASTPtr::make<BinaryAST>("/", lhs, rhs);
}

static inline ASTPtr operator>(const ASTPtr& lhs, const ASTPtr& rhs) {
    return ASTPtr::make<BinaryAST>(">", lhs, rhs);
}

static inline ASTPtr operator>=(const ASTPtr& lhs, const ASTPtr& rhs) {
    return ASTPtr::make<BinaryAST>(">=", lhs, rhs);
}

ASTPtr::ASTPtr(int imm) : m_ptr(std::make_shared<IntAST>(imm)) {}

ASTPtr::ASTPtr(float imm) : m_ptr(std::make_shared<FloatAST>(imm)) {}

ASTPtr::ASTPtr(float imm, CompNode::DeviceType cn_type, bool is_half)
        : m_ptr(std::make_shared<FloatAST>(imm, cn_type, is_half)) {}

using ElemMode = opr::Elemwise::Mode;
using ElemGeneratorMap = ThinHashMap<ElemMode, AstGenerator>;

//! mapping from elemwise mode to ast node generator
const ElemGeneratorMap& elem_opr_generator(CompNode::DeviceType type);

static inline bool check_elem_mode(ElemMode mode, CompNode::DeviceType type) {
    return elem_opr_generator(type).count(mode);
}

/*!
 * \brief Generate a AST node from the opr and the given ast inputs
 * \param opr the opr
 * \param inputs the AST inputs of the ASTs to be generate
 * \param device_type jit backend cn device type
 * \return AST nodes corresponding to opr value outputs
 */
ASTPtrArray opr2AST(
        cg::OperatorNodeBase* opr, const ASTPtrArray& inputs,
        CompNode::DeviceType device_type);

}  // namespace ast_c
}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
