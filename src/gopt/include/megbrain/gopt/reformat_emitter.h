#pragma once
#include <vector>
#include "megbrain/graph.h"
#include "megdnn/named_tensor.h"

namespace mgb {
namespace gopt {

class Emitter {
public:
    using Builder = thin_function<VarNode*(const VarNodeArray&)>;
    using Checker = thin_function<bool(const VarNodeArray&)>;
    using EmitResult = std::tuple<Builder, Checker>;
    virtual ~Emitter() = default;
    virtual EmitResult emit() const = 0;
};

class ModifyShapeMixin {
protected:
    using Pattern = SmallVector<std::tuple<int, int, bool>>;
    using Checker = Emitter::Checker;
    ModifyShapeMixin(
            const megdnn::NamedTensorShape& src, const megdnn::NamedTensorShape& dest)
            : m_src(src), m_dest(dest) {}
    Pattern mixin_analyze() const;
    Checker mixin_emit_checker(const Pattern& pattern) const;
    megdnn::NamedTensorShape m_src, m_dest;
};

class MakeShapeEmitter final : public Emitter, ModifyShapeMixin {
public:
    MakeShapeEmitter(
            const megdnn::NamedTensorShape& src, const megdnn::NamedTensorShape& dest)
            : ModifyShapeMixin(src, dest) {}
    EmitResult emit() const override;
};

class ReshapeEmitter final : public Emitter, ModifyShapeMixin {
public:
    ReshapeEmitter(
            const megdnn::NamedTensorShape& src, const megdnn::NamedTensorShape& dest)
            : ModifyShapeMixin(src, dest) {}
    EmitResult emit() const override;
};

class DimshuffleEmitter final : public Emitter {
public:
    DimshuffleEmitter(const std::vector<int>& pattern) : m_pattern{pattern} {}
    EmitResult emit() const override;

private:
    std::vector<int> m_pattern;
};

class ReformatEmitter final : public Emitter, ModifyShapeMixin {
public:
    ReformatEmitter(
            const megdnn::NamedTensorShape& src, const megdnn::NamedTensorShape& dest)
            : ModifyShapeMixin(src, dest) {}
    EmitResult emit() const override;

private:
    struct UnderlyingBuilders {
        Builder make_shape1, make_shape2, reshape1, reshape2, dimshuffle;
    };
    UnderlyingBuilders analyze() const;
};

class PaddingEmitter final : public Emitter {
public:
    PaddingEmitter(
            const megdnn::NamedTensorShape& padshp, size_t const_extent, size_t axis)
            : m_padshp{padshp}, m_const_extent{const_extent}, m_axis{axis} {}
    EmitResult emit() const override;

private:
    megdnn::NamedTensorShape m_padshp;
    size_t m_const_extent, m_axis;
};

class SubtensorEmitter final : public Emitter {
public:
    SubtensorEmitter(size_t const_extent, size_t axis)
            : m_const_extent{const_extent}, m_axis{axis} {}
    EmitResult emit() const override;

private:
    size_t m_const_extent, m_axis;
};
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
