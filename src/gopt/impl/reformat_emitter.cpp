/**
 * \file src/gopt/impl/reformat_emitter.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <numeric>
#include "megbrain/gopt/reformat_emitter.h"
#include "megbrain/opr/tensor_manip.h"

using namespace mgb;
using namespace gopt;
using Dimension = megdnn::Dimension;
using NamedTensorShape = megdnn::NamedTensorShape;

ReshapeEmitter::Operator ReshapeEmitter::emit() const {
    auto pattern = analyze();
    auto op = [pattern](VarNode* var) {
        auto sym_var = SymbolVar(var);
        auto shp = opr::GetVarShape::make(sym_var);
        auto cv = [&sym_var](int c) { return sym_var.make_scalar(c); };
        auto sub = [&shp, &cv](int ax) {
            return opr::IndexAt::make(shp, {{0, cv(ax)}});
        };
        SymbolVarArray axs;
        for (auto i : pattern) {
            if (std::get<0>(i) >= 0) {
                if (std::get<2>(i))
                    axs.emplace_back(sub(std::get<0>(i)) * std::get<1>(i));
                else
                    axs.emplace_back(sub(std::get<0>(i)) / std::get<1>(i));
            } else {
                axs.emplace_back(cv(std::get<1>(i)));
            }
        }
        auto tshp = opr::Concat::make(axs, 0);
        auto ovar = opr::Reshape::make(sym_var, tshp);
        return ovar.node();
    };
    return op;
}

SmallVector<std::tuple<int, int, bool>> ReshapeEmitter::analyze() const {
    static constexpr uint32_t UNDETERMINED_EXTENT =
            Dimension::UNDETERMINED_EXTENT;
    ThinHashMap<Dimension::Name, int> name2dominant;
    for (size_t i = 0; i < m_src.ndim; ++i) {
        auto name = m_src[i].name();
        if (m_src[i].extent() == UNDETERMINED_EXTENT) {
            auto insert = name2dominant.insert(std::make_pair(name, i));
            mgb_assert(insert.second);
        }
    }

    SmallVector<std::tuple<int, int, bool>> pattern(m_dest.ndim);
    for (size_t i = 0; i < m_dest.ndim; ++i) {
        auto name = m_dest[i].name();
        if (m_dest[i].extent() == UNDETERMINED_EXTENT) {
            int src_dim = name2dominant.at(name);
            bool mul = m_src[src_dim] < m_dest[i];
            int factor = mul ? (m_dest[i] / m_src[src_dim]).extent()
                             : (m_src[src_dim] / m_dest[i]).extent();
            pattern[i] = std::make_tuple(src_dim, factor, mul);
        } else {
            pattern[i] = std::make_tuple(-1, m_dest[i].extent(), false);
        }
    }
    return pattern;
}

DimshuffleEmitter::Operator DimshuffleEmitter::emit() const {
    auto pattern = m_pattern;
    auto op = [pattern](VarNode* var) {
        auto sym_var = SymbolVar(var);
        return opr::Dimshuffle::make(sym_var, pattern).node();
    };
    return op;
}

ReformatEmitter::Operator ReformatEmitter::emit() const {
    auto ops = analyze();
    auto op = [ops](VarNode* var) {
        VarNode* ovar = var;
        for (const auto& o : ops) {
            ovar = o(ovar);
        }
        return ovar;
    };
    return op;
}

SmallVector<ReformatEmitter::Operator> ReformatEmitter::analyze() const {
    struct Dim {
        Dimension dim;
        int index;
        Dim(Dimension dim_, int index_) : dim{dim_}, index{index_} {}
    };
    SmallVector<Dim> src_dims;
    SmallVector<Dim> dest_dims;
    for (size_t i = 0; i < m_src.ndim; ++i)
        src_dims.emplace_back(Dim(m_src[i], i));
    for (size_t i = 0; i < m_dest.ndim; ++i)
        dest_dims.emplace_back(Dim(m_dest[i], i));
    auto compare = [](const Dim& lhs, const Dim& rhs) {
        return lhs.dim < rhs.dim;
    };
    std::sort(src_dims.begin(), src_dims.end(), compare);
    std::sort(dest_dims.begin(), dest_dims.end(), compare);

    auto src_iter = src_dims.begin();
    auto dest_iter = dest_dims.begin();
    for (; src_iter != src_dims.end() && dest_iter != dest_dims.end();) {
        if (src_iter->dim == dest_iter->dim) {
            src_iter++;
            dest_iter++;
        } else if (src_iter->dim < dest_iter->dim) {
            auto split = dest_iter->dim / src_iter->dim;
            int dim_idx = dest_iter->index;
            dest_iter =
                    dest_dims.insert(dest_iter, Dim(src_iter->dim, dim_idx));
            dest_iter++;
            dest_iter->dim = split;
            dest_iter->index = dim_idx;
            src_iter++;
        } else {
            auto split = src_iter->dim / dest_iter->dim;
            int dim_idx = src_iter->index;
            src_iter = src_dims.insert(src_iter, Dim(dest_iter->dim, dim_idx));
            src_iter++;
            src_iter->dim = split;
            src_iter->index = dim_idx;
            dest_iter++;
        }
    }
    mgb_assert(src_dims.size() == dest_dims.size());
    std::vector<int> src_perm(src_dims.size());
    std::vector<int> permute(dest_dims.size());
    std::iota(src_perm.begin(), src_perm.end(), 0);
    std::iota(permute.begin(), permute.end(), 0);
    std::sort(src_perm.begin(), src_perm.end(), [&](const int a, const int b) {
        if (src_dims[a].index != src_dims[b].index)
            return src_dims[a].index < src_dims[b].index;
        return src_dims[a].dim < src_dims[b].dim;
    });
    std::sort(permute.begin(), permute.end(), [&](const int a, const int b) {
        int perm_a = src_perm[a];
        int perm_b = src_perm[b];
        if (dest_dims[perm_a].index != dest_dims[perm_b].index)
            return dest_dims[perm_a].index < dest_dims[perm_b].index;
        return dest_dims[perm_a].dim < dest_dims[perm_b].dim;
    });
    NamedTensorShape i1, i2;
    i1.ndim = src_dims.size(), i2.ndim = dest_dims.size();
    for (size_t i = 0; i < src_dims.size(); ++i) {
        i1[i] = src_dims[src_perm[i]].dim;
        i2[i] = src_dims[src_perm[permute[i]]].dim;
    }
    SmallVector<Operator> ops;
    if (!m_src.eq_shape(i1))
        ops.emplace_back(ReshapeEmitter(m_src, i1).emit());
    ops.emplace_back(DimshuffleEmitter(permute).emit());
    if (!m_dest.eq_shape(i2))
        ops.emplace_back(ReshapeEmitter(i2, m_dest).emit());
    return ops;
}

