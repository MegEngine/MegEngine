/**
 * \file src/opr/impl/loop/output_recorder.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./impl.h"
#include "./output_recorder.h"

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "megdnn/oprs.h"

#include <cmath>

using namespace mgb;

using LoopDesc = mgb::opr::intl::LoopImpl::Desc;
using OutputRecorderBase = LoopDesc::OutputRecorderBase;
using OutputMode = LoopDesc::OutputMode;

namespace {

/*!
 * \brief base class for output recorders whose output shape is the same as its
 *      input shape inferred
 */
class OutputRecorderOutputShapeSameAsInShape: public OutputRecorderBase {
    bool m_dest_var_allocated = false;
    int m_dest_var_is_static = -1;

    VarNode *m_src_var, *m_dest_var;

    bool has_shape_infer_desc() const override final {
        return true;
    }

    void register_infer_desc(
            SubgraphStaticInferHelper &helper) const override final {
        using namespace cg::static_infer;
        if (!helper.register_shape_infer_par(
                    m_dest_var, ShapeInferDesc::make_identity(m_src_var)))
            m_dest_var->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC);
    }

    protected:
        void bind_var(VarNode *var_sub, VarNode *var_out) override {
            m_src_var = var_sub;
            m_dest_var = var_out;
            if (!cg::is_static_var_shape(var_sub)) {
                var_out->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC);
            }
        }

        void on_exec_begin() override {
            m_dest_var_allocated = false;
            if (m_dest_var_is_static == -1)
                m_dest_var_is_static = cg::is_static_var_storage(m_dest_var);
        }


        /*!
         * \brief get tensor for output var to be written to
         */
        const DeviceTensorND& get_output_var_tensor(const TensorShape &tshape) {
            if (m_dest_var_is_static)
                return m_dest_var->dev_tensor();

            if (!m_dest_var_allocated) {
                if (m_dest_var->contain_flag(VarNode::Flag::NO_SYS_MEM_ALLOC)) {
                    m_dest_var->shape_alloc(tshape);
                } else {
                    // static shape but dynamic storage
                    mgb_assert(m_dest_var->shape().eq_shape(tshape));
                }
                m_dest_var_allocated = true;
            }
            return m_dest_var->dev_tensor();
        }

        VarNode* src_var() const {
            return m_src_var;
        }

        VarNode* dest_var() const {
            return m_dest_var;
        }
};


/*!
 * \brief record last output
 *
 * The shape of the output during each loop step must remain unchanged;
 * final output shape is the same as intermediate shapes
 */
class OutputRecorderLast final: public OutputRecorderOutputShapeSameAsInShape {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    void bind_var(VarNode *var_sub, VarNode *var_out) override {
        OutputRecorderOutputShapeSameAsInShape::bind_var(var_sub, var_out);

        // directly forward output var in sub graph to owner graph on exec end
        var_sub->add_flag(VarNode::Flag::NO_SYS_STATIC_MEM_ALLOC).
            add_flag(VarNode::Flag::NO_MEM_RECLAIM);
        var_out->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC);
    }

    void on_exec_end() override {
        auto succ = dest_var()->reset_dev_tensor_from_other_var(src_var());
        mgb_assert(succ);
    }

    virtual std::string name() const override {
        return "last";
    }

    SymbolVar get_outgrad_in_iter(
            SymbolVar loop_counter_down, SymbolVar loop_counter_up,
            SymbolVar outgrad) override {
        MGB_MARK_USED_VAR(loop_counter_down);
        return opr::switch_gt0(1 - loop_counter_up, outgrad);
    }

    OutputMode output_mode() const override {
        return OutputMode::LAST;
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(OutputRecorderLast);

/*!
 * \brief record all outputs
 */
class OutputRecorderAll final: public OutputRecorderBase {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
    static constexpr size_t MAX_OVERHEAD = 64 * 1024 * 1024, INIT_ALLOC = 5;
    mutable bool m_static_shape_succ = false;
    size_t m_used_size, m_max_size, m_max_overhead_nr;
    VarNode *m_src_var, *m_dest_var;
    TensorShape m_element_shape;

    void bind_var(VarNode *var_sub, VarNode *var_out) override {
        m_src_var = var_sub;
        m_dest_var = var_out;
    }

    bool has_shape_infer_desc() const override {
        return true;
    }

    static TensorShape extend_shape(TensorShape shp, size_t sz) {
        mgb_assert(sz);
        shp.ndim ++;
        mgb_assert(shp.ndim < TensorShape::MAX_NDIM);
        for (size_t i = shp.ndim - 1; i; i --)
            shp.shape[i] = shp.shape[i - 1];
        shp.shape[0] = sz;
        return shp;
    }

    void register_infer_desc(
            SubgraphStaticInferHelper &helper) const override final {
        using namespace cg::static_infer;

        auto infer_shp = [](TensorShape &dest, const InpVal &inp) {
            int loop_time = inp.val.at(1).value().ptr<int>()[0] + 1;
            mgb_assert(loop_time > 0);
            dest = extend_shape(inp.val[0].shape(), loop_time);
            return true;
        };

        auto &&loop = m_dest_var->owner_opr()->cast_final_safe<opr::Loop>();

        auto cnt_var = loop.output_counter_var();
        if (cg::is_static_var_value(cnt_var)) {
            ShapeInferDesc desc{
                SourceType::DEP,
                {{m_src_var, DepType::SHAPE}, {cnt_var, DepType::VALUE}},
                infer_shp};
            if (helper.register_shape_infer_par(m_dest_var, desc)) {
                m_static_shape_succ = true;
                return;
            }
        }

        m_static_shape_succ = false;
        m_dest_var->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC);
    }

    void on_exec_begin() override {
        m_used_size = 0;
    }

    void on_val_produced(const DeviceTensorND& val) override {
        if (!m_static_shape_succ)
            grow_output_storage(val.shape());

        auto &&dest = m_dest_var->dev_tensor();
        auto subs = Slice(m_used_size, m_used_size + 1).apply(dest.layout(), 0);
        subs = SubTensorSpec::make_from_offset_elem(
                subs.layout().remove_axis(0), subs.offset_elem());
        dest.sub(subs).copy_from_fixlayout(val);
        m_used_size ++;
    }

    void on_exec_end() override {
        if (m_static_shape_succ) {
            mgb_assert(m_used_size == m_dest_var->shape().shape[0]);
        } else {
            mgb_assert(m_used_size);
            auto shp = m_dest_var->shape();
            shp.shape[0] = m_used_size;
            m_dest_var->shape_alloc(shp);
        }
    }

    virtual std::string name() const override {
        return "all";
    }

    SymbolVar get_outgrad_in_iter(
            SymbolVar loop_counter_down, SymbolVar loop_counter_up,
            SymbolVar outgrad) override {
        MGB_MARK_USED_VAR(loop_counter_up);
        return opr::IndexAt::make(outgrad, {{0, loop_counter_down}});
    }

    void grow_output_storage(const TensorShape &elem_shape) {
        if (!m_used_size) {
            // first exec, allocate and init shape
            m_max_size = INIT_ALLOC;
            m_element_shape = elem_shape;
            m_dest_var->shape_alloc(extend_shape(m_element_shape, m_max_size));
            m_max_overhead_nr = std::max<size_t>(
                    m_max_size, MAX_OVERHEAD / m_element_shape.total_nr_elems());
        }

        mgb_assert(elem_shape.eq_shape(m_element_shape),
                "shape changed during recording output: expect=%s get=%s",
                m_element_shape.to_string().c_str(),
                elem_shape.to_string().c_str());

        if (m_used_size == m_max_size) {
            ptrdiff_t orig_max_size = m_max_size;
            m_max_size = std::min(
                    m_max_size * 2, m_max_size + m_max_overhead_nr);
            auto old_v = m_dest_var->dev_tensor();
            auto shp = old_v.shape();
            shp.shape[0] = m_max_size;
            m_dest_var->shape_alloc(shp);
            if (old_v.raw_ptr() != m_dest_var->dev_tensor().raw_ptr()) {
                m_dest_var->dev_tensor()[{{0, orig_max_size}}].
                    copy_from_fixlayout(old_v);
            }
        }
    }

    OutputMode output_mode() const override {
        return OutputMode::ALL;
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(OutputRecorderAll);

/*!
 * \brief record the reduced value of all history
 */
class OutputRecorderReduceHelper: public OutputRecorderOutputShapeSameAsInShape{
    bool m_first_exec = false;

    void on_exec_begin() override {
        OutputRecorderOutputShapeSameAsInShape::on_exec_begin();
        m_first_exec = true;
    }

    void on_val_produced(const DeviceTensorND& val) override {
        auto &&dest = get_output_var_tensor(val.shape());

        if (m_first_exec) {
            m_first_exec = false;
            dest.copy_from_fixlayout(val);
        } else
            do_reduce(dest, val);
    }

    protected:
        virtual void do_reduce(const DeviceTensorND& dest,
                               const DeviceTensorND& val) = 0;
};

class OutputRecorderSum final: public OutputRecorderReduceHelper {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
    opr::intl::UniqPtrWithCN<megdnn::Elemwise> m_adder_opr;

    SymbolVar get_outgrad_in_iter(
            SymbolVar loop_counter_down, SymbolVar loop_counter_up,
            SymbolVar outgrad) override {
        MGB_MARK_USED_VAR(loop_counter_down);
        MGB_MARK_USED_VAR(loop_counter_up);
        return outgrad;
    }

    void do_reduce(const DeviceTensorND& dest,
                   const DeviceTensorND& val) override {
        if (!m_adder_opr) {
            m_adder_opr = opr::intl::create_megdnn_opr<megdnn::Elemwise>(
                    dest.comp_node());
            m_adder_opr->param() = {megdnn::Elemwise::Mode::ADD};
        }
        mgb_assert(m_adder_opr.comp_node() == dest.comp_node());
        auto dm = dest.as_megdnn();
        m_adder_opr->exec({dm, val.as_megdnn()}, dm);
    }

    std::string name() const override {
        return "sum";
    }

    OutputMode output_mode() const override {
        return OutputMode::SUM;
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(OutputRecorderSum);

/*!
 * \brief dummy recorder for unused output
 */
class OutputRecorderDummy final: public OutputRecorderBase {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    void bind_var(VarNode *, VarNode *) override {
        mgb_assert(0);
    }

    bool has_shape_infer_desc() const override {
        mgb_assert(0);
    }

    void on_val_produced(const DeviceTensorND&) override {
        mgb_assert(0);
    }

    SymbolVar get_outgrad_in_iter(SymbolVar, SymbolVar , SymbolVar) override {
        mgb_assert(0);
    }

    std::string name() const override {
        return "dummy";
    }

    OutputMode output_mode() const override {
        mgb_assert(0);
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(OutputRecorderDummy);
OutputRecorderDummy global_dummy_recorder;

} // anonymous namespace

OutputRecorderBase* const
opr::intl::LoopImpl::OutputRecordSpecItem::m_dummy_recorder =
&global_dummy_recorder;

size_t LoopDesc::add_output(SymbolVar val, OutputMode mode) {
    std::unique_ptr<OutputRecorderBase> ptr;
    switch (mode) {
        case OutputMode::LAST:
            ptr.reset(new OutputRecorderLast());
            break;
        case OutputMode::ALL:
            ptr.reset(new OutputRecorderAll());
            break;
        case OutputMode::SUM:
            ptr.reset(new OutputRecorderSum());
            break;
        default:
            mgb_assert(0, "unknown output mode");
    }
    return do_add_output(val, std::move(ptr));
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
