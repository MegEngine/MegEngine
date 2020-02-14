/**
 * \file src/plugin/impl/opr_io_dump.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/plugin/opr_io_dump.h"
#include "megbrain/graph/event.h"
#include "megbrain/utils/debug.h"

#include "megdnn/tensor_iter.h"

#include <cmath>

using namespace mgb;

namespace {
template <typename T>
double as_double(T& a) {
    return static_cast<double>(a);
}

template <>
double as_double(megdnn::dt_quint8& a) {
    return static_cast<double>(a.as_uint8());
}
template <>
double as_double(megdnn::dt_qint8& a) {
    return static_cast<double>(a.as_int8());
}
template <>
double as_double(megdnn::dt_qint32& a) {
    return static_cast<double>(a.as_int32());
}

template <typename ctype>
void do_print_host_val(FILE* fout, size_t max_nr_print,
                       const megdnn::TensorND& val, bool print_stat) {
    bool first = true;
    fprintf(fout, "[");
    size_t nr_print = 0;
    for (ctype i : megdnn::tensor_iter_valonly<ctype>(val)) {
        if (first) {
            first = false;
        } else
            fprintf(fout, ", ");
        if (++nr_print > max_nr_print) {
            fprintf(fout, "...");
            break;
        }
        fprintf(fout, "%.4g", as_double(i));
    }
    fprintf(fout, "]");

    if (!print_stat)
        return;

    ctype min(megdnn::DTypeTrait<ctype>::max()),
            max(megdnn::DTypeTrait<ctype>::min());
    double sum1 = 0, sum2 = 0;
    auto update = [&](ctype i) {
        min = std::min(i, min);
        max = std::max(i, max);
        sum1 += as_double(i);
        sum2 += as_double(i) * as_double(i);
    };
    size_t nr = val.layout.total_nr_elems();
    if (val.layout.is_contiguous()) {
        ctype* ptr = val.ptr<ctype>();
        for (size_t i = 0; i < nr; ++i) {
            update(ptr[i]);
        }
    } else {
        for (ctype i : megdnn::tensor_iter_valonly<ctype>(val)) {
            update(i);
        }
    }
    fprintf(fout, "min=%.3g max=%.3g mean=%.3g l2=%.3g", as_double(min),
            as_double(max), sum1 / nr, std::sqrt(sum2 / nr));
    if (nr > 1) {
        fprintf(fout, " sd=%.3g",
                std::sqrt((sum2 * nr - sum1 * sum1) / (nr * (nr - 1))));
    } else {
        fprintf(fout, " sd=N/A");
    }
};

void print_host_val(FILE* fout, size_t max_nr_print,
                    const megdnn::TensorND& val, bool print_stat = false) {
    switch (val.layout.dtype.enumv()) {
#define cb(_dt)                                                              \
    case DTypeTrait<_dt>::enumv:                                             \
        return do_print_host_val<DTypeTrait<_dt>::ctype>(fout, max_nr_print, \
                                                         val, print_stat);
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
        default:
            mgb_throw(MegBrainError,
                      "can not handle dtype %s in "
                      "print_host_val",
                      val.layout.dtype.name());
    }
};

}  // anonymous namespace

/* =================== OprIODumpBase =================== */

struct OprIODumpBase::VarRecorderLazySync {
    class CompNodeRec {
        CompNode m_cn;

    public:
        ~CompNodeRec() { sync(); }

        void update(CompNode cn) {
            if (!m_cn.valid()) {
                m_cn = cn;
            } else {
                mgb_assert(m_cn == cn,
                           "there should be only one comp node in recorder "
                           "mode, got %s vs %s",
                           m_cn.to_string().c_str(), cn.to_string().c_str());
            }
        }

        void sync() const {
            mgb_assert(m_cn.valid());
            m_cn.sync();
        }
    };

    size_t id;
    TensorLayout prev_layout;
    DeviceTensorND dv_contig;  //!< used for creating a continuous temp storage
    HostTensorND val;
    std::string name;

    explicit VarRecorderLazySync(VarNode* var)
            : id{var->id()}, name{var->name()} {
        sync_value_from(var);
    }

    void sync_value_from(VarNode* var) {
        auto&& dv = var->dev_tensor();
        if (!prev_layout.ndim) {
            prev_layout = dv.layout();
        } else {
            mgb_assert(prev_layout.eq_layout(dv.layout()),
                       "tensor layout is not allowed to change in recording "
                       "mode with OprIODump plugin: var=%s layout: %s vs %s",
                       var->cname(), prev_layout.to_string().c_str(),
                       dv.layout().to_string().c_str());
        }
        if (!dv.layout().is_contiguous()) {
            dv_contig.copy_from(dv);
            val.copy_from(dv_contig);
        } else {
            val.copy_from(dv);
        }
    }
};

OprIODumpBase::OprIODumpBase(cg::ComputingGraph* graph) : PluginBase(graph) {
    using namespace cg::event;
    auto on_kern_finish = [this](const OprExecKernelEnd& event) {
        for (VarNode* var : event.opr->output()) {
            if (!var->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                auto run = [this, var]() {
                    dump_var(var, m_owner_graph->options()
                                          .comp_node_seq_record_level);
                };
                event.env->dispatch_on_comp_node(var->comp_node(), run);
            }
        }
    };
    add_event_handler(
            graph->event().register_receiver<OprExecKernelEnd>(on_kern_finish));
}

/* =================== TextOprIODump =================== */

class TextOprIODump::LazyValueRecorder {
    struct Opr : public NonCopyableObj {
        size_t id;
        Typeinfo* type = nullptr;
        std::string name;

        SmallVector<VarRecorderLazySync> outputs;

        Opr* ensure_init(cg::OperatorNodeBase* opr) {
            if (!type) {
                id = opr->id();
                type = opr->dyn_typeinfo();
                name = opr->name();
                return this;
            }
            return nullptr;
        }
    };
    ThinHashMap<cg::OperatorNodeBase*, Opr> m_opr_map;
    SmallVector<Opr*> m_oprs;
    VarRecorderLazySync::CompNodeRec m_cn;

public:
    //! return whether var is new
    bool record_var(VarNode* var) {
        auto&& item = m_opr_map[var->owner_opr()];
        if (auto opr = item.ensure_init(var->owner_opr())) {
            m_oprs.push_back(opr);
        }
        for (auto&& i : item.outputs) {
            if (i.id == var->id()) {
                // multiple exeutions because prev record is invalidated due to
                // plugin change or shape change
                i.sync_value_from(var);
                return false;
            }
        }
        item.outputs.emplace_back(var);
        m_cn.update(var->comp_node());
        return true;
    }

    void flush(TextOprIODump* iodump) const;
};

TextOprIODump::TextOprIODump(cg::ComputingGraph* graph,
                             const std::shared_ptr<FILE>& fout)
        : OprIODumpBase(graph), m_fout(fout) {}

void TextOprIODump::LazyValueRecorder::flush(TextOprIODump* iodump) const {
    auto fout = iodump->m_fout.get();
    m_cn.sync();
    fprintf(fout, "==== recorded values\n");
    for (size_t idx = 0; idx < m_oprs.size(); ++idx) {
        auto opr = m_oprs[idx];
        fprintf(fout, "#%zu: opr%zu %s{%s}\n", idx, opr->id, opr->name.c_str(),
                opr->type->name);
        for (auto&& ovar : opr->outputs) {
            fprintf(fout, "  var%zu: name=%s ", ovar.id, ovar.name.c_str());
            print_host_val(fout, iodump->m_max_size, ovar.val.as_megdnn(),
                           true);
            fprintf(fout, "\n");
        }
    }
}

void TextOprIODump::dump_var(VarNode* var, bool lazy_sync) {
    MGB_LOCK_GUARD(m_mtx);
    auto fout = m_fout.get();
    auto opr = var->owner_opr();
    bool valid = var->dev_tensor_valid();

    bool print_var_produce = true;
    if (lazy_sync) {
        if (!m_lazy_value) {
            m_lazy_value = std::make_unique<LazyValueRecorder>();
            fprintf(fout, "==== begin lazy value recording\n");
        }
        if (valid) {
            print_var_produce = m_lazy_value->record_var(var);
        } else {
            print_var_produce = false;
        }
    }

    if (print_var_produce) {
        fprintf(fout,
                "var%zd produced: name=%s layout=%s owner_opr=%s{%s} opr%zd\n",
                var->id(), var->cname(),
                valid ? var->layout().to_string().c_str() : "<invalid>",
                opr->cname(), opr->dyn_typeinfo()->name, opr->id());
    }

    if (!valid || !print_var_produce)
        return;

    auto print_var_val = [&](VarNode* var, bool print_stat = false) {
        if (lazy_sync) {
            fprintf(fout, "<see lazy value below>");
            return;
        }
        if (!var->dev_tensor_valid()) {
            fprintf(fout, "<invalid>");
            return;
        }

        HostTensorND hv;
        hv.copy_from(var->dev_tensor()).sync();
        print_host_val(fout, m_max_size, hv.as_megdnn(), print_stat);
        if (m_print_addr) {
            fprintf(fout, "  <%p>", var->dev_tensor().raw_ptr());
        }
    };

    fprintf(fout, " deps:\n");

    ThinHashMap<VarNode*, int> var2iid;
    for (size_t i = 0; i < opr->input().size(); ++i)
        var2iid[opr->input(i)] = i;

    using DT = cg::OperatorNodeBase::NodeProp::DepType;
    // [(input_id, var_id, var, dep_type)]
    SmallVector<std::tuple<int, size_t, VarNode*, DT>> dep_vars;
    for (auto&& dep_entry : opr->node_prop().dep_map()) {
        int inp_id = -1;
        VarNode* var = dep_entry.first;
        auto iter = var2iid.find(var);
        if (iter != var2iid.end()) {
            inp_id = iter->second;
        }
        dep_vars.emplace_back(inp_id, var->id(), var, dep_entry.second);
    }
    small_sort(dep_vars.begin(), dep_vars.end());

    for (auto&& dep_entry : dep_vars) {
        int input_id;
        VarNode* var;
        DT dep_type;
        std::tie(input_id, std::ignore, var, dep_type) = dep_entry;
        fprintf(fout, "  ");
        if (input_id != -1) {
            fprintf(fout, "[i%d]", input_id);
        }
        fprintf(fout, "var%zd: "_fmt, var->id());
        auto&& mgr = opr->owner_graph()->static_infer_manager();
        if (dep_type & DT::DEV_VALUE) {
            print_var_val(var);
        } else {
            if (dep_type & DT::SHAPE) {
                fprintf(fout, " <shape dep[%c]> %s",
                        cg::is_static_var_shape(var) ? 's' : 'd',
                        mgr.infer_shape(var).to_string().c_str());
            } else if (dep_type & DT::HOST_VALUE) {
                fprintf(fout, " <host value[%c]> ",
                        cg::is_static_var_value(var) ? 's' : 'd');
                print_host_val(fout, m_max_size,
                               mgr.infer_value(var).as_megdnn());
            } else {
                mgb_assert(dep_type == DT::DEV_COMP_ORDER);
                fprintf(fout, " <dev comp order>");
            }
        }
        fprintf(fout, " %c\n", cg::is_static_var_storage(var) ? 's' : 'd');
    }
    fprintf(fout, " val: ");
    print_var_val(var, true);
    fprintf(fout, " %c\n", cg::is_static_var_storage(var) ? 's' : 'd');
    fflush(fout);
}

void TextOprIODump::flush_lazy() {
    if (m_lazy_value) {
        m_lazy_value->flush(this);
    }
}

TextOprIODump::~TextOprIODump() {
    flush_lazy();
}

/* =================== BinaryOprIODump =================== */

class BinaryOprIODump::LazyValueRecorder {
    VarRecorderLazySync::CompNodeRec m_cn;
    SmallVector<VarRecorderLazySync> m_vals;
    ThinHashMap<VarNode*, size_t> m_var2idx_in_vals;

public:
    void record_var(std::string title, VarNode* var) {
        auto ins = m_var2idx_in_vals.insert({var, m_vals.size()});
        if (!ins.second) {
            m_vals.at(ins.first->second).sync_value_from(var);
            return;
        }
        m_vals.emplace_back(var);
        m_vals.back().name = std::move(title);
        m_cn.update(var->comp_node());
    }

    void flush(BinaryOprIODump* iodump) const {
        m_cn.sync();
        auto out_dir = iodump->m_output_dir.c_str();
        for (auto&& i : m_vals) {
            auto value = debug::dump_tensor(i.val, i.name);
            debug::write_to_file(ssprintf("%s%06zx", out_dir, i.id).c_str(),
                                 value);
        }
    }
};

BinaryOprIODump::BinaryOprIODump(cg::ComputingGraph* graph,
                                 std::string output_dir)
        : OprIODumpBase(graph), m_output_dir{std::move(output_dir)} {
    if (m_output_dir.back() != '/') {
        m_output_dir += '/';
    }
}

void BinaryOprIODump::dump_var(VarNode* var, bool lazy_sync) {
    auto do_dump = [ this, fid = var->id(), lazy_sync ](
            VarNode * var, const char* prefix, const char* suffix) {
        auto title =
                ssprintf("%svar=%s owner_opr_inputs=%s", prefix,
                         cg::dump_var_info({var}).c_str(),
                         cg::dump_var_info(var->owner_opr()->input()).c_str());
        if (lazy_sync) {
            if (!m_lazy_value) {
                m_lazy_value = std::make_unique<LazyValueRecorder>();
            }
            m_lazy_value->record_var(std::move(title), var);
            return;
        }
        auto value = debug::dump_tensor(var->dev_tensor(), title);
        debug::write_to_file(
                ssprintf("%s%06zx%s", m_output_dir.c_str(), fid, suffix)
                        .c_str(),
                value);
    };
    if (var->dev_tensor_valid()) {
        do_dump(var, "", "");
        if (MGB_GETENV("MGB_DUMP_INPUT")) {
            mgb_assert(
                    !lazy_sync,
                    "lazy sinc with MGB_DUMP_INPUT is currently not supported");
            for (size_t i = 0; i < var->owner_opr()->input().size(); ++i) {
                auto ivar = var->owner_opr()->input()[i];
                if (ivar->dev_tensor_valid()) {
                    do_dump(ivar, ssprintf("inp%zu: ", i).c_str(),
                            ssprintf("-inp%zu", i).c_str());
                }
            }
        }
    }
}

void BinaryOprIODump::flush_lazy() {
    if (m_lazy_value) {
        m_lazy_value->flush(this);
    }
}

BinaryOprIODump::~BinaryOprIODump() {
    flush_lazy();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
