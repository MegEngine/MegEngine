/**
 * \file src/plugin/include/megbrain/plugin/opr_io_dump.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <cstdio>
#include "megbrain/graph.h"
#include "megbrain/plugin/base.h"

namespace mgb {

class OprIODumpBase : public PluginBase {
protected:
    //! helper to record var value in lazy sync mode
    struct VarRecorderLazySync;

    /*!
     * \brief subclasses should override this method to dump the value of a
     *      single var
     * \param lazy_sync whether recorder is enabled, so we should synchronize
     *      and write to file only in the destructor.
     */
    virtual void dump_var(VarNode* var, bool lazy_sync) = 0;

    OprIODumpBase(cg::ComputingGraph* graph);

public:
    virtual ~OprIODumpBase() = default;

    /*!
     * \brief write lazy values due to comp_node_seq_record_level to file
     *
     * Note: this is only effective when comp_node_seq_record_level is set. If
     * compiled func is executed again without calling flush_lazy(), then
     * previously recorded values would be overwritten and nothing would be
     * recorded to file.
     */
    virtual void flush_lazy() = 0;
};

/*!
 * \brief dump opr input/output vars as text
 *
 *
 * In normal cases, the result would be written to the file at each execution.
 * When comp_node_seq_record_level is set, the result would be written to file
 * in destructor or when flush_lazy() is called.
 */
class TextOprIODump final : public OprIODumpBase {
    class LazyValueRecorder;

    bool m_print_addr = true;
    std::shared_ptr<FILE> m_fout;
    size_t m_max_size = 5;
    std::mutex m_mtx;
    std::unique_ptr<LazyValueRecorder> m_lazy_value;

    void dump_var(VarNode* var, bool lazy_sync) override;

public:
    TextOprIODump(cg::ComputingGraph* graph,
                  const std::shared_ptr<FILE>& fout =
                          std::shared_ptr<FILE>(stderr, [](FILE*) {}));

    TextOprIODump(cg::ComputingGraph* graph, const char* fpath)
            : TextOprIODump(graph,
                            std::shared_ptr<FILE>(fopen(fpath, "w"), fclose)) {}

    ~TextOprIODump();

    void flush_lazy() override;

    //! set whether to print var address
    TextOprIODump& print_addr(bool flag) {
        m_print_addr = flag;
        return *this;
    }

    //! set max number of entries to be printed for a single tensor
    TextOprIODump& max_size(size_t size) {
        m_max_size = size;
        return *this;
    }
};

/*!
 * \brief similar to TextOprIODump, but write to binary files in a directory
 *
 * The output directory must exist. An environment var MGB_DUMP_INPUT can be set
 * to also dump the input values accessed by each opr.
 *
 * The files can be parsed by the ``megbrain.plugin.load_tensor_binary`` python
 * function.
 */
class BinaryOprIODump final : public OprIODumpBase {
    class LazyValueRecorder;

    std::string m_output_dir;
    std::unique_ptr<LazyValueRecorder> m_lazy_value;

    void dump_var(VarNode* var, bool lazy_sync) override;

public:
    BinaryOprIODump(cg::ComputingGraph* graph, std::string output_dir);
    ~BinaryOprIODump();
    void flush_lazy() override;
};

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

