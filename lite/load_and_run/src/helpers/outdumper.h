#pragma once
#include "megbrain/serialization/serializer.h"

namespace lar {

/*!
 * \brief dumper for only output used for --bin-out-dump
 */
class OutputDumper {
public:
    struct DumpInfo {
        mgb::HostTensorND hv = {};
        std::string var_info;
        std::string owner_inputs_info;
        size_t id;
    };
    //! init the dump_file path
    OutputDumper(const char* file) { dump_file = file; }

    //! set the dump informations
    void set(mgb::SymbolVarArray& symb_var);

    //! callback function for specify output when compile computing graph
    mgb::ComputingGraph::Callback bind();

    //! write dumped output into dump_file
    void write_to_file();

private:
    mgb::SmallVector<DumpInfo> m_infos;
    size_t m_run_id = 0;
    size_t m_bind_id = 0;
    std::string dump_file;
};
}  // namespace lar