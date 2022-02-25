#include "outdumper.h"
#include "megbrain/utils/debug.h"

using namespace lar;

void OutputDumper::set(mgb::SymbolVarArray& symb_var) {
    for (auto&& i : symb_var) {
        auto&& var = i.node();
        DumpInfo info;
        info.var_info = mgb::cg::dump_var_info({var});
        info.owner_inputs_info = mgb::cg::dump_var_info(var->owner_opr()->input());
        info.id = var->id();
        m_infos.push_back(info);
    }
}

mgb::ComputingGraph::Callback OutputDumper::bind() {
    auto& info = m_infos.at(m_bind_id++);
    mgb::ComputingGraph::Callback cb = [&info](const mgb::DeviceTensorND& dv) {
        info.hv.copy_from(dv);
    };
    return cb;
}

void OutputDumper::write_to_file() {
    if (!dump_file.empty()) {
        for (auto&& info : m_infos) {
            auto value = mgb::debug::dump_tensor(
                    info.hv,
                    mgb::ssprintf(
                            "var=%s owner_opr_inputs= %s", info.var_info.c_str(),
                            info.owner_inputs_info.c_str()));
            mgb::debug::write_to_file(
                    mgb::ssprintf(
                            "%s/run%zu-var%zd", dump_file.c_str(), m_run_id, info.id)
                            .c_str(),
                    value);
        }
    }
    m_run_id++;
}
