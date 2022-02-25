#pragma once

#include "megbrain/plugin/base.h"

#include "megbrain/imperative/profiler.h"

namespace mgb::imperative::interpreter::intl {

class ProfilerPlugin : public PluginBase {
public:
    struct OprInfo {
        uint64_t id;
        CompNode comp_node;
        std::shared_ptr<std::unordered_map<std::string, std::string>> params;
    };

    struct VarInfo {
        uint64_t id;
        bool is_const;
        size_t ref_cnt;
        std::atomic_size_t rt_ref_cnt;
    };

private:
    std::unordered_map<cg::OperatorNodeBase*, OprInfo> m_opr_dict;
    std::unordered_map<cg::VarNode*, std::unique_ptr<VarInfo>> m_var_dict;

public:
    explicit ProfilerPlugin(cg::ComputingGraph* graph);
    void init_seq(cg::AsyncExecutable* comp_seq);
    OprInfo& register_opr(cg::OperatorNodeBase* opr);
    VarInfo& register_var(cg::VarNode* var);
    OprInfo& get_opr_info(cg::OperatorNodeBase* opr);
    VarInfo& get_var_info(cg::VarNode* var);
};

}  // namespace mgb::imperative::interpreter::intl
