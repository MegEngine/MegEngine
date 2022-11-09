#pragma once

#include <variant>

#include "megbrain/imperative.h"
#include "megbrain/test/helper.h"

namespace mgb {
namespace imperative {

class OprChecker {
public:
    using InputSpec = std::variant<HostTensorND, TensorShape>;
    OprChecker(std::shared_ptr<OpDef> opdef);
    void run(std::vector<InputSpec> inp_shapes, std::set<size_t> bypass = {});

    //! test the interface of apply_on_var_node
    VarNodeArray run_apply_on_var_node(std::vector<InputSpec> inp_shapes);

private:
    std::shared_ptr<OpDef> m_op;
};

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
