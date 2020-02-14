%inline {

    SymbolVarArray _get_owner_opr_inputs(SymbolVar var) {
        mgb_assert(var.node());
        return mgb::cg::to_symbol_var_array(var.node()->owner_opr()->input());
    }

    std::string _get_owner_opr_type(SymbolVar var) {
        mgb_assert(var.node());
        return var.node()->owner_opr()->dyn_typeinfo()->name;
    }

    SymbolVarArray _replace_vars(const SymbolVarArray& repl_src,
                                 const SymbolVarArray& repl_dst,
                                 const SymbolVarArray& vars) {
        mgb::ThinHashMap<SymbolVar, SymbolVar> varmap;
        for (size_t i = 0; i < repl_src.size(); ++i) {
            varmap[repl_src[i]] = repl_dst[i];
        }
        return mgb::cg::replace_vars(vars, varmap);
    }

    typedef std::vector<Operator> OperatorArray;
    SymbolVarArray _replace_oprs(const OperatorArray& repl_src,
                                 const OperatorArray& repl_dst,
                                 const SymbolVarArray& vars) {
        mgb::ThinHashMap<mgb::cg::OperatorNodeBase*, mgb::cg::OperatorNodeBase*>
                oprmap;
        for (size_t i = 0; i < repl_src.size(); ++i) {
            oprmap[repl_src[i].node()] = repl_dst[i].node();
        }
        return mgb::cg::replace_oprs(vars, oprmap);
    }
}
// vim: ft=swig foldmethod=marker foldmarker=f{{{,f}}}
