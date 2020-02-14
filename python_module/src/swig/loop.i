/*
 * $File: loop.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */

%{
#include "megbrain/opr/loop.h"
#include <cctype>
using LoopDesc = mgb::opr::Loop::Desc;
%}

%feature("autodoc",
"""An object used by callbacks for :func:`.make_loop` to describe the sub graph in
loop operator. See docs of :func:`.make_loop` for more explanation.
""") LoopDesc;

%feature("autodoc",
"""forward a variable belonging to the parent graph into the sub graph

:type input: :class:`.SymbolVar`
:param input: a variable in the parent graph
:type has_assign: bool
:param has_assign: whether this input var would be assigned later
:rtype: :class:`.SymbolVar`
:return: the corresponding variable in the sub graph
""") LoopDesc::add_input;

%feature("autodoc",
"""instructs that value of a variable in the sub graph should be replaced by
the new value at the end of each loop.

:type dest: :class:`.SymbolVar`
:param dest: the variable to be updated. It must be a return value of
    :meth:`add_input`.
:type val: :class:`.SymbolVar`
:param val: the new value
:return: self to be chained
""") LoopDesc::assign;

%feature("autodoc",
"""set a variable to indicate whether the loop should be repeated.

:type cond: :class:`.SymbolVar`
:param cond: loop would be repeated if the absolute value of *cond* is more
    than 1e-6; It must evaluates to a scalar.
:return: self to be chained
""") LoopDesc::set_loop_condition;

%feature("autodoc",
"""get the loop counter, which would indicate current loop count, starting from zero.

:rtype: :class:`.SymbolVar`
:return: the loop counter
""") LoopDesc::get_counter_var;

%feature("autodoc",
"""mark a variable to be copied as output value of the loop operator.

:type var: :class:`.SymbolVar`
:param var: a variable in sub graph whose value should be copied into the
    parent graph
:type mode: str
:param mode: output mode; possible values are:

    * ``'last'``: only the last value would be recorded
    * ``'all'``: all the value would be recorded; shape of the variable should
      not change during looping, and the output var would be prepended with an
      extra leading dimension to index the loop count.
    * ``'sum'``: sum of all values of this variable during looping would be
      copied to output
    * ``'product'``: product of all values of this variable during looping
      would be copied to output
:rtype: int
:return: call id, starting at 0 and increasing continuously
""") LoopDesc::add_output;

class LoopDesc {
    public:
        LoopDesc() = delete;
        ~LoopDesc() = delete;

        SymbolVar add_input(SymbolVar input, bool has_assign = false);
        LoopDesc& assign(SymbolVar dest, SymbolVar val);
        LoopDesc& set_loop_condition(SymbolVar cond);
        SymbolVar get_counter_var();

        %extend {
            size_t add_output(SymbolVar& var, std::string mode) {
                using Desc = mgb::opr::Loop::Desc;
                auto get_mode = [&]() {
                    using OM = Desc::OutputMode;
                    for (char &i: mode)
                        i = std::tolower(i);
                    if (mode == "last")
                        return OM::LAST;
                    if (mode == "all")
                        return OM::ALL;
                    if (mode == "sum")
                        return OM::SUM;
                    throw mgb::MegBrainError(
                            mgb::ssprintf("unrecognized loop mode: %s",
                                mode.c_str()));
                };
                return $self->add_output(var, get_mode());
            }
        }

};

%feature("director") _LoopDescMakerCallback;
%inline {
    class _LoopDescMakerCallback {
        public:
            virtual ~_LoopDescMakerCallback() = default;
            virtual void call(LoopDesc &desc) = 0;
    };

    static SymbolVarArray _make_loop(
            _LoopDescMakerCallback* callback, int swap_interval,
            const OperatorNodeConfig &config) {

        std::shared_ptr<_LoopDescMakerCallback> callbackptr{callback};

        auto desc_maker = [callbackptr](mgb::opr::Loop::Desc &loop_desc) {
            callbackptr->call(loop_desc);
        };
        return mgb::opr::Loop::make(desc_maker, swap_interval, config);
    }
} // %inline

// vim: ft=swig
