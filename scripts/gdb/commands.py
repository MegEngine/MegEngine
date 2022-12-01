import sys
import gdb

import re
import subprocess


_demangle_cache = {}


def demangle(name):
    if name not in _demangle_cache:
        _demangle_cache[name] = subprocess.run(['c++filt', "-t", name], stdout=subprocess.PIPE).stdout
    return _demangle_cache[name].decode('utf-8').strip()


def dynamic_cast(val):
    assert val.type.code == gdb.TYPE_CODE_REF
    val = val.cast(val.dynamic_type)
    return val


def eval_on_val(val, eval_str):
    if val.type.code == gdb.TYPE_CODE_REF:
        val = val.referenced_value()
    address = val.address
    eval_str = "(*({}){}){}".format(address.type, address, eval_str)
    return gdb.parse_and_eval(eval_str)


def is_subclass_of(subclass, baseclass):
    for field in subclass.fields():
        if field.is_base_class:
            if field.type == baseclass:
                return True
            elif is_subclass_of(field.type, baseclass):
                return True


def vector_size(vector):
    impl = vector["_M_impl"]
    return int(impl["_M_finish"] - impl["_M_start"])


def vector_item(vector, i):
    impl = vector["_M_impl"]
    return (impl["_M_start"] + i).dereference()


def shared_ptr_deref(ptr):
    return ptr["_M_ptr"].dereference()


def get_type_name(type_index):
    return gdb.lookup_global_symbol("mgb::imperative::debug::get_type_name(std::type_index const&)").value()(type_index).string()


mge_apply_transform_pattern = re.compile(r"^(.*)::apply_transform\(mgb::imperative::Operator const&, mgb::imperative::Span<mgb::imperative::ValueRef>\)$")
mge_op_fallback_pattern = re.compile(r"^(.*)::fallback\(mgb::imperative::Span<mgb::imperative::ValueRef>\) const$")


def is_mge_frame(frame):
    function = frame.function()
    if not (function and function.name):
        return False
    matcher = mge_apply_transform_pattern.match(function.name)
    if matcher:
        typename = matcher.group(1)
        return is_subclass_of(gdb.lookup_type(typename), gdb.lookup_type("mgb::imperative::Transform"))
    matcher = mge_op_fallback_pattern.match(function.name)
    if matcher:
        typename = matcher.group(1)
        return is_subclass_of(gdb.lookup_type(typename), gdb.lookup_type("mgb::imperative::Operator"))
    return False


def count_frame_level(frame):
    level = -1
    while frame:
        frame = frame.newer()
        level += 1
    return level


def print_mge_frame(frame):
    function = frame.function()
    op = eval_on_val(dynamic_cast(frame.read_var("op")), ".to_string().c_str()").string()
    inputs = str(frame.read_var("inputs"))
    matcher = mge_apply_transform_pattern.match(function.name)
    if matcher:
        name = matcher.group(1)
    else:
        name = mge_op_fallback_pattern.match(function.name).group(1)
    #TODO: span
    sal = frame.find_sal()
    filename = sal.symtab.filename
    line = sal.line
    print("#{} {} apply {} on {}, file: {}:{}".format(count_frame_level(frame), name, op, inputs, filename, line))


class MegengineBacktrace(gdb.Command):
    def __init__(self):
        super().__init__("mge-bt", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        frame = gdb.newest_frame()
        mge_stack = []
        while frame is not None:
            if is_mge_frame(frame):
                mge_stack.append(frame)
            frame = frame.older()
        for frame in reversed(mge_stack):
            print_mge_frame(frame)


class MegengineBreakApply(gdb.Command):
    def __init__(self):
        super().__init__("mge-brk-apply", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        gdb.Breakpoint("mgb::imperative::apply(mgb::imperative::Operator const&, mgb::imperative::Span<mgb::imperative::ValueRef>)")


class MegengineWatch(gdb.Command):
    def __init__(self):
        super().__init__("mge-watch", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        watch = gdb.lookup_global_symbol("mgb::imperative::debug::watch_value(mgb::imperative::ValueRef)").value()
        value = gdb.parse_and_eval(arg)
        watch(value)
        print("watching {}".format(str(value)))


class MegengineUp(gdb.Command):
    def __init__(self):
        super().__init__("mge-up", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        frame = gdb.selected_frame()
        if frame is None:
            print("Unable to find older dispatch frame")
            return
        frame = frame.older()
        while frame is not None and not is_mge_frame(frame):
            frame = frame.older()
        if frame is None:
            print("Unable to find older dispatch frame")
            return
        frame.select()
        print_mge_frame(frame)


class MegengineDown(gdb.Command):
    def __init__(self):
        super().__init__("mge-down", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        frame = gdb.selected_frame()
        if frame is None:
            print("Unable to find newer dispatch frame")
            return
        frame = frame.newer()
        while frame is not None and not is_mge_frame(frame):
            frame = frame.newer()
        if frame is None:
            print("Unable to find newer dispatch frame")
            return
        frame.select()
        print_mge_frame(frame)


class MegengineInfo(gdb.Command):
    def __init__(self):
        super().__init__("mge-info", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        if arg == "opr":
            registered_oprs = gdb.lookup_global_symbol("mgb::imperative::Operator::registered_types()").value()()
            size = vector_size(registered_oprs)
            for i in range(size):
                registered_opr = vector_item(registered_oprs, i)
                # name = eval_on_val(registered_opr, ".name()").string()
                name = get_type_name(registered_opr)
                print("{}: {}".format(i, demangle(name)))
        elif arg == "trf":
            dispatch_context = gdb.lookup_global_symbol("mgb::imperative::Transform::get_context()").value()()
            transformations = dispatch_context["transformations"]
            size = vector_size(transformations)
            for i in range(size):
                transformation = vector_item(transformations, i)
                # name = eval_on_val(transformation, ".get()->name().c_str()").string()
                name = shared_ptr_deref(transformation).dynamic_type.name
                # name = get_type_name(transformation)
                print("{}: {}".format(i, demangle(name)))
        else:
            print("Unsupported argument {}".format(arg))

    def complete(self, text, word):
        return ["opr", "trf"]


if sys.version_info.major > 2:
    MegengineBacktrace()
    MegengineBreakApply()
    MegengineUp()
    MegengineDown()
    MegengineInfo()
    MegengineWatch()
    
    gdb.Breakpoint("mgb::imperative::debug::notify_event(char const*)")
else:
    print("skip import commands")
