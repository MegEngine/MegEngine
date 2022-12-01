import sys

import gdb
import gdb.printing
import gdb.types


def dynamic_cast(val):
    assert val.type.code == gdb.TYPE_CODE_REF
    val = val.cast(val.dynamic_type)
    return val


def eval_on_val(val, eval_str):
    if val.type.code == gdb.TYPE_CODE_REF:
        val = val.referenced_value()
    address = val.address
    eval_str = "(*({}){}){}".format(address.type, int(address), eval_str)
    return gdb.parse_and_eval(eval_str)


class SmallVectorPrinter:
    def __init__(self, val):
        t = val.type.template_argument(0)
        self.begin = val["m_begin_ptr"].cast(t.pointer())
        self.end = val["m_end_ptr"].cast(t.pointer())
        self.size = self.end - self.begin
        self.capacity = val["m_capacity_ptr"].cast(t.pointer()) - val[
            "m_begin_ptr"
        ].cast(t.pointer())

    def to_string(self):
        return "SmallVector of Size {}".format(self.size)

    def display_hint(self):
        return "array"

    def children(self):
        for i in range(self.size):
            yield "[{}]".format(i), (self.begin + i).dereference()


class MaybePrinter:
    def __init__(self, val):
        self.val = val["m_ptr"]

    def to_string(self):
        if self.val:
            return "Some {}".format(self.val)
        else:
            return "None"

    def display_hint(self):
        return "array"

    def children(self):
        if self.val:
            yield "[0]", self.val.dereference()


class ToStringPrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        return eval_on_val(self.val, ".to_string().c_str()").string()


class ReprPrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        val = self.val
        if val.type.code == gdb.TYPE_CODE_REF:
            val = val.referenced_value()
        return eval_on_val(val, ".repr().c_str()").string()


class HandlePrinter:
    def __init__(self, val):
        impl = gdb.lookup_type("mgb::imperative::interpreter::intl::TensorInfo")
        self.val = val.cast(impl.pointer())

    def to_string(self):
        if self.val:
            return "Handle of TensorInfo at {}".format(self.val)
        else:
            return "Empty Handle"

    def display_hint(self):
        return "array"

    def children(self):
        if self.val:
            yield "[0]", self.val.dereference()


def print_small_tensor(device_nd):
    size = device_nd["m_storage"]["m_size"]
    ndim = device_nd["m_layout"]["ndim"]
    dim0 = device_nd["m_layout"]["shape"][0]
    stride0 = device_nd["m_layout"]["stride"][0]
    dtype = device_nd["m_layout"]["dtype"]
    if size == 0:
        return "<empty>"
    if ndim > 1:
        return "<ndim > 1>"
    if dim0 > 64:
        return "<size tool large>"
    raw_ptr = device_nd["m_storage"]["m_data"]["_M_ptr"]
    dtype_name = dtype["m_trait"]["name"].string()
    dtype_map = {
        "Float32": (gdb.lookup_type("float"), float),
        "Int32": (gdb.lookup_type("int"), int),
    }
    if dtype_name not in dtype_map:
        return "<dtype unsupported>"
    else:
        ctype, pytype = dtype_map[dtype_name]
    ptr = raw_ptr.cast(ctype.pointer())
    array = []
    for i in range(dim0):
        array.append((pytype)((ptr + i * int(stride0)).dereference()))
    return str(array)


class LogicalTensorDescPrinter:
    def __init__(self, val):
        self.layout = val["layout"]
        self.comp_node = val["comp_node"]
        self.value = val["value"]

    def to_string(self):
        return "LogicalTensorDesc"

    def children(self):
        yield "layout", self.layout
        yield "comp_node", self.comp_node
        yield "value", print_small_tensor(self.value)


class OpDefPrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        return self.val.dynamic_type.name

    def children(self):
        concrete_val = self.val.address.cast(
            self.val.dynamic_type.pointer()
        ).dereference()
        for field in concrete_val.type.fields():
            if field.is_base_class or field.artificial:
                continue
            if field.name == "sm_typeinfo":
                continue
            yield field.name, concrete_val[field.name]


class SpanPrinter:
    def __init__(self, val):
        self.begin = val["m_begin"]
        self.end = val["m_end"]
        self.size = self.end - self.begin

    def to_string(self):
        return "Span of Size {}".format(self.size)

    def display_hint(self):
        return "array"

    def children(self):
        for i in range(self.size):
            yield "[{}]".format(i), (self.begin + i).dereference()


if sys.version_info.major > 2:
    pp = gdb.printing.RegexpCollectionPrettyPrinter("MegEngine")
    # megdnn
    pp.add_printer(
        "megdnn::SmallVectorImpl",
        "^megdnn::SmallVector(Impl)?<.*>$",
        SmallVectorPrinter,
    )
    pp.add_printer("megdnn::TensorLayout", "^megdnn::TensorLayout$", ToStringPrinter)
    pp.add_printer("megdnn::TensorShape", "^megdnn::TensorShape$", ToStringPrinter)
    # megbrain
    pp.add_printer("mgb::CompNode", "^mgb::CompNode$", ToStringPrinter)
    pp.add_printer("mgb::Maybe", "^mgb::Maybe<.*>$", MaybePrinter)
    # imperative
    pp.add_printer(
        "mgb::imperative::LogicalTensorDesc",
        "^mgb::imperative::LogicalTensorDesc$",
        LogicalTensorDescPrinter,
    )
    pp.add_printer("mgb::imperative::OpDef", "^mgb::imperative::OpDef$", OpDefPrinter)
    pp.add_printer(
        "mgb::imperative::Subgraph", "^mgb::imperative::Subgraph$", ReprPrinter
    )
    pp.add_printer(
        "mgb::imperative::EncodedSubgraph",
        "^mgb::imperative::EncodedSubgraph$",
        ReprPrinter,
    )
    # imperative dispatch
    pp.add_printer(
        "mgb::imperative::ValueRef", "^mgb::imperative::ValueRef$", ToStringPrinter
    )
    pp.add_printer("mgb::imperative::Span", "^mgb::imperative::Span<.*>$", SpanPrinter)
    gdb.printing.register_pretty_printer(gdb.current_objfile(), pp)
else:
    print("skip import pretty printers")


def override_pretty_printer_for(val):
    type = val.type.strip_typedefs()
    if type.code == gdb.TYPE_CODE_PTR:
        if not val:
            return None
        target_typename = str(type.target().strip_typedefs())
        if target_typename == "mgb::imperative::OpDef":
            return OpDefPrinter(val.dereference())
        if target_typename == "mgb::imperative::interpreter::Interpreter::HandleImpl":
            return HandlePrinter(val)


if sys.version_info.major > 2:
    gdb.pretty_printers.append(override_pretty_printer_for)
