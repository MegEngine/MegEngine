#! /usr/bin/env python3
import argparse
import json
import math
import os

from graphviz import Digraph


class Node:
    def __init__(self, data):
        self.data = data
        self.label = ""
        self.output_labels = {i: "" for i in data["output"]}
        self.input_labels = {i: "" for i in data["input"]}

    def __str__(self):
        def quote(s):
            r = {
                "\\": "\\\\",
                "{": r"\{",
                "}": r"\}",
                "|": r"\|",
                "<": r"\<",
                ">": r"\>",
                "\n": r"\n",
            }
            for k, v in r.items():
                s = s.replace(k, v)
            return s

        def pport(d):
            return "|".join("<{}> {}".format(k, quote(v)) for k, v in d.items())

        in_ports = pport(self.input_labels)
        out_ports = pport(self.output_labels)

        return "{{%s}|%s|{%s}}" % (in_ports, quote(self.label), out_ports)


class CompGraphPlotter:
    _args = None

    _jgraph = None
    """original graph represented by json"""

    _jgraph_profile = None
    _profile_normalize = None
    _profile_max_size = 3
    _profile_min_size = 1

    _dest = None

    _finished_vars = None
    _finished_oprs = None
    _var_attr = None

    def __init__(self, args):
        self._finished_vars = set()
        self._finished_oprs = {}
        self._args = args

        self._load_data()
        self._do_plot()

    def _do_plot(self):
        self._node_commands = []
        self._edge_commands = []

        n0, c0 = map(len, [self._finished_oprs, self._finished_vars])
        if self._args.dest_nodes:
            for i in map(int, self._args.dest_nodes.split(",")):
                self._add_var(i)
        elif not self._args.prune_dangling_vars:
            for i in self._jgraph["var"].keys():
                self._add_var(i)
        else:
            for i in self._jgraph["operator"]:
                self._add_opr(i, 0)

        n1, c1 = map(len, [self._finished_oprs, self._finished_vars])
        print("plot with {} oprs, {} vars".format(n1 - n0, c1 - c0))

        for i in self._node_commands:
            i()
        for i in self._edge_commands:
            i()
        del self._node_commands
        del self._edge_commands

    @property
    def dot_graph(self):
        return self._dest

    def _make_node_attr_for_size(self, size):
        return dict(
            height=str(size / 2),
            width=str(size),
            fontsize=str(size * 5),
            fixedsize="true",
        )

    @classmethod
    def load_single_graph(cls, fpath):
        prof = None
        with open(fpath) as fin:
            data = json.load(fin)
            if "graph_exec" in data:
                prof = {int(k): v for k, v in data["profiler"]["device"].items()}
                data = data["graph_exec"]

            for t in ["operator", "var"]:
                data[t] = {int(i): j for i, j in data[t].items()}

            gvars = data["var"]
            for oid, i in data["operator"].items():
                i["input"] = list(map(int, i["input"]))
                out = i["output"] = list(map(int, i["output"]))
                for j in out:
                    gvars[j]["owner_opr"] = oid

            for var in data["var"].values():
                mp = var.get("mem_plan", None)
                if mp:
                    var["shape"] = "{" + ",".join(map(str, mp["layout"]["shape"])) + "}"
                else:
                    var["shape"] = "<?>"

        return data, prof

    def _load_data(self):
        args = self._args
        self._jgraph, prof = self.load_single_graph(args.input)
        if args.profile:
            for k, v in list(prof.items()):
                v = max(i["end"] - i["start"] for i in v.values())
                prof[k] = v
            self._jgraph_profile = prof
            self._profile_normalize = self._profile_max_size / max(
                map(math.sqrt, prof.values())
            )

        self._dest = Digraph(comment="plot for {}".format(args.input))

        if args.end_vars_from:
            eg, _ = self.load_single_graph(args.end_vars_from)
            for i in eg["operator"].keys():
                self._finished_oprs[i] = None
            for i in eg["var"].keys():
                self._finished_vars.add(i)

    def _add_opr(self, oprid, depth):
        name = "opr{}".format(oprid)
        if oprid in self._finished_oprs:
            return name
        oprobj = self._jgraph["operator"][oprid]
        if oprobj["type"] == "ImmutableTensor":
            self._finished_oprs[oprid] = None
            return name

        self._finished_oprs[oprid] = node = Node(oprobj)

        all_vars = self._jgraph["var"]
        dispname = [oprobj["name"], oprobj["type"]]
        for i in self._args.opr_attr:
            dispname.append("{}: {}".format(i, oprobj["extra"].get(i, "N/A")))

        attr = {}
        if self._jgraph_profile:
            time = self._jgraph_profile.get(oprid, 0)
            attr = self._make_node_attr_for_size(
                max(self._profile_normalize * time ** 0.5, self._profile_min_size)
            )
            dispname.append("time: {:.3f}ms".format(time * 1e3))

        node.label = "\n".join(dispname)

        self._node_commands.append(
            lambda: self._dest.node(name, str(node), shape="record", **attr)
        )

        for i in oprobj["input"]:
            inpopr = self._jgraph["operator"][all_vars[i]["owner_opr"]]
            if inpopr["type"] == "ImmutableTensor":
                node.input_labels[i] = "<const>"
                continue
            node.input_labels[i] = all_vars[i]["shape"]
            vi = self._add_var(i, depth)
            self._edge_commands.append(
                lambda vi=vi, name="{}:{}".format(name, i): self._dest.edge(vi, name)
            )

        return name

    def _add_var(self, varid, depth=0):
        varobj = self._jgraph["var"][varid]
        name = "opr{}:{}".format(varobj["owner_opr"], varid)
        if self._args.depth and depth > self._args.depth:
            return name
        if varid in self._finished_vars:
            return name
        self._finished_vars.add(varid)

        oprid = varobj["owner_opr"]
        oprobj = self._jgraph["operator"][oprid]
        dispname = [varobj["name"]] if varobj["name"] != oprobj["name"] else []
        dispname += [varobj["shape"]]
        dispname = "\n".join(dispname)

        self._add_opr(oprid, depth + 1)
        if self._finished_oprs[oprid] is not None:
            self._finished_oprs[oprid].output_labels[varid] = dispname

        return name


def main():
    parser = argparse.ArgumentParser(
        "plot megbrain computing graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dest-nodes",
        help="target var nodes; a comma-separated list of var ids. The "
        "dependency graph would be plotted. If not given, "
        "all nodes are plotted",
    )
    parser.add_argument("--end-vars-from", help="set end vars from another file")
    parser.add_argument(
        "-i", "--input", required=True, help="input computing graph file"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="write dot source to file"
    )
    parser.add_argument(
        "--profile", action="store_true", help="anonotate graph by profiling result"
    )
    parser.add_argument(
        "--prune-dangling-vars",
        action="store_true",
        help="remove vars not used by any opr",
    )
    parser.add_argument(
        "--opr-attr",
        action="append",
        default=[],
        help="extra opr attributes to be plotted",
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="max depth (i.e. distance from dest nodes) " "of nodes to be plotted",
    )
    parser.add_argument(
        "--output-format",
        default="dot",
        help="output file format, could be .dot/.png/.pdf",
    )
    args = parser.parse_args()

    graph = CompGraphPlotter(args).dot_graph
    if args.output:
        output_name = args.output.split(".")[0]
        graph.save("{}.dot".format(output_name))
        if args.output_format != "dot":
            os.system(
                "dot -T{} -o {}.{} {}.dot".format(
                    args.output_format, output_name, args.output_format, output_name
                )
            )
            os.system("rm -f {}.dot".format(output_name))


if __name__ == "__main__":
    main()
