#! /usr/bin/env python3
import argparse
import json
import math
import os

from megengine.utils.module_stats import sizeof_fmt
from megengine.utils.tensorboard import SummaryWriterExtend


def load_single_graph(fpath):
    with open(fpath) as fin:
        data = json.load(fin)

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

    return data


def comp_graph_plotter(input, writer):
    jgraph = load_single_graph(input)
    all_oprs = jgraph["operator"]
    all_vars = jgraph["var"]
    for i in all_oprs:
        opr = all_oprs[i]
        if opr["type"] == "ImmutableTensor":
            continue
        inputlist = []
        for var in opr["input"]:
            inpopr = all_oprs[all_vars[var]["owner_opr"]]
            if inpopr["type"] == "ImmutableTensor":
                continue
            inputlist.append(all_oprs[all_vars[var]["owner_opr"]]["name"])
        writer.add_node_raw(opr["name"], opr["type"], inputlist)
    writer.add_graph_by_node_raw_list()


def load_mem_info(fpath):
    with open(fpath) as fin:
        data = json.load(fin)

        oprs = data["opr"]
        for oid, i in oprs.items():
            i["size"] = 0

        for oid, i in data["chunk"].items():
            i["size"] = int(i["logic_addr_end"]) - int(i["logic_addr_begin"])

        data["peak_memory"] = 0
        data["weight_memory"] = 0
        for oid, i in data["chunk"].items():
            if i["type"] == "static_mem":
                i["owner_opr"] = oprs[i["time_begin"]]["name"]
                life_begin = int(i["time_begin"])
                life_end = int(i["time_end"])
                if i["overwrite_dest_id"] != "-1":
                    life_begin = life_begin + 1
                if data["peak_memory"] < int(i["logic_addr_end"]):
                    data["peak_memory"] = int(i["logic_addr_end"])
                for j in range(life_begin, life_end):
                    oprs[str(j)]["size"] = oprs[str(j)]["size"] + i["size"]
            elif i["type"] == "weight_mem":
                data["weight_memory"] += int(i["logic_addr_end"]) - int(
                    i["logic_addr_begin"]
                )
    return data


def peak_mem_regist(input, writer):
    jmem = load_mem_info(input)
    writer.add_text(
        "PEAK_MEMORY_SIZE",
        [sizeof_fmt(jmem["peak_memory"]) + "(" + str(jmem["peak_memory"]) + " B)"],
    )
    writer.add_text(
        "WEIGHT_MEMORY_SIZE",
        [sizeof_fmt(jmem["weight_memory"]) + "(" + str(jmem["weight_memory"]) + " B)"],
    )

    all_oprs = jmem["opr"]
    all_chunks = jmem["chunk"]

    max_size = 0
    max_size_oprs = []
    # get oprs that reach the max memory
    for oid, i in all_oprs.items():
        if i["size"] == max_size:
            max_size_oprs.append(int(i["id"]))
        elif i["size"] > max_size:
            max_size = i["size"]
            max_size_oprs.clear()
            max_size_oprs.append(int(i["id"]))
    # get component of chunks
    max_size_oprs.sort()
    opr2chunks = []
    num = len(max_size_oprs)
    for i in range(num):
        opr2chunks.append([])
    for oid, i in all_chunks.items():
        if i["type"] == "static_mem":
            life_begin = int(i["time_begin"])
            life_end = int(i["time_end"])
            if i["overwrite_dest_id"] != "-1":
                life_begin = life_begin + 1
            if max_size_oprs[0] >= life_end or max_size_oprs[-1] < life_begin:
                continue
            for j in range(num):
                if max_size_oprs[j] >= life_end:
                    break
                elif max_size_oprs[j] >= life_begin:
                    opr2chunks[j].append(i["id"])

    peak_num = 0
    for i in range(num):
        suffix_1 = "PEAK" + str(peak_num)
        if i - 1 > 0 and opr2chunks[i - 1] == opr2chunks[i]:
            continue
        max_num = 0
        opr2chunks[i] = sorted(
            opr2chunks[i],
            key=lambda chunk_id: all_chunks[chunk_id]["size"],
            reverse=True,
        )
        writer.add_text(
            suffix_1 + "/" + "<SUMMARY_INFO>",
            ["reached_max_opr_name:    " + all_oprs[str(max_size_oprs[i])]["name"]],
            0,
        )
        writer.add_text(
            suffix_1 + "/" + "<SUMMARY_INFO>",
            ["max_used_size:    " + sizeof_fmt(max_size)],
            1,
        )

        for j in opr2chunks[i]:
            suffix_2 = "MAX" + str(max_num)
            j_size = sizeof_fmt(all_chunks[j]["size"])
            j_percent = round(all_chunks[j]["size"] / max_size * 100, 3)

            writer.add_text(
                suffix_1 + "/" + suffix_2 + "_OPR",
                ["percent:    " + str(j_percent) + "%"],
                0,
            )
            writer.add_text(
                suffix_1 + "/" + suffix_2 + "_OPR", ["memory_size:    " + j_size], 1,
            )
            writer.add_text(
                suffix_1 + "/" + suffix_2 + "_OPR",
                ["owner_opr:    " + all_chunks[j]["owner_opr"]],
                2,
            )

            writer.add_node_raw_attributes(
                all_chunks[j]["owner_opr"],
                {
                    "memory_" + all_chunks[j]["id"]: j_size,
                    "memory_percent": str(j_percent) + "%",
                    "summary_memory_" + str(peak_num): sizeof_fmt(max_size),
                },
            )
            writer.add_node_raw_name_suffix(
                all_chunks[j]["owner_opr"], "_" + suffix_1 + "_" + suffix_2
            )
            max_num += 1
        peak_num += 1

    writer.add_graph_by_node_raw_list()


def convert(args):
    file_process_order = {
        "graph.json": comp_graph_plotter,
        "StaticMemoryInfo.json": peak_mem_regist,
    }
    g = os.walk(args.input)
    for path, dir_list, file_list in g:
        out_path = path.replace(args.input, args.output)
        writer = SummaryWriterExtend(out_path)
        for key, value in file_process_order.items():
            if key in file_list:
                value(os.path.join(path, key), writer)


def main():
    """`graph_info_analyze.py` is uesed to convert json dumped by `VisableDataSet`
    class to logs which can be read by python `tensorboard`.
    Now `get_static_memory_alloc_info()` support this feature,it will dump a dir
    which can be convert by `graph_info_analyze.py`.

    Examples:

        .. code-block:: shell

           graph_info_analyze.py -i <input_dir_name> -o <output_dir_name>
           tensorboard --logdir <output_dir_name>
    """
    parser = argparse.ArgumentParser(
        "convert json dumped by c to logs which can be read by python tensorboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True, help="input dirctor name(c tensorboard info)"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output dirctor name(python tensorboard info)",
    )
    args = parser.parse_args()

    convert(args)


if __name__ == "__main__":
    main()
