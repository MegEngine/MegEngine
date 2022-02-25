#! /usr/bin/env python3
import argparse
import collections
import json
import re
import textwrap

import numpy as np
from tabulate import tabulate

from megengine.utils.profile_analyzer import (
    NonExistNum,
    ProfileAnalyzer,
    TimeFuncHelper,
)


def _tabulate_ml(tab, **kwargs):
    r"""Tabulate profile output with multi-line support."""
    new_tab = []
    new_tab_is_row = []
    for row in tab:
        col_lines = [str(i).split("\n") for i in row]
        max_nr_line = max(map(len, col_lines))
        new_tab_is_row.append(True)
        if max_nr_line > 1:
            new_tab_is_row.extend([False] * (max_nr_line - 1))
            for i in col_lines:
                if len(i) < max_nr_line:
                    i.extend([""] * (max_nr_line - len(i)))
            new_tab.extend(zip(*col_lines))
        else:
            new_tab.append(row)

    assert len(new_tab_is_row) == len(new_tab)
    ret = [i + "\n" for i in tabulate(new_tab, **kwargs).split("\n")]
    for idx, val in enumerate(new_tab_is_row):
        if not val:
            ret[idx * 2 + 2] = ""
    return "".join(ret)[:-1]


def _tabulate_confluence(tab, **kwargs):
    r"""Tabulate profile output."""
    kwargs.pop("tablefmt", None)
    s = tabulate(tab, tablefmt="orgtbl", **kwargs)
    lines = s.split("\n")
    lines[1] = lines[1].replace("+", "|")
    return "\n".join(lines)


def main(passed_args=None):  # pylint: disable=too-many-statements
    r"""Analyses profile info from :mod:`~.utils.profile_analyzer` .
    Run this file with ``--help`` to get more usage.
    """
    parser = argparse.ArgumentParser(
        description="analyze analyzer result",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dump")
    parser.add_argument(
        "-t",
        "--top",
        type=int,
        default=3,
        help="number of most time-consuming operators to print",
    )
    parser.add_argument(
        "--type", action="append", help="filter oprs in the top list by type"
    )
    parser.add_argument(
        "--aggregate-by",
        default=None,
        choices=["type"],
        help="aggragate profiling result by",
    )
    parser.add_argument(
        "--opr-name", help="filter oprs in the top list by regex of name"
    )
    parser.add_argument(
        "--input-dtype", type=str, help="filter oprs in the top list by input dtype"
    )
    parser.add_argument(
        "--top-end-key",
        default="end",
        choices=["end", "kern"],
        help="how time in top is calculated; end corresponds "
        "to total device time, and kern corresponds to only "
        "wait time",
    )
    parser.add_argument(
        "--aggregate",
        default=None,
        help="aggregate operations",
        choices=["max", "min", "sum", "mean"],
    )
    parser.add_argument(
        "--order-by",
        default="time",
        help="sort result according to given column; the param can be "
        "<col_name> or +<col_name>, meaning sorting in descending or "
        "ascending order respectively",
    )
    parser.add_argument(
        "--copy-time", action="store_true", help="show copy time related result"
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=float("-inf"),
        help="minimal time of a result to be printed",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=float("inf"),
        help="maximal time of a result to be printed",
    )
    parser.add_argument(
        "--show-host", action="store_true", help="show host profiling info"
    )
    parser.add_argument(
        "--dump-only-opr",
        action="store_true",
        help="only dump operator info as plaintext; useful "
        "for diff between two filtered profile results",
    )
    parser.add_argument(
        "--confluence",
        "--wiki",
        action="store_true",
        help="output confluence-markdown-compatible table",
    )
    parser.add_argument(
        "--print-only",
        choices={"summary", "device", "host"},
        help="print only chosen info",
    )

    args = parser.parse_args(passed_args)

    opr_filters = []
    if args.type:
        opr_filters.append(lambda o, a, b: o["type"] in args.type)
    if args.opr_name:
        opr_filters.append(
            lambda o, a, b, r=re.compile(args.opr_name): r.match(o["name"])
        )
    if args.input_dtype:
        opr_filters.append(
            lambda o, a, b: any(
                [i["mem_plan"]["layout"]["dtype"] == args.input_dtype for i in a]
            )
        )
    if not opr_filters:

        def opr_filter(o, a, b):  # pylint: disable=unused-argument
            return True

    else:

        def opr_filter(o, a, b):
            return all(i(o, a, b) for i in opr_filters)

    with open(args.dump) as fin:
        dump = json.load(fin)

    analyzer = ProfileAnalyzer(dump, opr_filter)
    analyzer_tot = ProfileAnalyzer(dump, lambda _, __, ___: True)

    def summary():
        device_end_func = TimeFuncHelper.eval_time_func("device", "end", np.max)
        device_kern_func = TimeFuncHelper.eval_time_func("device", "kern", np.max)
        host_end_func = TimeFuncHelper.eval_time_func("host", "end", np.max)

        def get_tot_time(func):
            rec = analyzer_tot.select(func, aggregate=np.sum)
            if not rec:
                return "N/A"
            rec = rec[0]
            return rec.time

        tab = []
        tot_dev_time = get_tot_time(device_end_func)
        tot_host_time = get_tot_time(host_end_func)
        tab.append(("total device time", tot_dev_time))
        tab.append(("total host time", tot_host_time))
        if args.copy_time:

            def fmt(a, b):
                a = a[0]
                b = b[0]
                return "tot={:.4f} avg={:.4f}".format(a.time, b.time)

            tab.append(
                (
                    "copy time",
                    fmt(
                        analyzer.select(
                            device_end_func,
                            lambda opr: opr.opr_info["type"] == "Copy",
                            aggregate=np.sum,
                        ),
                        analyzer.select(
                            device_end_func,
                            lambda opr: opr.opr_info["type"] == "Copy",
                            aggregate=np.mean,
                        ),
                    ),
                )
            )
            tab.append(
                (
                    "copy wait time",
                    fmt(
                        analyzer.select(
                            device_kern_func,
                            lambda opr: opr.opr_info["type"] == "Copy",
                            aggregate=np.sum,
                        ),
                        analyzer.select(
                            device_kern_func,
                            lambda opr: opr.opr_info["type"] == "Copy",
                            aggregate=np.mean,
                        ),
                    ),
                )
            )

        if args.confluence:
            tab_str = _tabulate_confluence(tab, headers=["name", "value"])
        else:
            tab_str = tabulate(tab)

        return tab_str, tot_dev_time, tot_host_time

    def prof_details(prof_type, tot_time):
        tab = []

        def func(
            opr,
            *,
            f0=TimeFuncHelper.eval_time_func(prof_type, args.top_end_key, np.max)
        ):
            t = f0(opr)
            if t is not None and (t < args.min_time or t > args.max_time):
                return None
            return t

        records = analyzer.select(
            func,
            aggregate=args.aggregate,
            aggregate_by=args.aggregate_by,
            top_k=args.top,
            sort_by=args.order_by,
        )

        if args.dump_only_opr:
            ret = []
            for i in records:
                ret.append(" ".join(i.info.values()))
            return "\n".join(ret)

        def format_shapes(shapes, layouts=None, sep="\n"):
            if isinstance(shapes, NonExistNum) or shapes is None:
                return repr(shapes)
            if layouts is None:
                layouts = [None] * len(shapes)

            comp = []
            for i, j in zip(shapes, layouts):
                i = "{" + ",".join(map(str, i)) + "}"
                if j:
                    i += "\n -[" + ",".join(map(str, j)) + "]"
                comp.append(i)
            return sep.join(comp)

        def fix_num_and_find_unit(x, base):
            if isinstance(x, NonExistNum) or (
                isinstance(x, float) and not np.isfinite(x)
            ):
                return x, ""
            unit = iter(["", "K", "M", "G", "T", "P"])
            while x >= base:
                x /= base
                next(unit)
            return x, next(unit)

        def get_number_with_unit(num, unit, base, sep="\n"):
            num, unit_prefix = fix_num_and_find_unit(num, base)
            if isinstance(unit, list):
                unit = unit[int(unit_prefix != "")]
            return ("{:.2f}" + sep + "{}{}").format(num, unit_prefix, unit)

        if args.confluence:
            rows = []
            cum_time = 0

            max_time = max([r.time for r in records])
            max_bandwidth = max([r.bandwidth for r in records])
            max_flops = max(
                [r.flops for r in records if not isinstance(r.flops, NonExistNum)]
            )

            bar_length = 15
            for idx, record in enumerate(records):
                cum_time += record.time

                opr_info = [("opr " + k, v) for k, v in record.info.items()]

                row = collections.OrderedDict(
                    [
                        ("#", idx),
                        ("time", "{:.3}".format(record.time)),
                        ("ratio", "{:.1f}%".format(record.time / tot_time * 100)),
                        ("time bar", "#" * int(record.time / max_time * bar_length)),
                        ("cum-time", cum_time),
                        ("cum-time ratio", cum_time / tot_time),
                    ]
                    + opr_info
                    + [
                        (
                            "computation (MFLO)",
                            "{:.1f}".format(record.computation / 1000 ** 2),
                        ),
                        ("MFLOPS", "{:.1f}".format(record.flops / 1000 ** 2)),
                        (
                            "MFLOPS-bar",
                            ""
                            if isinstance(record.flops, NonExistNum)
                            else ("#" * int(record.flops / max_flops * bar_length)),
                        ),
                        ("memory (MB)", "{:.1f}".format(record.memory / 1024 ** 2)),
                        (
                            "bandwidth (MiB/s)",
                            "{:.1f}".format(record.bandwidth / 1024 ** 2),
                        ),
                        (
                            "bandwidth bar",
                            "#" * int(record.bandwidth / max_bandwidth * bar_length),
                        ),
                        (
                            "in_shapes",
                            format_shapes(
                                record.in_shapes, record.in_layouts, sep=", "
                            ),
                        ),
                        ("out_shapes", format_shapes(record.out_shapes, sep=", ")),
                    ]
                )
                rows.append(row)
            headers = list(rows[0].keys())
            tab = [[row[i] for i in headers] for row in rows]

            return _tabulate_confluence(tab, headers=headers)

        else:
            cum_time = 0
            for idx, record in enumerate(records):
                cum_time += record.time
                tab.append(
                    (
                        "#{}\n{:.3}\n{:.1f}%".format(
                            idx, record.time, record.time / tot_time * 100
                        ),
                        "{:.3}\n{:.1f}%".format(cum_time, cum_time / tot_time * 100),
                        "\n".join(
                            "\n-  ".join(textwrap.wrap(str(i), width=30))
                            for i in record.info.values()
                        ),
                        get_number_with_unit(record.computation, "FLO", 1000),
                        get_number_with_unit(record.flops, "FLOPS", 1000),
                        get_number_with_unit(record.memory, ["byte", "iB"], 1024),
                        get_number_with_unit(
                            record.bandwidth, ["byte/s", "iB/s"], 1024
                        ),
                        format_shapes(record.in_shapes, record.in_layouts),
                        format_shapes(record.out_shapes),
                    )
                )
            return _tabulate_ml(
                tab,
                headers=[
                    "{} self time".format(prof_type),
                    "cumulative",
                    "operator info",
                    "computation",
                    "FLOPS",
                    "memory",
                    "bandwidth",
                    "in_shapes",
                    "out_shapes",
                ],
                tablefmt="fancy_grid",
            )

    summary_tab, tot_dev_time, tot_host_time = summary()
    if args.print_only:
        print(
            {
                "summary": lambda: summary_tab,
                "device": lambda: prof_details("device", tot_dev_time),
                "host": lambda: prof_details("host", tot_host_time),
            }[args.print_only]()
        )
    else:
        print(summary_tab)
        print()
        print(prof_details("device", tot_dev_time))
        if args.show_host:
            print()
            print(prof_details("host", tot_host_time))


if __name__ == "__main__":
    main()
