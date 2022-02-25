# -*- coding: utf-8 -*-
import argparse

import numpy as np
import yaml
from megengine import jit, tensor
from megengine.module.external import ExternOprSubgraph


# "1,3,224,224" -> (1,3,224,224)
def str2tuple(x):
    x = x.split(",")
    x = [int(a) for a in x]
    x = tuple(x)
    return x


def main():
    parser = argparse.ArgumentParser(
        description="load a .pb model and convert to corresponding "
        "load-and-run model"
    )
    parser.add_argument("--input", help="mace model file")
    parser.add_argument("--param", help="mace param file")
    parser.add_argument(
        "--output", help="converted mge model"
    )
    parser.add_argument("--config", help="config file with yaml format")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = yaml.load(f)

    for model_name in configs["models"]:
        # ignore several sub models currently
        sub_model = configs["models"][model_name]["subgraphs"][0]

        # input/output shapes
        isizes = [str2tuple(x) for x in sub_model["input_shapes"]]

        # input/output names
        input_names = sub_model["input_tensors"]
        if "check_tensors" in sub_model:
            output_names = sub_model["check_tensors"]
            osizes = [str2tuple(x) for x in sub_model["check_shapes"]]
        else:
            output_names = sub_model["output_tensors"]
            osizes = [str2tuple(x) for x in sub_model["output_shapes"]]

        with open(args.input, "rb") as fin:
            raw_model = fin.read()
        with open(args.param, "rb") as fin:
            raw_param = fin.read()

        model_size = (len(raw_model)).to_bytes(4, byteorder="little")
        param_size = (len(raw_param)).to_bytes(4, byteorder="little")

        n_inputs = (len(input_names)).to_bytes(4, byteorder="little")
        n_outputs = (len(output_names)).to_bytes(4, byteorder="little")

        names_buffer = n_inputs + n_outputs
        for iname in input_names:
            names_buffer += (len(iname)).to_bytes(4, byteorder="little")
            names_buffer += str.encode(iname)
        for oname in output_names:
            names_buffer += (len(oname)).to_bytes(4, byteorder="little")
            names_buffer += str.encode(oname)

        shapes_buffer = n_outputs
        for oshape in osizes:
            shapes_buffer += (len(oshape)).to_bytes(4, byteorder="little")
            for oi in oshape:
                shapes_buffer += oi.to_bytes(4, byteorder="little")

        # raw content contains:
        # input/output names + output shapes + model buffer + param buffer
        wk_raw_content = (
            names_buffer
            + shapes_buffer
            + model_size
            + raw_model
            + param_size
            + raw_param
        )

        net = ExternOprSubgraph(osizes, "mace", wk_raw_content)
        net.eval()

        @jit.trace(record_only=True)
        def inference(inputs):
            return net(inputs)

        inputs = [
            tensor(np.random.random(isizes[i]).astype(np.float32)) for i in range(len(isizes))
        ]
        inference(*inputs)
        inference.dump(args.output)


if __name__ == "__main__":
    main()
