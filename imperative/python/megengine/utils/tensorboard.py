#!/usr/bin/env python
# -*-coding=utf-8-*-

from megengine.logger import get_logger

logger = get_logger(__name__)

try:
    from tensorboardX import SummaryWriter
    from tensorboardX.proto.attr_value_pb2 import AttrValue
    from tensorboardX.proto.graph_pb2 import GraphDef
    from tensorboardX.proto.node_def_pb2 import NodeDef
    from tensorboardX.proto.plugin_text_pb2 import TextPluginData
    from tensorboardX.proto.step_stats_pb2 import (
        DeviceStepStats,
        RunMetadata,
        StepStats,
    )
    from tensorboardX.proto.summary_pb2 import Summary, SummaryMetadata
    from tensorboardX.proto.tensor_pb2 import TensorProto
    from tensorboardX.proto.tensor_shape_pb2 import TensorShapeProto
    from tensorboardX.proto.versions_pb2 import VersionDef
except ImportError:
    logger.error(
        "TensorBoard and TensorboardX are required for visualize.", exc_info=True,
    )


def tensor_shape_proto(shape):
    """Creates an object matching
    https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor_shape.proto
    """
    return TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in shape])


def attr_value_proto(shape, dtype, attr):
    """Creates a dict of objects matching
    https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/attr_value.proto
    specifically designed for a NodeDef. The values have been
    reverse engineered from standard TensorBoard logged data.
    """
    attr_proto = {}
    if shape is not None:
        shapeproto = tensor_shape_proto(shape)
        attr_proto["_output_shapes"] = AttrValue(
            list=AttrValue.ListValue(shape=[shapeproto])
        )
    if dtype is not None:
        attr_proto["dtype"] = AttrValue(s=dtype.encode(encoding="utf-8"))
    if attr is not None:
        for key in attr.keys():
            attr_proto[key] = AttrValue(s=attr[key].encode(encoding="utf-8"))

    return attr_proto


def node_proto(
    name, op="UnSpecified", input=None, outputshape=None, dtype=None, attributes={}
):
    """Creates an object matching
    https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/node_def.proto
    """
    if input is None:
        input = []
    if not isinstance(input, list):
        input = [input]
    return NodeDef(
        name=name.encode(encoding="utf_8"),
        op=op,
        input=input,
        attr=attr_value_proto(outputshape, dtype, attributes),
    )


def node(
    name, op="UnSpecified", input=None, outputshape=None, dtype=None, attributes={}
):
    return node_proto(name, op, input, outputshape, dtype, attributes)


def graph(node_list):
    graph_def = GraphDef(node=node_list, versions=VersionDef(producer=22))
    stepstats = RunMetadata(
        step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")])
    )
    return graph_def, stepstats


def text(tag, text):
    plugin_data = SummaryMetadata.PluginData(
        plugin_name="text", content=TextPluginData(version=0).SerializeToString()
    )
    smd = SummaryMetadata(plugin_data=plugin_data)
    string_val = []
    for item in text:
        string_val.append(item.encode(encoding="utf_8"))
    tensor = TensorProto(
        dtype="DT_STRING",
        string_val=string_val,
        tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=len(text))]),
    )

    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])


class NodeRaw:
    def __init__(self, name, op, input, outputshape, dtype, attributes):
        self.name = name
        self.op = op
        self.input = input
        self.outputshape = outputshape
        self.dtype = dtype
        self.attributes = attributes


class SummaryWriterExtend(SummaryWriter):
    def __init__(
        self,
        logdir=None,
        comment="",
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix="",
        write_to_disk=True,
        log_dir=None,
        **kwargs
    ):
        self.node_raw_dict = {}
        super().__init__(
            logdir,
            comment,
            purge_step,
            max_queue,
            flush_secs,
            filename_suffix,
            write_to_disk,
            log_dir,
            **kwargs,
        )

    def add_text(self, tag, text_string_list, global_step=None, walltime=None):
        """Add text data to summary.

        Args:
            tag (string): Data identifier
            text_string_list (string list): String to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
                seconds after epoch of event

        Examples:

            .. code-block:: python

               # text can be divided into three levels by tag and global_step
               from writer import SummaryWriterExtend
               writer = SummaryWriterExtend()

               writer.add_text('level1.0/level2.0', ['text0'], 0)
               writer.add_text('level1.0/level2.0', ['text1'], 1)
               writer.add_text('level1.0/level2.1', ['text2'])
               writer.add_text('level1.1', ['text3'])
        """

        self._get_file_writer().add_summary(
            text(tag, text_string_list), global_step, walltime
        )

    def add_node_raw(
        self,
        name,
        op="UnSpecified",
        input=[],
        outputshape=None,
        dtype=None,
        attributes={},
    ):
        """Add node raw datas that can help build graph.

        After add all nodes, call ``add_graph_by_node_raw_list()`` to build
        graph and add graph data to summary.

        Args:
            name (string): opr name.
            op (string): opr class name.
            input (string list): input opr name.
            outputshape (list): output shape.
            dtype (string): output data dtype.
            attributes (dict): attributes info.

        Examples:

            .. code-block:: python

               from writer import SummaryWriterExtend
               writer = SummaryWriterExtend()

               writer.add_node_raw('node1', 'opr1', outputshape=[6, 2, 3], dtype="float32", attributes={
                       "peak_size": "12MB", "mmory_alloc": "2MB, percent: 16.7%"})
               writer.add_node_raw('node2', 'opr2', outputshape=[6, 2, 3], dtype="float32", input="node1",  attributes={
                                   "peak_size": "12MB", "mmory_alloc": "2MB, percent: 16.7%"})
               writer.add_graph_by_node_raw_list()

        """
        # self.node_raw_list.append(
        #     node(name, op, input, outputshape, dtype, attributes))
        self.node_raw_dict[name] = NodeRaw(
            name, op, input, outputshape, dtype, dict(attributes)
        )

    def add_node_raw_name_suffix(self, name, suffix):
        """Give node name suffix in order to finding this node by 'search nodes'
        Args:
            name (string): opr name.
            suffix (string): nam suffix.
        """
        old_name = self.node_raw_dict[name].name
        new_name = old_name + suffix
        # self.node_raw_dict[new_name] = self.node_raw_dict.pop(name)
        self.node_raw_dict[name].name = new_name
        for node_name, node in self.node_raw_dict.items():
            node.input = [new_name if x == old_name else x for x in node.input]

    def add_node_raw_attributes(self, name, attributes):
        """
        Args:
            name (string): opr name.
            attributes (dict): attributes info that need to be added.
        """
        for key, value in attributes.items():
            self.node_raw_dict[name].attributes[key] = value

    def add_graph_by_node_raw_list(self):
        """Build graph and add graph data to summary."""
        node_raw_list = []
        for key, value in self.node_raw_dict.items():
            node_raw_list.append(
                node(
                    value.name,
                    value.op,
                    value.input,
                    value.outputshape,
                    value.dtype,
                    value.attributes,
                )
            )
        self._get_file_writer().add_graph(graph(node_raw_list))
