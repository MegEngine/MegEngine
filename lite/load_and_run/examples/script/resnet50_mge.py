#!/usr/bin/env python3
import argparse
import math

import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import numpy as np
from megengine import jit, tensor


class BasicBlock(M.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        norm=M.BatchNorm2d,
    ):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = M.Conv2d(
            in_channels, channels, 3, stride, padding=dilation, bias=False
        )
        self.bn1 = norm(channels)
        self.conv2 = M.Conv2d(channels, channels, 3, 1, padding=1, bias=False)
        self.bn2 = norm(channels)
        self.downsample = (
            M.Identity()
            if in_channels == channels and stride == 1
            else M.Sequential(
                M.Conv2d(in_channels, channels, 1, stride, bias=False), norm(channels),
            )
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(identity)
        x += identity
        x = F.relu(x)
        return x


class Bottleneck(M.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        norm=M.BatchNorm2d,
    ):
        super().__init__()
        width = int(channels * (base_width / 64.0)) * groups
        self.conv1 = M.Conv2d(in_channels, width, 1, 1, bias=False)
        self.bn1 = norm(width)
        self.conv2 = M.Conv2d(
            width,
            width,
            3,
            stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = norm(width)
        self.conv3 = M.Conv2d(width, channels * self.expansion, 1, 1, bias=False)
        self.bn3 = norm(channels * self.expansion)
        self.downsample = (
            M.Identity()
            if in_channels == channels * self.expansion and stride == 1
            else M.Sequential(
                M.Conv2d(in_channels, channels * self.expansion, 1, stride, bias=False),
                norm(channels * self.expansion),
            )
        )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        identity = self.downsample(identity)

        x += identity
        x = F.relu(x)

        return x


class ResNet(M.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm=M.BatchNorm2d,
    ):
        super().__init__()
        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = M.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm(self.in_channels)
        self.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0], norm=norm)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            norm=norm,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            norm=norm,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            norm=norm,
        )
        self.fc = M.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)
            elif isinstance(m, M.Linear):
                M.init.msra_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block
        # behaves like an identity. According to https://arxiv.org/abs/1706.02677
        # This improves the model by 0.2~0.3%.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    M.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    M.init.zeros_(m.bn2.weight)

    def _make_layer(
        self, block, channels, blocks, stride=1, dilate=False, norm=M.BatchNorm2d
    ):
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        layers.append(
            block(
                self.in_channels,
                channels,
                stride,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm=norm,
            )
        )
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm=norm,
                )
            )

        return M.Sequential(*layers)

    def extract_features(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        outputs["stem"] = x

        x = self.layer1(x)
        outputs["res2"] = x
        x = self.layer2(x)
        outputs["res3"] = x
        x = self.layer3(x)
        outputs["res4"] = x
        x = self.layer4(x)
        outputs["res5"] = x
        return outputs

    def forward(self, x):
        x = F.reshape(x, (1,3,224,224))
        x = self.extract_features(x)["res5"]

        x = F.avg_pool2d(x, 7)
        x = F.flatten(x, 1)
        x = self.fc(x)

        return x



@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/resnet50_fbaug_76254_4e14b7d1.pkl"
)
def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="dump mge model for resnet50",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b", "--batch-size", help="batch size of the model", default=1, type=int
    )
    parser.add_argument(
        "-d",
        "--dtype",
        help="the dtype of the model,which includes float32 and uint8",
        default="float32",
        type=str,
    )
    parser.add_argument(
        "--inputs",
        help="set the inputs data to get a model with testcase",
        default="",
        type=str,
    )

    parser.add_argument(
        "--dir",
        help="set the dir where the model to dump",
        default=".",
        type=str,
    )

    parser.add_argument(
        "--enable-nchw4",
        help="enable-nchw4 for NVIDIA CUDNN",
        action='store_true'
    )

    parser.add_argument(
        "--enable-chwn4",
        help="enable-chwn4 for NVIDIA CUDNN",
        action='store_true'
    )

    args = parser.parse_args()
    net = resnet50()
    net.eval()

    @jit.trace(symbolic=True, capture_as_const=True)
    def fun(data):
        return net(data)
    
    if args.dtype == "float32" or args.dtype == "uint8":
        # dump float32
        data_type=np.float32
        if args.dtype == "uint8":
            data_type =np.uint8
        data = tensor(
            (np.random.random([args.batch_size, 3, 224, 224])*255).astype(data_type)
        )
        fun(data)
        if args.inputs == "":
            fun.dump(
                
                args.dir + "/resnet50_b" + str(args.batch_size) + "_"+ args.dtype +"_without_data.mge", arg_names=["data"],
                no_assert=True, enable_nchw4=args.enable_nchw4, enable_chwn4=args.enable_chwn4,
                
            )
        else:
            fun.dump(
                args.dir + "/resnet50_b" + str(args.batch_size) + "_"+ args.dtype +"_with_data.mge", arg_names=["data"],
                input_data=[args.inputs], no_assert=True, enable_nchw4=args.enable_nchw4,
            )

    else:
        raise TypeError("dtype should be float32 or uint8")