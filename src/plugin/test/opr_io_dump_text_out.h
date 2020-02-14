/**
 * \file src/plugin/test/opr_io_dump_text_out.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

namespace {

const char* EXPECTED_TEXT_OUT_REC[3] = {
        // rec level 0
        R"OUTPUT(
var3 produced: name=var3 layout={1(1)} owner_opr=opr2{ImmutableTensor} opr2
 deps:
 val: [2]min=2 max=2 mean=2 l2=2 sd=N/A s
var1 produced: name=var1 layout={2(3),3(1)} owner_opr=opr0{Host2DeviceCopy} opr0
 deps:
 val: [2.352, 0.1114, -0.2721, 0.7569, -0.2438, ...]min=-0.272 max=2.35 mean=0.471 l2=1.02 sd=0.994 s
var17 produced: name=var17 layout={2(3),3(1)} owner_opr=opr16{Elemwise} opr16
 deps:
  [i0]var1: [2.352, 0.1114, -0.2721, 0.7569, -0.2438, ...] s
 val: [2.352, 0.1114, 0, 0.7569, 0, ...]min=0 max=2.35 mean=0.557 l2=1.01 sd=0.924 s
var11 produced: name=var11 layout={1(3),3(1)} owner_opr=opr10{Subtensor} opr10
 deps:
  [i0]var1: [2.352, 0.1114, -0.2721, 0.7569, -0.2438, ...] s
  [i1]var5:  <host value[s]> [0] s
  [i2]var7:  <host value[s]> [1] s
 val: [2.352, 0.1114, -0.2721]min=-0.272 max=2.35 mean=0.731 l2=1.37 sd=1.42 s
var13 produced: name=var13 layout={2(0),3(1)} owner_opr=opr12{Broadcast} opr12
 deps:
  [i0]var11: [2.352, 0.1114, -0.2721] s
  [i1]var9:  <host value[s]> [2, 3] s
 val: [2.352, 0.1114, -0.2721, 2.352, 0.1114, ...]min=-0.272 max=2.35 mean=0.731 l2=1.37 sd=1.27 s
var15 produced: name=var15 layout={2(3),3(1)} owner_opr=opr14{Elemwise} opr14
 deps:
  [i0]var3: [2] s
  [i1]var13: [2.352, 0.1114, -0.2721, 2.352, 0.1114, ...] s
 val: [4.352, 2.111, 1.728, 4.352, 2.111, ...]min=1.73 max=4.35 mean=2.73 l2=2.97 sd=1.27 s
var19 produced: name=var19 layout={2(3),3(1)} owner_opr=opr18{Elemwise} opr18
 deps:
  [i0]var15: [10.24, 0.2352, 0, 3.294, 0, ...] s
  [i1]var17: [2.352, 0.1114, 0, 0.7569, 0, ...] s
 val: [10.24, 0.2352, 0, 3.294, 0, ...]min=0 max=10.2 mean=2.33 l2=4.39 sd=4.08 s
var3 produced: name=var3 layout={1(1)} owner_opr=opr2{ImmutableTensor} opr2
 deps:
 val: [2]min=2 max=2 mean=2 l2=2 sd=N/A s
var1 produced: name=var1 layout={2(3),3(1)} owner_opr=opr0{Host2DeviceCopy} opr0
 deps:
 val: [0.05521, 0.724, 1.134, -0.2697, -1.545, ...]min=-1.54 max=1.13 mean=-0.105 l2=0.895 sd=0.974 s
var17 produced: name=var17 layout={2(3),3(1)} owner_opr=opr16{Elemwise} opr16
 deps:
  [i0]var1: [0.05521, 0.724, 1.134, -0.2697, -1.545, ...] s
 val: [0.05521, 0.724, 1.134, 0, 0, ...]min=0 max=1.13 mean=0.319 l2=0.55 sd=0.491 s
var11 produced: name=var11 layout={1(3),3(1)} owner_opr=opr10{Subtensor} opr10
 deps:
  [i0]var1: [0.05521, 0.724, 1.134, -0.2697, -1.545, ...] s
  [i1]var5:  <host value[s]> [0] s
  [i2]var7:  <host value[s]> [1] s
 val: [0.05521, 0.724, 1.134]min=0.0552 max=1.13 mean=0.638 l2=0.778 sd=0.545 s
var13 produced: name=var13 layout={2(0),3(1)} owner_opr=opr12{Broadcast} opr12
 deps:
  [i0]var11: [0.05521, 0.724, 1.134] s
  [i1]var9:  <host value[s]> [2, 3] s
 val: [0.05521, 0.724, 1.134, 0.05521, 0.724, ...]min=0.0552 max=1.13 mean=0.638 l2=0.778 sd=0.487 s
var15 produced: name=var15 layout={2(3),3(1)} owner_opr=opr14{Elemwise} opr14
 deps:
  [i0]var3: [2] s
  [i1]var13: [0.05521, 0.724, 1.134, 0.05521, 0.724, ...] s
 val: [2.055, 2.724, 3.134, 2.055, 2.724, ...]min=2.06 max=3.13 mean=2.64 l2=2.68 sd=0.487 s
var19 produced: name=var19 layout={2(3),3(1)} owner_opr=opr18{Elemwise} opr18
 deps:
  [i0]var15: [0.1135, 1.972, 3.556, 0, 0, ...] s
  [i1]var17: [0.05521, 0.724, 1.134, 0, 0, ...] s
 val: [0.1135, 1.972, 3.556, 0, 0, ...]min=0 max=3.56 mean=0.94 l2=1.66 sd=1.5 s
var3 produced: name=var3 layout={1(1)} owner_opr=opr2{ImmutableTensor} opr2
 deps:
 val: [2]min=2 max=2 mean=2 l2=2 sd=N/A s
var1 produced: name=var1 layout={2(3),3(1)} owner_opr=opr0{Host2DeviceCopy} opr0
 deps:
 val: [-0.5069, 0.4525, 0.1695, -0.02793, -0.1907, ...]min=-0.507 max=1.32 mean=0.203 l2=0.616 sd=0.637 s
var17 produced: name=var17 layout={2(3),3(1)} owner_opr=opr16{Elemwise} opr16
 deps:
  [i0]var1: [-0.5069, 0.4525, 0.1695, -0.02793, -0.1907, ...] s
 val: [0, 0.4525, 0.1695, 0, 0, ...]min=0 max=1.32 mean=0.324 l2=0.574 sd=0.52 s
var11 produced: name=var11 layout={1(3),3(1)} owner_opr=opr10{Subtensor} opr10
 deps:
  [i0]var1: [-0.5069, 0.4525, 0.1695, -0.02793, -0.1907, ...] s
  [i1]var5:  <host value[s]> [0] s
  [i2]var7:  <host value[s]> [1] s
 val: [-0.5069, 0.4525, 0.1695]min=-0.507 max=0.453 mean=0.0384 l2=0.404 sd=0.493 s
var13 produced: name=var13 layout={2(0),3(1)} owner_opr=opr12{Broadcast} opr12
 deps:
  [i0]var11: [-0.5069, 0.4525, 0.1695] s
  [i1]var9:  <host value[s]> [2, 3] s
 val: [-0.5069, 0.4525, 0.1695, -0.5069, 0.4525, ...]min=-0.507 max=0.453 mean=0.0384 l2=0.404 sd=0.441 s
var15 produced: name=var15 layout={2(3),3(1)} owner_opr=opr14{Elemwise} opr14
 deps:
  [i0]var3: [2] s
  [i1]var13: [-0.5069, 0.4525, 0.1695, -0.5069, 0.4525, ...] s
 val: [1.493, 2.453, 2.17, 1.493, 2.453, ...]min=1.49 max=2.45 mean=2.04 l2=2.08 sd=0.441 s
var19 produced: name=var19 layout={2(3),3(1)} owner_opr=opr18{Elemwise} opr18
 deps:
  [i0]var15: [0, 1.11, 0.3678, 0, 0, ...] s
  [i1]var17: [0, 0.4525, 0.1695, 0, 0, ...] s
 val: [0, 1.11, 0.3678, 0, 0, ...]min=0 max=2.87 mean=0.724 l2=1.26 sd=1.13 s
var3 produced: name=var3 layout={1(1)} owner_opr=opr2{ImmutableTensor} opr2
 deps:
 val: [2]min=2 max=2 mean=2 l2=2 sd=N/A s
var1 produced: name=var1 layout={2(3),3(1)} owner_opr=opr0{Host2DeviceCopy} opr0
 deps:
 val: [-0.03637, 2.111, 0.3236, -0.4861, -2.071, ...]min=-2.07 max=2.11 mean=0.0589 l2=1.25 sd=1.37 s
var17 produced: name=var17 layout={2(3),3(1)} owner_opr=opr16{Elemwise} opr16
 deps:
  [i0]var1: [-0.03637, 2.111, 0.3236, -0.4861, -2.071, ...] s
 val: [0, 2.111, 0.3236, 0, 0, ...]min=0 max=2.11 mean=0.491 l2=0.897 sd=0.822 s
var11 produced: name=var11 layout={1(3),3(1)} owner_opr=opr10{Subtensor} opr10
 deps:
  [i0]var1: [-0.03637, 2.111, 0.3236, -0.4861, -2.071, ...] s
  [i1]var5:  <host value[s]> [0] s
  [i2]var7:  <host value[s]> [1] s
 val: [-0.03637, 2.111, 0.3236]min=-0.0364 max=2.11 mean=0.799 l2=1.23 sd=1.15 s
var13 produced: name=var13 layout={2(0),3(1)} owner_opr=opr12{Broadcast} opr12
 deps:
  [i0]var11: [-0.03637, 2.111, 0.3236] s
  [i1]var9:  <host value[s]> [2, 3] s
 val: [-0.03637, 2.111, 0.3236, -0.03637, 2.111, ...]min=-0.0364 max=2.11 mean=0.799 l2=1.23 sd=1.03 s
var15 produced: name=var15 layout={2(3),3(1)} owner_opr=opr14{Elemwise} opr14
 deps:
  [i0]var3: [2] s
  [i1]var13: [-0.03637, 2.111, 0.3236, -0.03637, 2.111, ...] s
 val: [1.964, 4.111, 2.324, 1.964, 4.111, ...]min=1.96 max=4.11 mean=2.8 l2=2.95 sd=1.03 s
var19 produced: name=var19 layout={2(3),3(1)} owner_opr=opr18{Elemwise} opr18
 deps:
  [i0]var15: [0, 8.675, 0.7518, 0, 0, ...] s
  [i1]var17: [0, 2.111, 0.3236, 0, 0, ...] s
 val: [0, 8.675, 0.7518, 0, 0, ...]min=0 max=8.68 mean=1.77 l2=3.59 sd=3.42 s
var3 produced: name=var3 layout={1(1)} owner_opr=opr2{ImmutableTensor} opr2
 deps:
 val: [2]min=2 max=2 mean=2 l2=2 sd=N/A s
var1 produced: name=var1 layout={5(4),4(1)} owner_opr=opr0{Host2DeviceCopy} opr0
 deps:
 val: [-1.199, -1.02, 1.098, -1.472, -0.3848, ...]min=-2.24 max=1.25 mean=-0.347 l2=1.04 sd=1.01 s
var17 produced: name=var17 layout={5(4),4(1)} owner_opr=opr16{Elemwise} opr16
 deps:
  [i0]var1: [-1.199, -1.02, 1.098, -1.472, -0.3848, ...] s
 val: [0, 0, 1.098, 0, 0, ...]min=0 max=1.25 mean=0.262 l2=0.471 sd=0.402 s
var11 produced: name=var11 layout={1(4),4(1)} owner_opr=opr10{Subtensor} opr10
 deps:
  [i0]var1: [-1.199, -1.02, 1.098, -1.472, -0.3848, ...] s
  [i1]var5:  <host value[s]> [0] s
  [i2]var7:  <host value[s]> [1] s
 val: [-1.199, -1.02, 1.098, -1.472]min=-1.47 max=1.1 mean=-0.648 l2=1.21 sd=1.18 s
var13 produced: name=var13 layout={5(0),4(1)} owner_opr=opr12{Broadcast} opr12
 deps:
  [i0]var11: [-1.199, -1.02, 1.098, -1.472] s
  [i1]var9:  <host value[s]> [5, 4] s
 val: [-1.199, -1.02, 1.098, -1.472, -1.199, ...]min=-1.47 max=1.1 mean=-0.648 l2=1.21 sd=1.05 s
var15 produced: name=var15 layout={5(4),4(1)} owner_opr=opr14{Elemwise} opr14
 deps:
  [i0]var3: [2] s
  [i1]var13: [-1.199, -1.02, 1.098, -1.472, -1.199, ...] s
 val: [0.8006, 0.9802, 3.098, 0.5279, 0.8006, ...]min=0.528 max=3.1 mean=1.35 l2=1.69 sd=1.05 s
var19 produced: name=var19 layout={5(4),4(1)} owner_opr=opr18{Elemwise} opr18
 deps:
  [i0]var15: [0, 0, 3.401, 0, 0, ...] s
  [i1]var17: [0, 0, 1.098, 0, 0, ...] s
 val: [0, 0, 3.401, 0, 0, ...]min=0 max=3.86 mean=0.549 l2=1.23 sd=1.13 s
)OUTPUT",

        // rec level 1
        R"OUTPUT(
==== begin lazy value recording
var3 produced: name=var3 layout={1(1)} owner_opr=opr2{ImmutableTensor} opr2
 deps:
 val: <see lazy value below> s
var1 produced: name=var1 layout={2(3),3(1)} owner_opr=opr0{Host2DeviceCopy} opr0
 deps:
 val: <see lazy value below> s
var17 produced: name=var17 layout={2(3),3(1)} owner_opr=opr16{Elemwise} opr16
 deps:
  [i0]var1: <see lazy value below> s
 val: <see lazy value below> s
var11 produced: name=var11 layout={1(3),3(1)} owner_opr=opr10{Subtensor} opr10
 deps:
  [i0]var1: <see lazy value below> s
  [i1]var5:  <host value[s]> [0] s
  [i2]var7:  <host value[s]> [1] s
 val: <see lazy value below> s
var13 produced: name=var13 layout={2(0),3(1)} owner_opr=opr12{Broadcast} opr12
 deps:
  [i0]var11: <see lazy value below> s
  [i1]var9:  <host value[s]> [2, 3] s
 val: <see lazy value below> s
var15 produced: name=var15 layout={2(3),3(1)} owner_opr=opr14{Elemwise} opr14
 deps:
  [i0]var3: <see lazy value below> s
  [i1]var13: <see lazy value below> s
 val: <see lazy value below> s
var19 produced: name=var19 layout={2(3),3(1)} owner_opr=opr18{Elemwise} opr18
 deps:
  [i0]var15: <see lazy value below> s
  [i1]var17: <see lazy value below> s
 val: <see lazy value below> s
==== recorded values
#0: opr2 opr2{ImmutableTensor}
  var3: name=var3 [2]min=2 max=2 mean=2 l2=2 sd=N/A
#1: opr0 opr0{Host2DeviceCopy}
  var1: name=var1 [1.084, -1.283, -0.07331, 0.5371, -0.1351, ...]min=-1.28 max=1.08 mean=-0.17 l2=0.862 sd=0.925
#2: opr16 opr16{Elemwise}
  var17: name=var17 [1.084, 0, 0, 0.5371, 0, ...]min=0 max=1.08 mean=0.27 l2=0.494 sd=0.453
#3: opr10 opr10{Subtensor}
  var11: name=var11 [1.084, -1.283, -0.07331]min=-1.28 max=1.08 mean=-0.0909 l2=0.971 sd=1.18
#4: opr12 opr12{Broadcast}
  var13: name=var13 [1.084, -1.283, -0.07331, 1.084, -1.283, ...]min=-1.28 max=1.08 mean=-0.0909 l2=0.971 sd=1.06
#5: opr14 opr14{Elemwise}
  var15: name=var15 [3.084, 0.7167, 1.927, 3.084, 0.7167, ...]min=0.717 max=3.08 mean=1.91 l2=2.14 sd=1.06
#6: opr18 opr18{Elemwise}
  var19: name=var19 [3.343, 0, 0, 1.656, 0, ...]min=0 max=3.34 mean=0.833 l2=1.52 sd=1.4
==== recorded values
#0: opr2 opr2{ImmutableTensor}
  var3: name=var3 [2]min=2 max=2 mean=2 l2=2 sd=N/A
#1: opr0 opr0{Host2DeviceCopy}
  var1: name=var1 [0.1777, -0.6396, -1.422, 0.9506, -0.2105, ...]min=-1.69 max=0.951 mean=-0.472 l2=1.02 sd=0.993
#2: opr16 opr16{Elemwise}
  var17: name=var17 [0.1777, 0, 0, 0.9506, 0, ...]min=0 max=0.951 mean=0.188 l2=0.395 sd=0.38
#3: opr10 opr10{Subtensor}
  var11: name=var11 [0.1777, -0.6396, -1.422]min=-1.42 max=0.178 mean=-0.628 l2=0.906 sd=0.8
#4: opr12 opr12{Broadcast}
  var13: name=var13 [0.1777, -0.6396, -1.422, 0.1777, -0.6396, ...]min=-1.42 max=0.178 mean=-0.628 l2=0.906 sd=0.716
#5: opr14 opr14{Elemwise}
  var15: name=var15 [2.178, 1.36, 0.5778, 2.178, 1.36, ...]min=0.578 max=2.18 mean=1.37 l2=1.52 sd=0.716
#6: opr18 opr18{Elemwise}
  var19: name=var19 [0.387, 0, 0, 2.07, 0, ...]min=0 max=2.07 mean=0.41 l2=0.86 sd=0.828
)OUTPUT",

        // rec level 2
        R"OUTPUT(
==== begin lazy value recording
var3 produced: name=var3 layout={1(1)} owner_opr=opr2{ImmutableTensor} opr2
 deps:
 val: <see lazy value below> s
var1 produced: name=var1 layout={2(3),3(1)} owner_opr=opr0{Host2DeviceCopy} opr0
 deps:
 val: <see lazy value below> s
var17 produced: name=var17 layout={2(3),3(1)} owner_opr=opr16{Elemwise} opr16
 deps:
  [i0]var1: <see lazy value below> s
 val: <see lazy value below> s
var11 produced: name=var11 layout={1(3),3(1)} owner_opr=opr10{Subtensor} opr10
 deps:
  [i0]var1: <see lazy value below> s
  [i1]var5:  <host value[s]> [0] s
  [i2]var7:  <host value[s]> [1] s
 val: <see lazy value below> s
var13 produced: name=var13 layout={2(0),3(1)} owner_opr=opr12{Broadcast} opr12
 deps:
  [i0]var11: <see lazy value below> s
  [i1]var9:  <host value[s]> [2, 3] s
 val: <see lazy value below> s
var15 produced: name=var15 layout={2(3),3(1)} owner_opr=opr14{Elemwise} opr14
 deps:
  [i0]var3: <see lazy value below> s
  [i1]var13: <see lazy value below> s
 val: <see lazy value below> s
var19 produced: name=var19 layout={2(3),3(1)} owner_opr=opr18{Elemwise} opr18
 deps:
  [i0]var15: <see lazy value below> s
  [i1]var17: <see lazy value below> s
 val: <see lazy value below> s
==== recorded values
#0: opr2 opr2{ImmutableTensor}
  var3: name=var3 [2]min=2 max=2 mean=2 l2=2 sd=N/A
#1: opr0 opr0{Host2DeviceCopy}
  var1: name=var1 [-0.5252, 1.477, 0.00676, 0.9276, -0.5487, ...]min=-0.549 max=1.87 mean=0.534 l2=1.09 sd=1.04
#2: opr16 opr16{Elemwise}
  var17: name=var17 [0, 1.477, 0.00676, 0.9276, 0, ...]min=0 max=1.87 mean=0.713 l2=1.04 sd=0.834
#3: opr10 opr10{Subtensor}
  var11: name=var11 [-0.5252, 1.477, 0.00676]min=-0.525 max=1.48 mean=0.319 l2=0.905 sd=1.04
#4: opr12 opr12{Broadcast}
  var13: name=var13 [-0.5252, 1.477, 0.00676, -0.5252, 1.477, ...]min=-0.525 max=1.48 mean=0.319 l2=0.905 sd=0.927
#5: opr14 opr14{Elemwise}
  var15: name=var15 [1.475, 3.477, 2.007, 1.475, 3.477, ...]min=1.47 max=3.48 mean=2.32 l2=2.47 sd=0.927
#6: opr18 opr18{Elemwise}
  var19: name=var19 [0, 5.134, 0.01357, 1.368, 0, ...]min=0 max=5.13 mean=1.71 l2=2.65 sd=2.22
==== recorded values
#0: opr2 opr2{ImmutableTensor}
  var3: name=var3 [2]min=2 max=2 mean=2 l2=2 sd=N/A
#1: opr0 opr0{Host2DeviceCopy}
  var1: name=var1 [0.2565, -0.1118, -0.1181, 1.641, 0.2665, ...]min=-0.118 max=1.64 mean=0.333 l2=0.69 sd=0.663
#2: opr16 opr16{Elemwise}
  var17: name=var17 [0.2565, 0, 0, 1.641, 0.2665, ...]min=0 max=1.64 mean=0.371 l2=0.687 sd=0.634
#3: opr10 opr10{Subtensor}
  var11: name=var11 [0.2565, -0.1118, -0.1181]min=-0.118 max=0.257 mean=0.00886 l2=0.175 sd=0.214
#4: opr12 opr12{Broadcast}
  var13: name=var13 [0.2565, -0.1118, -0.1181, 0.2565, -0.1118, ...]min=-0.118 max=0.257 mean=0.00886 l2=0.175 sd=0.192
#5: opr14 opr14{Elemwise}
  var15: name=var15 [2.257, 1.888, 1.882, 2.257, 1.888, ...]min=1.88 max=2.26 mean=2.01 l2=2.02 sd=0.192
#6: opr18 opr18{Elemwise}
  var19: name=var19 [0.5788, 0, 0, 3.703, 0.5032, ...]min=0 max=3.7 mean=0.817 l2=1.54 sd=1.44
)OUTPUT"};

}  // anonymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

