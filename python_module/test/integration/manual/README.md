# Regression test
* [How to run](#how-to-run)
* [Correctness](#correctness)
* [Performance](#performance)
* [Debug tools](#debug-tools)
* [To do list](#to-do-list)

## How to run

1. Run correctness regression test by

```
rlaunch --cpu=4 --memory=15000 --gpu=1 -- python3 verify_correctness.py
```

2. Run performance regression test by

```
rlaunch --cpu=4 --memory=15000 --gpu=1 -- python3 run_resnet50_perf.py
```

Compare with the [reference result](#performance) to verify the performance change.

3. [Temporary]: Run dynamic graph test

```
cd python_module/megengine/examples/cifar10/resnet_example
rlaunch --cpu=4 --memory=15000 --gpu=1 -- MGE_DISABLE_TRACE=1 python3 main.py --mode train --backend megengine-dynamic
```

Be sure to run a few epochs to verify the CPU/GPU memory usage and the result tends to converge. The complete run takes around 2 hours.

## Correctness

Pre-trained Resnet18 model on cifar10 dataset is used.

The test set contains
* forward run with static graph
* forward run with  dynamic graph
* forward + backward + parameter update with static graph
* forward + backward + parameter update with dynamic graph

Sample output:

```
Running fwd static ...
Success
Running fwd dynamic ...
Success
Running train static ...
Success
Running train dynamic ...
Failed!!!
import megengine operator
[INFO] load /home/zhangfan/.local/lib/python3.6/site-packages/megengine/examples/cifar10/resnet_example/checkpoint/pytorch_init.pth done
calculated loss: [2.3731833, 34.4626]
expect: [ 2.3731833 34.460594 ]
```

## Performance

Test cases run Resnet 50 training with batch size = 64.

Run `python3 resnet50_perf.py --help` for valid options.

Example script:

* Run `python3 run_resnet50_perf.py`
* You may want to submit the job to a remote server by  `rlaunch --cpu=16 --memory=100384 --gpu=8 -- python3 run_resnet50_perf.py`
* Sample output
```
**************************************
Run ResNet 50 performance test with batch size = 64
**************************************
Run static graph with default opt level
Finish with GPU Usage 6710MiB
Wall time per iter 283 ms
Run status: finished
**************************************
Run static graph with conv fastrun
Finish with GPU Usage 6540MiB
Wall time per iter 265 ms
Run status: finished
**************************************
Run static graph with conv fastrun and JIT
Finish with GPU Usage 6540MiB
Wall time per iter 267 ms
Run status: finished
**************************************
Run static graph with JIT, conv fastrun and without running step
Finish with GPU Usage 6540MiB
Wall time per iter 223 ms
Run status: finished
**************************************
```

## Debug tools 

You can pass `--run-debug-tool` to script `run_resnet50_perf.py`. Opr-level profiling result and valgrind will be invoked.

### How much overhead time will it take due to usage of the profiler

Please compare the same job with/without profiler. The timing statistic reported by profiler does not include the overhead time from itself.

### How can I get more information from profiler?

Refer to the main function in `megengine.utils.profile_analyze`.

### How can I profile main memory usage?

Valgrind massif tool can be used. The script also prints memory usage summary on screen as:

```

    GB
1.836^                                                             #          
     |                                                           @@#::::::@:::
     |                                                         @@@ #::::::@:::
     |                                 ::::::::::::@:::::::::@:@@@ #::::::@:::
     |                                ::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |                              @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |                            ::@@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |                          @:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |                        @@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |                       :@@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |                     @::@@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |                    @@::@@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |                  @:@@::@@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |                 :@ @@::@@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |              ::::@ @@::@@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |            :::: :@ @@::@@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |          :@: :: :@ @@::@@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |         :@@: :: :@ @@::@@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |       @@:@@: :: :@ @@::@@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
     |      @@ :@@: :: :@ @@::@@@:: @@::::: :::::: @ ::: ::: @:@@@ #::::::@:::
   0 +----------------------------------------------------------------------->Gi
     0                                                                   19.39

```
You can change "--run-iter" value to adjust iters to profile.
The detailed profiling is printed to `massif.out.ms_print`.

### How can I understand the profiler result?

The dumped profiling file `prof.json` can be interpolated by [megengine/utils/profile_analyze.py](../../utils/profile_analyze.py).
The following information is printed from the profiler:

```
-----------------  --------
total device time  0.318062
total host time    0.275643
-----------------  --------

╒════════════════════╤══════════════╤═══════════════════════════╤═══════════════╤═════════╤══════════╤═════════════╤═════════════╤══════════════╕
│ device self time   │ cumulative   │ operator info             │ computation   │ FLOPS   │ memory   │ bandwidth   │ in_shapes   │ out_shapes   │
╞════════════════════╪══════════════╪═══════════════════════════╪═══════════════╪═════════╪══════════╪═════════════╪═════════════╪══════════════╡
│ #0                 │ 0.114        │ Elemwise                  │ 6.53          │ 57.40   │ 51.63    │ 454.02      │ None        │ None         │
│ 0.114              │ 35.8%        │ 1481                      │ GFLO          │ GFLOPS  │ GiB      │ GiB/s       │             │              │
│ 35.8%              │              │ N/A                       │               │         │          │             │             │              │
├────────────────────┼──────────────┼───────────────────────────┼───────────────┼─────────┼──────────┼─────────────┼─────────────┼──────────────┤
│ #1                 │ 0.176        │ ConvolutionBackwardFilter │ 523.15        │ 8.35    │ 5.28     │ 84.24       │ None        │ None         │
│ 0.0627             │ 55.5%        │ 53                        │ GFLO          │ TFLOPS  │ GiB      │ GiB/s       │             │              │
│ 19.7%              │              │ N/A                       │               │         │          │             │             │              │
├────────────────────┼──────────────┼───────────────────────────┼───────────────┼─────────┼──────────┼─────────────┼─────────────┼──────────────┤
│ #2                 │ 0.221        │ ConvolutionBackwardData   │ 508.05        │ 11.31   │ 5.05     │ 112.42      │ None        │ None         │
│ 0.0449             │ 69.6%        │ 52                        │ GFLO          │ TFLOPS  │ GiB      │ GiB/s       │             │              │
│ 14.1%              │              │ N/A                       │               │         │          │             │             │              │
├────────────────────┼──────────────┼───────────────────────────┼───────────────┼─────────┼──────────┼─────────────┼─────────────┼──────────────┤
```
Please read [megengine/utils/profile_analyze.py](../../utils/profile_analyze.py) for more usages.

## To do list

* Change numerical tolerance after XPU-280 is done
* Add scripts to facilitate log analysis
* Profile GPU memory
* Incorporate with QA system
* Add more regression tests
