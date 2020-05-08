# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import math
import multiprocessing
import queue
import random
import time

import numpy as np

from ..logger import get_logger
from ..random.rng import _random_seed_generator
from .collator import Collator
from .dataset import Dataset
from .sampler import Sampler, SequentialSampler
from .transform import PseudoTransform, Transform

logger = get_logger(__name__)


MP_QUEUE_GET_TIMEOUT = 5


class DataLoader:
    __initialized = False

    def __init__(
        self,
        dataset: Dataset,
        sampler: Sampler = None,
        transform: Transform = None,
        collator: Collator = None,
        num_workers: int = 0,
        timeout: int = 0,
        divide: bool = False,
    ):
        r"""Provides a convenient way to iterate on a given dataset.

        `DataLoader` combines a dataset with sampler, transform and collator,
        make it flexible to get minibatch continually from a dataset.

        :type dataset: Dataset
        :param dataset: dataset from which to load the minibatch.
        :type sampler: Sampler
        :param sampler: defines the strategy to sample data from the dataset.
            If specified, :attr:`shuffle` must be ``False``.
        :type transform: Transform
        :param transform: defined the transforming strategy for a sampled batch.
            (default: ``None``)
        :type collator: Collator
        :param collator: defined the merging strategy for a transformed batch.
            (default: ``None``)
        :type num_workers: int
        :param num_workers: the number of sub-process to load, transform and collate
            the batch. ``0`` means using single-process. (default: ``0``)
        :type timeout: int
        :param timeout: if positive, means the timeout value(second) for collecting a
            batch from workers. (default: 0)
        :type divide: bool
        :param divide: define the paralleling strategy in multi-processing mode.
            ``True`` means one batch is divided into :attr:`num_workers` pieces, and
            the workers will process these pieces parallelly. ``False`` means
            different sub-process will process different batch. (default: ``False``)

        """

        if num_workers < 0:
            raise ValueError("num_workers should not be negative")

        if timeout < 0:
            raise ValueError("timeout should not be negative")

        if divide and num_workers <= 1:
            raise ValueError("divide should not be set to True when num_workers <= 1")

        self.dataset = dataset
        self.num_workers = num_workers
        self.timeout = timeout

        self.divide = divide

        if sampler is None:
            self.sampler = SequentialSampler(dataset, batch_size=1, drop_last=False)
        else:
            self.sampler = sampler

        if divide:
            if self.sampler.batch_size <= self.num_workers:
                raise ValueError(
                    "batch size must not smaller than num_workers in divide mode."
                )
            elif self.sampler.batch_size % self.num_workers:
                logger.warning(
                    "batch size is not divisible by num_workers, may lose performance in divide mode."
                )

        if transform is None:
            self.transform = PseudoTransform()
        else:
            self.transform = transform

        if collator is None:
            self.collator = Collator()
        else:
            self.collator = collator

        self.__initialized = True

    def __iter__(self):
        if self.num_workers == 0:
            return _SerialDataLoaderIter(self)
        else:
            return _ParallelDataLoaderIter(self)

    def __len__(self):
        return len(self.sampler)


class _BaseDataLoaderIter:
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.sampler = loader.sampler
        self.seed = _random_seed_generator().__next__()
        self.transform = loader.transform
        self.collator = loader.collator
        self.num_workers = loader.num_workers
        self.timeout = loader.timeout
        self.divide = loader.divide
        self.num_processed = 0

    def _get_next_batch(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_processed >= len(self):
            raise StopIteration
        minibatch = self._get_next_batch()
        self.num_processed += 1
        return minibatch


class _SerialDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SerialDataLoaderIter, self).__init__(loader)
        self.indices_iter = iter(self.sampler)

    def _get_next_batch(self):
        indices = next(self.indices_iter)
        items = [self.dataset[idx] for idx in indices]
        trans_items = self.transform.apply_batch(items)
        return self.collator.apply(trans_items)


class _ParallelDataLoaderIter(_BaseDataLoaderIter):
    __initialized = False

    def __init__(self, loader):
        super(_ParallelDataLoaderIter, self).__init__(loader)

        self.task_queues = [
            multiprocessing.Queue(maxsize=2) for _ in range(self.num_workers)
        ]

        self.feed_batch_idx = multiprocessing.Value("i", 0)
        self.target_batch_idx = multiprocessing.Value("i", 0)
        self.shutdown_flag = multiprocessing.Value("i", 0)

        self.trans_data_queues = [
            multiprocessing.Queue(maxsize=1) for _ in range(self.num_workers)
        ]

        # use shared-memory queue implemented by pyarrow plasma store.
        from ._queue import PlasmaShmQueue

        self.batch_queue = PlasmaShmQueue(maxsize=2)

        self.task_feeding_worker = multiprocessing.Process(
            target=_task_feeding_loop,
            args=(
                iter(self.sampler),
                self.task_queues,
                self.num_workers,
                self.divide,
                self.shutdown_flag,
                self.feed_batch_idx,
            ),
            daemon=True,
        )
        self.task_feeding_worker.start()

        self.workers = []
        for worker_id in range(self.num_workers):
            worker = multiprocessing.Process(
                target=_worker_loop,
                args=(
                    self.dataset,
                    self.task_queues[worker_id],
                    self.trans_data_queues[worker_id],
                    self.transform,
                    self.seed + worker_id + 1,
                    self.shutdown_flag,
                ),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

        if self.divide:
            self.data_collecting_worker = multiprocessing.Process(
                target=_data_gathering_loop,
                args=(
                    self.trans_data_queues,
                    self.batch_queue,
                    self.collator,
                    len(self),
                    self.num_workers,
                    self.shutdown_flag,
                    self.target_batch_idx,
                ),
                daemon=True,
            )
        else:
            self.data_collecting_worker = multiprocessing.Process(
                target=_data_selecting_loop,
                args=(
                    self.trans_data_queues,
                    self.batch_queue,
                    self.collator,
                    len(self),
                    self.num_workers,
                    self.shutdown_flag,
                    self.target_batch_idx,
                ),
                daemon=True,
            )
        self.data_collecting_worker.start()

        self.__initialized = True

    def _check_workers(self):
        # Check the status of each worker.
        if not self.data_collecting_worker.is_alive():
            exitcode = self.task_feeding_worker.exitcode
            if exitcode != 0:
                raise RuntimeError("data collecting worker died. {}".format(exitcode))

        if not self.task_feeding_worker.is_alive():
            exitcode = self.task_feeding_worker.exitcode
            if exitcode != 0:
                raise RuntimeError("task feeding worker died. {}".format(exitcode))

        for worker_id, worker in enumerate(self.workers):
            if not worker.is_alive():
                exitcode = worker.exitcode
                if exitcode != 0:
                    raise RuntimeError("worker:{} died. {}".format(worker_id, exitcode))

        logger.debug("all workers are alive.")

    def _try_get_next_batch(self):
        start_time = time.time()
        while True:
            self._check_workers()
            try:
                return self.batch_queue.get(timeout=1)
            except queue.Empty:
                logger.debug("batch queue empty!")
            waited_time = time.time() - start_time
            if self.timeout > 0:
                if waited_time > self.timeout:
                    raise RuntimeError("get_next_batch timeout!")

    def _get_next_batch(self):
        batch_data = self._try_get_next_batch()
        return batch_data

    def _shutdown(self):
        with self.shutdown_flag.get_lock():
            self.shutdown_flag.value = 1

        if self.task_feeding_worker.is_alive():
            self.task_feeding_worker.terminate()
        self.task_feeding_worker.join()

        if self.data_collecting_worker.is_alive():
            self.data_collecting_worker.terminate()
        self.data_collecting_worker.join()

        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
            worker.join()

        for q in self.trans_data_queues:
            q.cancel_join_thread()
            q.close()

        for q in self.task_queues:
            q.cancel_join_thread()
            q.close()

        self.batch_queue.cancel_join_thread()
        self.batch_queue.close()

    def __del__(self):
        if self.__initialized:
            self._shutdown()


def _task_feeding_loop(
    indices_iter, task_queues, num_workers, divide, shutdown_flag, feed_batch_idx
):
    # Feed the indices into the task queues
    while True:
        if shutdown_flag.value == 1:
            break
        batch_idx = feed_batch_idx.value
        try:
            indices = next(indices_iter)
        except StopIteration:
            break
        if divide:
            # make sure all task_queues is ready for put
            while any([q.full() for q in task_queues]):
                if shutdown_flag.value == 1:
                    return
            # divide into small pieces, feed to different workers.
            sub_num = math.ceil(len(indices) / num_workers)
            for worker_id in range(num_workers):
                sub_indices = indices[worker_id * sub_num : (worker_id + 1) * sub_num]
                task_queues[worker_id].put((batch_idx, sub_indices))
        else:
            # distribute tasks to different workers uniformly.
            target_id = batch_idx % num_workers
            while task_queues[target_id].full():
                if shutdown_flag.value == 1:
                    return
            task_queues[target_id].put((batch_idx, indices))
        with feed_batch_idx.get_lock():
            feed_batch_idx.value += 1


def _worker_loop(dataset, task_queue, trans_data_queue, transform, seed, shutdown_flag):
    # Get dataset items and do the transform
    random.seed(seed)
    np.random.seed(seed)
    while True:
        if shutdown_flag.value == 1:
            break
        try:
            batch_idx, indices = task_queue.get(timeout=MP_QUEUE_GET_TIMEOUT)
        except queue.Empty:
            continue
        if len(indices) > 0:
            items = [dataset[idx] for idx in indices]
            trans_items = transform.apply_batch(items)
        else:
            # in case of incomplete last batch
            trans_items = ()
        while True:
            try:
                trans_data_queue.put((batch_idx, trans_items), timeout=1)
                break
            except queue.Full:
                if shutdown_flag.value == 1:
                    break
                logger.debug("batch part queue is full!")


def _data_gathering_loop(
    trans_data_queues,
    batch_queue,
    collator,
    length,
    num_workers,
    shutdown_flag,
    target_idx,
):
    # Gathering the small pieces of batch data into full batch data
    while True:
        if shutdown_flag.value == 1:
            break

        target_batch_idx = target_idx.value

        if target_batch_idx >= length:
            break

        full_trans_items = []
        for worker_id in range(num_workers):
            while True:
                try:
                    batch_idx, trans_items = trans_data_queues[worker_id].get(
                        timeout=MP_QUEUE_GET_TIMEOUT
                    )
                    break
                except queue.Empty:
                    if shutdown_flag.value == 1:
                        break
                    logger.debug(
                        "worker:{} data queue get timeout! target batch idx:{}".format(
                            worker_id, target_batch_idx
                        )
                    )
            if batch_idx != target_batch_idx:
                raise RuntimeError(
                    "Unexperted batch_idx in data gathering loop. worker_id:{}.".format(
                        worker_id
                    )
                )
            else:
                full_trans_items.extend(trans_items)

        # Merge different parts into a batch.
        full_batch = collator.apply(full_trans_items)

        while True:
            try:
                batch_queue.put(full_batch, timeout=1)
                break
            except queue.Full:
                if shutdown_flag.value == 1:
                    break
                logger.debug("batch queue is full!")

        with target_idx.get_lock():
            target_idx.value += 1

    batch_queue.disconnect_client()


def _data_selecting_loop(
    trans_data_queues,
    batch_queue,
    collator,
    length,
    num_workers,
    shutdown_flag,
    target_idx,
):
    # Make sure that batch is generated exactly with the same order as generated indices
    while True:
        if shutdown_flag.value == 1:
            break

        target_batch_idx = target_idx.value

        if target_batch_idx >= length:
            break

        target_worker_id = target_batch_idx % num_workers
        while True:
            try:
                batch_idx, trans_items = trans_data_queues[target_worker_id].get(
                    timeout=MP_QUEUE_GET_TIMEOUT
                )
                batch_data = collator.apply(trans_items)
                break
            except queue.Empty:
                if shutdown_flag.value == 1:
                    break
                logger.debug(
                    "worker:{} data queue get timeout! target batch idx:{}".format(
                        target_worker_id, target_batch_idx
                    )
                )

        if batch_idx != target_batch_idx:
            raise RuntimeError(
                "batch_idx {} mismatch the target_batch_idx {}".format(
                    batch_idx, target_batch_idx
                )
            )

        while True:
            try:
                batch_queue.put(batch_data, timeout=1)
                break
            except queue.Full:
                if shutdown_flag.value == 1:
                    break
                logger.debug("batch queue is full!")

        with target_idx.get_lock():
            target_idx.value += 1

    batch_queue.disconnect_client()
