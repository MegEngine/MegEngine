# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Tuple


class Dataset(ABC):
    r"""An abstract base class for all map-style datasets.

    .. admonition:: Abstract methods
    
       All subclasses should overwrite these two methods:

       * ``__getitem__()``: fetch a data sample for a given key.
       * ``__len__()``: return the size of the dataset.

       They play roles in the data pipeline, see the description below.

    .. admonition:: Dataset in the Data Pipline

       Usually a dataset works with :class:`~.DataLoader`, :class:`~.Sampler`, :class:`~.Collator` and other components.

       For example, the sampler generates **indexes** of batches in advance according to the size of the dataset (calling ``__len__``),
       When dataloader need to yield a batch of data, pass indexes into the ``__getitem__`` method, then collate them to a batch.

       * Highly recommended reading :ref:`dataset-guide` for more details;
       * It might helpful to read the implementation of :class:`~.MNIST`, :class:`~.CIFAR10` and other existed subclass.

    .. warning::

       By default, all elements in a dataset would be :class:`numpy.ndarray`.
       It means that if you want to do Tensor operations, it's better to do the conversion explicitly, such as:

       .. code-block:: python

          dataset = MyCustomDataset()  # A subclass of Dataset
          data, label = MyCustomDataset[0]  # equals to MyCustomDataset.__getitem__[0]
          data = Tensor(data, dtype="float32")  # convert to MegEngine Tensor explicitly

          megengine.functional.ops(data)

       Tensor ops on ndarray directly are undefined behaviors.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


class StreamDataset(Dataset):
    r"""An abstract class for stream data.

    __iter__ method is aditionally needed.

    Examples:

        .. code-block:: python

            from megengine.data.dataset import StreamDataset
            from megengine.data.dataloader import DataLoader, get_worker_info
            from megengine.data.sampler import StreamSampler

            class MyStream(StreamDataset):
                def __init__(self):
                    self.data = [iter([1, 2, 3]), iter([4, 5, 6]), iter([7, 8, 9])]
                def __iter__(self):
                    worker_info = get_worker_info()
                    data_iter = self.data[worker_info.idx]
                    while True:
                        yield next(data_iter)

            dataloader = DataLoader(
                dataset = MyStream(),
                sampler = StreamSampler(batch_size=2),
                num_workers=3,
                parallel_stream = True,
            )

            for step, data in enumerate(dataloader):
                print(data)
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def __getitem__(self, idx):
        raise AssertionError("can not get item from StreamDataset by index")

    def __len__(self):
        raise AssertionError("StreamDataset does not have length")


class ArrayDataset(Dataset):
    r"""ArrayDataset is a dataset for numpy array data.

    One or more numpy arrays are needed to initiate the dataset.
    And the dimensions represented sample number are expected to be the same.

    Examples:

        .. code-block:: python

            from megengine.data.dataset import ArrayDataset
            from megengine.data.dataloader import DataLoader
            from megengine.data.sampler import SequentialSampler

            rand_data = np.random.randint(0, 255, size=(sample_num, 1, 32, 32), dtype=np.uint8)
            label = np.random.randint(0, 10, size=(sample_num,), dtype=int)
            dataset = ArrayDataset(rand_data, label)
            seque_sampler = SequentialSampler(dataset, batch_size=2)

            dataloader = DataLoader(
                dataset,
                sampler = seque_sampler,
                num_workers=3,
            )

            for step, data in enumerate(dataloader):
                print(data)

    """

    def __init__(self, *arrays):
        super().__init__()
        if not all(len(arrays[0]) == len(array) for array in arrays):
            raise ValueError("lengths of input arrays are inconsistent")
        self.arrays = arrays

    def __getitem__(self, index: int) -> Tuple:
        return tuple(array[index] for array in self.arrays)

    def __len__(self) -> int:
        return len(self.arrays[0])
