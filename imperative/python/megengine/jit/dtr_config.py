# -*- coding: utf-8 -*-


class DTRConfig:
    r"""Configuration for DTR memory optimization.

    Args:
        eviction_threshold: eviction threshold in bytes. When GPU memory usage
            exceeds this value, DTR will heuristically select and evict resident
            tensors until the amount of used memory falls below this threshold.
        evictee_minimum_size: memory threshold of tensors in bytes. Only tensors
            whose size exceeds this threshold will be added to the candidate set.
            A tensor that is not added to the candidate set will never be evicted
            during its lifetime. Default: 1048576.
        recomp_memory_factor: hyperparameter of the estimated memory of recomputing
            the tensor. The larger this value is, the less memory-consuming
            tensor will be evicted in heuristic strategies. This value is greater
            than or equal to 0. Default: 1.
        recomp_time_factor: hyperparameter of the estimated time of recomputing
            the tensor. The larger this value is, the less time-consuming
            tensor will be evicted in heuristic strategies. This value is greater
            than or equal to 0. Default: 1.
    """

    def __init__(
        self,
        eviction_threshold: int = 0,
        evictee_minimum_size: int = 1 << 20,
        recomp_memory_factor: float = 1,
        recomp_time_factor: float = 1,
    ):
        assert eviction_threshold > 0, "eviction_threshold must be greater to zero"
        self.eviction_threshold = eviction_threshold
        assert (
            evictee_minimum_size >= 0
        ), "evictee_minimum_size must be greater or equal to zero"
        self.evictee_minimum_size = evictee_minimum_size
        assert (
            recomp_memory_factor >= 0
        ), "recomp_memory_factor must be greater or equal to zero"
        self.recomp_memory_factor = recomp_memory_factor
        assert (
            recomp_time_factor >= 0
        ), "recomp_time_factor must be greater or equal to zero"
        self.recomp_time_factor = recomp_time_factor
