# -*- coding: utf-8 -*-
from ..device import get_device_count


class SublinearMemoryConfig:
    r"""Configuration for sublinear memory optimization.

    Args:
        thresh_nr_try: number of samples both for searching in linear space
            and around current thresh in sublinear memory optimization. Default: 10.
            It can also be set through the environmental variable 'MGB_SUBLINEAR_MEMORY_THRESH_NR_TRY'.
        genetic_nr_iter: number of iterations to find the best checkpoints in genetic algorithm.
            Default: 0.
            It can also be set through the environmental variable 'MGB_SUBLINEAR_MEMORY_GENETIC_NR_ITER'.
        genetic_pool_size: number of samples for the crossover random selection
            during genetic optimization. Default: 20.
            It can also be set through the environmental variable 'MGB_SUBLINEAR_MEMORY_GENETIC_POOL_SIZE'.
        lb_memory_mb: memory lower bound of bottleneck size in MB for sublinear memory optimization.
            It can be used to perform manual tradeoff between memory and speed. Default: 0.
            It can also be set through the environmental variable 'MGB_SUBLINEAR_MEMORY_LOWER_BOUND_MB'.
        num_worker: number of thread workers to search the optimum checkpoints
            in sublinear memory optimization. Default: half of cpu number in the system.
            Note: the value must be greater or equal to one.
            It can also be set through the environmental variable 'MGB_SUBLINEAR_MEMORY_WORKERS'.
    
    Note that the environmental variable MGB_COMP_GRAPH_OPT must be set to 'enable_sublinear_memory_opt=1'
    in order for the above environmental variable to be effective.
    """

    def __init__(
        self,
        thresh_nr_try: int = 10,
        genetic_nr_iter: int = 0,
        genetic_pool_size: int = 20,
        lb_memory_mb: int = 0,
        num_worker: int = max(1, get_device_count("cpu") // 2),
    ):
        assert thresh_nr_try >= 0, "thresh_nr_try must be greater or equal to zero"
        self.thresh_nr_try = thresh_nr_try
        assert genetic_nr_iter >= 0, "genetic_nr_iter must be greater or equal to zero"
        self.genetic_nr_iter = genetic_nr_iter
        assert (
            genetic_pool_size >= 0
        ), "genetic_pool_size must be greater or equal to zero"
        self.genetic_pool_size = genetic_pool_size
        self.lb_memory_mb = lb_memory_mb
        assert num_worker > 0, "num_worker must be greater or equal to one"
        self.num_worker = num_worker
