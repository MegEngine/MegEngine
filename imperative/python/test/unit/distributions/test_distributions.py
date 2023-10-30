import numpy as np

from megengine.distributions.exponential import Exponential


def test_exponential():
    # check shape
    rate = np.abs(np.random.randn(5, 5))
    rate_1d = np.abs(np.random.randn(1))

    def assert_shape_equal(actual, expect):
        if isinstance(actual, tuple):
            assert actual == expect
        else:
            assert all(actual.numpy() == np.array(expect))

    assert_shape_equal(Exponential(rate).sample().shape, (5, 5))
    assert_shape_equal(Exponential(rate).sample((7,)).shape, (7, 5, 5))
    assert_shape_equal(Exponential(rate_1d).sample((1,)).shape, (1, 1))
    assert_shape_equal(Exponential(rate_1d).sample().shape, (1,))
    assert_shape_equal(Exponential(0.2).sample((1,)).shape, (1,))
    assert_shape_equal(Exponential(50.0).sample((1,)).shape, (1,))

    # check mean and var
    def mean_var(rate, sample_len):
        m = Exponential(rate)
        expected_mean = m.mean
        expected_var = m.variance
        samples = m.sample((sample_len,))
        assert np.fabs(samples.numpy().mean() - expected_mean) < 1e-1
        assert np.fabs(samples.numpy().var() - expected_var) < 1e-1

    for rate in [0.5, 1.0, 1.5, 2.0, 5.0]:
        sample_len = 200000
        mean_var(rate, sample_len)
