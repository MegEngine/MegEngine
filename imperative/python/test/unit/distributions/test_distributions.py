from megengine.distributions.exponential import Exponential
import numpy as np


def test_exponential():
    # check shape
    rate = np.abs(np.random.randn(5, 5))
    rate_1d = np.abs(np.random.randn(1))
    assert Exponential(rate).sample().shape == (5, 5)
    assert Exponential(rate).sample((7,)).shape == (7, 5, 5)
    assert Exponential(rate_1d).sample((1,)).shape == (1, 1)
    assert Exponential(rate_1d).sample().shape == (1,)
    assert Exponential(0.2).sample((1,)).shape == (1,)
    assert Exponential(50.0).sample((1,)).shape == (1,)

    # check mean and var
    def mean_var(rate, sample_len):
        m = Exponential(rate)
        expected_mean = m.mean
        expected_var = m.variance
        samples = m.sample((sample_len,))
        assert np.fabs(samples.numpy().mean() - expected_mean) < 1e-1
        assert np.fabs(samples.numpy().var() - expected_var) < 1e-1

    for rate in [0.5, 1., 1.5, 2., 5.]:
        sample_len = 200000
        mean_var(rate, sample_len)

    # check cdf and icdf
    m = Exponential(2.)
    samples = m.sample((200000,))
    cdf = m.cdf(samples)
    icdf = m.icdf(cdf)
    assert np.fabs(samples.numpy() - icdf.numpy()).max() < 1e-1

    # check log_prob
    m = Exponential(1.)
    samples = m.sample((200000,))
    log_prob = m.log_prob(samples)
    assert np.fabs(log_prob.numpy() + samples.numpy()).max() < 1e-1
