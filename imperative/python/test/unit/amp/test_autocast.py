from megengine import amp
from megengine.core.tensor import amp as origin_amp


def test_autocast():
    def check(enabled, low, high):
        assert amp.enabled == enabled
        assert origin_amp._enabled == enabled
        assert amp.low_prec_dtype == low
        assert origin_amp._get_amp_low_prec_dtype() == low
        assert amp.high_prec_dtype == high
        assert origin_amp._get_amp_high_prec_dtype() == high

    origin_enabled = amp.enabled
    origin_high = amp.high_prec_dtype
    origin_low = amp.low_prec_dtype
    with amp.autocast(low_prec_dtype="float16", high_prec_dtype="float32"):
        check(True, "float16", "float32")
    check(origin_enabled, origin_low, origin_high)
    amp.enabled = True
    amp.high_prec_dtype = "float32"
    amp.low_prec_dtype = "float16"
    check(True, "float16", "float32")
    amp.enabled = origin_enabled
    amp.high_prec_dtype = origin_high
    amp.low_prec_dtype = origin_low
    check(origin_enabled, origin_low, origin_high)
