from megengine.core.tensor.multipledispatch import Dispatcher


def test_register_many():
    f = Dispatcher("f")

    log = []

    @f.register()
    def _(x: int):
        log.append("a")
        return log[-1]

    @f.register()
    def _(x: int):
        log.append("b")
        return log[-1]

    assert f(0) == "b"
    assert log == ["b"]


def test_return_not_implemented():
    f = Dispatcher("f")

    log = []

    @f.register()
    def _(x: int):
        log.append("a")
        return log[-1]

    @f.register()
    def _(x: int):
        log.append("b")
        return NotImplemented

    assert f(0) == "a"
    assert log == ["b", "a"]


def test_super():
    f = Dispatcher("f")

    log = []

    @f.register()
    def _(x: int):
        log.append("a")
        return log[-1]

    @f.register()
    def _(x: int):
        log.append("b")
        return f.super(x)

    assert f(0) == "a"
    assert log == ["b", "a"]
