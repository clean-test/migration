# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import pytest

from migrate import base


testdata_connect = [
    ("a, b", "a << b", [" << "]),
    ("a, b, c, d", "a, b}, X{c, d", [", ", "}, X{", ", "]),
]


def _connect(lines, **kwargs):
    def _adapter(tree):
        return base.insert_connectors(root=tree, **kwargs)

    return base.transform_tree(lines=lines, adapter=_adapter)


@pytest.mark.parametrize("case,expected,connectors", testdata_connect)
def test_connect(case, expected, connectors):
    assert _connect(lines=[base.Line(indent="", content=case)], connectors=connectors)[0].content == expected


testdata_lift = [
    ("foo()", "ct::expect(foo());", {}),
    ("true and foo()", "ct::expect(ct::lift(true) and foo());", {}),
    ("not foo()", "ct::expect(not ct::lift(foo()));", {}),
    ('not not foo(), "hi" << you()', 'ct::expect(not not ct::lift(foo())) << "hi" << you();', {"connectors": [" << "]}),
    ("not normal()", "ct::expect(not ct::lift(normal()));", {}),
    ('n::f(a, "x") == "x"', 'ct::expect(ct::lift(n::f(a, "x")) == "x");', {}),
    ('f(a, "x") == "x"', 'ct::expect(ct::lift(f(a, "x")) == "x");', {}),
    ("f(g(h()))", "ct::expect(f(g(h())));", {}),
    ("a, 2", "ct::expect(a == 2_i);", {"connectors": [" == "]}),
    ("!f()", "ct::expect(!ct::lift(f()));", {}),
    ("a != b", "ct::expect(ct::lift(a) != b);", {}),
    ("a, message", "ct::expect(a) << message;", {"connectors": [" << "]}),
    ("""u == u"123" """, """ct::expect(ct::lift(u) == u"123");""", {}),
    ("""x == "x"sv""", """ct::expect(ct::lift(x) == "x"sv);""", {}),
    (
        """\
        U"X"
        U"Y"
    !=
        z""",
        """\
        ct::expect(ct::lift(U"X"
        U"Y")
    !=
        z);""",
        {},
    ),
    ("f().size(), g()", "ct::expect(ct::lift(f().size()) == g());", {"connectors": [" == "]}),
    ("f().g()", "ct::expect(f().g());", {}),
    ("X::f()[5], g()", "ct::expect(ct::lift(X::f()[5]) == g());", {"connectors": [" == "]}),
    ("X::f()()(2), g()", "ct::expect(ct::lift(X::f()()(2)) == g());", {"connectors": [" == "]}),
    ("x.f().f()(2), g()", "ct::expect(ct::lift(x.f().f()(2)) == g());", {"connectors": [" == "]}),
    ("(e.x[i] == c.x[i])", "ct::expect((ct::lift(e.x[i]) == c.x[i]));", {}),
    ("(e.x[i].b == c.x[i].b)", "ct::expect((ct::lift(e.x[i].b) == c.x[i].b));", {}),
    ("c2->isError()", "ct::expect(c2->isError());", {}),
    ("duration(t, act) > ms::zero()", "ct::expect(ct::lift(duration(t, act)) > ms::zero());", {}),
    ("!check(x, *user)", "ct::expect(!ct::lift(check(x, *user)));", {}),
    ('f({}, {"nt"})', 'ct::expect(f({}, {"nt"}));', {}),
    ("get<Counter>() == Counter{asserted}", "ct::expect(ct::lift(get<Counter>()) == Counter{asserted});", {}),
    (
        'x, R"EOS({"origin": "app","value": 42})EOS"',
        'ct::expect(ct::lift(x) != R"EOS({"origin": "app","value": 42})EOS");',
        {"connectors": [" != "]},
    ),
    (".15 >= f()", "ct::expect(.15_d >= f());", {}),
    ("2. != f()", "ct::expect(2._d != f());", {}),
    ("0.15f >= f()", "ct::expect(0.15_f >= f());", {}),
]


@pytest.mark.parametrize("case,expected,config", testdata_lift)
def test_lift(case, expected, config):
    kwargs = {"namespace": "ct"}
    kwargs.update(config)
    assert base.lift(lines=base.split_lines(case), **kwargs) == base.split_lines(expected)
