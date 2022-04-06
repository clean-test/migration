# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import pytest

from migrate import base


testdata_connect = [
    ("a, b", "a << b", [" << "]),
    ("a, b, c, d", "a, b}, X{c, d", [", ", "}, X{", ", "]),
]


@pytest.mark.parametrize("case,expected,connectors", testdata_connect)
def test_connect(case, expected, connectors):
    assert base.connect(lines=[base.Line(indent="", content=case)], connectors=connectors)[0].content == expected


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
]


@pytest.mark.parametrize("case,expected,config", testdata_lift)
def test_lift(case, expected, config):
    kwargs = {"namespace": "ct"}
    kwargs.update(config)
    assert base.lift(lines=base.split_lines(case), **kwargs) == base.split_lines(expected)
