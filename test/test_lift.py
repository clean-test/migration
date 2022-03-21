# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import pytest

from migrate import base


testdata = [
    ("foo()", "ct::expect(foo());", []),
    ("true and foo()", "ct::expect(ct::lift(true) and foo());", []),
    ("not foo()", "ct::expect(not ct::lift(foo()));", []),
    ('not not foo(), "hi" << you()', 'ct::expect(not not ct::lift(foo())) << "hi" << you();', ["<<"]),
    ('not normal()', 'ct::expect(not ct::lift(normal()));', []),
    ('n::f(a, "x") == "x"', 'ct::expect(ct::lift(n::f(a, "x")) == "x");', []),
    ('f(a, "x") == "x"', 'ct::expect(ct::lift(f(a, "x")) == "x");', []),
    ('f(g(h()))', 'ct::expect(f(g(h())));', []),
    ('a, 2', 'ct::expect(a == 2_i);', ["=="]),
]


@pytest.mark.parametrize("case,expected,connectors", testdata)
def test_lift(case, expected, connectors):
    kwargs = {"namespace": "ct", "connectors": connectors}
    assert base.lift(lines=[base.Line(indent="", content=case)], **kwargs)[0].content == expected
