# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import pytest

from migrate import base


testdata = [
    ("foo()", "ct::expect(foo());", []),
    ("true and foo()", "ct::expect(ct::lift(true) and foo());", []),
    ("not foo()", "ct::expect(not ct::lift(foo()));", []),
    ('not not foo(), "hi" << you()', 'ct::expect(not not ct::lift(foo())) << "hi" << you();', ['<<']),
]


@pytest.mark.parametrize("case,expected,connectors", testdata)
def test_lift(case, expected, connectors):
    assert base.lift(lines=[base.Line(indent="", content=case)], namespace="ct", connectors=connectors)[0].content == expected
