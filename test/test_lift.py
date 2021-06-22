# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import pytest

from migrate import base


testdata = [
    ("foo()", "ct::expect(foo());"),
    ("true and foo()", "ct::expect(ct::lift(true) and foo());"),
    ("not foo()", "ct::expect(not ct::lift(foo()));"),
]


@pytest.mark.parametrize("case,expected", testdata)
def test_lift(case, expected):
    assert base.lift(lines=[base.Line(indent="", content=case)], namespace="ct")[0].content == expected
