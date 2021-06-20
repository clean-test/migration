# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import pytest

from migrate import base


testdata = [
    ("foo()", "foo()"),
    ("true and foo()", "ct::lift(true) and foo()"),
]


@pytest.mark.parametrize("case,expected", testdata)
def test_lift(case, expected):
    assert base.lift(lines=[base.Line(indent="", content=case)])[0].content == expected
