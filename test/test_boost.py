# Copyright (c) m8mble 2022.
# SPDX-License-Identifier: BSL-1.0

import pytest

from migrate import boost, base

import textwrap


testdata = [
    (
        """\
    BOOST_TEST(
        function(
            U"weird",
            &var)
        == U"fancy");""",
        """\
    ct::expect(
        ct::lift(function(
            U"weird",
            &var))
        == U"fancy");""",
    ),
]


def split_lines(data: str) -> list[base.Line]:
    raw_lines = data.splitlines()
    stripped_lines = [l.strip() for l in raw_lines]
    return [
        base.Line(indent=raw[: -len(stripped)], content=stripped) for raw, stripped in zip(raw_lines, stripped_lines)
    ]


def join_lines(lines: list[base.Line]) -> str:
    return "\n".join(f"{l.indent}{l.content}" for l in lines)


@pytest.mark.parametrize("case,expected", testdata)
def test_lift(case, expected):
    kwargs = {"namespace": "ct"}

    lines = split_lines(case)
    parse = boost.ExpectationConverter("BOOST_TEST")
    assert join_lines(parse(lines=lines, **kwargs)) == expected
