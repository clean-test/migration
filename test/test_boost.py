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
    (
        """\
        {
            BOOST_TEST(true);
            BOOST_TEST(false);
        }
        """,
        """\
        {
            ct::expect(true);
            ct::expect(false);
        }
        """,
    ),
    (
        """\
        {
            BOOST_TEST(true);
            foo;
            BOOST_TEST(false);
        }
        """,
        """\
        {
            ct::expect(true);
            foo;
            ct::expect(false);
        }
        """,
    ),
    (
        """\
        {
            BOOST_TEST(true);

            BOOST_TEST(false);
        }
        """,
        """\
        {
            ct::expect(true);

            ct::expect(false);
        }
        """,
    ),
    (
        """\
        BOOST_TEST(true);


        x = f();
        """,
        """\
        ct::expect(true);


        x = f();
        """,
    ),
]


@pytest.mark.parametrize("case,expected", testdata)
def test_lift(case, expected):
    kwargs = {"namespace": "ct"}
    parse = boost.ExpectationConverter("BOOST_TEST")
    assert parse(lines=base.split_lines(case), **kwargs) == base.split_lines(expected)
