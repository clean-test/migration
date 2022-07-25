# Copyright (c) m8mble 2022.
# SPDX-License-Identifier: BSL-1.0

import pytest

from migrate import boost, base, migrate

import textwrap


expectation_testdata = [
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


@pytest.mark.parametrize("case,expected", expectation_testdata)
def test_expectation(case, expected):
    kwargs = {"namespace": "ct"}
    parse = boost.ExpectationConverter("BOOST_TEST")
    assert parse(lines=base.split_lines(case), **kwargs) == base.split_lines(expected)


throw_testdata = [
    (
        """BOOST_REQUIRE_THROW(reduce(map, V{}), std::exception); // yea?""",
        """ct::expect(ct::throws<std::exception>([&]() { reduce(map, V{}); })) << ct::asserted; // yea?""",
    ),
    (
        """BOOST_WARN_THROW((parse(token, f())), std::runtime_error);""",
        """ct::expect(ct::throws<std::runtime_error>([&]() { (parse(token, f())); })) << ct::flaky;""",
    ),
    (
        """\
        BOOST_CHECK_NO_THROW(
            const auto c = compute(a, b);
            ct::expect(c == "really?");
        );""",
        """\
        ct::expect(not ct::throws([&]() {
            const auto c = compute(a, b);
            ct::expect(c == "really?");
        }));""",
    ),
    (
        """\
        BOOST_REQUIRE_NO_THROW(
            const auto c = compute(a, b);

            ct::expect(c == "really?");
        );""",
        """\
        ct::expect(not ct::throws([&]() {
            const auto c = compute(a, b);

            ct::expect(c == "really?");
        })) << ct::asserted;""",
    ),
]


@pytest.mark.parametrize("case,expected", throw_testdata)
def test_throw(case, expected):
    kwargs = {"namespace": "ct"}
    parsers = [
        boost.ThrowExpectationConverter("BOOST_REQUIRE_THROW", terminator=f" << {kwargs['namespace']}::asserted"),
        boost.ThrowExpectationConverter("BOOST_WARN_THROW", terminator=f" << {kwargs['namespace']}::flaky"),
        boost.ThrowExpectationConverter("BOOST_CHECK_NO_THROW"),
        boost.ThrowExpectationConverter("BOOST_REQUIRE_NO_THROW", terminator=f" << {kwargs['namespace']}::asserted"),
    ]
    lines = migrate.convert_lines(lines=base.split_lines(case), handlers=parsers, **kwargs)
    assert lines == base.split_lines(expected)
