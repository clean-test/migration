# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import re

from . import base

# TODO: tests
# TODO: lifiting of literls
# TODO: licenses


class SuiteConverter(base.SingleLineConverter):
    _rx_suite_begin = re.compile("^BOOST_AUTO_TEST_SUITE\((?P<name>\w+)\)$")

    def handle_line(self, line: base.Line, **kwargs) -> base.Line:
        m = SuiteConverter._rx_suite_begin.match(line.content)
        if m:
            m = m.groupdict()
            return base.Line(
                indent=line.indent,
                content=f'auto const {m["name"]} = {kwargs["namespace"]}::Suite{{"{m["name"]}", [] {{',
            )

        if line.content == "BOOST_AUTO_TEST_SUITE_END()":
            return base.Line(indent=line.indent, content="}};")

        return line


class CaseConverter(base.SingleLineConverter):
    _rx_case_begin = re.compile("BOOST_AUTO_TEST_CASE\((?P<name>\w+)\)")

    def __init__(self):
        self._finisher = None

    def handle_line(self, line: base.Line, **kwargs) -> base.Line:
        if self._finisher is not None:
            assert not line.content or self._finisher.level <= line.level
            if line.level == self._finisher.level and not line.content.startswith("{") and line.content:
                assert line.content.startswith("}")
                line.content = self._finisher.content + line.content[1:]
                self._finisher = None
            return line

        m = CaseConverter._rx_case_begin.match(line.content)
        if not m:
            return line
        m = m.groupdict()

        if kwargs["use_literals"]:
            content = f'"{m["name"]}"_test = []'
        else:
            content = f'{kwargs["namespace"]}::Test{{"{m["name"]}"}} = []'
        self._finisher = base.Line(indent=line.indent, content="};")
        return base.Line(indent=line.indent, content=content)


def load_handlers():
    return [
        base.ReFilterHandler({re.compile('^#include\s+[<"]boost/test')}),
        base.FilterHandler(forbidden={"#define BOOST_TEST_MAIN"}),
        SuiteConverter(),
        CaseConverter(),
    ]
