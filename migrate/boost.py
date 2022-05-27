# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import re

from . import base, log


def _macro_pattern(name: str, *properties) -> str:
    assert properties
    body = ",".join(rf"\s*(?P<{property}>[^,)]+)\s*" for property in properties)
    return rf"{name}\({body}(?P<extra>,.+)?\)(?P<trailer>\s*" + r"{?\s*(//.*|/\*.*)?)$"


class SuiteConverter(base.SingleLineConverter):
    _rx_suite_begin = re.compile(_macro_pattern("BOOST_AUTO_TEST_SUITE", "name"))

    def handle_line(self, line: base.Line, namespace: str, **kwargs) -> base.Line:
        m = SuiteConverter._rx_suite_begin.match(line.content)
        if m:
            m = m.groupdict()
            return base.Line(
                indent=line.indent,
                content=f'auto const {m["name"]} = {namespace}::Suite{{"{m["name"]}", [] {{{m["trailer"]}',
            )

        if line.content == "BOOST_AUTO_TEST_SUITE_END()":
            return base.Line(indent=line.indent, content="}};")

        return line


class FixtureSuiteConverter(base.SingleLineConverter):
    _rx_suite_begin = re.compile(_macro_pattern("BOOST_FIXTURE_TEST_SUITE", "name", "fixture"))
    _rx_auto_case_begin = re.compile(_macro_pattern("BOOST_AUTO_TEST_CASE", "name"))
    _rx_data_case_begin = re.compile(_macro_pattern("BOOST_DATA_TEST_CASE", "name"))

    def __init__(self):
        self._fixture_stack = []

    def handle_line(self, line: base.Line, **kwargs) -> base.Line:
        if (
            self._fixture_stack
            and line.indent == self._fixture_stack[-1]["indent"]
            and line.content == "BOOST_AUTO_TEST_SUITE_END()"
        ):
            self._fixture_stack.pop()
            return line

        if self._fixture_stack:
            fixture = self._fixture_stack[-1]['fixture']
            m = FixtureSuiteConverter._rx_auto_case_begin.match(line.content)
            if m:
                line.content = f"BOOST_FIXTURE_TEST_CASE({m.group('name')}, {fixture}{self._extras(m.groupdict())}){m.group('trailer')}"
                return line
            m = FixtureSuiteConverter._rx_data_case_begin.match(line.content)
            if m:
                line.content = f"BOOST_DATA_TEST_CASE_F({fixture}, {m.group('name')}{self._extras(m.groupdict())}){m.group('trailer')}"
                return line

        m = FixtureSuiteConverter._rx_suite_begin.match(line.content)
        if not m:
            return line
        self._fixture_stack.append({"indent": line.indent, "fixture": m.group("fixture")})
        return base.Line(indent=line.indent, content=f"BOOST_AUTO_TEST_SUITE({m.group('name')})")

    def _extras(self, details: dict) -> str:
        extras = details["extra"]
        if extras is None:
            extras = ""
        return extras


class CaseConverterBase(base.MultiLineConverter):
    def __init__(self, pattern: str):
        self._rx_case_begin = re.compile(pattern)
        self._insertion = []  # List[str]: What to insert once the test really stats (not including indentation)
        self._finisher = None  # Optional[base.Line]: How the case is supposed to be ended (including indentation)

    def handle_line(self, line: base.Line, **kwargs) -> list[base.Line]:
        if self._insertion and line.content and not line.content.startswith("{"):
            return self._generate_insertion(line=line)

        if self._finisher is not None:
            assert not line.content or self._finisher.level <= line.level
            lines = [line]
            if line.level == self._finisher.level and line.content.startswith("}"):
                lines = self._generate_insertion(line=line)
                lines[-1].content = self._finisher.content + line.content[1:]
                self._finisher = None
            return lines

        m = self._rx_case_begin.match(line.content)
        if not m:
            return [line]

        content, self._insertion, self._finisher = self.handle_case(line=line, details=m.groupdict(), **kwargs)
        return [base.Line(indent=line.indent, content=content)]

    def _generate_insertion(self, line: base.Line) -> list[base.Line]:
        lines = [base.Line(indent=line.indent, content=i) for i in self._insertion] + [line]
        self._insertion = []
        return lines


class CaseConverter(CaseConverterBase):
    def __init__(self):
        super().__init__(pattern=_macro_pattern("BOOST_AUTO_TEST_CASE", "name"))

    def handle_case(self, line: base.Line, details: dict, namespace: str, use_literals: bool, **kwargs):
        if details["extra"]:
            log.warning(f"Found BOOST_AUTO_TEST_CASE with '{details['extra'].lstrip(', ')}' (unsupported and ignored).")
        if use_literals:
            content = f'"{details["name"]}"_test = []'
        else:
            content = f'{namespace}::Test{{"{details["name"]}"}} = []'
        content += details["trailer"]
        finisher = base.Line(indent=line.indent, content="};")
        return content, [], finisher


class FixtureCaseConverter(CaseConverterBase):
    def __init__(self):
        super().__init__(pattern=_macro_pattern("BOOST_FIXTURE_TEST_CASE", "name", "fixture"))

    def handle_case(self, line: base.Line, details: dict, namespace: str, use_literals: bool, **kwargs):
        if details["extra"]:
            log.warning(
                f"Found BOOST_FIXTURE_TEST_CASE with '{details['extra'].lstrip(', ')}' (unsupported and ignored)."
            )
        if use_literals:
            content = f'"{details["name"]}"_test = []'
        else:
            content = f'{namespace}::Test{{"{details["name"]}"}} = []'
        content += details["trailer"]
        finisher = base.Line(indent=line.indent, content="};")
        return content, [f'{details["fixture"]} fixture{{}};'], finisher


class EqualCollectionExpectationConverter(base.MacroCallConverter):
    _rx_macro_start = re.compile(r"BOOST_(?P<lvl>WARN|CHECK|REQUIRE)_EQUAL_COLLECTIONS\(")

    def check_start(self, content: str) -> str:
        m = self._rx_macro_start.match(content)
        return m.group()[:-1] if m else None

    def handle_macro(self, macro: str, lines: list[base.Line], **kwargs) -> list[base.Line]:
        lines = base.connect(lines=lines, connectors=[", ", "}, std::ranges::subrange{", ", "], **kwargs)
        lvl = self._rx_macro_start.match(f"{macro}(").group("lvl")
        lines[0].content = f"BOOST_{lvl}_EQUAL(std::ranges::equal(std::ranges::subrange{{{lines[0].content}"
        lines[-1].content = f"{lines[-1].content}}}));"
        return lines


class ExpectationConverter(base.MacroCallConverter):
    def __init__(self, macro, *, connectors=[], terminator: str = ""):
        super().__init__()
        self._rx_macro = re.compile(rf"{macro}\(")
        self._connectors = connectors
        self._terminator = terminator

    def handle_macro(self, macro: str, lines: list[base.Line], **kwargs) -> list[base.Line]:
        lines = base.lift(lines=lines, connectors=self._connectors, **kwargs)
        lines[-1].content = f"{lines[-1].content[:-1]}{self._terminator};"
        return lines

    def check_start(self, content: str) -> str:
        m = self._rx_macro.match(content)
        return m.group()[:-1] if m else None


def load_handlers(namespace: str):
    return [
        base.ReFilterHandler({re.compile(r'^#include\s+[<"]boost/test')}),
        base.FilterHandler(forbidden={"#define BOOST_TEST_MAIN"}),
        FixtureSuiteConverter(),
        SuiteConverter(),
        CaseConverter(),
        FixtureCaseConverter(),
        ExpectationConverter("BOOST_TEST"),
        EqualCollectionExpectationConverter(),
    ] + [
        ExpectationConverter(
            f"BOOST_{lvl}{macro}",
            connectors=connectors,
            terminator=(f" << {namespace}::{term}" if term else ""),
        )
        for lvl, term in (("WARN", "flaky"), ("CHECK", ""), ("REQUIRE", "asserted"))
        for macro, connectors in [
            ("", []),
            ("_EQUAL", [" == "]),
            ("_MESSAGE", [" << "]),
            ("_GE", [" >= "]),
            ("_GT", [" > "]),
            ("_LE", [" <= "]),
            ("_LT", [" < "]),
            ("_NE", [" != "]),
        ]
    ]
