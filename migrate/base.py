# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import re
import pathlib

from dataclasses import dataclass, field


@dataclass
class Line:
    indent: str
    content: str

    def __post_init__(self):
        self.content = self.content.rstrip()

    @property
    def level(self):
        return len(self.indent)


_rx_split_line = re.compile("^(?P<indent>\s*)(?P<content>([^\s].*)?)$")


def load_lines(path: pathlib.Path):
    return [Line(**_rx_split_line.match(l).groupdict()) for l in path.read_text().splitlines()]


def write_lines(lines: list, path: pathlib.Path):
    path.write_text("".join(f"{l.indent}{l.content}\n" for l in lines))


@dataclass
class FilterHandler:
    forbidden: set[str] = field(default_factory=set)

    def __call__(self, lines: list[Line], **kwargs) -> list[Line]:
        return [l for l in lines if l.content not in self.forbidden]


@dataclass
class ReFilterHandler:
    forbidden: list[re.Pattern] = field(default_factory=list)

    def __call__(self, lines: list[Line], **kwargs) -> list[Line]:
        return [l for l in lines if not any(rx.match(l.content) for rx in self.forbidden)]


class SingleLineConverter:
    def __call__(self, lines: list[Line], **kwargs) -> list[Line]:
        return [self.handle_line(line, **kwargs) for line in lines]


class MultiLineConverter:
    def __call__(self, lines: list[Line], **kwargs) -> list[Line]:
        lines = [self.handle_line(l, **kwargs) for l in lines]
        return [line for l in lines for line in l]


class MacroCallConverter(MultiLineConverter):
    def __init__(self):
        self._buffer = []
        self._macro = None

    def handle_line(self, line: Line, **kwargs) -> list[Line]:
        is_first_macro_line = False
        result = [line]
        if not self._macro:
            self._macro = self.check_start(content=line.content)
            is_first_macro_line = True
        if self._macro:
            if is_first_macro_line:
                assert line.content[len(self._macro)] == "("
                line.content = line.content[len(self._macro) + 1 :]
            self._buffer.append(line)
            result = []
        if self._buffer and self._buffer[-1].content.endswith(");"):
            line = self._buffer[-1]
            line.content = line.content[: -len(");")]
            result = self.handle_macro(macro=self._macro, lines=self._buffer, **kwargs)
            self._buffer = []
            self._macro = None
        return result


class IncludeAdder:
    def __call__(self, lines: list[Line], **kwargs) -> list[Line]:
        include_position = self._include_position(lines)
        ct_includes = [
            Line(indent="", content="#include <clean-test/clean-test.h>"),
            Line("", ""),
        ]

        ct_extras = []
        if kwargs["use_namespace_alias"]:
            ct_extras.append(Line(indent="", content="namespace ct = clean_test;"))
        if kwargs["use_literals"]:
            ct_extras.append(Line(indent="", content=f"using namespace {kwargs['namespace']}::literals;"))
        if ct_extras:
            ct_extras.append(Line("", ""))
        lines[include_position:include_position] = ct_includes + ct_extras
        return lines

    def _include_position(self, lines):
        for n, line in enumerate(lines):
            if line.content and not line.content.startswith("#include"):
                return n
        return 0
