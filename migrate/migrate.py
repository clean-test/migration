# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import pathlib

from . import base, boost


class _StripWhitespace:
    def __call__(self, lines: list[base.Line], **kwargs):
        first_non_empty = self._first_non_empty(lines)
        if first_non_empty is None:
            return lines

        lines = lines[first_non_empty:]
        last_non_empty = self._first_non_empty(reversed(lines))
        assert last_non_empty is not None
        if last_non_empty > 0:
            lines = lines[:-last_non_empty]
        return lines

    def _first_non_empty(self, lines):
        for n, l in enumerate(lines):
            if l.content:
                return n


def load_handlers(**kwargs):
    result = []
    result += boost.load_handlers(**kwargs)
    result += [
        base.IncludeAdder(),
        _StripWhitespace(),
    ]
    return result


def convert(path: pathlib.Path, *, handlers: list = [], **kwargs):
    if not handlers:
        handlers = load_handlers(**kwargs)
    lines = base.load_lines(path)
    lines = convert_lines(lines, handlers=handlers, **kwargs)
    base.write_lines(lines, path=path)


def convert_lines(lines: list, handlers: list = [], **kwargs):
    for hdl in handlers:
        lines = hdl(lines=lines, **kwargs)
    return lines
