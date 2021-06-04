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


def _load_handlers():
    result = []
    result += boost.load_handlers()
    result += [
        base.IncludeAdder(),
        _StripWhitespace(),
    ]
    return result


_handlers = _load_handlers()


def convert(path: pathlib.Path, **kwargs):
    lines = base.load_lines(path)
    lines = convert_lines(lines, **kwargs)
    base.write_lines(lines, path=path)


def convert_lines(lines: list, **kwargs):
    for hdl in _handlers:
        lines = hdl(lines=lines, **kwargs)
    return lines
