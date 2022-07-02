# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import re
import pathlib

from dataclasses import dataclass, field
from enum import Enum


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


########################################################################################################################


@dataclass
class Token:
    Kind = Enum("Kind", ["string_literal", "operator", "call_begin", "parenthesis_begin", "end", "unknown"])

    content: str
    line_idx: int
    kind: Kind


def _tokenize(lines: list[Line]) -> list[Token]:
    # TODO: handling of ternary (=> some sort of if then else?)
    splitters = [
        (r"""(?<!\\)"(([^"]|(?<=\\)")*)(?<!\\)"|(?<!\\)'([^']|\\.)(?<!\\)'""", Token.Kind.string_literal),
        (r"\s*\b(not|&&|and|\|\||or|!=|==|(?<!\+)\+(?!\+)|(?<!-)-(?!-)|[!*/%,~])\b\s*", Token.Kind.operator),
        (r"\w[\w0-9_<>\.:]+\s*[\({]", Token.Kind.call_begin),
        (r"\(", Token.Kind.parenthesis_begin),
        (r"\)|}", Token.Kind.end),
    ]
    result = [Token(content=line.content, line_idx=l, kind=Token.Kind.unknown) for l, line in enumerate(lines)]
    for regex, kind in splitters:
        splits = []
        for t in result:
            m = re.search(regex, t.content)
            if not m or t.kind != Token.Kind.unknown:
                splits.append(t)
                continue
            start, end = m.span()
            if start > 0:
                splits.append(Token(content=t.content[:start], kind=t.kind, line_idx=t.line_idx))
            splits.append(Token(content=m.group(), kind=kind, line_idx=t.line_idx))
            if end < len(t.content):
                splits.append(Token(content=t.content[end:], kind=t.kind, line_idx=t.line_idx))
        result = splits
    return result


@dataclass
class Node:
    Kind = Enum("Kind", ["unary", "binary", "scope", "call", "raw"])

    parent: "Node"
    kind: Kind
    precedence: int = 0
    tokens: list[Token] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)


def compute_precedence(token: Token, unary: bool) -> int:
    if token.kind != Token.Kind.operator:
        return 0
    operator = token.content.strip()
    if unary and operator in {"+", "-", "~", "!", "not"}:
        return 3
    return {
        "*": 4,
        "/": 4,
        "%": 4,
        "+": 5,
        "-": 5,
        "&&": 14,
        "and": 14,
        "||": 15,
        "or": 15,
    }[operator]


def _load_tree(tokens: list[Token]) -> Node:
    root = None
    last = None

    def make_node(**kwargs):
        nonlocal root
        node = Node(**kwargs)
        if node.parent is not None:
            node.parent.children.append(node)
        else:
            root = node
        return node

    kind_map = {
        Token.Kind.unknown: Node.Kind.raw,
        Token.Kind.string_literal: Node.Kind.raw,
        Token.Kind.call_begin: Node.Kind.call,
        Token.Kind.parenthesis_begin: Node.Kind.scope,
    }
    for token in tokens:
        if token.kind in kind_map:
            last = make_node(parent=last, kind=kind_map[token.kind], precedence=0, tokens=[token])
        elif token.kind == Token.Kind.end:
            while last.kind not in {Node.Kind.scope, Node.Kind.call}:
                last = last.parent
            last.tokens.append(token)
            last = last.parent
        elif token.kind == Token.Kind.operator:
            is_unary = True  # TODO (last.kind in {...})
            precedence = compute_precedence(token, is_unary)
            parent = last
            while parent is not None and parent.precedence <= precedence:
                parent = parent.parent
            children = []
            if parent:
                children = parent.children
                parent.children = []
            assert len(children) < 2

            last = make_node(
                parent=parent,
                kind=(Node.Kind.unary if is_unary else Node.Kind.binary),
                precedence=precedence,
                tokens=[token],
                children=children,
            )
        else:
            assert False, f"Unsupported token kind: {token}"
    return root


# TODO: parse string / view literals with suffixes (s / sv)


def _lift_node(node: Node, namespace: str, **kwargs):
    # Kind = Enum("Kind", ["unary", "binary", "scope", "call", "raw"])
    # Kind = Enum("Kind", ["string_literal", "operator", "call_begin", "parenthesis_begin", "end", "unknown"])
    if node.kind == Node.Kind.raw:
        raw = node.tokens[0].content
        # TODO: support all literal types (if literals are enabled)

    node.tokens[0].content = f"{namespace}::lift({node.tokens[0].content}"
    node.tokens[-1].content += ")"


def _supports_preferential_lifting(node: Node) -> bool:
    return node.kind == Node.Kind.raw and False  # TODO


_preferential_lift_number_handlers = [
    # Integers
    (re.compile(r"^(?P<number>\d+)(?P<suffix>u|l|ul|ll|ull)?$"), {None: "i"}),
    # Floats
    (re.compile(r"^(?P<number>\d*\.\d+)(?P<suffix>f|ld)?$"), {None: "d"}),
]


def _preferential_lift_number(content: str) -> optional[str]:
    """ Compute preferential i.e. literal-based lift of content; return None iff this is not possible. """
    for rx, overrides in _preferential_lift_number_handlers:
        m = rx.match(content)
        if m:
            number, suffix = m.group("number"), m.group("suffix")
            suffix = overrides.get(suffix, suffix)
            return f"{number}_{suffix}"


def _is_automatically_lifted(node: Node) -> bool:
    return node.kind in {Node.Kind.unary, Node.Kind.binary, Node.Kind.scope}


def _lift_tree(root: Node, **kwargs) -> Node:
    # both auto -> nothing here, descend both
    # one auto: nothing
    # none auto: lift according to preference
    num_auto_lifted_children = 0
    for child in root.children:
        if _is_automatically_lifted(node=child):
            _lift_tree(root=child, **kwargs)
            num_auto_lifted_children += 1

    if num_auto_lifted_children == 0:
        lifting_preference = [
            c for c in root.children if _supports_preferential_lifting(node=c) and not _is_automatically_lifted(node=c)
        ]
        _lift_node(node=(lifting_preference + root.children)[0], **kwargs)
    return root


def _display_tree(root: Node, depth=0):
    if root is not None:
        print(f'{" " * (2 * depth + 3)} {root.precedence:02d} {root.kind} {" ".join(t.content for t in root.tokens)}')
        ignored = [_display_tree(c, depth=depth + 1) for c in root.children]
    else:
        print("   None")


def _collect_tokens(root: Node) -> list[Token]:
    if root.kind in {Node.Kind.scope, Node.Kind.call}:
        child_tokens = [_collect_tokens(root=child) for child in root.children]
        return [root.tokens[0]] + [t for ts in child_tokens for t in ts] + [root.tokens[1]]
    if root.kind == Node.Kind.raw:
        return [root.tokens[0]]
    if root.kind == Node.Kind.unary:
        return [root.tokens[0]] + _collect_tokens(root=root.children[0])
    if root.kind == Node.Kind.binary:
        return _collect_tokens(root=root.children[0]) + [root.tokens[0]] + _collect_tokens(root=root.children[1])
    assert False, "Unsupported note kind!"


def _reconstruct_lines(tokens: list[Token], original: list[Line]) -> list[Line]:
    return [
        Line(content="".join(t.content for t in tokens if t.line_idx == lid), indent=line.indent)
        for lid, line in enumerate(original)
    ]


def lift(lines: list[Line], connectors: list[str] = [], **kwargs) -> list[Line]:
    assert lines
    print(f' Input: {"~".join(l.content for l in lines)}')
    tokens = _tokenize(lines=lines)
    print(f'  Tokens: {"~".join(f"{{{t.content}-{t.kind}}}" for t in tokens)}')
    tree = _load_tree(tokens=tokens)
    _display_tree(root=tree)
    if tree.kind not in {Node.Kind.raw, Node.Kind.call}:  # at least two nodes in total
        tree = _lift_tree(root=tree, **kwargs)
    _display_tree(root=tree)
    tokens = _collect_tokens(root=tree)
    lines = _reconstruct_lines(tokens=tokens, original=lines)
    # TODO: alter toplevel commas according to connectors

    # note: we currently assume there are no comments contained.
    lines[0].content = f"{kwargs['namespace']}::expect({lines[0].content}"
    lines[-1].content += ");"
    print(f' Output: {"~".join(l.content for l in lines)}')
    return lines


# TODO: make name of alias namespace configurable
